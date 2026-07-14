# pinn.py

"""
2D Parameterized Stenosis
    Navier-Stokes PINN using DeepXDE library
    Functions to define BCs, PDE loss, building, training, etc.

Evan Hackstadt
Rugonyi Lab
"""


import os
import re
import json
import time, datetime
import glob

import numpy as np
import torch
import deepxde as dde

from config import StenosisConfig

# ———————————— GLOBAL CONSTANTS ————————————
# need these for functions called by DeepXDE, since it can't pass more args
L = H_MAX = X_C = Y_C = P1 = P2 = RE = A = B = 0    # placeholder

def set_global_constants(cfg, a, b):
    globals()['L']     = cfg.L
    globals()['H_MAX'] = cfg.H_max
    globals()['X_C']   = cfg.x_c
    globals()['Y_C']   = cfg.y_c
    globals()['P1']    = cfg.P1
    globals()['P2']    = cfg.P2
    globals()['RE']    = cfg.Re
    globals()['A']     = a
    globals()['B']     = b


# ———————————— PINN FUNCTIONS ————————————

# --- Define the Boundary Conditions (BCs) ---
# no need to define ellipse here since DeepXDE samples BC pts from the defined CSG geometry

# Left wall
def inlet(x, on_boundary):
    return on_boundary and (np.isclose(x[0], -L/2))

# Right wall
def outlet(x, on_boundary):
    return on_boundary and (np.isclose(x[0], L/2))

# Top/Bottom walls
def walls(x, on_boundary):
    return on_boundary and not inlet(x, on_boundary) and not outlet(x, on_boundary)


# --- Define the PDE Residual ---
def pde_loss(x, u):
    """
    x: collocation points (x, y)
    u: model output (u, v, p) = (x-vel, y-vel, pressure)
    Returns the residual between the model-predicted values and the governing PDEs.
    """
    # unpack data
    u_pred = u[:, 0:1]
    v_pred = u[:, 1:2]
    p_pred = u[:, 2:3]
    
    # compute derivatives using auto-diff
    du_x = dde.grad.jacobian(u, x, i=0, j=0)
    du_y = dde.grad.jacobian(u, x, i=0, j=1)
    dv_x = dde.grad.jacobian(u, x, i=1, j=0)
    dv_y = dde.grad.jacobian(u, x, i=1, j=1)
    dp_x = dde.grad.jacobian(u, x, i=2, j=0)
    dp_y = dde.grad.jacobian(u, x, i=2, j=1)
    
    du_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
    
    # compute residuals per Navier-Stokes
    continuity = du_x + dv_y
    x_momentum = u_pred*du_x + v_pred*du_y + dp_x - (1/RE)*(du_xx + du_yy)
    y_momentum = u_pred*dv_x + v_pred*dv_y + dp_y - (1/RE)*(dv_xx + dv_yy)
    
    # return a list of residuals
    return [continuity, x_momentum, y_momentum]


# --- Add Geometry to Inputs ---
def augment_inputs(x: torch.Tensor):
    """
    Concatenates channel height parameter to each row.
    Args:
        x: tensor of coordinates (N, 2), cols [x_coord, y_coord]
    
    Return: augmented input tensor (N, 3), cols [x, y, h(x)]
    """
    dtype  = x.dtype    # respect whatever DeepXDE passes in (float32 on MPS)
    device = x.device
    
    x_coords = x[:, 0:1]
    y_coords = x[:, 1:2]
    dx = x_coords - X_C
    
    # compute h(x) using tensors - same logic as ellipse_bottom()
    radicand = (1.0 - (dx / A) ** 2).clamp(min=1e-6)
    y_ellipse = torch.tensor(Y_C, dtype=dtype, device=device) - B * torch.sqrt(radicand)
    
    # Smooth mask: 1 inside ellipse footprint, 0 outside
    # Use sigmoid approximation instead of hard torch.where for differentiability
    sharpness = torch.tensor(1000.0, dtype=dtype, device=device)
    a_tensor  = torch.tensor(A,      dtype=dtype, device=device)
    H_tensor  = torch.tensor(H_MAX,  dtype=dtype, device=device)
    
    inside = torch.sigmoid(sharpness * (a_tensor - dx.abs()))
    
    h = inside * y_ellipse + (1.0 - inside) * H_tensor.expand_as(x_coords)
    
    return torch.cat([x_coords, y_coords, h], dim=1)


# --- Custom Callback to save the model at specified epochs ---
class EpochSaver(dde.callbacks.Callback):
    def __init__(self, cfg, save_prefix):
        super().__init__()
        self.iterations_to_save = cfg.iterations_to_save
        self.save_prefix = save_prefix

    def on_epoch_end(self):
        current_iter = self.model.train_state.iteration
        if current_iter in self.iterations_to_save:
            print(f"Saving epoch {current_iter}...")
            self.model.save(self.save_prefix)


# --- Instantiate Model Object ---
def build_model(cfg, a, b):
    """
    Constructs model object based on geometry, BCs, data, and network config.
    Args:
        cfg: custom config class object
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
    Returns:
        model: DeepXDE model object
    """
    
    # augment_inputs() needs global constnats
    set_global_constants(cfg, a, b)
    
    # Construct the geometry: base channel rectangle - obstructing ellipse
    channel = dde.geometry.Rectangle([-cfg.L/2, 0], [cfg.L/2, cfg.H_max])
    obstruction = dde.geometry.Ellipse([cfg.x_c, cfg.y_c], a, b, cfg.angle)
    geometry = dde.geometry.CSGDifference(channel, obstruction)

    # Define the boundary conditions
    bc_inlet_p = dde.DirichletBC(geometry, lambda x: cfg.P1, inlet, component=2)    # inlet pressure=1.0
    bc_outlet_p = dde.DirichletBC(geometry, lambda x: cfg.P2, outlet, component=2)   # outlet pressure=0.0
    bc_wall_u = dde.DirichletBC(geometry, lambda x: 0, walls, component=0)      # no-slip walls
    bc_wall_v = dde.DirichletBC(geometry, lambda x: 0, walls, component=1)

    # Instantiate data object
    data = dde.data.PDE(
            geometry=geometry,
            pde=pde_loss,
            bcs=[bc_inlet_p, bc_outlet_p, bc_wall_u, bc_wall_v],
            num_domain=cfg.n_interior,
            num_boundary=cfg.n_boundary,
            num_test=cfg.n_test
        )

    # Instantiate network object
    net = dde.nn.FNN(
            layer_sizes=cfg.layers,
            activation="tanh",
            kernel_initializer="Glorot uniform"
            )
    net.apply_feature_transform(augment_inputs)     # add h to (x,y) input data

    # Build the model
    return dde.Model(data, net)
    

# --- Core Training Function ---
def train_model(cfg, a, b, model_prefix):
    """
    Constructs model object and trains until convergence, saving model and metadata.
    Args:
        cfg: custom config class object
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
        model_prefix: filename prefix for saved model, preferrably an absolute path
        model (optional): pre-built DeepXDE model object
    Returns:
        loss_history: DeepXDE loss history object of all training
    """
    # pde_loss(), inlet(), outlet(), and walls() need global constants
    set_global_constants(cfg, a, b)
    
    # Build model
    model = build_model(cfg, a, b)
    
    model.compile("adam", lr=cfg.lr, loss_weights=cfg.loss_weights_adam)


    # FIRST TRAINING (Adam)
    
    # Create callbacks
    epoch_saver = EpochSaver(cfg, model_prefix)
    resampler = dde.callbacks.PDEPointResampler(period=1000)   # RAR - resamples more training pts at difficult areas

    # Train
    start_time = time.time()
    start_timestamp = datetime.datetime.now().isoformat()
    
    loss_history_1, train_state_1 = model.train(iterations=cfg.n_adam,
                                                callbacks=[epoch_saver, resampler],
                                                display_every=1000,
                                                model_save_path=model_prefix)
    
    
    # SECOND TRAINING (L-BFGS)
    
    # Set params
    model.compile("L-BFGS", loss_weights=cfg.loss_weights_lbfgs)
    # dde.config.set_default_float("float64")       # causes MPS errors on pinn_predict
    dde.optimizers.config.set_LBFGS_options(gtol=cfg.gtol_lbfgs,
                                            ftol=cfg.ftol_lbfgs,
                                            maxiter=cfg.n_lbfgs,
                                            maxfun=cfg.n_lbfgs * 10)
    
    # Train
    loss_history_2, train_state_2 = model.train(display_every=1000,
                                                model_save_path=model_prefix,)

    end_time = time.time()
    elapsed_seconds = int(end_time - start_time)
    elapsed_minutes, elapsed_seconds_remainder = divmod(elapsed_seconds, 60)
    
    # Log results
    dde.saveplot(loss_history_2, train_state_2, 
                 issave=True, isplot=False, output_dir=cfg.case_dirs(a,b)["pinn"])
    
    metadata = {
            "start_timestamp": start_timestamp,
            "end_timestamp": datetime.datetime.now().isoformat(),
            "elapsed_time_seconds": elapsed_seconds,
            "elapsed_time": f"{elapsed_minutes}m {elapsed_seconds_remainder}s",
            "training_iterations_adam": cfg.n_adam,
            "training_iterations_lbfgs": getattr(train_state_2, "iteration", None) - cfg.n_adam,
            "training_iterations_total": getattr(train_state_2, "iteration", None),
            "adam_steps": len(getattr(loss_history_1, "steps", [])),
            "lbfgs_steps": len(getattr(loss_history_2, "steps", [])) - len(getattr(loss_history_1, "steps", [])),
            "total_steps": len(getattr(loss_history_2, "steps", [])),
            "num_domain_points": cfg.n_interior,
            "num_boundary_points": cfg.n_boundary,
            "num_test_points": cfg.n_test
        }
    config = {
        "L": cfg.L,
        "H_max": cfg.H_max,
        "x_c": cfg.x_c,
        "y_c": cfg.y_c,
        "angle": cfg.angle,
        "a": a,
        "b": b,
        "Re": cfg.Re,
        "P1": cfg.P1,
        "P2": cfg.P2,
    }
    
    metadata_path = os.path.join(cfg.case_dirs(a,b)["pinn"], "training_log.json")
    config_path = os.path.join(cfg.case_dirs(a,b)["base"], "config_log.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training completed in {elapsed_minutes}m {elapsed_seconds_remainder}s")
    print(f"Saved config and training metadata")
    
    return model, loss_history_2


# --- Restore a Model ---
def restore_model(cfg, a, b, model_prefix):
    """
    Restores a saved model based on model_prefix and returns it.
    Args:
        model: DeepXDE model object (instantiated with matching Data and Net)
        cfg: custom config class object
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
        model_prefix: filename prefix for saved model, preferrably an absolute path
    Returns:
        model: DeepXDE model object, restored
    """
    
    # Must compile before restore
    model = build_model(cfg, a, b)
    model.compile("adam", lr=cfg.lr, loss_weights=cfg.loss_weights_adam)

    # Find the latest saved checkpoint
    checkpoints = glob.glob(f"{model_prefix}-*.pt")
    latest = max(checkpoints, key=lambda f: int(f.split("-")[-1].split(".")[0]))
    
    # Manually load only the network weights
    # DeepXDE model.restore() loads optimizer state, causing errors
    checkpoint = torch.load(latest, map_location="cpu")
    model.net.load_state_dict(checkpoint["model_state_dict"])
    print(f"Restored weights from {latest}")

    return model


# --- Get Model Outputs ---
def pinn_predict(model, query, cfg, a, b):
    """
    Computes h(x) for query points and passes through the model, returning its predictions.
    Args:
        model: DeepXDE model object used for prediction
        query: array of N (x,y) points, shape (N, 2)
        cfg: custom config class object
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
    Returns: ndarray of shape (N, 5) with columns = [x, y, u_pred, v_pred, p_pred]
    """
    
    query_f32 = query.astype(np.float32)
    pred = model.predict(query_f32)    # (N, 3) = (u,v,p)
    
    return np.concatenate([query_f32, pred], axis=1)