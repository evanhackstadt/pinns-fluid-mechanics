# pinn.py

"""
2D Stenosis Geometry-Conditioned PINN
    Navier-Stokes PINN using DeepXDE library
    Functions to define BCs, data, PDE loss, building, training, etc.

Evan Hackstadt
Rugonyi Lab
"""


import json
import time, datetime
from pathlib import Path

import numpy as np
import torch
import deepxde as dde

from data import build_interior_dataset, build_labeled_dataset


# ———————————— GLOBAL CONSTANTS ————————————
# need these for functions called by DeepXDE, since it can't pass more args

# declare vars with placeholder:
L = H_MAX = X_C = Y_C = U_IN_MAX = P_OUT = U_REF = RE = 0

def set_global_constants(cfg):
    globals()['L']        = cfg.L
    globals()['H_MAX']    = cfg.H_max
    globals()['X_C']      = cfg.x_c
    globals()['Y_C']      = cfg.y_c
    globals()['U_IN_MAX'] = cfg.u_in_max
    globals()['P_OUT']    = cfg.P_out
    globals()['U_REF']    = cfg.U_ref
    globals()['RE']       = cfg.Re


# ———————————— PINN HELPER FUNCTIONS ————————————

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

# Inlet x-velocity profile
def inlet_u(x):
    y = x[:, 1:2]
    # Poiseuille parabola, nondimensionalized by U_ref
    # Zero at y=0 and y=H_MAX, peak = u_in_max/U_ref at y=H_MAX/2
    return (U_IN_MAX / U_REF) * 4.0 * (y / H_MAX) * (1.0 - y / H_MAX)


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


# --- Helper: Construct PointSetBCs ---
def build_pointsetbcs(boundary_data, cfg):
    
    # Manually extract relevant points for each BC (inlet, outlet, walls)
    tol = 1e-8
    x = boundary_data[:, 0]
    y = boundary_data[:, 1]

    inlet_mask = np.isclose(x, -cfg.L / 2.0, atol=tol)
    outlet_mask = np.isclose(x, cfg.L / 2.0, atol=tol)
    wall_mask = np.isclose(y, 0.0, atol=tol) | np.isclose(y, cfg.H_max, atol=tol)

    inlet_pts = boundary_data[inlet_mask]
    outlet_pts = boundary_data[outlet_mask]
    wall_pts = boundary_data[wall_mask]

    if inlet_pts.size == 0:
        raise ValueError("No inlet boundary points found in boundary_data.")
    if outlet_pts.size == 0:
        raise ValueError("No outlet boundary points found in boundary_data.")
    if wall_pts.size == 0:
        raise ValueError("No wall boundary points found in boundary_data.")

    # Define BCs using PointSetBCs
    inlet_vals_u = inlet_u(inlet_pts[:, :2])
    inlet_vals_v = np.zeros((inlet_pts.shape[0], 1))
    wall_vals = np.zeros((wall_pts.shape[0], 1))
    outlet_vals_p = np.full((outlet_pts.shape[0], 1), cfg.P_out / cfg.U_ref ** 2)

    bc_inlet_u  = dde.PointSetBC(inlet_pts,  inlet_vals_u,  component=0)    # parabolic profile
    bc_inlet_v  = dde.PointSetBC(inlet_pts,  inlet_vals_v,  component=1)    # v=0
    bc_wall_u   = dde.PointSetBC(wall_pts,   wall_vals,     component=0)    # u=0
    bc_wall_v   = dde.PointSetBC(wall_pts,   wall_vals,     component=1)    # v=0
    bc_outlet_p = dde.PointSetBC(outlet_pts, outlet_vals_p, component=2)    # p=0 (from config)

    return [bc_inlet_u, bc_inlet_v, bc_wall_u, bc_wall_v, bc_outlet_p]


# --- Loss Re-Weighter custom callback ---
class LossMagnitudeReweighter(dde.callbacks.Callback):
    """
    Reweights loss terms so their magnitudes stay balanced. 
    Inspired by weight annealing (Wang et al. 2021).
    Every `period` steps, sets weight_i = median_loss_reference / median_loss_i,
    where reference is the geometric mean across terms.
    """
    def __init__(self, period=2000, alpha=0.8, min_w=0.1, max_w=500.0):
        super().__init__()
        self.period = period
        self.alpha  = alpha
        self.min_w  = min_w
        self.max_w  = max_w

    def on_epoch_end(self):
        it = self.model.train_state.iteration
        if it % self.period != 0 or it == 0:
            return

        loss_arr = np.array(self.model.train_state.loss_train)   # (n_terms,)
        if loss_arr.ndim != 1 or np.any(loss_arr <= 0):
            return

        ref       = np.exp(np.mean(np.log(loss_arr)))            # geometric mean
        new_w     = np.clip(ref / loss_arr, self.min_w, self.max_w)
        old_w     = np.array(self.model.loss_weights, dtype=float)
        blended   = self.alpha * old_w + (1 - self.alpha) * new_w

        self.model.loss_weights = blended.tolist()
        print(f"\n[AdaptiveWeights @ iter {it}] {np.round(blended, 2).tolist()}")




# ———————————— PINN WORKHORSE FUNCTIONS / ENTRY POINTS ————————————

# --- Instantiate Model Object ---
def build_model(interior_data, boundary_data, labeled_data, cfg):
    """
    Constructs model object based on geometry, BCs, data, and network config.
    Args:
        interior_data: array of shape (n_interior * n_train_geometries, 4) = [x, y, a, b]
        boundary_data: array of shape (n_boundary * n_train_geometries, 4) = [x, y, a, b]
        labeled_data: array of shape (n_labeled_train, 7) = [x, y, a, b, u, v, p]
        cfg: custom config class object
    Returns:
        DeepXDE model object built with a PDE dataset and boundary conditions.
    """

    # inlet_u() requires global constants
    set_global_constants(cfg)
    
    bcs = build_pointsetbcs(boundary_data, cfg)

    # Add labeled data to BCs
    if labeled_data is not None and labeled_data.shape[0] > 0:
        obs_xyab = labeled_data[:, 0:4]
        obs_u = labeled_data[:, 4:5]
        obs_v = labeled_data[:, 5:6]
        obs_p = labeled_data[:, 6:7]

        bc_obs_u = dde.PointSetBC(obs_xyab, obs_u, component=0)
        bc_obs_v = dde.PointSetBC(obs_xyab, obs_v, component=1)
        bc_obs_p = dde.PointSetBC(obs_xyab, obs_p, component=2)

        bcs.extend([bc_obs_u, bc_obs_v, bc_obs_p])

    # Build base geometry. Pretend 4D to match input dimensionality (x,y,a,b)
    a_min = np.min(interior_data[:, 2])
    a_max = np.max(interior_data[:, 2])
    b_min = np.min(interior_data[:, 3])
    b_max = np.max(interior_data[:, 3])
    geometry = dde.geometry.Hypercube(
        xmin=[-cfg.L/2, 0.0, a_min, b_min],
        xmax=[ cfg.L/2, cfg.H_max, a_max, b_max]
    )

    # Instantiate data and network objects    
    data = dde.data.PDE(
        geometry=geometry,
        pde=pde_loss,
        bcs=bcs,
        num_domain=0,
        num_boundary=0,
        num_test=cfg.n_test,
        anchors=interior_data
    )
    
    net = dde.nn.FNN(
        layer_sizes=cfg.layers,
        activation="tanh",
        kernel_initializer="Glorot uniform",
    )

    return dde.Model(data, net)
    

# --- Core Training Function ---
def train_model(model, model_prefix, cfg):
    """
    Constructs model object and trains until convergence, saving model and metadata.
    Args:
        model: DeepXDE model object instantiated with data and network
        model_prefix: filename prefix for saved model, preferrably an absolute path
        cfg: custom config class object
    Returns:
        loss_history: DeepXDE loss history object of all training
    """
    # pde_loss(), inlet(), outlet(), and walls() need global constants
    set_global_constants(cfg)
    
    # exclude last 3 loss weights if we don't have labeled points
    loss_weights = cfg.loss_weights_adam[:-3] if cfg.n_labeled_train <= 0 else cfg.loss_weights_adam
    model.compile("adam", lr=cfg.lr, loss_weights=loss_weights)


    # FIRST TRAINING (Adam)
    
    # Create callbacks
    resampler = dde.callbacks.PDEPointResampler(period=1000)   # resample training pts at difficult areas (RAR) every 1000 iterations
    reweighter = LossMagnitudeReweighter(period=2000)   # balance loss weights every 2000 iteations

    # Train
    start_time = time.time()
    start_timestamp = datetime.datetime.now().isoformat()
    
    loss_history_1, train_state_1 = model.train(iterations=cfg.n_adam,
                                                callbacks=[resampler, reweighter],
                                                display_every=1000)
    handoff_loss_weights = model.loss_weights   # pass to L-BFGS
    
    
    # SECOND TRAINING (L-BFGS)
    
    # Set params
    model.compile("L-BFGS", loss_weights=handoff_loss_weights)
    # dde.config.set_default_float("float64")       # causes MPS errors
    dde.optimizers.config.set_LBFGS_options(gtol=cfg.gtol_lbfgs,
                                            ftol=cfg.ftol_lbfgs,
                                            maxiter=cfg.n_lbfgs,
                                            maxfun=cfg.n_lbfgs * 10)
    
    # Train
    loss_history_2, train_state_2 = model.train(callbacks=[resampler, reweighter],
                                                display_every=1000,
                                                model_save_path=model_prefix)

    end_time = time.time()
    elapsed_seconds = int(end_time - start_time)
    elapsed_minutes, elapsed_seconds_remainder = divmod(elapsed_seconds, 60)
    
    
    # Log results
    dde.saveplot(loss_history_2, train_state_2, 
                 issave=True, isplot=False, output_dir=str(cfg.pinn_dir))
    
    # Log training metadata
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
        }
    
    metadata_path = cfg.pinn_dir / "training_log.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    
    print(f"Training completed in {elapsed_minutes}m {elapsed_seconds_remainder}s")
    print(f"Saved config and training metadata")
    
    return model


# --- Restore a Model ---
def restore_model(model, model_prefix, cfg):
    """
    Restores a saved model based on model_prefix and returns it.
    Args:
        model: DeepXDE model object instantiated with data and network
        model_prefix: filename prefix for saved model, preferrably an absolute path
        cfg: custom config class object
    Returns:
        model: DeepXDE model object, restored
    """
    
    # Must compile before restore
    model.compile("adam", lr=cfg.lr, loss_weights=cfg.loss_weights_adam)

    # Find the latest saved checkpoint
    model_prefix = Path(model_prefix)
    checkpoints = list(model_prefix.parent.glob(f"{model_prefix.name}-*.pt"))
    latest = max(checkpoints, key=lambda p: int(p.stem.split("-")[-1]))
    
    # Manually load only the network weights
    # DeepXDE model.restore() loads optimizer state, causing errors
    checkpoint = torch.load(latest, map_location="cpu")
    model.net.load_state_dict(checkpoint["model_state_dict"])
    print(f"Restored weights from {latest}")

    return model


# --- One-Shot Prediction ---
def pinn_predict(model, query):
    """
    Computes h(x) for query points and passes through the model, returning its predictions.
    Args:
        model: DeepXDE model object used for prediction
        query: model input array of shape (N, 4) = [x,y,a,b]
    Returns: ndarray of shape (N, 5) with columns = [x, y, u_pred, v_pred, p_pred]
    """
    
    query_f32 = query.astype(np.float32)
    pred = model.predict(query_f32)    # (N, 3) = (u,v,p)
    
    return np.concatenate([query_f32, pred], axis=1)


# --- Fine-Tuning ---
def pinn_finetune(pretrained_model, observed_data,
                  obs_xy,                 # (3, 2) array of observation locations
    obs_u,                  # (3, 1) observed u-velocity
    cfg, a, b,
    n_finetune=1000,
    lambda_pde=10.0,
    lambda_obs=100.0,
    lambda_anchor=1.0,
    layers_to_adapt=[-1, -2],   # indices into net.linears
):
    """
    Fine-tunes a generally-trained model to patient-specific observations; optionally anchor weights.
    Args:
        pretrained_model: DeepXDE model from general training across geometries
        observed_data: possibly-sparse array of shape (m_observations, 7) = [x, y, a, b, u, v, p]
    """
    # Snapshot pretrained weights
    theta_0 = {name: p.clone().detach() 
                for name, p in pretrained_model.net.named_parameters()}
    
    # Optionally freeze early layers
    for i, layer in enumerate(pretrained_model.net.linears):
        if i not in layers_to_adapt:
            for param in layer.parameters():
                param.requires_grad = False
    
    # Anchor regularization callback
    class AnchorRegCallback(dde.callbacks.Callback):
        def on_batch_begin(self):
            anchor_loss = sum(
                ((p - theta_0[n]) ** 2).sum()
                for n, p in self.model.net.named_parameters()
                if p.requires_grad
            )
            # inject as additional loss term -- requires custom training loop
            # or handled via loss_weights and a PointSetBC trick
    
    # Observation BC
    bc_obs = dde.PointSetBC(obs_xy, obs_u, component=0)
    
    # Rebuild data with new geometry + observation BC
    new_geometry = dde.geometry.Hypercube(xmin=[-cfg.L/2, 0.0, a, b],
                                          xmax=[ cfg.L/2, cfg.H_max, a, b])
    data = dde.data.PDE(geometry=new_geometry, pde=pde_loss, bcs=[])
    
    pretrained_model.data = data
    pretrained_model.compile("adam", lr=1e-4, 
                             loss_weights=[lambda_pde]*3 + [...] + [lambda_obs])
    pretrained_model.train(iterations=n_finetune)
    
    return pretrained_model