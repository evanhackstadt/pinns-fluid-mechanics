# pinn.py

"""
2D Parameterized Stenosis
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


# ———————————— GLOBAL CONSTANTS ————————————
# need these for functions called by DeepXDE, since it can't pass more args

# declare vars with placeholder:
L = H_MAX = X_C = Y_C = U_IN_MAX = P_OUT = U_REF = RE = A = B = 0

def set_global_constants(cfg, a, b):
    globals()['L']        = cfg.L
    globals()['H_MAX']    = cfg.H_max
    globals()['X_C']      = cfg.x_c
    globals()['Y_C']      = cfg.y_c
    globals()['U_IN_MAX'] = cfg.u_in_max
    globals()['P_OUT']    = cfg.P_out
    globals()['U_REF']    = cfg.U_ref
    globals()['RE']       = cfg.Re
    globals()['A']        = a
    globals()['B']        = b


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


# --- Sample Labeled Data ---
def sample_labeled_data(fem_data, n_points, cfg, uniform_frac=0.3):
    """
    Sample labeled FEM points biased toward high-gradient regions.
    Args:
        fem_data:     (M, 5) array [x, y, u, v, p], ellipse-masked
        n_points:     total number of points to sample
        cfg:          config object
        uniform_frac: fraction of n_points drawn uniformly (rest drawn by gradient score)
    Returns:
        labeled_data: (n_points, 5) subset of fem_data
    """
    
    if n_points <= 0:
        return None
    
    M = len(fem_data)
    u, v, p = fem_data[:, 2], fem_data[:, 3], fem_data[:, 4]

    # Approximate pointwise gradient magnitudes using finite differences on
    # the flat (unstructured) masked array. This is a rough proxy.
    du = np.abs(np.gradient(u)) + np.abs(np.gradient(np.gradient(u)))
    dv = np.abs(np.gradient(v)) + np.abs(np.gradient(np.gradient(v)))
    dp = np.abs(np.gradient(p))

    raw_scores = du + dv + dp
    
    # Clip outliers before normalizing to avoid one extreme point dominating
    raw_scores = np.clip(raw_scores, 0, np.percentile(raw_scores, 95))
    scores = raw_scores / raw_scores.sum()

    # Split sample budget
    n_uniform = int(n_points * uniform_frac)
    n_scored  = n_points - n_uniform

    # Sample indices
    generator = np.random.default_rng(seed=cfg.seed)
    idx_uniform = np.random.Generator.choice(generator, M, size=n_uniform, replace=False)
    idx_scored  = np.random.Generator.choice(generator, M, size=n_scored, replace=False, p=scores)

    # Merge and deduplicate
    all_idx = np.unique(np.concatenate([idx_scored, idx_uniform]))

    # If deduplication reduced count, top up with additional uniform samples
    if len(all_idx) < n_points:
        remaining = np.setdiff1d(np.arange(M), all_idx)
        top_up = np.random.Generator.choice(generator, remaining, size=n_points-len(all_idx), replace=False)
        all_idx = np.concatenate([all_idx, top_up])

    return fem_data[all_idx]


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
def build_model(fem_data, cfg, a, b, n_labeled):
    """
    Constructs model object based on geometry, BCs, data, and network config.
    Args:
        fem_data: ground-truth array of shape (N, 5) with columns = [x, y, u_fem, v_fem, p_fem]
        cfg: custom config class object
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
        n_labeled: number of labeled data points to sample
    Returns:
        model, labeled_pts: tuple containing DeepXDE model object, labeled data sampled from FEM data, shape (N, 5) with columns = [x, y, u_fem, v_fem, p_fem]
    """
    
    # augment_inputs() needs global constnats
    set_global_constants(cfg, a, b)
    
    # Construct the geometry: base channel rectangle - obstructing ellipse
    channel = dde.geometry.Rectangle([-cfg.L/2, 0], [cfg.L/2, cfg.H_max])
    obstruction = dde.geometry.Ellipse([cfg.x_c, cfg.y_c], a, b, cfg.angle)
    geometry = dde.geometry.CSGDifference(channel, obstruction)

    # Define the boundary conditions
    bc_inlet_u  = dde.DirichletBC(geometry, inlet_u, inlet, component=0)         # inlet x-velocity=parabolic profile
    bc_inlet_v  = dde.DirichletBC(geometry, lambda x: 0, inlet, component=1)     # inlet y-velocity=0
    bc_wall_u   = dde.DirichletBC(geometry, lambda x: 0, walls, component=0)      # no-slip walls
    bc_wall_v   = dde.DirichletBC(geometry, lambda x: 0, walls, component=1)
    bc_outlet_p = dde.DirichletBC(geometry, lambda x: cfg.P_out/cfg.U_ref**2, outlet, component=2)   # outlet pressure=0.0
    
    bcs = [bc_inlet_u, bc_inlet_v, bc_wall_u, bc_wall_v, bc_outlet_p]
    
    # Sample labeled data points
    labeled_pts = sample_labeled_data(fem_data, n_labeled, cfg, uniform_frac=0.3)
    
    if n_labeled > 0:
        obs_xy  = labeled_pts[:, 0:2]   # (N, 2)
        obs_u   = labeled_pts[:, 2:3]   # (N, 1)
        obs_v   = labeled_pts[:, 3:4]
        obs_p   = labeled_pts[:, 4:5]

        bc_obs_u = dde.PointSetBC(obs_xy, obs_u, component=0)
        bc_obs_v = dde.PointSetBC(obs_xy, obs_v, component=1)
        bc_obs_p = dde.PointSetBC(obs_xy, obs_p, component=2)
        
        bcs.append(bc_obs_u)
        bcs.append(bc_obs_v)
        bcs.append(bc_obs_p)
    

    # Instantiate data object
    data = dde.data.PDE(
            geometry=geometry,
            pde=pde_loss,
            bcs=bcs,
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
    return dde.Model(data, net), labeled_pts
    

# --- Core Training Function ---
def train_model(fem_data, cfg, a, b, n_labeled, model_prefix):
    """
    Constructs model object and trains until convergence, saving model and metadata.
    Args:
        fem_data: ground-truth array of shape (N, 5) with columns = [x, y, u_fem, v_fem, p_fem]
        cfg: custom config class object
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
        n_labeled: number of labeled data points to sample
        model_prefix: filename prefix for saved model, preferrably an absolute path
        model (optional): pre-built DeepXDE model object
    Returns:
        loss_history: DeepXDE loss history object of all training
    """
    # pde_loss(), inlet(), outlet(), and walls() need global constants
    set_global_constants(cfg, a, b)
    
    # Build model
    model, labeled_pts = build_model(fem_data, cfg, a, b, n_labeled)
    
    # exclude last 3 loss weights if we don't have labeled points
    loss_weights = cfg.loss_weights_adam[:-3] if n_labeled <= 0 else cfg.loss_weights_adam
    model.compile("adam", lr=cfg.lr, loss_weights=loss_weights)


    # FIRST TRAINING (Adam)
    
    # Create callbacks
    resampler = dde.callbacks.PDEPointResampler(period=1000)   # RAR - resamples more training pts at difficult areas
    reweighter = LossMagnitudeReweighter()

    # Train
    start_time = time.time()
    start_timestamp = datetime.datetime.now().isoformat()
    
    loss_history_1, train_state_1 = model.train(iterations=cfg.n_adam,
                                                callbacks=[resampler, reweighter],
                                                display_every=1000,
                                                model_save_path=model_prefix)
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
    pinn_dir = Path(cfg.case_dirs(a, b, n_labeled)["pinn"])
    
    dde.saveplot(loss_history_2, train_state_2, 
                 issave=True, isplot=False, output_dir=str(pinn_dir))
    
    # Log labeled points
    if n_labeled > 0:
        np.savetxt(pinn_dir / "labeled_points.csv",
                labeled_pts, delimiter=",")
    
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
    
    metadata_path = pinn_dir / "training_log.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    
    print(f"Training completed in {elapsed_minutes}m {elapsed_seconds_remainder}s")
    print(f"Saved config and training metadata")
    
    return model


# --- Restore a Model ---
def restore_model(fem_data, cfg, a, b, n_labeled, model_prefix):
    """
    Restores a saved model based on model_prefix and returns it.
    Args:
        fem_data: ground-truth array of shape (N, 5) with columns = [x, y, u_fem, v_fem, p_fem]
        cfg: custom config class object
        a: ellipse semimajor (half width)
        b: ellipse semiminor (half height)
        n_labeled: number of labeled data points to sample
        model_prefix: filename prefix for saved model, preferrably an absolute path
    Returns:
        model: DeepXDE model object, restored
    """
    
    # Must compile before restore
    model, _ = build_model(fem_data, cfg, a, b, n_labeled)
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


# --- Get Model Outputs ---
def pinn_predict(model, query):
    """
    Computes h(x) for query points and passes through the model, returning its predictions.
    Args:
        model: DeepXDE model object used for prediction
        query: array of N (x,y) points, shape (N, 2)
    Returns: ndarray of shape (N, 5) with columns = [x, y, u_pred, v_pred, p_pred]
    """
    
    query_f32 = query.astype(np.float32)
    pred = model.predict(query_f32)    # (N, 3) = (u,v,p)
    
    return np.concatenate([query_f32, pred], axis=1)