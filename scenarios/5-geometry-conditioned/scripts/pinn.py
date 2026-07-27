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


# ———————————— GLOBAL CONSTANTS ————————————
# need these for functions called by DeepXDE, since it can't pass in custom args (such as cfg)

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

# Inlet x-velocity profile
def inlet_u(cfg):
    U_IN_MAX = float(cfg.u_in_max)
    U_REF = float(cfg.U_ref)
    H_MAX = float(cfg.H_max)
    
    def function(x):
        y = x[:, 1:2]
        # Poiseuille parabola, nondimensionalized by U_ref
        # Zero at y=0 and y=H_MAX, peak = u_in_max/U_ref at y=H_MAX/2
        return (U_IN_MAX / U_REF) * 4.0 * (y / H_MAX) * (1.0 - y / H_MAX)
    
    return function


# Hard BC output transformation
def make_hard_bc_transform(cfg):
    """
    Hard-imposes only the exact, algebraically tractable BCs:
        - Inlet u-velocity: Poiseuille parabola at x = -L/2
        - Inlet v-velocity: v = 0 at x = -L/2
        - Outlet pressure:  p = 0 at x = L/2

    No-slip wall conditions remain as soft loss terms.
    """

    L     = float(cfg.L)
    H     = float(cfg.H_max)
    u_max = float(cfg.u_in_max)
    U_ref = float(cfg.U_ref)
    p_out = float(cfg.P_out)
    
    def transform(x, u_raw):
        xc = x[:, 0:1]
        yc = x[:, 1:2]

        u_raw_u = u_raw[:, 0:1]
        u_raw_v = u_raw[:, 1:2]
        u_raw_p = u_raw[:, 2:3]

        # Distance from inlet, normalized: 0 at x=-L/2, 1 at x=L/2
        d_inlet = (xc + L / 2.0) / L

        # Distance from outlet, normalized: 0 at x=L/2, 1 at x=-L/2
        d_outlet = (L / 2.0 - xc) / L

        # Poiseuille inlet profile, nondimensionalized
        u_inlet = (u_max / U_ref) * 4.0 * (yc / H) * (1.0 - yc / H)

        # u: blends from inlet profile (at x=-L/2) to network output (moving right)
        #   At inlet (d_inlet=0): u = u_inlet  ✓
        #   Interior/outlet:      u = u_inlet*(1-d_inlet) + d_inlet*u_raw
        #                           = network free, with inlet profile decaying away
        u_hard = (1.0 - d_inlet) * u_inlet + d_inlet * u_raw_u

        # v: zero at inlet, network free elsewhere
        #   At inlet (d_inlet=0): v = 0  ✓
        #   Interior/outlet:      v = d_inlet * v_raw (network free)
        v_hard = d_inlet * u_raw_v

        # p: zero at outlet, network free elsewhere
        #   At outlet (d_outlet=0): p = p_out  ✓
        #   Interior/inlet:         p = p_out + d_outlet * p_raw (network free)
        p_hard = p_out + d_outlet * u_raw_p

        return torch.cat([u_hard, v_hard, p_hard], dim=1)
    
    return transform


# TEMP TESTING
def verify_hard_bc_transform(transform, cfg):
    dtype = torch.float32

    # Inlet points: x = -L/2, y uniform in [0, H]
    y_test = torch.linspace(0, cfg.H_max, 20, dtype=dtype).unsqueeze(1)
    x_test = torch.full_like(y_test, -cfg.L / 2)
    x_inlet = torch.cat([x_test, y_test], dim=1)
    u_raw   = torch.ones(20, 3, dtype=dtype)
    out     = transform(x_inlet, u_raw)
    u_expected = (cfg.u_in_max / cfg.U_ref) * 4.0 * (y_test / cfg.H_max) * (1.0 - y_test / cfg.H_max)
    print(f"Inlet u max error: {(out[:, 0:1] - u_expected).abs().max():.2e}")  # should be ~0
    print(f"Inlet v max error: {out[:, 1:2].abs().max():.2e}")                 # should be ~0

    # Outlet points: x = L/2, y uniform
    x_out = torch.full_like(y_test, cfg.L / 2)
    x_outlet = torch.cat([x_out, y_test], dim=1)
    out = transform(x_outlet, u_raw)
    print(f"Outlet p max error: {(out[:, 2:3] - cfg.P_out).abs().max():.2e}") # should be ~0
    


# --- Define the PDE Residual ---
def pde_loss(cfg):
    
    RE = cfg.Re
    
    def function(x, u):
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
    
    return function


# --- Helper: Construct PointSetBCs ---
def build_pointsetbcs(boundary_data, cfg,
                      requested_bcs=['inlet_u', 'inlet_v', 'wall_u', 'wall_v', 'outlet_p']):
    """
    boundary_data: array of shape (n_boundary * n_train_geometries, 4) = [x, y, a, b]
    """
    # Manually extract relevant points for each BC (inlet, outlet, walls)
    tol = 1e-8
    x = boundary_data[:, 0]
    y = boundary_data[:, 1]

    inlet_mask  = np.isclose(x, -cfg.L / 2.0, atol=tol)
    outlet_mask = np.isclose(x,  cfg.L / 2.0, atol=tol)
    wall_mask   = np.isclose(y, 0.0, atol=tol) | np.isclose(y, cfg.H_max, atol=tol)

    inlet_pts  = boundary_data[inlet_mask]
    outlet_pts = boundary_data[outlet_mask]
    wall_pts   = boundary_data[wall_mask]

    if inlet_pts.size == 0:
        raise ValueError("No inlet boundary points found in boundary_data.")
    if outlet_pts.size == 0:
        raise ValueError("No outlet boundary points found in boundary_data.")
    if wall_pts.size == 0:
        raise ValueError("No wall boundary points found in boundary_data.")

    # Define values for each BC
    inlet_vals_u = inlet_u(cfg)(inlet_pts[:, :2])
    inlet_vals_v = np.zeros((inlet_pts.shape[0], 1))
    wall_vals = np.zeros((wall_pts.shape[0], 1))
    outlet_vals_p = np.full((outlet_pts.shape[0], 1), cfg.P_out / cfg.U_ref ** 2)

    # Construct using PointSets, only for requested
    label_map = {}
    label_map['inlet_u']  = dde.PointSetBC(inlet_pts,  inlet_vals_u,  component=0)    # parabolic profile
    label_map['inlet_v']  = dde.PointSetBC(inlet_pts,  inlet_vals_v,  component=1)    # v=0
    label_map['wall_u']   = dde.PointSetBC(wall_pts,   wall_vals,     component=0)    # u=0
    label_map['wall_v']   = dde.PointSetBC(wall_pts,   wall_vals,     component=1)    # v=0
    label_map['outlet_p'] = dde.PointSetBC(outlet_pts, outlet_vals_p, component=2)    # p=0 (from config)

    return [bc for name, bc in label_map.items() if name in requested_bcs]


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
        pde=pde_loss(cfg),
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


# --- Fine-Tuning custom weight anchoring callback ---
# Generated by Claude 4.6 Sonnet
class AnchorRegularizationCallback(dde.callbacks.Callback):
    """
    Gradient-injection anchor regularization for fine-tuning a pretrained PINN.

    After each backward pass, adds the gradient of:
        L_anchor = (lambda_anchor / 2) * sum_i ||theta_i - theta_0_i||^2

    which is simply: d L_anchor / d theta_i = lambda_anchor * (theta_i - theta_0_i)

    This is equivalent to L2 regularization toward the pretrained weights,
    preventing catastrophic forgetting during fine-tuning.

    Args:
        pretrained_weights: dict of {name: tensor} from model.net.named_parameters()
                            captured BEFORE fine-tuning begins. Pass a deep copy.
        lambda_anchor:      regularization strength. Higher = stay closer to pretrained.
                            Start around 1e-2 to 1e-1 and tune based on obs fit vs. 
                            global accuracy tradeoff.
        frozen_prefixes:    list of parameter name prefixes to exclude from anchoring
                            (e.g. ["linears.3"] to exclude the last layer entirely,
                            allowing it to adapt freely). None = anchor all params.
    """

    def __init__(self, pretrained_weights, lambda_anchor: float = 0.01,
                 frozen_prefixes: list = None):
        super().__init__()
        self.lambda_anchor = lambda_anchor
        self.frozen_prefixes = frozen_prefixes or []

        # Store pretrained weights on CPU; we'll move to device on first use
        self.theta_0 = {
            name: tensor.detach().clone()
            for name, tensor in pretrained_weights
        }
        self._device = None

    def _should_anchor(self, name: str) -> bool:
        """Returns True if this parameter should be anchored."""
        return not any(name.startswith(pfx) for pfx in self.frozen_prefixes)

    def on_batch_end(self):
        # Lazily resolve device from model on first call
        if self._device is None:
            first_param = next(self.model.net.parameters())
            self._device = first_param.device

        for name, param in self.model.net.named_parameters():
            if not param.requires_grad:
                continue
            if not self._should_anchor(name):
                continue
            if param.grad is None:
                continue  # parameter wasn't reached in backward pass

            theta_0 = self.theta_0[name].to(self._device)

            # Gradient of (lambda/2)||theta - theta_0||^2 w.r.t. theta
            anchor_grad = self.lambda_anchor * (param.data - theta_0)
            param.grad.add_(anchor_grad)


# --- Fine-Tuning function ---
def pinn_finetune(pretrained_model, observation_data, query, cfg, a, b,
                  layers_to_adapt=[-1, -2], weight_anchor=False, hard_bc=False):
    """
    Fine-tunes a generally-trained model to patient-specific observations.
    Options to anchor weights and enforce hard BCs, representing different prediction strategies.
    Args:
        pretrained_model: DeepXDE model from general training across geometries
        observed_data: possibly-sparse (NaNs or Nones okay) array of shape (m_observations, 7) = [x, y, a, b, u, v, p]
        query: model input array of shape (N, 4) = [x,y,a,b]
        cfg: custom config object
        layers_to_adapt: indices into model.net.linears (reverse recommended)
        anchor: whether or not to anchor the weights (regularization term)
        hardbc: whether or not to enforce hard boundary conditions
    Returns:
        finetuned_model: DeepXDE model with weights fine-tuned to the given query
    """
    # Snapshot pretrained weights
    theta_0 = {name: p.clone().detach() 
                for name, p in pretrained_model.net.named_parameters()}
    
    # Optionally freeze early layers
    for i, layer in enumerate(pretrained_model.net.linears):
        if i not in layers_to_adapt:
            for param in layer.parameters():
                param.requires_grad = False
    
    # Prep BCs (all unless hard bcs)
    boundary_data = np.loadtxt(cfg.data_dir / "boundary_data.csv", delimiter=",")
    requested_bcs = ['wall_u', 'wall_v'] if hard_bc else ['inlet_u', 'inlet_v', 'wall_u', 'wall_v', 'outlet_p']
    bcs = build_pointsetbcs(boundary_data, cfg, requested_bcs=requested_bcs)
    
    # Parse observation data, add existing to BCs
    obs_u = observation_data[:, 4]
    obs_v = observation_data[:, 5]
    obs_p = observation_data[:, 6]
    candidates = {"u": obs_u, "v": obs_v, "p": obs_p}
    
    # only use non-null observation components
    for component, (var, obs) in enumerate(candidates.items()):
        if not np.all(np.isnan(obs)):
            obs_xyab = observation_data[~np.isnan(obs), 0:4]
            a_check = np.unique(obs_xyab[:, 2])
            b_check = np.unique(obs_xyab[:, 3])
            if len(a_check) > 1 or len(b_check) > 1:
                raise ValueError(f"Multiple geometries found in observation_data: unique a = {a_check}, unique b = {b_check}")
            if component not in cfg.test_observation_components:
                raise ValueError(f"Mismatch between component from data ({component}) and config test components ({cfg.test_observation_components})")
            print(f"Extracting observed {var} from observation data.")
            bcs.append(dde.PointSetBC(obs_xyab, obs, component=component))
    
    # Rebuild data with new geometry + observation BC
    # expect only a single (a, b) so create a dummy range for domain
    new_geometry = dde.geometry.Hypercube(xmin=[-cfg.L/2, 0.0, a*0.9, b*0.9],
                                          xmax=[ cfg.L/2, cfg.H_max, a*1.1, b*1.1])
    data = dde.data.PDE(geometry=new_geometry, 
                        pde=pde_loss(cfg), 
                        bcs=bcs)
    
    # Construct loss terms
    old_loss_weights = pretrained_model.loss_weights    # 3 pde + 4 bc + 3 labeled data
    loss_weights = old_loss_weights[:3]     # always have PDE
    if hard_bc:
        loss_weights += old_loss_weights[5:7]   # only wall_u, wall_v
    else:
        loss_weights += old_loss_weights[3:8]   # all 4 bc terms
    loss_weights += [cfg.test_observation_loss_weight] * len(cfg.test_observation_components)
    
    pretrained_model.data = data
    pretrained_model.compile("adam", lr=cfg.lr_finetune, 
                            loss_weights=loss_weights)
    
    if hard_bc:
        
        verify_hard_bc_transform(make_hard_bc_transform(cfg), cfg)      # TEMP
        
        pretrained_model.net.apply_output_transform(make_hard_bc_transform(cfg))
    
    if weight_anchor:
        weights = pretrained_model.net.named_parameters()
        anchor_callback = AnchorRegularizationCallback(weights,
                                                       lambda_anchor=cfg.lambda_anchor,
                                                       frozen_prefixes=None)
        pretrained_model.train(iterations=cfg.n_finetune,
                               callbacks=[anchor_callback])
    else:
        pretrained_model.train(iterations=cfg.n_finetune)
    
    return pretrained_model     # now fine tuned