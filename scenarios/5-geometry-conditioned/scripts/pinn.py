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
def build_pointsetbcs(boundary_data, cfg, hard_bc=False):
    """
    boundary_data: array of shape (n_boundary * n_train_geometries, 4) = [x, y, a, b]
    """
    # Manually extract relevant points for each BC (inlet, outlet, walls)
    tol = 1e-8
    x = boundary_data[:, 0]
    y = boundary_data[:, 1]

    inlet_mask    = np.isclose(x, -cfg.L / 2.0, atol=tol)
    outlet_mask   = np.isclose(x,  cfg.L / 2.0, atol=tol)
    wall_mask     = np.isclose(y, 0.0, atol=tol) | np.isclose(y, cfg.H_max, atol=tol)
    obstacle_mask = ~inlet_mask & ~outlet_mask & ~wall_mask

    inlet_pts    = boundary_data[inlet_mask]
    outlet_pts   = boundary_data[outlet_mask]
    wall_pts     = boundary_data[wall_mask]
    obstacle_pts = boundary_data[obstacle_mask]
    
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
    obstacle_vals = np.zeros((obstacle_pts.shape[0], 1))
    outlet_vals_p = np.full((outlet_pts.shape[0], 1), cfg.P_out / cfg.U_ref ** 2)

    # Construct using PointSets
    bc_inlet_u  = dde.PointSetBC(inlet_pts,  inlet_vals_u,  component=0)    # parabolic profile
    bc_inlet_v  = dde.PointSetBC(inlet_pts,  inlet_vals_v,  component=1)    # v=0
    bc_wall_u   = dde.PointSetBC(wall_pts,   wall_vals,     component=0)    # u=0
    bc_wall_v   = dde.PointSetBC(wall_pts,   wall_vals,     component=1)    # v=0
    bc_obstacle_u = dde.PointSetBC(obstacle_pts, obstacle_vals, component=0)
    bc_obstacle_v = dde.PointSetBC(obstacle_pts, obstacle_vals, component=1)
    bc_outlet_p = dde.PointSetBC(outlet_pts, outlet_vals_p, component=2)    # p=0 (from config)

    # if we will be enforcing hard BCs, don't return inlet/outlet
    if hard_bc:
        bcs = [bc_wall_u, bc_wall_v, bc_obstacle_u, bc_obstacle_v]
    else:
        bcs = [bc_inlet_u, bc_inlet_v, bc_wall_u, bc_wall_v, 
               bc_obstacle_u, bc_obstacle_v, bc_outlet_p]
    
    return bcs


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
    
    bcs = build_pointsetbcs(boundary_data, cfg, hard_bc=False)

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
    a_max = np.max(interior_data[:, 2])
    b_max = np.max(interior_data[:, 3])
    geometry = dde.geometry.Hypercube(
        xmin=[-cfg.L/2, 0.0, 0.0, 0.0],
        xmax=[ cfg.L/2, cfg.H_max, a_max, b_max]
    )

    # Instantiate data and network objects    
    data = dde.data.PDE(
        geometry=geometry,
        pde=pde_loss(cfg),
        bcs=bcs,
        num_domain=0,
        num_boundary=0,
        num_test=None,
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
    Trains given model object until convergence, saving model and training metadata.
    Args:
        model: DeepXDE model object instantiated with data and network
        model_prefix: filename prefix for saved model, preferrably an absolute path
        cfg: custom config class object
    Returns:
        model: trained DeepXDE model object
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
def restore_model(model, model_prefix, cfg, learning_rate=None, loss_weights=None):
    """
    Restores a saved model based on model_prefix and returns it.
    Args:
        model: DeepXDE model object instantiated with data and network
        model_prefix: filename prefix for saved model, preferrably an absolute path
        cfg: custom config class object
        learning_rate: optionally override cfg.lr
        loss_weights: optionally override cfg.loss_weights_adam
    Returns:
        model: DeepXDE model object, restored
    """
    
    # Must compile before restore
    lr = cfg.lr if learning_rate is None else learning_rate
    lw = cfg.loss_weights_adam if loss_weights is None else loss_weights
    model.compile("adam", lr=lr, loss_weights=lw)

    # Find the latest saved checkpoint
    model_prefix = Path(model_prefix)
    checkpoints = list(model_prefix.parent.glob(f"{model_prefix.name}-*.pt"))
    latest = max(checkpoints, key=lambda p: int(p.stem.split("-")[-1]))
    
    # Manually load only the network weights
    # DeepXDE model.restore() loads optimizer state, causing errors
    checkpoint = torch.load(latest, map_location="cpu")
    model.net.load_state_dict(checkpoint["model_state_dict"])
    print(f"Restored weights from {latest.name}")

    return model


# --- One-Shot Prediction ---
def pinn_predict(model, query):
    """
    Computes h(x) for query points and passes through the model, returning its predictions.
    Args:
        model: DeepXDE model object used for prediction
        query: model input array of shape (N, 4) = [x,y,a,b]
    Returns: ndarray of shape (N, 5) with columns = [x,y, u_pred,v_pred,p_pred]
    """
    
    query_f32 = query.astype(np.float32)
    pred = model.predict(query_f32)    # (N, 3) = (u,v,p)
    
    return np.concatenate([query_f32[:, :2], pred], axis=1)


# ————————— PINN FINE-TUNING FUNCTIONS —————————

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


# Hard BC output transformation
def make_hard_bc_transform(cfg):
    """
    Hard-imposes only the exact, algebraically tractable BCs:
        - Inlet u-velocity: Poiseuille parabola at x = -L/2
        - Inlet v-velocity: v = 0 at x = -L/2
        - Outlet pressure:  p = 0 at x = L/2

    No-slip wall/obstacle conditions remain as soft loss terms.
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

        # Use a sharper decay than linear so the hard BC influence drops off faster
        # Low blend --> enforce BC; high blend --> network is free
        # So lessen the strength of hard BC by making blend rise quicker
        blend_inlet = d_inlet ** 0.25
        blend_outlet = d_outlet ** 0.25

        # Poiseuille inlet profile, nondimensionalized
        u_inlet = (u_max / U_ref) * 4.0 * (yc / H) * (1.0 - yc / H)

        # u: blends from inlet profile (at x=-L/2) to network output (moving right)
        #   At inlet (d_inlet=0): u = u_inlet  ✓
        #   Interior/outlet:      u = u_inlet*(1-blend_inlet) + blend_inlet*u_raw
        #                           = network free faster than linear interpolation
        u_hard = (1.0 - blend_inlet) * u_inlet + blend_inlet * u_raw_u

        # v: zero at inlet, network free elsewhere
        #   At inlet (d_inlet=0): v = 0  ✓
        #   Interior/outlet:      v = blend_inlet * v_raw (network free)
        v_hard = blend_inlet * u_raw_v

        # p: zero at outlet, network free elsewhere
        #   At outlet (blend_outlet=0): p = p_out  ✓
        #   Interior/inlet:         p = p_out + blend_outlet * p_raw (network free)
        p_hard = p_out + blend_outlet * u_raw_p

        return torch.cat([u_hard, v_hard, p_hard], dim=1)
    
    return transform


# --- Fine-Tuning Build Function ---
#        valid entry point
def build_model_finetune(pretrained_model, interior_data, boundary_data, observation_data, 
                         cfg, layers_to_adapt=[-1, -2], hard_bc=False):
    """
    Modifies pretrained model object with relevant BCs and data, and constructs loss weights.
    Args:
        pretrained_model: DeepXDE model from general training across geometries
        interior_data: array of shape (n_interior * n_train_geometries, 4) = [x, y, a, b]
        boundary_data: array of shape (n_boundary * n_train_geometries, 4) = [x, y, a, b]
        observation_data: array of shape (n_labeled_test * n_geometries, 7) = [x, y, a, b, u, v, p]
        cfg: custom config object
        layers_to_adapt: indices into model.net.linears (reverse recommended)
        hard_bc: whether or not to enforce hard boundary conditions
    Returns:
        built_model: fresh DeepXDE model object with old weights and new Data object (BCs, geometry)
        loss_weights: list of loss weights of the correct length for the model
    """
    
    # Snapshot pretrained weights
    theta_0 = {name: p.clone().detach() 
                for name, p in pretrained_model.net.named_parameters()}
    
    # Prep BCs (all unless hard bcs)
    bcs = build_pointsetbcs(boundary_data, cfg, hard_bc=hard_bc)
    
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
    
    
    # Manually rebuild data with new geometry + observation BC
    a_max = np.max(observation_data[:, 2])
    b_max = np.max(observation_data[:, 3])
    geometry = dde.geometry.Hypercube(xmin=[-cfg.L/2, 0.0, 0.0, 0.0],
                                      xmax=[ cfg.L/2, cfg.H_max, a_max, b_max])
    
    data = dde.data.PDE(geometry=geometry, 
                        pde=pde_loss(cfg), 
                        bcs=bcs,
                        anchors=interior_data)
    
    # Instantiate fresh model object with old model's weights
    new_model = dde.Model(data, pretrained_model.net)
    
    # Optionally freeze early layers
    for i, layer in enumerate(new_model.net.linears):
        if i not in layers_to_adapt:
            for param in layer.parameters():
                param.requires_grad = False
    
    # Construct loss weights
    loss_weights = cfg.loss_weights_adam[:10]   # exclude labeled data weights
    
    if hard_bc:
        # enforce BC output transform
        new_model.net.apply_output_transform(make_hard_bc_transform(cfg))
        # remove indices 3,4,9 = bc_inlet_u, bc_inlet_v, bc_outlet_p
        loss_weights = loss_weights[0:3] + loss_weights[5:9]
    
    loss_weights.extend(cfg.test_observation_weights)   # add test obs weights to end
    
    return new_model, loss_weights


# --- Fine-Tuning Training Function ---
#          valid entry point
def finetune_model(model, loss_weights, model_prefix, cfg, weight_anchor=False):
    """
    Fine-tunes a generally-trained model to patient-specific observations.
    Options to anchor weights and enforce hard BCs, representing different prediction strategies.
    Args:
        model: DeepXDE model object from build_model_finetune() - pretrained weights, new Data object
        model_prefix: filename prefix for saved model, preferrably an absolute path
        cfg: custom config object
        anchor: whether or not to anchor the weights (regularization term)
        hardbc: whether or not to enforce hard boundary conditions
    Returns:
        finetuned_model: DeepXDE model with weights fine-tuned to the given query
    """
    
    model.compile("adam", lr=cfg.lr_finetune, loss_weights=loss_weights)
    reweighter = LossMagnitudeReweighter()
    
    if weight_anchor:
        weights = model.net.named_parameters()
        anchor_callback = AnchorRegularizationCallback(weights,
                                                       lambda_anchor=cfg.lambda_anchor,
                                                       frozen_prefixes=None)
        
        loss_history, train_state = model.train(iterations=cfg.n_adam_finetune,
                                                callbacks=[reweighter, anchor_callback],
                                                display_every=100,
                                                model_save_path=model_prefix)
    else:
        loss_history, train_state = model.train(iterations=cfg.n_adam_finetune,
                                                callbacks=[reweighter],
                                                display_every=100,
                                                model_save_path=model_prefix)
    
    output_dir = model_prefix.parent
    dde.saveplot(loss_history, train_state, 
                 issave=True, isplot=False, output_dir=str(output_dir))
    
    return model     # now fine-tuned