# deeponet.py

"""
2D Stenosis — Physics-Informed Deep Operator Network
    Implementation using PDEOperator (non-Cartesian formulation).

Architecture
------------
    Branch net : SDF field sampled on fixed sensor grid (N_sensors,) --> latent (p,)
    Trunk net  : query point (x, y) --> latent (p,)
    Output     : inner product + bias --> (u, v, p) at query point

Key design decisions vs. prior Cartesian Product implementation
---------------------------------------------------------------

1.  PDEOperator (non-Cartesian) instead of PDEOperatorCartesianProd.
    Each training function (geometry) gets its own independently sampled
    trunk points from the full channel Rectangle.  No shared trunk; no
    intersection-masking workaround.

2.  SDF as auxiliary variable injected into the PDE residual.
    The per-point signed distance to the current geometry's ellipse is
    appended to x inside `auxiliary_var_function`. The PDE closure reads
    this column to:
        (a) suppress the NS residual inside the solid   (w_pde weight)
        (b) enforce no-slip on the obstacle surface     (w_obs weight)
    This replaces all geometry-dependent PointSetBCs on the obstacle.

3.  Channel wall / inlet / outlet BCs remain as standard DeepXDE BCs
    on the full Rectangle geometry, because those boundaries are fixed
    across all geometries.

4.  FunctionSpace carries (a, b) state for the current batch so that
    auxiliary_var_function can decode the geometry without a second
    argument.

SDF weighting scheme
--------------------
    d   = signed distance to ellipse surface (positive = fluid, negative = solid)
    σ   = obstacle_sigma  (width of the no-slip soft zone, ~1 mesh element)
    ε   = fluid_offset    (small negative buffer before declaring interior)
    τ   = fluid_sharpness (transition width for the fluid mask sigmoid)

    w_pde(d) = sigmoid( (d - ε) / τ )          # 0 inside solid, 1 in fluid
    w_obs(d) = exp( -d² / (2σ²) )              # peaks on surface, decays both ways

    PDE residual loss  : w_pde(d) * [continuity, x_mom, y_mom]
    Obstacle BC loss   : λ_obs * w_obs(d) * [u_pred², v_pred²]

    The channel wall BCs (y=0 and y=H_max) are handled by DeepXDE DirichletBCs
    on the full-channel Rectangle geometry, not by SDF weights.

Evan Hackstadt
Rugonyi Lab
"""

import json
import time
import datetime
from pathlib import Path

import numpy as np
import torch
import deepxde as dde

from config import StenosisConfig


# ─────────────────────────────────────────────────────────────
# SECTION 1: SDF UTILITIES
# ─────────────────────────────────────────────────────────────

def make_sensor_grid(cfg: StenosisConfig) -> np.ndarray:
    """
    Fixed sensor grid over the full channel [-L/2, L/2] x [0, H_max].
    Sensors include interior ellipse points — negative SDF there is informative.
    Returns (N_sensors, 2) float32.
    """
    xs = np.linspace(-cfg.L / 2, cfg.L / 2, cfg.sensor_nx)
    ys = np.linspace(0.0, cfg.H_max, cfg.sensor_ny)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)


def compute_sdf(points: np.ndarray, cfg: StenosisConfig,
                a: float, b: float) -> np.ndarray:
    """
    Approximate signed distance from each point to the ellipse surface.
    Positive = outside/fluid, negative = inside/solid.

    Uses the normalised radial distance  f = sqrt((dx/a)^2 + (dy/b)^2) - 1,
    which is smooth and cheap. Not a true metric SDF, but sufficient for
    soft weighting.

    Parameters
    ----------
    points : (N, 2) array of (x, y) coordinates
    Returns : (N,) float32
    """
    dx = points[:, 0] - cfg.x_c
    dy = points[:, 1] - cfg.y_c
    return (np.sqrt((dx / a) ** 2 + (dy / b) ** 2) - 1.0).astype(np.float32)


def compute_sdf_torch(points: torch.Tensor, x_c: float, y_c: float,
                      a: float, b: float) -> torch.Tensor:
    """
    Same signed-distance formula in PyTorch, for use inside loss functions.
    points : (N, ≥2) tensor; uses columns 0 (x) and 1 (y).
    Returns : (N, 1) tensor.
    """
    dx = points[:, 0:1] - x_c
    dy = points[:, 1:2] - y_c
    return torch.sqrt((dx / a) ** 2 + (dy / b) ** 2) - 1.0


# ─────────────────────────────────────────────────────────────
# SECTION 2: FUNCTION SPACE
# ─────────────────────────────────────────────────────────────

class StenosisGeometrySpace(dde.data.function_spaces.FunctionSpace):
    """
    Function space over stenosis geometries parameterised by (a, b).
    Each "function" is the SDF field of one geometry evaluated at the sensor grid.

    Stores the most recently sampled (a, b) pairs in `self.last_sampled` so that
    `auxiliary_var_function` can decode the geometry for each training function
    without needing an extra argument.
    """

    def __init__(self, cfg: StenosisConfig, geometries: list):
        self.cfg = cfg
        self.geometries = geometries
        self.rng = np.random.default_rng(cfg.seed)
        self.last_sampled: list = []   # [(a, b), ...] for current batch

    def random(self, size: int) -> np.ndarray:
        """
        Sample `size` geometries (with replacement) from the training set.
        Returns (size, 2) — each row is an (a, b) pair, also stored in last_sampled.
        """
        idx = self.rng.integers(0, len(self.geometries), size=size)
        sampled = [self.geometries[i] for i in idx]
        self.last_sampled = sampled
        return np.array(sampled, dtype=np.float32)  # (size, 2) = features

    def eval_one(self, feature: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Evaluate SDF for one geometry at sensor points X.
        feature : (2,) = [a, b]
        X       : (N_sensors, 2)
        Returns : (N_sensors,)
        """
        a, b = float(feature[0]), float(feature[1])
        return compute_sdf(X, self.cfg, a, b)

    def eval_batch(self, features: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Evaluate SDF for N geometries at sensor points X.
        features : (N, 2)
        X        : (N_sensors, 2)
        Returns  : (N, N_sensors)
        """
        return np.stack(
            [self.eval_one(f, X) for f in features], axis=0
        ).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# SECTION 3: BOUNDARY CONDITIONS (channel walls, inlet, outlet)
# ─────────────────────────────────────────────────────────────

def inlet_u_values(cfg: StenosisConfig):
    """Returns a function (pts: np.ndarray) -> np.ndarray for the Poiseuille profile."""
    U_IN_MAX = cfg.u_in_max
    U_REF = cfg.u_ref
    H_MAX = cfg.H_max

    def fn(pts: np.ndarray) -> np.ndarray:
        y = pts[:, 1:2]
        return (U_IN_MAX / U_REF) * 4.0 * (y / H_MAX) * (1.0 - y / H_MAX)

    return fn


def _is_inlet(x, on_boundary, L):
    return on_boundary and np.isclose(x[0], -L / 2)


def _is_outlet(x, on_boundary, L):
    return on_boundary and np.isclose(x[0], L / 2)


def _is_channel_wall(x, on_boundary, L, H):
    """Top and bottom flat walls only; excludes inlet and outlet."""
    return (
        on_boundary
        and not np.isclose(x[0], -L / 2)
        and not np.isclose(x[0],  L / 2)
    )


def build_channel_bcs(geometry, cfg: StenosisConfig) -> list:
    """
    Static BCs for the fixed channel boundaries (inlet, outlet, top/bottom walls).
    The obstacle (ellipse) BC is handled dynamically via SDF weighting in the loss.
    """
    L, H = cfg.L, cfg.H_max
    p_out = cfg.p_out / cfg.u_ref ** 2

    inlet_fn  = lambda x, ob: _is_inlet(x, ob, L)
    outlet_fn = lambda x, ob: _is_outlet(x, ob, L)
    wall_fn   = lambda x, ob: _is_channel_wall(x, ob, L, H)

    bc_inlet_u  = dde.DirichletBC(geometry, inlet_u_values(cfg), inlet_fn,  component=0)
    bc_inlet_v  = dde.DirichletBC(geometry, lambda x: 0,         inlet_fn,  component=1)
    bc_wall_u   = dde.DirichletBC(geometry, lambda x: 0,         wall_fn,   component=0)
    bc_wall_v   = dde.DirichletBC(geometry, lambda x: 0,         wall_fn,   component=1)
    bc_outlet_p = dde.DirichletBC(geometry, lambda x: p_out,     outlet_fn, component=2)

    return [bc_inlet_u, bc_inlet_v, bc_wall_u, bc_wall_v, bc_outlet_p]


# ─────────────────────────────────────────────────────────────
# SECTION 4: PDE LOSS WITH SDF WEIGHTING
# ─────────────────────────────────────────────────────────────

def make_pde_loss(cfg: StenosisConfig,
                  function_space: StenosisGeometrySpace,
                  obstacle_sigma: float = 0.05,
                  fluid_offset: float = -0.02,
                  fluid_sharpness: float = 0.01,
                  lambda_obs: float = 50.0):
    """
    Returns the PDE residual function for PDEOperator.

    Signature required by DeepXDE for PDEOperator (non-Cartesian):
        fn(x, outputs, inputs)
        x       : (M, n_x_cols) trunk points; col 0 = x_coord, col 1 = y_coord,
                  col 2 = SDF value (appended by auxiliary_var_function)
        outputs : (M, 3) model predictions (u, v, p)
        aux_sdf  : (N_sensors,) SDF values for the current geometry

    The SDF column in x is populated once per function evaluation by
    `auxiliary_var_function` (see `make_auxiliary_var_fn`).

    SDF weighting:
        w_pde(d) = sigmoid((d - fluid_offset) / fluid_sharpness)
                   ≈ 1 in fluid, 0 inside ellipse solid
        w_obs(d) = exp(-d² / (2 * obstacle_sigma²))
                   ≈ 1 on ellipse surface, 0 elsewhere

    Returns a list of four residuals:
        [w_pde * continuity,
         w_pde * x_momentum,
         w_pde * y_momentum,
         lambda_obs * w_obs * (u² + v²)]   ← obstacle no-slip
    """
    RE = cfg.Re

    def pde(x, outputs, aux_sdf):
        u_pred = outputs[:, 0:1]
        v_pred = outputs[:, 1:2]
        p_pred = outputs[:, 2:3]

        # --- NS derivatives ---
        du_x  = dde.grad.jacobian(outputs, x, i=0, j=0)
        du_y  = dde.grad.jacobian(outputs, x, i=0, j=1)
        dv_x  = dde.grad.jacobian(outputs, x, i=1, j=0)
        dv_y  = dde.grad.jacobian(outputs, x, i=1, j=1)
        dp_x  = dde.grad.jacobian(outputs, x, i=2, j=0)
        dp_y  = dde.grad.jacobian(outputs, x, i=2, j=1)

        du_xx = dde.grad.hessian(outputs, x, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(outputs, x, component=0, i=1, j=1)
        dv_xx = dde.grad.hessian(outputs, x, component=1, i=0, j=0)
        dv_yy = dde.grad.hessian(outputs, x, component=1, i=1, j=1)

        continuity = du_x + dv_y
        x_momentum = (u_pred * du_x + v_pred * du_y
                      + dp_x - (1.0 / RE) * (du_xx + du_yy))
        y_momentum = (u_pred * dv_x + v_pred * dv_y
                      + dp_y - (1.0 / RE) * (dv_xx + dv_yy))

        # Fluid mask — smoothly suppresses residual inside the solid
        w_pde = torch.sigmoid((aux_sdf - fluid_offset) / fluid_sharpness)

        # Obstacle surface weight — soft Gaussian peaked at d=0
        w_obs = torch.exp(-aux_sdf ** 2 / (2.0 * obstacle_sigma ** 2))

        # Obstacle no-slip residual: enforce u=0, v=0 where d≈0
        obs_noslip = lambda_obs * w_obs * (u_pred ** 2 + v_pred ** 2)

        return [
            w_pde * continuity,
            w_pde * x_momentum,
            w_pde * y_momentum,
            obs_noslip,
        ]

    return pde


# ─────────────────────────────────────────────────────────────
# SECTION 5: AUXILIARY VARIABLE FUNCTION
# ─────────────────────────────────────────────────────────────

def make_auxiliary_var_fn(cfg: StenosisConfig,
                          function_space: StenosisGeometrySpace):
    """
    Returns the `auxiliary_var_function` required by dde.data.PDE.

    DeepXDE calls this as  fn(X)  where X: (M, 2) trunk points for this batch

    We need to evaluate the SDF at X for the current geometry. Since `inputs`
    is the SDF-at-sensors array and not (a, b) directly, we recover (a, b) from
    `function_space.last_sampled`.

    The function must return (M, n_aux) — one auxiliary value per trunk point.
    We return (M, 1): the SDF at each trunk point.

    Note on batch structure: PDEOperator feeds one function at a time through
    the PDE residual (it does NOT vectorise over functions like the Cartesian
    product formulation). So `function_space.last_sampled` always contains the
    current geometry during training when called from within the PDE evaluation.

    For robustness, if `last_sampled` has exactly one entry use it; if it has
    more (shouldn't happen in PDEOperator's single-function-at-a-time mode) use
    the first entry and log a warning.
    """

    def aux_fn(X):
        """
        X      : (M, 2) numpy array — trunk spatial coordinates
        Returns : (M, 1) numpy float32
        """
        sampled = function_space.last_sampled

        if len(sampled) == 0:
            # Fallback: no geometry sampled yet (e.g. test evaluation)
            # Return zeros — the PDE mask will treat these as fluid points.
            return np.zeros((X.shape[0], 1), dtype=np.float32)

        # PDEOperator calls the PDE once per function; use the first sampled.
        if len(sampled) > 1:
            # Should not occur with PDEOperator, but be defensive.
            pass

        a, b = float(sampled[0][0]), float(sampled[0][1])
        sdf = compute_sdf(X, cfg, a, b).reshape(-1, 1)  # (M, 1)
        return sdf.astype(np.float32)

    return aux_fn


# ─────────────────────────────────────────────────────────────
# SECTION 6: LABELED DATA / SUPERVISED TRAINING
# ─────────────────────────────────────────────────────────────


class PDEOperatorSemiSupervised(dde.data.PDEOperator):
    """
    Extends PDEOperator with a supervised data loss term.
    
    At each training step, after sampling geometries via function_space.random(),
    we look up precomputed FEM solutions at the trunk points for those geometries
    and compute MSE(u_pred, u_fem) alongside the PDE and BC losses.
    
    Parameters
    ----------
    labeled_data_dict : {(a,b): {"query": [x,y], "targets": [u,v,p]} }
                        Keys must match function_space.geometries exactly.
    **kwargs      : passed through to PDEOperator.__init__
    """
    
    def __init__(self, *args, labeled_data_dict: dict, cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.labeled_data_dict = labeled_data_dict   # {(a,b): {"query", "targets"}}
        self.cfg = cfg
        self._current_batch_y = None         # set during losses(), read from train_y
        

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        # base PDE and BC loss
        base_losses = super().losses(targets, outputs, loss_fn, inputs, model, aux)
        
        # identify current geometries
        sampled = self.func_space.last_sampled   # [(a,b), ...]
        if len(sampled) == 0:
            return base_losses
        
        # Predict on labeled data for the sampled geometries
        loss_u_terms = []
        loss_v_terms = []
        loss_p_terms = []
        
        for (a, b) in sampled:
            query   = self.labeled_data_dict[(a, b)]["query"]
            targets = self.labeled_data_dict[(a, b)]["targets"]
            branch = self.func_space.eval_one(np.array([a, b], dtype=np.float32), self.eval_pts)
            
            # cast to tensors
            branch = torch.tensor(branch[None, :], 
                                  dtype=outputs.dtype, device=outputs.device)
            
            trunk = torch.tensor(query.astype(np.float32), 
                                 dtype=outputs.dtype, device=outputs.device)
            true = torch.tensor(targets, dtype=outputs.dtype, device=outputs.device)
            
            # forward pass --> compute loss
            pred = model.net((branch, trunk))
            
            loss_u_terms.append(loss_fn(true[:, 0:1], pred[:, 0:1]))
            loss_v_terms.append(loss_fn(true[:, 1:2], pred[:, 1:2]))
            loss_p_terms.append(loss_fn(true[:, 2:3], pred[:, 2:3]))
        
        # aggregate across all sampled geometries
        loss_u = sum(loss_u_terms) / len(loss_u_terms)
        loss_v = sum(loss_v_terms) / len(loss_v_terms)
        loss_p = sum(loss_p_terms) / len(loss_p_terms)

        return base_losses + [loss_u, loss_v, loss_p]


# ─────────────────────────────────────────────────────────────
# SECTION 7: MODEL CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def build_deeponet_model(
    cfg: StenosisConfig,
    sensors: np.ndarray,
    function_space: StenosisGeometrySpace,
    labeled_data_dict: dict,
    p: int = 128,
    obstacle_sigma: float = 0.05,
    fluid_offset: float = -0.02,
    fluid_sharpness: float = 0.01,
    lambda_obs: float = 50.0,
) -> dde.Model:
    """
    Build the PI-DeepONet model using custom PDEOperatorSemiSupervised.

    Parameters
    ----------
    cfg              : StenosisConfig
    sensors          : (N_sensors, 2) fixed sensor grid
    function_space   : StenosisGeometrySpace instance (shared with aux_fn)
    labeled_data_dict: dict mapping (a,b) -> dict with "branch_sdf", "trunk_pts", "targets"
    p                : inner product latent dimension
    obstacle_sigma   : Gaussian σ for obstacle BC soft weight
    fluid_offset     : sigmoid offset for fluid mask (negative = buffer inside solid)
    fluid_sharpness  : sigmoid transition width for fluid mask
    lambda_obs       : weight on the obstacle no-slip term

    Returns
    -------
    dde.Model ready to compile and train.
    """

    # --- Channel geometry (full Rectangle, no ellipse cut) ---
    geometry = dde.geometry.Rectangle(
        xmin=[-cfg.L / 2, 0.0],
        xmax=[ cfg.L / 2, cfg.H_max],
    )

    # --- Auxiliary variable function (injects SDF into trunk x) ---
    aux_fn = make_auxiliary_var_fn(cfg, function_space)

    # --- PDE loss (4 residuals: cont, x_mom, y_mom, obs_noslip) ---
    pde_fn = make_pde_loss(
        cfg, function_space,
        obstacle_sigma=obstacle_sigma,
        fluid_offset=fluid_offset,
        fluid_sharpness=fluid_sharpness,
        lambda_obs=lambda_obs,
    )

    # --- Static channel BCs ---
    channel_bcs = build_channel_bcs(geometry, cfg)

    # --- PDE data object ---
    # auxiliary_var_function appends SDF as x[:, 2]; trunk x becomes (M, 3).
    # Autodiff in the PDE runs through columns 0 and 1 (x_coord, y_coord);
    # column 2 (SDF) is treated as a non-differentiable annotation — this is
    # fine because we only need spatial derivatives of (u,v,p), not of the SDF.
    pde_data = dde.data.PDE(
        geometry=geometry,
        pde=pde_fn,
        bcs=channel_bcs,
        num_domain=cfg.n_interior,
        num_boundary=cfg.n_boundary,
        num_test=None,
        auxiliary_var_function=aux_fn,
    )

    # --- PDEOperator data object (non-Cartesian) ---
    data = PDEOperatorSemiSupervised(
        pde=pde_data,
        function_space=function_space,
        evaluation_points=sensors,       # branch discretisation (N_sensors, 2)
        num_function=cfg.n_functions,    # functions sampled per training step
        num_test=None,
        labeled_data_dict=labeled_data_dict,
        cfg=cfg,
    )

    # --- Network ---
    # Branch: N_sensors -> hidden -> p*3 (split_branch strategy)
    # Trunk:  2         -> hidden -> p   (x_coord, y_coord)
    N_sensors = sensors.shape[0]
    branch_layers = [N_sensors] + cfg.branch_net_hidden_layers + [p * 3]
    trunk_layers  = [2]         + cfg.trunk_net_hidden_layers  + [p]

    net = dde.nn.DeepONet(
        branch_layers,
        trunk_layers,
        activation="tanh",
        kernel_initializer="Glorot uniform",
        num_outputs=3,                       # u, v, p
        multi_output_strategy="split_branch",
    )

    return dde.Model(data, net)


# ─────────────────────────────────────────────────────────────
# SECTION 8: TRAINING
# ─────────────────────────────────────────────────────────────

class LossMagnitudeReweighter(dde.callbacks.Callback):
    """
    Geometric-mean reweighter (same as in pinn.py).
    Every `period` Adam steps, adjusts weights so all loss terms
    sit near the same order of magnitude.
    """

    def __init__(self, period: int = 2000, alpha: float = 0.8,
                 min_w: float = 0.1, max_w: float = 500.0):
        super().__init__()
        self.period = period
        self.alpha  = alpha
        self.min_w  = min_w
        self.max_w  = max_w

    def on_epoch_end(self):
        it = self.model.train_state.iteration
        if it % self.period != 0 or it == 0:
            return
        loss_arr = np.array(self.model.train_state.loss_train)
        if loss_arr.ndim != 1 or np.any(loss_arr <= 0):
            return
        ref     = np.exp(np.mean(np.log(loss_arr)))
        new_w   = np.clip(ref / loss_arr, self.min_w, self.max_w)
        blended = self.alpha * np.array(self.model.loss_weights) + (1 - self.alpha) * new_w
        self.model.loss_weights = blended.tolist()
        print(f"\n[AdaptiveWeights @ iter {it}]  {np.round(blended, 2).tolist()}")


def train_deeponet(
    model: dde.Model,
    model_prefix: Path,
    cfg: StenosisConfig,
) -> dde.Model:
    """
    Two-stage Adam → L-BFGS training.

    Loss term ordering (matches make_pde_loss + build_channel_bcs):
        0  continuity (PDE, SDF-weighted)
        1  x-momentum (PDE, SDF-weighted)
        2  y-momentum (PDE, SDF-weighted)
        3  obstacle no-slip (SDF-weighted soft BC)
        4  BC: inlet u
        5  BC: inlet v
        6  BC: wall u
        7  BC: wall v
        8  BC: outlet p
        9  data supervision
    """

    reweighter = LossMagnitudeReweighter(period=2000)

    start_time = time.time()
    start_ts   = datetime.datetime.now().isoformat()

    # Stage 1: pretrain with labeled data strong
    print(f"[DeepONet] Adam training for {cfg.n_adam_1} iterations...")
    model.compile("adam", lr=cfg.lr, loss_weights=cfg.loss_weights_1)
    loss_h1, state1 = model.train(
        iterations=cfg.n_adam_1,
        display_every=1000,
    )
    
    # Stage 2: train with balanced loss weights
    print(f"[DeepONet] Adam training for {cfg.n_adam_2} iterations...")
    model.compile("adam", lr=cfg.lr, loss_weights=cfg.loss_weights_2)
    loss_h2, state2 = model.train(
        iterations=cfg.n_adam_2,
        callbacks=[reweighter],
        display_every=1000,
    )

    # Stage 3: local refinement
    print(f"[DeepONet] L-BFGS fine-tuning for up to {cfg.n_lbfgs} iterations...")
    model.compile("L-BFGS", loss_weights=model.loss_weights)
    dde.optimizers.config.set_LBFGS_options(
        gtol=cfg.gtol_lbfgs,
        ftol=cfg.ftol_lbfgs,
        maxiter=cfg.n_lbfgs,
        maxfun=cfg.n_lbfgs * 10,
    )
    loss_h3, state3 = model.train(
        display_every=1000,
        model_save_path=str(model_prefix),
    )

    elapsed  = int(time.time() - start_time)
    mm, ss   = divmod(elapsed, 60)

    dde.saveplot(loss_h2, state2,
                 issave=True, isplot=False,
                 output_dir=str(model_prefix.parent))

    metadata = {
        "start_timestamp":  start_ts,
        "end_timestamp":    datetime.datetime.now().isoformat(),
        "elapsed_time":     f"{mm}m {ss}s",
        "n_adam_1":           cfg.n_adam_1,
        "n_adam_2":           cfg.n_adam_2,
        "n_lbfgs_actual":   getattr(state3, "iteration", None),
        "n_sensors":        cfg.sensor_nx * cfg.sensor_ny,
        "n_functions_train": cfg.n_functions,
        "architecture":     "PDEOperator (non-Cartesian) + DeepONet",
        "obstacle_bc":      "SDF-weighted soft no-slip in PDE residual",
        "channel_bc":       "Standard DirichletBC on Rectangle",
    }
    log_path = model_prefix.parent / "training_log.json"
    log_path.write_text(json.dumps(metadata, indent=2))

    print(f"[DeepONet] Training complete in {mm}m {ss}s.")
    return model


# ─────────────────────────────────────────────────────────────
# SECTION 9: INFERENCE
# ─────────────────────────────────────────────────────────────

def deeponet_predict(
    model: dde.Model,
    sensors: np.ndarray,
    cfg: StenosisConfig,
    a: float,
    b: float,
    query_pts: np.ndarray,
    function_space: StenosisGeometrySpace,
) -> np.ndarray:
    """
    Zero-shot prediction for a geometry (a, b) at arbitrary query points.

    Sets `function_space.last_sampled` so that the auxiliary_var_function
    correctly evaluates the SDF for this geometry during the forward pass.

    Parameters
    ----------
    model         : trained dde.Model
    sensors       : (N_sensors, 2) — must match training sensor grid
    cfg           : StenosisConfig
    a, b          : ellipse semi-axes of the target geometry
    query_pts     : (M, 2) evaluation locations (should be in the fluid domain)
    function_space: the same StenosisGeometrySpace instance used at training

    Returns
    -------
    (M, 5) array with columns [x, y, u, v, p]
    """
    # Set the geometry state so aux_fn returns the correct SDF
    function_space.last_sampled = [(a, b)]

    sdf_field  = compute_sdf(sensors, cfg, a, b)
    branch_in  = sdf_field[np.newaxis, :].astype(np.float32)   # (1, N_sensors)
    trunk_in   = query_pts[:, :2].astype(np.float32)            # (M, 2)

    pred = model.predict((branch_in, trunk_in))

    return np.concatenate([trunk_in, pred], axis=1)             # (M, 5)


def restore_deeponet(
    model: dde.Model,
    model_prefix: Path,
    cfg: StenosisConfig,
) -> dde.Model:
    """Restore network weights into an already-built model."""
    model.compile("adam", lr=cfg.lr)
    checkpoints = list(model_prefix.parent.glob(f"{model_prefix.name}-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found at {model_prefix}")
    latest = max(checkpoints, key=lambda p: int(p.stem.split("-")[-1]))
    ckpt = torch.load(latest, map_location="cpu")
    model.net.load_state_dict(ckpt["model_state_dict"])
    print(f"[DeepONet] Restored weights from {latest.name}")
    return model
