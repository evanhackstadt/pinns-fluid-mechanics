# deeponet.py

"""
2D Stenosis — Physics-Informed Deep Operator Network
    Learns the operator: geometry SDF field --> (u, v, p) solution field
    Uses DeepXDE's DeepONetCartesianProd + PDEOperatorCartesianProd

Architecture
------------
    Branch net : SDF field sampled on fixed sensor grid (N_sensors,) --> latent (p,)
    Trunk net  : query point (x, y) --> latent (p,)
    Output     : inner product + bias --> (u, v, p) at query point,
                 evaluated for all N_geom x M_trunk pairs (Cartesian product)

Trunk point strategy — intersection masking
--------------------------------------------
    The Cartesian product formulation requires one SHARED trunk array used
    identically for every geometry. The ellipse obstruction means that for a
    small geometry (a=0.3), a point near the top wall may be in the fluid,
    while for a large geometry (a=0.65) that same point is inside solid.

    Solution: the shared trunk is built from the LARGEST training ellipse mask.
    Any point outside the largest ellipse is guaranteed to be in the fluid
    domain for ALL training geometries (since all smaller ellipses are subsets).
    This is geometrically exact and requires no per-point loss masking.

    Consequence: the "annular" region between each smaller ellipse surface and
    the largest ellipse surface has no interior collocation points for the
    smaller geometries. The PDE loss is zero there. This is the honest tradeoff
    vs. the alternative of padding with incorrect labels.

    Obstacle BCs are handled separately: each geometry contributes its own
    ellipse surface points as labeled pairs in the supervised output array
    (zero velocity), aligned to the shared trunk via nearest-neighbour lookup.
    See build_shared_trunk_and_obstacle_bcs() below.

Evan Hackstadt
Rugonyi Lab
"""

import json
import time, datetime
from pathlib import Path

import numpy as np
import torch
import deepxde as dde

from config import StenosisConfig



# ─────────────────────────────────────────────────────────────
# SECTION 1: BOUNDARY CONDITIONS
# ─────────────────────────────────────────────────────────────

def inlet_u_values(cfg):
    U_IN_MAX = cfg.u_in_max
    U_REF = cfg.u_ref
    H_MAX = cfg.H_max
    
    def function(pts: np.ndarray) -> np.ndarray:
        """Poiseuille profile at inlet, nondimensionalised. pts shape (N, 2)."""
        y = pts[:, 1:2]
        return (U_IN_MAX / U_REF) * 4.0 * (y / H_MAX) * (1.0 - y / H_MAX)
    
    return function


def build_channel_bcs(trunk_pts: np.ndarray, cfg: StenosisConfig) -> list:
    """
    Construct PointSetBCs for the geometry-INDEPENDENT boundary conditions
    (inlet, bottom wall, top wall, outlet) from the shared trunk points.
 
    Obstacle BCs are NOT included here — they are handled via the supervised
    output array (zero-velocity labels on ellipse surface points appended to trunk).
    This is correct because obstacle surface points differ per geometry and
    cannot be a single shared PointSetBC.
 
    Parameters
    ----------
    trunk_pts : (M, 2) shared trunk from build_shared_trunk()
    cfg       : StenosisConfig
 
    Returns
    -------
    list of dde.PointSetBC for inlet (u, v), bottom wall (u, v),
    top-wall outside-ellipse (u, v), outlet (p).
    """
    tol = 1e-8
    x_q, y_q = trunk_pts[:, 0], trunk_pts[:, 1]
 
    inlet_mask  = np.isclose(x_q, -cfg.L / 2, atol=tol)
    outlet_mask = np.isclose(x_q,  cfg.L / 2, atol=tol)
    bot_mask    = np.isclose(y_q, 0.0, atol=tol)
    # Top wall: only points at y=H_max that are NOT inlet/outlet
    top_mask    = np.isclose(y_q, cfg.H_max, atol=tol) & ~inlet_mask & ~outlet_mask
 
    inlet_pts   = trunk_pts[inlet_mask]
    outlet_pts  = trunk_pts[outlet_mask]
    bot_pts     = trunk_pts[bot_mask]
    top_pts     = trunk_pts[top_mask]
 
    zeros = lambda pts: np.zeros((pts.shape[0], 1), dtype=np.float32)
    outlet_p_vals = np.full((outlet_pts.shape[0], 1), 
                            cfg.p_out / cfg.u_ref**2, dtype=np.float32)
 
    bcs = [
        dde.PointSetBC(inlet_pts,  inlet_u_values(cfg)(inlet_pts), component=0),
        dde.PointSetBC(inlet_pts,  zeros(inlet_pts), component=1),
        dde.PointSetBC(bot_pts,    zeros(bot_pts),   component=0),
        dde.PointSetBC(bot_pts,    zeros(bot_pts),   component=1),
        dde.PointSetBC(top_pts,    zeros(top_pts),   component=0),
        dde.PointSetBC(top_pts,    zeros(top_pts),   component=1),
        dde.PointSetBC(outlet_pts, outlet_p_vals,    component=2),
    ]
    return bcs


# ─────────────────────────────────────────────────────────────
# SECTION 2: PDE LOSS
# ─────────────────────────────────────────────────────────────
def pde_loss_operator(cfg):
    
    RE = cfg.Re
    
    def function(x, outputs, X_func):
        """
        Navier-Stokes residuals for PDEOperatorCartesianProd.

        Signature differs from PINN pde_loss: receives X_func (branch inputs for the
        batch) as a third argument. X_func is not used in the residual — geometry
        information is already encoded in `outputs` via the branch net.
        Autodiff is through the TRUNK coordinates x = (x_coord, y_coord).

        x       : (M, 2) trunk points — spatial coords, autodiff runs through here
        outputs : (N*M, 3) or (N, M, 3) — DeepXDE handles reshape internally
        X_func  : (N, N_sensors) — branch inputs, unused in PDE residual
        """
        u_pred = outputs[:, 0:1]
        v_pred = outputs[:, 1:2]
        p_pred = outputs[:, 2:3]
    
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
        x_momentum = u_pred*du_x + v_pred*du_y + dp_x - (1/RE)*(du_xx + du_yy)
        y_momentum = u_pred*dv_x + v_pred*dv_y + dp_y - (1/RE)*(dv_xx + dv_yy)
    
        return [continuity, x_momentum, y_momentum]
    
    return function


# ─────────────────────────────────────────────────────────────
# SECTION 3: SENSOR GRID AND BRANCH INPUTS (SDF fields)
# ─────────────────────────────────────────────────────────────


def make_sensor_grid(cfg: StenosisConfig) -> np.ndarray:
    """
    Fixed sensor grid over the full channel [-L/2, L/2] x [0, H_max].
    Sensors include points inside the obstruction — the SDF value there
    is negative, which is meaningful information for the branch encoder.
    The sensor grid does NOT need to match the trunk grid.

    Returns
    -------
    sensors : (SENSOR_NX * SENSOR_NY, 2)  float32
    """
    xs = np.linspace(-cfg.L / 2, cfg.L / 2, cfg.sensor_nx)
    ys = np.linspace(0.0, cfg.H_max, cfg.sensor_ny)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)


def compute_sdf(sensors: np.ndarray, cfg: StenosisConfig, a: float, b: float) -> np.ndarray:
    """
    Approximate signed distance from each sensor point to the ellipse surface.
    Positive = outside/fluid, negative = inside/solid.

    Uses the normalised radial distance  f = sqrt((dx/a)^2 + (dy/b)^2) - 1,
    which is smooth and differentiable but not a true metric SDF. Sufficient
    for the branch encoder; a true SDF would require iterative projection.

    Parameters
    ----------
    sensors : (N_sensors, 2) from make_sensor_grid()
    cfg     : StenosisConfig
    a, b    : ellipse semi-axes

    Returns
    -------
    sdf : (N_sensors,) float32
    """
    dx = sensors[:, 0] - cfg.x_c
    dy = sensors[:, 1] - cfg.y_c
    f  = np.sqrt((dx / a) ** 2 + (dy / b) ** 2) - 1.0
    return f.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# SECTION 4: TRUNK DATA
# ─────────────────────────────────────────────────────────────

# --- Obstacle Points ---

def find_largest_ellipse(geometries: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Identify the largest training ellipse by enclosed area (pi*a*b).
    This ellipse defines the most restrictive fluid-domain mask —
    all other ellipses are contained within it.

    For geometries where a != b-dominant, area is the right metric
    because the intersection of fluid domains is determined by the
    ellipse with the largest footprint at every x-slice.
    """
    return max(geometries, key=lambda ab: ab[0] * ab[1])


def sample_obstacle_points(cfg: StenosisConfig,
                           geometries: list[tuple[float, float]]):
    """
    Collect per-geometry obstacle boundary points.

    Parameters
    ----------
    cfg             : StenosisConfig
    geometries      : list of (a, b) training geometries
    n_obstacle_pts  : number of surface points sampled per ellipse for obstacle BC

    Returns
    -------
    obstacle_pts_by_geom: dict mapping (a, b) --> (K, 2) array of points along each ellipse geometry
    """

    # Sample K points on each ellipse arc (lower half only — the fluid-facing surface).
    # Parameterise the lower half: y = y_c - b*sin(theta), x = x_c + a*cos(theta),
    # theta in [0, pi] gives the lower boundary from right tip to left tip.
    obstacle_pts_by_geom = {}
    thetas = np.linspace(0, np.pi, cfg.n_obstacle)
    
    for (a, b) in geometries:
        obs_x = cfg.x_c + a * np.cos(thetas)
        obs_y = cfg.y_c - b * np.sin(thetas)
        # Clip to channel bounds (safety — should be within [0, H_max])
        obs_y = np.clip(obs_y, 0.0, cfg.H_max)
        obstacle_pts_by_geom[(a, b)] = np.column_stack([obs_x, obs_y]).astype(np.float32)
    
    return obstacle_pts_by_geom


# --- Labeled Output Array ---

def build_output_array(
    fem_data_dict: dict,
    trunk_pts: np.ndarray,
    obstacle_pts_by_geom: dict,
    u_sol_dict: dict,
    p_sol_dict: dict,
    msh_dict: dict,
    cfg: StenosisConfig,
) -> np.ndarray:
    """
    Build the supervised output array (N_geom, M_trunk, 3) by evaluating
    FEM solutions at the shared trunk points.

    For interior and channel-boundary trunk points: query FEM via bb_tree.
    For obstacle surface points (per-geometry): append zero-velocity rows,
    since these are no-slip BC points not in the shared trunk.

    NOTE: This function requires the live FEM solution objects (u_sol, p_sol, msh)
    from fem.py because fem_predict() does arbitrary-point evaluation.
    If you only have the cached .npz FEM data (not the live objects), use
    build_output_array_from_cache() below instead.

    Parameters
    ----------
    fem_data_dict        : {(a,b): (N_fem_pts, 5)} — cached FEM data (for reference)
    trunk_pts            : (M, 2) shared trunk from build_shared_trunk()
    obstacle_pts_by_geom : {(a,b): (K, 2)} from build_shared_trunk()
    u_sol_dict           : {(a,b): dolfinx velocity solution object}
    p_sol_dict           : {(a,b): dolfinx pressure solution object}
    msh_dict             : {(a,b): dolfinx mesh object}
    cfg                  : StenosisConfig

    Returns
    -------
    output_arr : (N_geom, M_trunk + K, 3)  float32
        where K is the number of obstacle surface points per geometry.
        First M_trunk rows correspond to trunk_pts; last K rows to obstacle_pts.
        These are stored contiguously so the full array can be passed to DeepXDE.
    extended_trunk : (M_trunk + K, 2)
        Trunk points extended with per-geometry obstacle points — NOT shared.
        Used only for constructing the full supervised data object.
    """
    from fem import fem_predict   # local import to avoid circular dependency

    geom_list  = list(fem_data_dict.keys())
    M_trunk    = trunk_pts.shape[0]
    output_list = []

    for (a, b) in geom_list:
        msh   = msh_dict[(a, b)]
        u_sol = u_sol_dict[(a, b)]
        p_sol = p_sol_dict[(a, b)]

        # Query FEM at shared trunk points
        fem_vals_trunk = fem_predict(u_sol, p_sol, msh, trunk_pts, cfg)   # (M_trunk, 5)
        uvp_trunk = fem_vals_trunk[:, 2:5]   # (M_trunk, 3)

        # Obstacle surface: zero velocity, pressure from FEM (or zero for simplicity)
        obs_pts = obstacle_pts_by_geom[(a, b)]   # (K, 2)
        K = obs_pts.shape[0]
        uvp_obs = np.zeros((K, 3), dtype=np.float32)
        # Optionally query pressure on obstacle surface too:
        # fem_vals_obs = fem_predict(u_sol, p_sol, msh, obs_pts, cfg)
        # uvp_obs[:, 2] = fem_vals_obs[:, 4]

        output_list.append(np.concatenate([uvp_trunk, uvp_obs], axis=0))

    output_arr = np.stack(output_list, axis=0).astype(np.float32)  # (N_geom, M_trunk+K, 3)
    
    # Extended trunk (same for all geometries structurally, but obstacle pts differ
    # per geometry — use first geometry's obstacle pts as canonical for the data object,
    # since obstacle BC is enforced via supervised labels not collocation)
    first_obs = obstacle_pts_by_geom[geom_list[0]]
    extended_trunk = np.concatenate([trunk_pts, first_obs], axis=0).astype(np.float32)
 
    return output_arr, extended_trunk


def build_output_array_from_cache(
    fem_data_dict: dict,
    trunk_pts: np.ndarray,
    obstacle_pts_by_geom: dict,
    cfg: StenosisConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the supervised output array using only cached FEM .npz data
    (no live FEniCS objects required). Uses nearest-neighbour lookup from
    the cached FEM grid to the shared trunk points.

    Accuracy note: nearest-neighbour interpolation introduces small errors
    near the obstacle boundary where the FEM grid spacing is coarsest.
    Use build_output_array() with live FEM objects for production runs;
    use this version for rapid prototyping when FEM objects aren't available.

    Parameters
    ----------
    fem_data_dict        : {(a,b): (N_fem_pts, 5)} with columns [x,y,u,v,p]
    trunk_pts            : (M, 2) shared trunk from build_shared_trunk()
    obstacle_pts_by_geom : {(a,b): (K, 2)} from build_shared_trunk()
    cfg                  : StenosisConfig

    Returns
    -------
    output_arr     : (N_geom, M_trunk + K, 3)  float32
    extended_trunk : (M_trunk + K, 2)  float32
    """
    from scipy.spatial import cKDTree

    geom_list   = list(fem_data_dict.keys())
    output_list = []

    for (a, b) in geom_list:
        fem_data  = fem_data_dict[(a, b)]
        fem_xy    = fem_data[:, 0:2]
        fem_uvp   = fem_data[:, 2:5]

        # Nearest-neighbour lookup: trunk_pts -> FEM grid
        tree = cKDTree(fem_xy)
        _, idx_trunk = tree.query(trunk_pts)
        uvp_trunk = fem_uvp[idx_trunk].astype(np.float32)   # (M_trunk, 3)

        # Obstacle surface: zero velocity (no-slip)
        obs_pts = obstacle_pts_by_geom[(a, b)]
        uvp_obs = np.zeros((obs_pts.shape[0], 3), dtype=np.float32)

        output_list.append(np.concatenate([uvp_trunk, uvp_obs], axis=0))

    output_arr = np.stack(output_list, axis=0)  # (N_geom, M_trunk + K, 3)
    
    first_obs = obstacle_pts_by_geom[geom_list[0]]
    extended_trunk = np.concatenate([trunk_pts, first_obs], axis=0).astype(np.float32)
 
    return output_arr, extended_trunk


def build_pde_data_object(geometries: list[tuple],
                          cfg: StenosisConfig):
    """Construct the data.PDE object, which handles DeepONet trunk points"""
    
    # Build trunk geometry (intersection-masked)
    channel     = dde.geometry.Rectangle([-cfg.L/2, 0], [cfg.L/2, cfg.H_max])
    largest_a, largest_b = find_largest_ellipse(geometries)
    obstruction = dde.geometry.Ellipse([cfg.x_c, cfg.y_c], largest_a, largest_b)
    trunk_geom  = dde.geometry.CSGDifference(channel, obstruction)

    pde_data = dde.data.PDE(
        geometry=trunk_geom,
        pde=pde_loss_operator(cfg),       # 3-argument form: (x, outputs, X_func)
        bcs=[],
        num_domain=cfg.n_interior,
        num_boundary=cfg.n_boundary,
        num_test=cfg.n_test,
    )
    
    # inject PointSetBCs after construction since we need PDE's points
    trunk_pts = pde_data.train_x_all
    bcs = build_channel_bcs(trunk_pts, cfg)
    pde_data.bcs = bcs
    
    return pde_data


# ─────────────────────────────────────────────────────────────
# SECTION 5: MODEL BUILD
# ─────────────────────────────────────────────────────────────

class StenosisGeometrySpace(dde.data.function_spaces.FunctionSpace):
    """
    Function space over stenosis geometries, parameterised by ellipse (a, b).
    Each "function" is the SDF field of one geometry, evaluated at the sensor grid.
    
    random(size)         --> samples (a,b) pairs from the training set
    eval_one(feature, X) --> evaluates SDF for one geometry at points X
    eval_batch(features, X) --> evaluates SDF for N geometries at points X
    """

    def __init__(self, cfg: StenosisConfig, geometries: list[tuple]):
        self.cfg = cfg
        self.geometries = geometries        # the finite training set
        self.rng = np.random.default_rng(cfg.seed)

    def random(self, size: int) -> np.ndarray:
        """
        Sample `size` geometries (with replacement) from the training set.
        Returns an array of shape (size, 2) — each row is an (a, b) pair.
        These are the "features" that parameterise each function instance.
        """
        idx = self.rng.integers(0, len(self.geometries), size=size)
        return np.array(self.geometries)[idx]   # (size, 2)

    def eval_one(self, feature: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the SDF for one geometry at points X.
        feature : (2,) array = [a, b]
        X       : (n_points, 2) = sensor grid
        Returns : (n_points,) SDF values
        """
        a, b = feature[0], feature[1]
        return compute_sdf(X, self.cfg, a, b)

    def eval_batch(self, features: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Evaluate SDF for N geometries at the same points X.
        features : (N, 2) array of (a, b) pairs
        X        : (n_points, 2) sensor grid
        Returns  : (N, n_points) SDF values — this is the branch input array
        """
        return np.stack([self.eval_one(f, X) for f in features],
                        axis=0,).astype(np.float32)

def build_deeponet_model(
    geometries: list[tuple],
    pde_data: dde.data.PDE,
    sensors: np.ndarray,
    extended_trunk: np.ndarray,
    output_arr: np.ndarray,
    cfg: StenosisConfig,
    p: int = 120,               # TODO CHANGE BACK TO 128
) -> dde.Model:
    """
    Instantiate the PI-DeepONet model.

    Parameters
    ----------
    geometries       : list of (a, b) to instantiate the model function space with
    pde_data         : DeepXDE PDE data object handling trunk points (geometry, PDE loss, channel BCs)
    sensors          : (n_sensors, 2)       - fixed sensor grid points [x,y]
    extended_trunk   : (M_trunk + K, 2)     - shared trunk + first-geometry obstacle pts
    output_arr       : (N_geom, M + K, 3)   - FEM- and BC-labeled data [u,v,p] for each geometry
    cfg              : StenosisConfig
    p                : inner product latent dimension
    """
    
    # PDE Operator data object
    data = dde.data.PDEOperatorCartesianProd(
        pde=pde_data,
        function_space=StenosisGeometrySpace(cfg, geometries),
        evaluation_points=sensors,              # (800, 2) — branch discretization
        num_function=len(geometries),
    )
    
    # Manual insertion of labeled and trunk data
    data.train_x = (data.train_x[0], extended_trunk)    # extended trunk creates (M+K) dimension
    data.train_y = output_arr   # (N_geom, M_trunk + K, 3) — supervised labels
    
    # layer sizes
    branch_layers = [sensors.shape[0]] + cfg.branch_net_hidden_layers + [p*3]   # last layer = 3 * num_outputs
    trunk_layers = [2] + cfg.trunk_net_hidden_layers + [p]                      # matches PINN hidden width
    
    # Neural Operator object
    net = dde.nn.DeepONetCartesianProd(
        branch_layers,
        trunk_layers,
        activation="tanh",
        kernel_initializer="Glorot uniform",
        num_outputs=3,                    # u, v, p — each gets its own inner product + bias
        multi_output_strategy="split_branch",
    )
    
    print(data.train_x[1])
    
    return dde.Model(data, net)


# ─────────────────────────────────────────────────────────────
# SECTION 6: MODEL TRAINING & INFERENCE
# ─────────────────────────────────────────────────────────────

class LossMagnitudeReweighter(dde.callbacks.Callback):
    """Rebalances loss term magnitudes every `period` steps (same as pinn.py)."""
    def __init__(self, period=2000, alpha=0.8, min_w=0.1, max_w=500.0):
        super().__init__()
        self.period = period; self.alpha = alpha
        self.min_w = min_w; self.max_w = max_w

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
        print(f"\n[AdaptiveWeights @ iter {it}] {np.round(blended, 2).tolist()}")

def train_deeponet(
    model: dde.Model,
    model_prefix: Path,
    cfg: StenosisConfig,
    n_adam: int = None,
    n_lbfgs: int = None,
) -> dde.Model:
    """
    Two-stage Adam → L-BFGS training. Loss weight ordering matches build_deeponet_model()
    docstring. Verify with model.data.losses_names before tuning weights.
    """
    n_adam  = n_adam  or cfg.n_adam
    n_lbfgs = n_lbfgs or cfg.n_lbfgs

    model.compile("adam", lr=cfg.lr, loss_weights=cfg.loss_weights_deeponet)
    reweighter = LossMagnitudeReweighter(period=2000)

    start_time = time.time()
    start_ts   = datetime.datetime.now().isoformat()

    loss_history_1, train_state_1 = model.train(
        iterations=n_adam,
        callbacks=[reweighter],
        display_every=1000,
    )

    model.compile("L-BFGS", loss_weights=model.loss_weights)
    dde.optimizers.config.set_LBFGS_options(
        gtol=cfg.gtol_lbfgs, ftol=cfg.ftol_lbfgs,
        maxiter=n_lbfgs, maxfun=n_lbfgs * 10,
    )
    loss_history_2, train_state_2 = model.train(
        display_every=1000,
        model_save_path=model_prefix,
    )

    elapsed = int(time.time() - start_time)
    mm, ss  = divmod(elapsed, 60)

    dde.saveplot(loss_history_2, train_state_2,
                 issave=True, isplot=False, output_dir=str(model_prefix.parent))

    metadata = {
        "start_timestamp": start_ts,
        "end_timestamp":   datetime.datetime.now().isoformat(),
        "elapsed_time":    f"{mm}m {ss}s",
        "n_adam":          n_adam,
        "n_lbfgs":         getattr(train_state_2, "iteration", None) - n_adam,
        "sensor_nx":       cfg.sensor_nx,
        "sensor_ny":       cfg.sensor_ny,
        "n_sensors":       cfg.sensor_nx * cfg.sensor_ny,
        "architecture":    "DeepONetCartesianProd + PDEOperatorCartesianProd",
        "trunk_strategy":  "intersection masking (largest ellipse)",
    }
    (model_prefix.parent / "training_log.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"DeepONet training completed in {mm}m {ss}s")
    return model


def deeponet_predict(
    model: dde.Model,
    sensors: np.ndarray,
    cfg: StenosisConfig,
    a: float,
    b: float,
    query_pts: np.ndarray,
) -> np.ndarray:
    """
    Zero-shot prediction for a new geometry (a, b) at arbitrary query points.

    Parameters
    ----------
    model      : trained dde.Model
    sensors    : (N_sensors, 2) — must match the sensor grid used at training time
    cfg        : StenosisConfig
    a, b       : ellipse semi-axes of the target geometry
    query_pts  : (M, 2) evaluation locations (should be in the fluid domain)

    Returns
    -------
    (M, 5) array with columns [x, y, u, v, p]
    """
    sdf_field = compute_sdf(sensors, cfg, a, b)
    branch_in = sdf_field[np.newaxis, :].astype(np.float32)   # (1, N_sensors)
    trunk_in  = query_pts.astype(np.float32)                   # (M, 2)

    pred = model.predict((branch_in, trunk_in))   # (1, M, 3) or (M, 3)
    if pred.ndim == 3:
        pred = pred[0]

    return np.concatenate([query_pts[:, :2], pred], axis=1)


def restore_deeponet(model: dde.Model, model_prefix: Path, cfg: StenosisConfig) -> dde.Model:
    """Restore saved weights into a model built with matching architecture."""
    model.compile("adam", lr=cfg.lr)
    checkpoints = list(model_prefix.parent.glob(f"{model_prefix.name}-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints at {model_prefix}")
    latest = max(checkpoints, key=lambda p: int(p.stem.split("-")[-1]))
    ckpt = torch.load(latest, map_location="cpu")
    model.net.load_state_dict(ckpt["model_state_dict"])
    print(f"Restored DeepONet weights from {latest.name}")
    return model
