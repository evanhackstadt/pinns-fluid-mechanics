# config.py

"""
2D Stenosis - Physics-Informed Deep Operator Network
    Custom config defining key parameters for the problem setup.

Evan Hackstadt
Rugonyi Lab
"""


from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict


@dataclass
class StenosisConfig:
    
    # --- Geometry Constants ---
    L: float = 2.0          # length
    H_max: float = 1.0      # height of channel unobstructed
    x_c: float = 0.0        # ellipse center x
    y_c: float = 1.0        # ellipse center y
    
    # --- Geometry Variables ---
    # List of (a, b) = ellipse semimajor, semiminor; where a>b
    
    train_geometries: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.3, 0.25),
            (0.55, 0.45),
            (0.65, 0.60)
        ]
    )
    
    test_geometries: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.25, 0.20),   # extrapolation
            (0.45, 0.35),   # interpolation
            (0.60, 0.50),   # interpolation
            (0.65, 0.65),   # extrapolation
        ]
    )
    
    
    # --- Physics ---
    Re: float = 100         # Reynold's number = rho•U•L/µ, for nondimensionalization
    u_in_max: float = 1.0   # max inlet velocity (will be at H/2 centerline)
    p_out: float = 0.0      # outlet pressure
    u_ref: float = 1.0      # rerence x-velocity for nondimensionalization
                            # P_ref = rho * U_ref^2, with rho=1 --> P_ref = 1.0
    
    
    # --- DeepONet ---
    seed: int = 0
    sensor_nx = 40      # fixed grid to sample SDF input function,
    sensor_ny = 20      # held constant across geos b/c Cartesian Product
    # Default 40x20 = 800 sensors
    
    n_interior: int = 2000      # default 2000, can tune. Fed to PDE loss.
    n_boundary: int = 800       # default 800, can tune. Fed to BC loss.
    n_obstacle: int = 100       # default 200, can tune.
    n_test: int = 800
    
    n_functions: int = 10        # geometries sampled per training step
    n_functions_test: int = 5    # geometries used for test PDE loss

    loss_weights_deeponet: List[float] = field(
        default_factory=lambda: [10, 10, 10,   # PDE cont, xm, ym
                                100,          # obstacle no-slip
                                5, 5,         # BC inlet u, v
                                25, 25,       # BC wall u, v
                                5]            # BC outlet p
    )
    
    branch_net_hidden_layers: List[int] = field(
        default_factory=lambda: [256, 256, 128]     # neurons between input and latent dimension
        # Larger first layer because N_sensors (800) is the raw input
    )
    trunk_net_hidden_layers: List[int] = field(
        default_factory=lambda: [128, 128, 128]     # neurons between input and latent dimension
    )
    loss_weights_deeponet: List[float] = field(     # will be reweighted dynamically during training
        default_factory=lambda: [10, 10, 10,    # pde_cont, pde_xm, pde_ym
                                 5, 5,          # bc_inlet_u, bc_inlet_v
                                 25, 25,        # bc_wall_u, bc_wall_v
                                 5]             # bc_outlet_p
    )
    
    # train adam
    n_adam: int = 50000         # train for N iterations with Adam
    lr: float = 1e-3            # Adam learning rate
    
    # train l-bfgs
    n_lbfgs: int = 25000        # max iterations on L-BFGS
    gtol_lbfgs: float = 1e-10   # tight gradient tolerance stopping criteria for L-BFGS, default=1e-7
    ftol_lbfgs: float = 0.0
    
    
    # --- FEM ---
    mesh_size: float = 0.04
    
    
    # --- Visualization ---
    query_nx = 200    # FEM and heatmap mesh
    query_ny = 100
    
    
    # --- Path Management ---
    '''
    6-deeponet/
        fem/
            labeled_data_train_geometries.csv
            labeled_data_test_geometries.csv
            solution_{geo_tag}.npz
        meshes/
            stenosis_{geo_tag}.msh
        results/
            config_log.json
            error_summary/
            deeponet/
                training_log.json
                *.pt
                *.dat
                loss_curves*.png
            test_geometries/...geos.../
            train_geometries/...geos.../
        scripts/
    '''
    
    base_dir: Path = Path(__file__).resolve().parents[1]
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"
    
    @property
    def fem_dir(self) -> Path:
        return self.base_dir / "fem"
    
    @property
    def meshes_dir(self) -> Path:
        return self.base_dir / "meshes"

    @property
    def results_dir(self) -> Path:
        return self.base_dir / "results"
    
    @property
    def deeponet_dir(self) -> Path:
        return self.results_dir / "deeponet"
    
    @property 
    def summary_dir(self) -> Path:
        return self.results_dir / "error_summary"
    
    
    def geo_tag(self, a, b):
        return f"a{a:.2f}_b{b:.2f}"
    
    def infer_geo(self, geo_tag: str):
        a_str, b_str = geo_tag.split("_")
        a = a_str[1:]
        b = b_str[1:]
        return a, b
    
    def train_or_test(self, a, b):
        if (a, b) in self.train_geometries:
            return "train"
        elif (a, b) in self.test_geometries:
            return "test"
        else:
            return None
    
    def geo_dir(self, a, b):
        if self.train_or_test(a, b) is not None:
            return self.results_dir / self.train_or_test(a, b) / self.geo_tag(a, b)
        else:
            return None

    
    def make_all_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fem_dir.mkdir(parents=True, exist_ok=True)
        self.meshes_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.deeponet_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
        for (a, b) in self.train_geometries:
            self.geo_dir(a, b).mkdir(parents=True, exist_ok=True)
        for (a, b) in self.test_geometries:
            self.geo_dir(a, b).mkdir(parents=True, exist_ok=True)
    
    def clear_dir(self, target):
        target.mkdir(parents=True, exist_ok=True)
        for f in target.iterdir():
            if f.is_file():
                f.unlink()
    
    
    def config_as_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}