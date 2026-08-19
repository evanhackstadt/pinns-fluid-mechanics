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
    
    # Sampling (overriden by manual lists below)
    n_train_geometries: int = 100
    n_test_geometries: int = 5
    a_range: List[int] = field(default_factory=lambda: [0.15, 0.65])
    b_range: List[int] = field(default_factory=lambda: [0.15, 0.65])
    

    # Manual lists of (a, b) = ellipse semimajor, semiminor; where a>b
    # When either of these exist, they override the geometry sampling above
    '''
    train_geometries: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.3, 0.25),
            (0.55, 0.45),
            (0.65, 0.60)
        ]
    )
    '''
    
    test_geometries: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.10, 0.10),   # extrapolation, beyond
            (0.20, 0.15),   # extrapolation, border
            (0.35, 0.30),   # interpolation
            (0.60, 0.50),   # interpolation
            (0.70, 0.60),   # extrapolation, border
            (0.70, 0.70),   # extrapolation, beyond
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
    sensor_nx = 20     # fixed grid to sample SDF input function,
    sensor_ny = 10     # held constant across geos b/c Cartesian Product
    # Default 40x20 = 800 sensors
    
    n_interior: int = 2000     # default 2000, can tune. Fed to PDE loss.
    n_boundary: int = 600      # default 800, can tune. Fed to BC loss.
    n_obstacle: int = 100      # default 200, can tune.
    n_labeled_train: int = 100
    uniform_frac: float = 0.3
    
    n_functions: int = 10        # geometries sampled per training step (batch size)
    
    branch_net_hidden_layers: List[int] = field(
        default_factory=lambda: [256, 128, 64]     # neurons between input and latent dimension
        # Larger first layer because N_sensors (800) is the raw input
    )
    trunk_net_hidden_layers: List[int] = field(
        default_factory=lambda: [128, 128, 128]     # neurons between input and latent dimension
    )
    latent_dim_p: int = 64
    
    # Training (stage 1)
    n_adam_1: int = 2000     # iterations
    lr_1: float = 5e-4      # learning rate
    loss_weights_1: List[float] = field(
        default_factory=lambda: [
            # ignore PDE loss and BCs
            0, 0, 0, 0,     # PDE cont, xm, ym, obstacle no-slip
            0, 0, 0, 0, 0,  # BCs
            25, 25, 25   # labeled u, v, p
        ]
    )
    
    # Training (stage 2)
    n_adam_2: int = 20000
    lr_2: float = 1e-3
    lr_2_min: float = 1e-4     # min lr at the end of cosine decay (pytorch eta_min)
    loss_weights_2: List[float] = field(
        default_factory=lambda: [
            10, 10, 10,   # PDE cont, xm, ym
            100,          # obstacle no-slip
            5, 5,         # BC inlet u, v
            25, 25,       # BC wall u, v
            5,            # BC outlet p
            25, 25, 25]   # labeled u, v, p
    )
    
    # Training (stage 3)
    n_lbfgs: int = 10000        # max iterations on L-BFGS
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
        return f"a{a:.3g}_b{b:.3g}"
    
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