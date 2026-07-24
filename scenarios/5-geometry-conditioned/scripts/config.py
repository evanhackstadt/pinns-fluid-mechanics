# config.py

"""
2D Stenosis Geometry-Conditioned PINN
    Custom config defining key parameters for the problem setup.

Evan Hackstadt
Rugonyi Lab
"""


from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple


@dataclass
class StenosisConfig:
    
    # --- Geometry Constants ---
    L: float = 2.0          # length
    H_max: float = 1.0      # height of channel unobstructed
    x_c: float = 0.0        # ellipse center x
    y_c: float = 1.0        # ellipse center y
    
    # --- Geometry Variables ---
    # List of (a, b) = ellipse semimajor, semiminor
    
    train_geometries: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.4, 0.3),
            (0.6, 0.5)
        ]
    )
    
    test_geometries: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.5, 0.4)
        ]
    )
    
    
    # --- Physics ---
    Re: float = 100         # Reynold's number = rho•U•L/µ, for nondimensionalization
    u_in_max: float = 1.0   # max inlet velocity (will be at H/2 centerline)
    P_out: float = 0.0      # outlet pressure
    U_ref: float = 1.0      # rerence x-velocity for nondimensionalization
                            # P_ref = rho * U_ref^2, with rho=1 --> P_ref = 1.0
    
    
    # --- PINN ---
    seed: int = 0
    n_interior: int = 2000      # default 2000, can tune. Fed to PDE loss.
    n_boundary: int = 800       # default 800, can tune. Fed to BC loss.
    n_test: int = 1400          # default 2000, can tune. Sampled from both interior & boundary.
    n_labeled_train: int = 10   # default 10, can tune.
    n_labeled_test:  int = 3    # default 3, can tune.
    uniform_frac: float = 0.3
    
    layers: List[int] = field(default_factory=lambda: [4, 128, 128, 3])     # (x,y,a,b)->...->(u,v,p)
    
    # train adam
    n_adam: int = 25000         # train for N iterations with Adam
    lr: float = 1e-3            # Adam learning rate
    loss_weights_adam: List[float] = field(     # will be reweighted dynamically during training
        default_factory=lambda: [10, 10, 10,    # pde_cont, pde_xm, pde_ym
                                 5, 5,          # bc_inlet_u, bc_inlet_v
                                 25, 25,        # bc_wall_u, bc_wall_v  <-- important hard constraints
                                 25,            # bc_outlet_p
                                 10, 10, 10]    # bc_obs_u, bc_obs_v, bc_obs_p
    )
    
    # train l-bfgs
    n_lbfgs: int = 25000        # max iterations on L-BFGS
    gtol_lbfgs: float = 1e-10   # tight gradient tolerance stopping criteria for L-BFGS, default=1e-7
    ftol_lbfgs: float = 0.0
    
    # fine-tune adam
    n_adam_finetune: int = 1000
    lr_finetune: float = 1e-5
    loss_weights_finetune: List[float] = field(     # TODO
        default_factory=lambda: [1, 1, 1,   # PDE terms
                                 5, 5,      # bc inlet u, v
                                 50, 50,    # bc walls u, v
                                 5,         # bc outlet p
                                 100,       # observed velocity bc
                                 10]        # weight anchor (regularization)
    )
    
    
    # --- FEM ---
    mesh_size: float = 0.04
    
    
    # --- Visualization ---
    nx = 200    # heatmap mesh
    ny = 100
    
    
    # --- Path Management ---
    '''
    5-geometry-conditioned/
        fem/
            labeled_data_train_geometries.csv
            labeled_data_test_geometries.csv
            solution_{geo_tag}.npz
        meshes/
            stenosis_{geo_tag}.msh
        results/
            errors.json
            config_log.json
            pinn/
                training_log.json
                *.pt
                *.dat
            plots/
                train/...geos...
                test/...geos...
                summary/
        scripts/
    '''
    
    base_dir: Path = Path(__file__).resolve().parents[1]
    
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
    def pinn_dir(self) -> Path:
        return self.results_dir / "pinn"
    
    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plots"
    
    @property 
    def summary_dir(self) -> Path:
        return self.plots_dir / "summary"
    
    
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
    
    def plots_geo_dir(self, a, b):
        if self.train_or_test(a, b) is not None:
            return self.plots_dir / self.train_or_test(a, b) / self.geo_tag(a, b)
        else:
            return None

    
    def make_all_dirs(self):
        self.fem_dir.mkdir(parents=True, exist_ok=True)
        self.meshes_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.pinn_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        for (a, b) in self.train_geometries:
            self.plots_geo_dir(a, b).mkdir(parents=True, exist_ok=True)
        for (a, b) in self.test_geometries:
            self.plots_geo_dir(a, b).mkdir(parents=True, exist_ok=True)
    
    def clear_dir(self, target):
        target.mkdir(parents=True, exist_ok=True)
        for f in target.iterdir():
            if f.is_file():
                f.unlink()
    
    
    def config_as_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}