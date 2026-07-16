# config.py


from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class StenosisConfig:
    
    # --- Geometry Constants ---
    L: float = 2.0          # length
    H_max: float = 1.0      # height of channel unobstructed
    x_c: float = 0.0        # ellipse center x
    y_c: float = 1.0        # ellipse center y
    angle: float = 0.0      # ellipse angle, keep 0 for now (can add to variables)
    
    # --- Geometry Variables ---
    # List of (a, b) ellipse semi-axis pairs to train/evaluate over.
    cases: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.4, 0.1),    # wide and shallow, easy
                                #  (0.4, 0.4)     # wide and deeper, hard
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
    n_interior: int = 1000  # default 2000, can tune. Fed to PDE loss.
    n_boundary: int = 500   # default 800, can tune. Fed to BC loss.
    n_test: int = 500      # default 2000, can tune. Sampled from both interior & boundary.
    n_labeled: List[int] = field(
        # try different N of labeled data.
        default_factory=lambda: [0, 3, 5, 10, 25, 100, 250]
    )
    
    layers: List[int] = field(default_factory=lambda: [3, 128, 128, 3])     # (x,y,h)->...->(u,v,p)
    
    # adam
    n_adam: int = 15000         # train for N iterations with Adam
    lr: float = 1e-3            # Adam learning rate
    loss_weights_adam: List[float] = field(     # will be reweighted dynamically during training
        default_factory=lambda: [10, 10, 10,    # pde_cont, pde_xm, pde_ym
                                 5, 5,          # bc_inlet_u, bc_inlet_v
                                 50, 50,        # bc_wall_u, bc_wall_v  <-- important hard constraints
                                 5,             # bc_outlet_p
                                 10, 10, 10]    # bc_obs_u, bc_obs_v, bc_obs_p
    )
    
    # l-bfgs
    n_lbfgs: int = 15000        # max iterations on L-BFGS
    gtol_lbfgs: float = 1e-10    # tight gradient tolerance stopping criteria for L-BFGS, default=1e-7
    ftol_lbfgs: float = 0.0
    
    
    # --- FEM ---
    mesh_size: float = 0.04
    
    
    # --- Visualization ---
    nx = 200
    ny = 100
    
    
    # --- Path Management ---
    '''
    4-supervised/
        meshes/
        scripts/
        results/
            a0.5_b0.3_n25/
                fem/
                pinn/
                plots/
            ...more cases...
            summary_plots/
            summary.json
    '''
    
    base_dir: Path = Path(__file__).resolve().parents[1]
    
    @property
    def meshes_dir(self) -> Path:
        return self.base_dir / "meshes"

    @property
    def results_dir(self) -> Path:
        return self.base_dir / "results"

    def case_tag(self, a, b, n):
        return f"n{n}_a{a:.2f}_b{b:.2f}"

    def case_dirs(self, a, b, n):
        tag = self.case_tag(a, b, n)
        base = self.results_dir / tag
        summary = self.results_dir / "summary_plots"
        return {
            "base":   base,
            "fem":    base / "fem",
            "pinn":   base / "pinn",
            "plots":  base / "plots",
            "summary_plots": summary
        }

    def make_dirs(self, a, b, n):
        for d in self.case_dirs(a, b, n).values():
            d.mkdir(parents=True, exist_ok=True)
    
    def clear_pinn(self, a, b, n):
        target = self.case_dirs(a, b, n)["pinn"]
        target.mkdir(parents=True, exist_ok=True)
        for f in target.iterdir():
            if f.is_file():
                f.unlink()
    
    def config_as_dict(self, a=None, b=None, n=None):
        return {
            "Geometry": {
                "L": self.L,
                "H_max": self.H_max,
                "x_c": self.x_c,
                "y_c": self.y_c,
                "angle": self.angle,
                "a": a,
                "b": b
            },
            "Physics": {
                "Re": self.Re,
                "n_in_max": self.u_in_max,
                "P_out": self.P_out,
                "U_ref": self.U_ref
            },
            "PINN": {
                "n_labeled": n,
                "n_interior": self.n_interior,
                "n_boundary": self.n_boundary,
                "n_test": self.n_test,
                "n_adam": self.n_adam,
                "lr": self.lr,
                "loss_weights_adam": self.loss_weights_adam,
                "n_lbfgs": self.n_lbfgs,
                "gtol_lbfgs": self.gtol_lbfgs,
                "ftol_lbfgs": self.ftol_lbfgs
            },
            "FEM": {
                "mesh_size": self.mesh_size
            },
            "Misc": {
                "nx": self.nx,
                "ny": self.ny
            }
        }
        
        