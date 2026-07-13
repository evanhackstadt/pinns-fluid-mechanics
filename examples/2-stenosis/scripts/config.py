# config.py


from dataclasses import dataclass, field
from typing import List, Tuple
import os
import json


@dataclass
class StenosisConfig:
    
    # --- Geometry Constants ---
    L: float = 2.0          # length
    H_max: float = 1.0      # height of channel unobstructed
    x_c: float = 0.0        # ellipse center x
    y_c: float = 1.0        # ellipse center y
    angle: float = 0.0      # ellipse angle, keep 0 for now (can add to variables)
    
    # --- Geometry Variables ---
    # List of (a, b) ellipse semi-axis pairs to train/evaluate over. Comment out as desired.
    
    # Single case:
    cases: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.4, 0.3)]
    )
    
    # 3 cases:
    # cases: List[Tuple[float, float]] = field(
    #     default_factory=lambda: [(0.35, 0.2), (0.5, 0.3), (0.65, 0.4)]
    # )
    
    # --- Physics ---
    Re: float = 100    # Reynold's number = rho•U•L/µ, for nondimensionalization
    P1: float = 1.0    # inlet pressure
    P2: float = 0.0    # outlet pressure
    
    # --- PINN ---
    n_interior: int = 2000   # default 4000, can tune. Fed to PDE loss.
    n_boundary: int = 800    # default 800, can tune. Fed to data loss.
    n_test: int = 2000       # default 2000, can tune. Sampled from both interior & boundary.
    
    layers: List[int] = field(default_factory=lambda: [3, 128, 128, 3])     # (x,y,h)->...->(u,v,p)
    loss_weights: List[float] = field(
        default_factory=lambda: [1, 100, 1, 100, 100, 100, 100]   # pde_cont, pde_xm, pde_ym, bc_p_in, bc_p_out, bc_wall_u, bc_wall_v
    )
    
    n_adam: int = 10000         # train for N iterations with Adam
    lr: float = 1e-3            # Adam learning rate
    
    n_lbfgs: int = 20000        # max iterations on L-BFGS
    gtol_lbfgs: float = 1e-10    # tight gradient tolerance stopping criteria for L-BFGS, default=1e-7
    ftol_lbfgs: float = 1e-15    # tight function tolerance, near machine epsilon for float64
    
    iterations_to_save: List[int] = field(
        default_factory=lambda: [1000, 5000, 10000, 20000]
    )
    
    # --- FEM ---
    mesh_size: float = 0.04
    
    
    # --- Path Management ---
    '''
    2-stenosis/
        meshes/
        scripts/
        results/
            a0.5_b0.3_Re100/
                fem/
                pinn/
                plots/
            ...
    '''
    
    base_dir: str = "/Users/evan/Documents/GitHub/pinns-fluid-mechanics/examples/2-stenosis/"
    
    @property
    def meshes_dir(self):
        return os.path.join(self.base_dir, "meshes")

    @property
    def results_dir(self):
        return os.path.join(self.base_dir, "results")

    def case_tag(self, a, b):
        return f"a{a:.2f}_b{b:.2f}_Re{self.Re:.0f}"

    def case_dirs(self, a, b):
        tag = self.case_tag(a, b)
        base = os.path.join(self.results_dir, tag)
        return {
            "base":   base,
            "fem":    os.path.join(base, "fem"),
            "pinn":   os.path.join(base, "pinn"),
            "plots":  os.path.join(base, "plots"),
        }

    def make_dirs(self, a, b):
        for d in self.case_dirs(a, b).values():
            os.makedirs(d, exist_ok=True)
    
    def clear_pinn(self, a, b):
        target = self.case_dirs(a, b)["pinn"]
        for f in os.listdir(target):
            os.remove(os.path.join(target, f))