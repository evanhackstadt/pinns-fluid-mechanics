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
    # List of (a, b) ellipse semi-axis pairs to train/evaluate over.
    cases: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.2, 0.1), (0.4, 0.3)]
    )
    
    
    # --- Physics ---
    Re: float = 100         # Reynold's number = rho•U•L/µ, for nondimensionalization
    u_in_max: float = 1.0   # max inlet velocity (will be at H/2 centerline)
    P_out: float = 0.0      # outlet pressure
    U_ref: float = 1.0      # rerence x-velocity for nondimensionalization
                            # P_ref = rho * U_ref^2, with rho=1 --> P_ref = 1.0
    
    
    # --- PINN ---
    n_interior: int = 2000   # default 4000, can tune. Fed to PDE loss.
    n_boundary: int = 800    # default 800, can tune. Fed to data loss.
    n_test: int = 2000       # default 2000, can tune. Sampled from both interior & boundary.
    
    layers: List[int] = field(default_factory=lambda: [3, 128, 128, 3])     # (x,y,h)->...->(u,v,p)
    
    # adam
    n_adam: int = 20000         # train for N iterations with Adam
    lr: float = 1e-3            # Adam learning rate
    loss_weights_adam: List[float] = field(
        # pde_cont, pde_xm, pde_ym, bc_u_in, bc_v_in, bc_wall_u, bc_wall_v, bc_p_out
        default_factory=lambda: [1, 10, 1, 10, 10, 10, 10, 10]
    )
    
    # l-bfgs
    n_lbfgs: int = 20000        # max iterations on L-BFGS
    gtol_lbfgs: float = 1e-10    # tight gradient tolerance stopping criteria for L-BFGS, default=1e-7
    ftol_lbfgs: float = 0.0
    loss_weights_lbfgs: List[float] = field(
        # pde_cont, pde_xm, pde_ym, bc_u_in, bc_v_in, bc_wall_u, bc_wall_v, bc_p_out
        default_factory=lambda: [1, 10, 1, 100, 100, 100, 100, 100]
    )
    
    iterations_to_save: List[int] = field(
        default_factory=lambda: [1000, 5000, 10000, 20000]
    )
    
    
    # --- FEM ---
    mesh_size: float = 0.04
    
    
    # --- Visualization ---
    nx = 200
    ny = 100
    
    
    # --- Path Management ---
    '''
    3-stenosis-inlet-u/
        meshes/
        scripts/
        results/
            a0.5_b0.3_Re100/
                fem/
                pinn/
                plots/
            ...
    '''
    
    base_dir: str = "/Users/evan/Documents/GitHub/pinns-fluid-mechanics/examples/3-stenosis-inlet-u/"
    
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