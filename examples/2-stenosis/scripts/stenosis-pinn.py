"""
Navier-Stokes PINN - 2D Parameterized Stenosis.

Added Complexity from Poiseuille Flow:
    Geometry - variable top wall height defined by ellipse
    Inputs - position and h(x) for eventual training across geometries
    Outputs - predict v, p, and 
    Loss - non-dimensionalization to normalize magnitudes (use Re)

Spatial domain:
    2D rectangle (L, H_MAX) obstructed by an ellipse on the top wall
    Explicitly defined with CSG Difference

Known Values:
    L, H_MAX, ellipse params
    Reynold's number (Re) = 100
    Inlet pressure = 1.0
    Outlet pressure = 0.0

Explicit Navier-Stokes PDE:
    ∂u/∂x + ∂v/∂y = 0
    u•∂u/∂x + v•∂u/∂y + ∂p/∂x - (1/RE)•(∂2u/∂x2 + ∂2u/∂y2) = 0
    u•∂v/∂x + v•∂v/∂y + ∂p/∂y - (1/RE)•(∂2v/∂x2 + ∂2v/∂y2) = 0

PINN Model
    - Inputs: (x, y, h) = (x-position, y-position, channel height)
    - Outputs: (u, v, p) = (x-velocity, y-velocity, pressure)
    - Data:
        Interior collocation points (x,y) --> u,v,p --> auto-diff --> PDE loss
        Boundary condition points (x,y) --> u,v,p --> BC Loss
    - Loss:
        L_pde = residuals from the NS PDEs above
        L_bc = residuals from conditions (u=0 at walls, inlet pressure, outlet pressure)
        L_total = sum(L_pde) + sum(L_bc)

Evan Hackstadt
Rugonyi Lab
"""


import os
import re
import json, yaml
import time, datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse

import torch
import deepxde as dde


# ———————————— SETUP ————————————

# --- Load Config Variables ---
config_path = '/Users/evan/Documents/GitHub/pinns-fluid-mechanics/examples/2-stenosis/config/config.yaml'

with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

L, H_MAX = cfg['channel']['L'], cfg['channel']['H_MAX']
P1, P2, RE = cfg['channel']['P1'], cfg['channel']['P2'], cfg['channel']['RE']
ELLIPSE_X_C, ELLIPSE_Y_C = cfg['ellipse']['ELLIPSE_X_C'], cfg['ellipse']['ELLIPSE_Y_C']
ELLIPSE_A, ELLIPSE_B = cfg['ellipse']['ELLIPSE_A'], cfg['ellipse']['ELLIPSE_B']
ELLIPSE_ANGLE = cfg['ellipse']['ELLIPSE_ANGLE']

N_INTERIOR_PTS, N_BOUNDARY_PTS, N_TEST_PTS = cfg['data']['N_INTERIOR_PTS'], cfg['data']['N_BOUNDARY_PTS'], cfg['data']['N_TEST_PTS']
N_ITERATIONS, MAX_ITER = cfg['training']['N_ITERATIONS'], cfg['training']['MAX_ITER']
ITERATIONS_TO_SAVE = cfg['training']['ITERATIONS_TO_SAVE']

# --- Establish Paths ---
parent_dir = f'/Users/evan/Documents/GitHub/pinns-fluid-mechanics/examples/2-stenosis/n={N_INTERIOR_PTS+N_BOUNDARY_PTS}/'
outputs_dir = os.path.join(parent_dir, 'outputs/')
plots_dir = os.path.join(parent_dir, 'plots/')
models_dir = os.path.join(parent_dir, 'models/')
model_save_prefix = os.path.join(models_dir, 'model')
metadata_path = os.path.join(outputs_dir, 'training_log.json')

for d in [parent_dir, outputs_dir, plots_dir, models_dir]:
    if not os.path.exists(d):
        os.mkdir(d)


# ———————————— MODEL FUNCTIONS ————————————

# --- Define Ellipse Curve / Channel Height ---
def ellipse_bottom(x_arr, x_c, y_c, a, b):
    """
    Lower boundary of the ellipse at each x location.
    Returns y_bottom of ellipse if x in ellipse domain, else H.
    Note that channel height = H(x) = ellipse_bottom(x).
    """
    dx = x_arr - x_c
    inside = np.abs(dx) < a
    y_bottom = np.where(
        inside,
        y_c - b * np.sqrt(np.maximum(1 - (dx / a) ** 2, 0)),
        H_MAX
    )
    return y_bottom


# --- Define the Boundary Conditions (BCs) ---
# no need to define ellipse here since DeepXDE samples BC pts from the defined CSG geometry

# Left wall
def inlet(x, on_boundary, L):
    return on_boundary and (np.isclose(x[0], -L/2))

# Right wall
def outlet(x, on_boundary, L):
    return on_boundary and (np.isclose(x[0], L/2))

# Top/Bottom walls
def walls(x, on_boundary):
    return on_boundary and not inlet(x, on_boundary) and not outlet(x, on_boundary)


# --- Add Geometry to Inputs ---
def augment_inputs(x):
    """
    x: tensor of coordinates (N, 2), cols [x_coord, y_coord]
    Concatenates channel height parameter to each row
    Returns: augmented input tensor (N, 3), cols [x, y, h(x)]
    """
    x_coords = x[:, 0:1]
    y_coords = x[:, 1:2]
    x_c, y_c, a, b = ELLIPSE_X_C, ELLIPSE_Y_C, ELLIPSE_A, ELLIPSE_B
    
    # compute h(x) using tensors
    dx = x_coords - x_c
    radicand = (1.0 - (dx / a) ** 2).clamp(min=0.0)
    inside = dx.abs() < a
    y_bottom = torch.where(
        inside,
        y_c - b * radicand.sqrt(),
        torch.full_like(x_coords, H_MAX)
    )
    # concatenate to input
    return torch.cat((x_coords, y_coords, y_bottom), dim=1)
    


# --- Define the PDE Residual ---
def pde_loss(x, u):
    '''
    x: collocation points (x, y)
    u: model output (u, v, p) = (x-vel, y-vel, pressure)
    Returns the residual between the model-predicted values and the governing PDEs.
    '''
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


# --- Custom Callback to save the model at specified epochs ---
class EpochSaver(dde.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self):
        current_iter = model.train_state.iteration
        if current_iter in ITERATIONS_TO_SAVE:
            print(f"Saving epoch {current_iter}...")
            model.save(model_save_prefix)



# ———————————— ANALYSIS FUNCTIONS ————————————
'''
- Pred vs. true velocity curve at a vertical slice
- Pred vs. true velocity heatmap across the 2D domain
- Absolute error heatmap across the 2D domain
- Train/test loss curves over epochs
'''

CMAP_VEL   = "rainbow"
CMAP_ERR   = "hot_r"
COLOR_PINN = "#185FA5"   # blue  – PINN prediction
COLOR_TRUE = "#E8593C"   # coral – analytical ground truth
FIG_DPI    = 150


# --- Plot Domain ---
def plot_domain(L, H_max, x_c, y_c, semimajor, semiminor, angle):
    """
    Visualizes the spatial domain (ellipse obstruction on the 2D channel)
    """
    # create box
    fig, ax = plt.subplots(figsize=(6, 4), dpi=FIG_DPI)
    plt.xlim(-L/2, L/2)
    plt.ylim(0, H_max)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    
    # add the ellipse patch
    ellipse = Ellipse(xy=(x_c, y_c), 
                      width=semimajor*2, 
                      height=semiminor*2, 
                      angle=angle,
                      color='black')
    ax.add_patch(ellipse)
    
    # save plot
    plt.savefig(os.path.join(plots_dir, f"geometry.png"), dpi=FIG_DPI)
    # plt.show()
    plt.close()


# --- Grid Construction ---
def make_grid(L, H, nx=200, ny=100):
    """
    Build a dense uniform meshgrid over [0,L] x [0,H].
 
    Returns
    -------
    X, Y  : 2-D arrays of shape (ny, nx) — for plotting with pcolormesh
    pts   : 2-D array of shape (ny*nx, 2) — for model.predict() and analytical()
    """
    x_vals = np.linspace(0, L, nx)
    y_vals = np.linspace(0, H, ny)
    X, Y   = np.meshgrid(x_vals, y_vals)              # (ny, nx) each
    pts    = np.column_stack([X.ravel(), Y.ravel()])   # (N, 2)
    return X, Y, pts


# --- Velocity Slice ---
def plot_velocity_slice(model, L, H, model_iter=20000, ny=300, x_loc=None):
    """
    Plot vx(y) at a fixed x-slice: PINN prediction vs. analytical parabola.
 
    Parameters
    ----------
    x_loc : float, x-coordinate of the slice (default: L/2)
    ax    : existing Axes to draw on (creates a new figure if None)
    """
    if x_loc is None:
        x_loc = 0.0

    # get values
    y_vals = np.linspace(0, H, ny)
    pts    = np.column_stack([np.full(ny, x_loc), y_vals])   # (ny, 2)
 
    pred = model.predict(pts)        # (ny, 1)
    u_pred = pred[:, 0:1]
    v_pred = pred[:, 1:2]
    p_pred = pred[:, 2:3]
    # vx_true = analytical(pts)           # (ny, 1)

    # plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=FIG_DPI)
    # ax.plot(vx_true.ravel(), y_vals, color=COLOR_TRUE,
    #         lw=2.5, label="Analytical", zorder=3)
    ax.plot(u_pred.ravel(), y_vals, color=COLOR_PINN,
            lw=1.8, ls="--", label="PINN", zorder=4)

    # format plot
    ax.set_xlabel("$v_x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"Velocity profile at $x = {x_loc:.2f}$ \n(iterations={model_iter})")
    ax.legend(framealpha=0.9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", lw=0.4, alpha=0.4)

    # save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{model_iter}_velocity_slice.png"),
                dpi=FIG_DPI)
    # plt.show()
    plt.close()


# --- Velocity 2D Heatmap ---
def plot_output_heatmaps(model, L, H, x_c=None, y_c=None, 
                         semimajor=None, semiminor=None, angle=None, 
                         model_iter=20000, nx=200, ny=100):
    """
    2-D heatmap of each output over the full domain.
    """
    X, Y, pts = make_grid(L, H, nx, ny)

    pred = model.predict(pts)
    u_pred = pred[:, 0:1].reshape(ny, nx)
    v_pred = pred[:, 1:2].reshape(ny, nx)
    p_pred = pred[:, 2:3].reshape(ny, nx)
    
    labels = {
        "u": u_pred,
        "v": v_pred,
        "p": p_pred
    }

    for label, pred in labels.items():
        # plot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=FIG_DPI)
        pcm = ax.pcolormesh(X, Y, pred, cmap=CMAP_VEL, shading="auto")
        plt.colorbar(pcm, ax=ax, label=f"${label}$")

        # draw obstruction boundary as a dashed ellipse outline, if provided
        if x_c and y_c and semimajor and semiminor and angle:
            ellipse = Ellipse(
                xy=(x_c, y_c),
                width=semimajor * 2,
                height=semiminor * 2,
                angle=np.degrees(angle),
                edgecolor="black",
                facecolor="none",
                linestyle="--",
                linewidth=1.25,
                zorder=10,
            )
            ax.add_patch(ellipse)

        # format
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(f"PINN predicted field ${label}(x, y)$ \n(iterations={model_iter})")
        ax.set_aspect("equal")

        # save plot
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{model_iter}_{label}_heatmap.png"),
                    dpi=FIG_DPI)
        # plt.show()
        plt.close()


# --- Loss Curves ---
def plot_loss_curves(losshistory, fname="loss_curves"):
    """
    Plot PDE residual loss and BC loss separately over training iterations.
 
    losshistory : object returned by model.train() in DeepXDE.
                  losshistory.loss_train is shape (iters, n_loss_terms).
                  Columns: PDE_continuity, PDE_x_momentum, PDE_y_momentum, 
                           BC_inlet_p, BC_outlet_p, BC_wall_u, BC_wall_v
    """
    steps      = np.array(losshistory.steps)
    loss_train = np.array(losshistory.loss_train)   # (iters, n_terms)
    loss_test  = np.array(losshistory.loss_test)   # (iters, n_terms)
    
    labels = {"train": loss_train, "test": loss_test}
    
    for label, loss in labels.items():
            
        # extract individual loss terms
        pde_cont = loss[:, 0]
        pde_x_m = loss[:, 1]
        pde_y_m = loss[:, 2]
        bc_i_p = loss[:, 3]
        bc_o_p = loss[:, 4]
        bc_w_u = loss[:, 5]
        bc_w_v = loss[:, 6]

        fig, ax = plt.subplots(figsize=(7, 4), dpi=FIG_DPI)

        ax.semilogy(steps, pde_cont, color="#280CAC", lw=1.5, label="PDE (continuity)")
        ax.semilogy(steps, pde_x_m, color="#1C3FDE", lw=1.5, label="PDE (x-momentum)")
        ax.semilogy(steps, pde_y_m, color="#2A58E1", lw=1.5, label="PDE (y-momentum)")
        ax.semilogy(steps, bc_i_p,  color="#B40925", lw=1.5, label="BC (inlet pressure)")
        ax.semilogy(steps, bc_o_p,  color="#CB2D11", lw=1.5, label="BC (outlet pressure)")
        ax.semilogy(steps, bc_w_u,  color="#F44E30", lw=1.5, label="BC (wall x-velocity)")
        ax.semilogy(steps, bc_w_v,  color="#F47530", lw=1.5, label="BC (wall y-velocity)")
    
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title(f"Loss history - {label}")
        ax.legend(framealpha=0.9)
        ax.grid(True, which="both", lw=0.35, alpha=0.4)
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))
    
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{fname}_{label}_terms.png"),
                    dpi=FIG_DPI)
        # plt.show()
        plt.close()
        
        # Also save summarized PDE and BC curves
        pde_total = pde_cont + pde_x_m + pde_y_m
        bc_total = bc_i_p + bc_o_p + bc_w_u + bc_w_v

        fig, ax = plt.subplots(figsize=(7, 4), dpi=FIG_DPI)
        
        ax.semilogy(steps, pde_total, color=COLOR_PINN, lw=1.5, label="PDE Loss")
        ax.semilogy(steps, bc_total,  color=COLOR_TRUE, lw=1.5, label="BC Loss")
    
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title(f"Loss history - {label}")
        ax.legend(framealpha=0.9)
        ax.grid(True, which="both", lw=0.35, alpha=0.4)
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))
    
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{fname}_{label}_summed.png"),
                    dpi=FIG_DPI)
        # plt.show()
        plt.close()



# ———————————— MAIN ————————————

# Construct the geometry: base channel rectangle - obstructing ellipse
channel = dde.geometry.Rectangle([-L/2, 0], [L/2, H_MAX])
obstruction = dde.geometry.Ellipse([ELLIPSE_X_C, ELLIPSE_Y_C], 
                                   ELLIPSE_A,
                                   ELLIPSE_B, 
                                   ELLIPSE_ANGLE)

geometry = dde.geometry.CSGDifference(channel, obstruction)

# Define the boundary conditions
bc_inlet_p = dde.DirichletBC(geometry, lambda x: P1, inlet, component=2)    # inlet pressure=1.0
bc_outlet_p = dde.DirichletBC(geometry, lambda x: P2, outlet, component=2)   # outlet pressure=0.0
bc_wall_u = dde.DirichletBC(geometry, lambda x: 0, walls, component=0)      # no-slip walls
bc_wall_v = dde.DirichletBC(geometry, lambda x: 0, walls, component=1)


# Instantiate data object
data = dde.data.PDE(
        geometry=geometry,
        pde=pde_loss,
        bcs=[bc_inlet_p, bc_outlet_p, bc_wall_u, bc_wall_v],
        num_domain=N_INTERIOR_PTS,
        num_boundary=N_BOUNDARY_PTS,
        num_test=N_TEST_PTS
    )

# Instantiate network object
net = dde.nn.FNN(
        layer_sizes=[3] + [50]*4 + [3],     # (x, y, h) --> (u, v, p)
        activation="tanh",
        kernel_initializer="Glorot uniform"
        )
net.apply_feature_transform(augment_inputs)     # add h to (x, y) input data

# Create callbacks
epoch_saver = EpochSaver()
resampler = dde.callbacks.PDEPointResampler()

# Create the model

start_time = time.time()
start_timestamp = datetime.datetime.now().isoformat()

model = dde.Model(data, net)
model.compile("adam", 
              lr=1e-3, 
              loss_weights=[1, 1, 1, 100, 100, 100, 100]
              )

# Train the model

loss_history_1, train_state_1 = model.train(iterations=N_ITERATIONS,
                                            callbacks=[epoch_saver, resampler],
                                            display_every=1000,
                                            model_save_path=model_save_prefix)

# train with L-BFGS
dde.optimizers.config.set_LBFGS_options(maxiter=MAX_ITER)
model.compile("L-BFGS")
loss_history, train_state = model.train(display_every=1000,
                                        model_save_path=model_save_prefix)

end_time = time.time()
elapsed_seconds = int(end_time - start_time)
elapsed_minutes, elapsed_seconds_remainder = divmod(elapsed_seconds, 60)

metadata = {
    "start_timestamp": start_timestamp,
    "end_timestamp": datetime.datetime.now().isoformat(),
    "elapsed_time_seconds": elapsed_seconds,
    "elapsed_time": f"{elapsed_minutes}m {elapsed_seconds_remainder}s",
    "training_iterations_adam": N_ITERATIONS,
    "training_iterations_lbfgs": getattr(train_state, "iteration", None),
    "adam_steps": len(getattr(loss_history_1, "steps", [])),
    "lbfgs_steps": len(getattr(loss_history, "steps", [])),
    "num_domain_points": N_INTERIOR_PTS,
    "num_boundary_points": N_BOUNDARY_PTS,
    "num_test_points": N_TEST_PTS
}

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Training completed in {elapsed_minutes}m {elapsed_seconds_remainder}s")
print(f"Saved training metadata to {metadata_path}")

# dde.saveplot(loss_history_1, train_state_1, issave=True, 
#             isplot=True, output_dir=outputs_dir)
dde.saveplot(loss_history, train_state, issave=True, 
            isplot=True, output_dir=outputs_dir)


# One-time plots
plot_domain(L, H_MAX, ELLIPSE_X_C, ELLIPSE_Y_C, ELLIPSE_A, ELLIPSE_B, ELLIPSE_ANGLE)
plot_loss_curves(loss_history, fname="loss_curves")

# Analyze models over epochs
model_paths = [os.path.join(models_dir, f) 
               for f in os.listdir(models_dir) if '.pt' in f]

for model_path in model_paths:
    model_iter = int(re.search(r"\d+", os.path.basename(model_path)).group())
    print(f"\nLoading n={model_iter} model from {model_path}")
    
    model.restore(model_path)

    plot_velocity_slice(model, L, H_MAX, model_iter=model_iter)
    
    plot_output_heatmaps(model, L, H_MAX, x_c=ELLIPSE_X_C, y_c=ELLIPSE_Y_C,
                         semimajor=ELLIPSE_A, semiminor=ELLIPSE_B,
                         angle=ELLIPSE_ANGLE, model_iter=model_iter)