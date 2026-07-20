"""
Navier-Stokes PINN - Simple 2D Poisseuille Flow example.

Spatial Domain:
    2D rectangle of length L and height H
    Explicitly defined with DeepXDE Rectangle

Known Constants:
    L, H, pressure P, viscosity µ

Partial Differential Equation (PDE):
    Simplified Navier-Stokes
    -∆P/L + µ•(d2vx / dy2) = 0

    with analytical solution
    vx = (1/2µ)•(-dP/dx)•(y)•(H-y)

PINN Model
- Inputs:
    x position
    y position
- Outputs:
    vx (predicted x-velocity)
- Data:
    Interior collocation points (x,y) --> u,v,p --> auto-diff --> PDE loss
    Boundary condition points (x,y) --> u,v,p --> BC Loss
- Loss:
    L_pde = residuals from the NS PDEs above
    L_bc = residuals from conditions (u=0 at walls, inlet pressure, outlet pressure)
    L_total = L_pde + L_bc
    

Evan Hackstadt
Rugonyi Lab
"""


import os
import re
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import deepxde as dde


# ———————————— CONFIG ————————————

# --- DOMAIN CONSTANTS ---
L = 1.0     # length
H = 2.0     # height
P1 = 8.0    # inlet pressure
P2 = 0.0    # outlet pressure
MU = 1.0    # viscosity

geometry = dde.geometry.Rectangle([0, 0], [L, H])

# --- DATA CONSTANTS ---
N_INTERIOR_PTS = 2000   # default 2000, can reduce. Fed to PDE loss.
N_BOUNDARY_PTS = 200    # default 200, can reduce. Fed to data loss.
N_TEST_PTS = 500        # default 500, should reduce in scale to interior/boundary pts

# --- TRAINING CONSTANTS ---
N_ITERATIONS = 10000        # train for N iterations
ITERATIONS_TO_SAVE = [1, 5, 10, 100, 200, 1000, 10000]   # specify which iters after which to save + plot

# --- Establish Paths ---
parent_dir = '/Users/evan/Documents/GitHub/pinns-fluid-mechanics/examples/1-poiseuille-flow/n=2200/'
outputs_dir = os.path.join(parent_dir, 'outputs/')
plots_dir = os.path.join(parent_dir, 'plots/')
models_dir = os.path.join(parent_dir, 'models/')
model_save_prefix = os.path.join(models_dir, 'model')

if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
if not os.path.exists(outputs_dir):
    os.mkdir(outputs_dir)
if not os.path.exists(plots_dir):
    os.mkdir(plots_dir)
if not os.path.exists(models_dir):
    os.mkdir(models_dir)


# ———————————— MODEL FUNCTIONS ————————————

# --- Define the PDE Residual ---
def pde_loss(x, u):
    '''
    x: collocation points (x, y)
    u: model output (vx)
    Returns the residual between the model-predicted PDE and the known PDE.
    ''' 
    # compute first term (d^2vx / dy^2) using auto-diff
    du_yy = dde.grad.hessian(u, x, i=1, j=1)    # hessian = 2nd deriv matrix
    
    # compute second term (∆P / L)
    dPdx = (P2 - P1) / L
    
    # return the residual: difference is known to =0
    return (MU * du_yy) - dPdx


# --- Define the Boundary Conditions (BCs) ---
def walls(x, on_boundary: bool):
    '''
    x: single point (x, y)
    on_boundary: passed in by deepxde, True for sampled boundary pts
    Returns 1 if on or very close to a wall, 0 otherwise
    '''
    return on_boundary and (np.isclose(x[1], 0) or np.isclose(x[1], H))


# --- Ground Truth PDE Analytical Solution ---
def analytical(x):
    '''
    x: array of coordinates (x, y) with shape = (N, 2)
    Returns the true value of vx at y
    '''
    y = x[:, 1:2]   # shape (N, 1) to match DeepXDE
    dPdx = (P2 - P1) / L
    return (1 / (2 * MU)) * (-dPdx) * y * (H - y)   # analytical soln


# --- Create the data and neural network objects for DeepXDE
def init_data_and_net(geometry, pde_loss, bc, N_INTERIOR_PTS,
                      N_BOUNDARY_PTS, analytical, N_TEST_PTS):
    
    # Generate data by sampling points from the domain
    data = dde.data.PDE(
        geometry=geometry,
        pde=pde_loss,
        bcs=[bc],
        num_domain=N_INTERIOR_PTS,
        num_boundary=N_BOUNDARY_PTS,
        solution=analytical,    # used so we can track L2 error
        num_test=N_TEST_PTS
    )

    print(f"Training data shape = {data.train_x_all.shape}")
    print(f"Testing data shape = {data.test_x.shape}")

    # Define the neural network
    net = dde.nn.FNN(
        layer_sizes=[2, 32, 32, 32, 1],     # input (x, y) --> vx output
        activation="tanh",
        kernel_initializer="Glorot uniform"
    )
    
    return data, net

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
CMAP_VEL   = "viridis"
CMAP_ERR   = "hot_r"
COLOR_PINN = "#185FA5"   # blue  – PINN prediction
COLOR_TRUE = "#E8593C"   # coral – analytical ground truth
FIG_DPI    = 150

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
def plot_velocity_slice(model, L, H, model_iter=10000, ny=300, x_loc=None):
    """
    Plot vx(y) at a fixed x-slice: PINN prediction vs. analytical parabola.
 
    Parameters
    ----------
    x_loc : float, x-coordinate of the slice (default: L/2)
    ax    : existing Axes to draw on (creates a new figure if None)
    """
    if x_loc is None:
        x_loc = L / 2.0

    # get values
    y_vals = np.linspace(0, H, ny)
    pts    = np.column_stack([np.full(ny, x_loc), y_vals])   # (ny, 2)
 
    vx_pred = model.predict(pts)        # (ny, 1)
    vx_true = analytical(pts)           # (ny, 1)

    # plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=FIG_DPI)
    ax.plot(vx_true.ravel(), y_vals, color=COLOR_TRUE,
            lw=2.5, label="Analytical", zorder=3)
    ax.plot(vx_pred.ravel(), y_vals, color=COLOR_PINN,
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
def plot_velocity_field(model, L, H, model_iter=10000, nx=200, ny=100):
    """
    2-D heatmap of predicted vx over the full domain.
    """
    X, Y, pts = make_grid(L, H, nx, ny)

    vx_pred = model.predict(pts).reshape(ny, nx)

    # plot
    fig, ax = plt.subplots(figsize=(5, 6), dpi=FIG_DPI)
    pcm = ax.pcolormesh(X, Y, vx_pred, cmap=CMAP_VEL, shading="auto")
    plt.colorbar(pcm, ax=ax, label="$v_x$")

    # format
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"PINN velocity field $v_x(x, y)$ \n(iterations={model_iter})")
    ax.set_aspect("equal")

    # save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{model_iter}_velocity_field.png"),
                dpi=FIG_DPI)
    # plt.show()
    plt.close()


# --- Error 2D Heatmap ---
def plot_error_map(model, L, H, model_iter=10000, nx=200, ny=100):
    """
    Spatial map of |PINN - analytical|.
    Reveals where error concentrates: walls, centerline, inlet, outlet.
    """
    X, Y, pts = make_grid(L, H, nx, ny)
 
    vx_pred = model.predict(pts).reshape(ny, nx)
    vx_true = analytical(pts).reshape(ny, nx)
    err     = np.abs(vx_pred - vx_true)

    # plot
    fig, ax = plt.subplots(figsize=(5, 6), dpi=FIG_DPI)
    pcm = ax.pcolormesh(X, Y, err, cmap=CMAP_ERR, shading="auto")
    cbar = plt.colorbar(pcm, ax=ax, label="|error|")
    
    # format
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
 
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Absolute error  $|\\hat{v}_x - v_x^*|$ " + f"\n(iterations={model_iter})")
    ax.set_aspect("equal")

    # save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{model_iter}_error_map.png"),
                dpi=FIG_DPI)
    # plt.show()
    plt.close()


# --- Loss Curves ---
def plot_loss_curves(losshistory):
    """
    Plot PDE residual loss and BC loss separately over training iterations.
 
    losshistory : object returned by model.train() in DeepXDE.
                  losshistory.loss_train is shape (iters, n_loss_terms).
                  Column 0 = PDE loss, column 1 = BC loss (standard ordering).
    """
    steps      = np.array(losshistory.steps)
    loss_train = np.array(losshistory.loss_train)   # (iters, n_terms)
 
    pde_loss = loss_train[:, 0]
    bc_loss  = loss_train[:, 1]
 
    fig, ax = plt.subplots(figsize=(7, 4), dpi=FIG_DPI)
 
    ax.semilogy(steps, pde_loss, color=COLOR_PINN, lw=1.5, label="PDE residual loss")
    ax.semilogy(steps, bc_loss,  color=COLOR_TRUE, lw=1.5, label="BC loss")
 
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training loss history")
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", lw=0.35, alpha=0.4)
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))
 
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_curves.png"),
                dpi=FIG_DPI)
    # plt.show()
    plt.close()


# --- L2 Test Error ---
def evaluate_l2(model, L, H, model_iter=10000, nx=200, ny=100, verbose=True):
    """
    Compute L² relative error and L∞ (max absolute) error on a dense grid.
 
    Returns
    -------
    l2_rel  : float — L² relative error (dimensionless, lower is better)
    l_inf   : float — max absolute error
    """
    _, _, pts = make_grid(L, H, nx, ny)
 
    vx_pred = model.predict(pts)        # (N, 1)
    vx_true = analytical(pts)           # (N, 1)
 
    diff   = vx_pred - vx_true
    l2_rel = np.linalg.norm(diff) / np.linalg.norm(vx_true)
    l_inf  = np.max(np.abs(diff))
 
    if verbose:
        print(f"ITERATIONS={model_iter}")
        print(f"L² relative error : {l2_rel:.4e}")
        print(f"L∞ max abs error  : {l_inf:.4e}")
        if l2_rel < 0.01:
            print("  ✓ < 1% — model has converged well.")
        elif l2_rel < 0.05:
            print("  ~ 1-5% — acceptable; consider more iterations or collocation pts.")
        else:
            print("  ✗ > 5% — check loss weighting, network depth, or training length.")
 
    return l2_rel, l_inf



# ———————————— MAIN ————————————

bc = dde.DirichletBC(geometry, lambda x: 0, walls)

data, net = init_data_and_net(geometry, pde_loss, bc, N_INTERIOR_PTS,
                              N_BOUNDARY_PTS, analytical, N_TEST_PTS)

model = dde.Model(data, net)
model.compile(optimizer="adam",
              lr=1e-3,
              metrics=['l2 relative error'])

epoch_saver = EpochSaver()


# Train the model
# Comment out if you want to restore a previous model

loss_history, train_state = model.train(iterations=N_ITERATIONS,
                                        callbacks=[epoch_saver],
                                        display_every=1000,
                                        model_save_path=model_save_prefix)

dde.saveplot(loss_history, train_state, issave=True, 
            isplot=True, output_dir=outputs_dir)


# Analyze the models
plot_loss_curves(loss_history)

# Analyze models over epochs
model_paths = [os.path.join(models_dir, f) 
               for f in os.listdir(models_dir) if '.pt' in f]

l_errors = {}

for model_path in model_paths:
    model_iter = int(re.search(r"\d+", os.path.basename(model_path)).group())
    print(f"\nLoading n={model_iter} model from {model_path}")
    
    model.restore(model_path)

    plot_velocity_slice(model, L, H, model_iter=model_iter)
    plot_velocity_field(model, L, H, model_iter=model_iter)
    plot_error_map(model, L, H, model_iter=model_iter)
    l2_rel, l_inf = evaluate_l2(model, L, H, model_iter=model_iter)
    
    l_errors[model_iter] = {
        'L2 Norm Error': l2_rel,
        'Max Absolute (L_inf) Error': l_inf
    }

l_errors = dict(sorted(l_errors.items()))
with open(os.path.join(outputs_dir, 'norm_error.json'), 'w') as f:
    f.write(json.dumps(l_errors, indent=2))