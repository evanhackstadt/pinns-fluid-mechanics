"""
Simple 2D Poisseuille Flow Example of a Navier-Stokes PINN.

Spatial domain: 2D rectangle with length L and height H.

Partial Differential Equation (PDE):
Simplified Navier-Stokes
0 = -∆P/L + µ•(d2vx / dy2)

Therefore, 
vx = (1/2µ)•(-dP/dx)•(y)•(H-y)

PINN Model
- Inputs: position x and y
- Outputs: predicted vx (x-component of velocity)
- Loss = L_data + L_pde, where L_pde = 

Evan Hackstadt
Rugonyi Lab
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import deepxde as dde


# --- Establish Paths ---
parent_dir = '/Users/evan/Documents/GitHub/pinns-fluid-mechanics/examples/poiseuille-flow/'
outputs_dir = os.path.join(parent_dir, 'outputs/')
plots_dir = os.path.join(parent_dir, 'plots/')
model_save_prefix = os.path.join(outputs_dir, 'model')


# --- DOMAIN CONSTANTS ---

L = 1.0     # length
H = 2.0     # height
P1 = 8.0    # inlet pressure
P2 = 0.0    # outlet pressure
MU = 1.0    # viscosity

geometry = dde.geometry.Rectangle([0, 0], [L, H])


# --- DATA CONSTANTS ---
N_INTERIOR_PTS = 2000   # default 2000, can reduce for faster debug runs
N_BOUNDARY_PTS = 200    # default 200, can reduce for faster debug runs
N_TEST_PTS = 500        # default 500, should reduce in scale to interior/boundary pts



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
    
    # return the residual since difference is known to =0
    return (MU * du_yy) - dPdx


# --- Define the Boundary Conditions (BCs) ---
def walls(x, on_boundary: bool):
    '''
    x: single point (x, y)
    on_boundary: passed in by deepxde, True for sampled boundary pts
    Returns 1 if on or very close to a wall, 0 otherwise
    '''
    return on_boundary and (np.isclose(x[1], 0) or np.isclose(x[1], H))

bc = dde.DirichletBC(geometry, lambda x: 0, walls)


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
def init_data_and_net():
    
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
def plot_velocity_slice(model, L, H, ny=300, x_loc=None):
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
    fig, ax = plt.subplots(figsize=(5, 5), dpi=FIG_DPI)
    ax.plot(vx_true.ravel(), y_vals, color=COLOR_TRUE,
            lw=2.5, label="Analytical", zorder=3)
    ax.plot(vx_pred.ravel(), y_vals, color=COLOR_PINN,
            lw=1.8, ls="--", label="PINN", zorder=4)

    # format plot
    ax.set_xlabel("$v_x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"Velocity profile at $x = {x_loc:.2f}$")
    ax.legend(framealpha=0.9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", lw=0.4, alpha=0.4)

    # save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "velocity_slice.png"),
                dpi=FIG_DPI)
    # plt.show()


# --- Velocity 2D Heatmap ---
def plot_velocity_field(model, L, H, nx=200, ny=100):
    """
    2-D heatmap of predicted vx over the full domain.
    """
    X, Y, pts = make_grid(L, H, nx, ny)

    vx_pred = model.predict(pts).reshape(ny, nx)

    # plot
    fig, ax = plt.subplots(figsize=(4, 6), dpi=FIG_DPI)
    pcm = ax.pcolormesh(X, Y, vx_pred, cmap=CMAP_VEL, shading="auto")
    plt.colorbar(pcm, ax=ax, label="$v_x$")

    # format
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("PINN velocity field $v_x(x, y)$")
    ax.set_aspect("equal")

    # save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "velocity_field.png"),
                dpi=FIG_DPI)
    # plt.show()


# --- Error 2D Heatmap ---
def plot_error_map(model, L, H, nx=200, ny=100):
    """
    Spatial map of |PINN - analytical|.
    Reveals where error concentrates: walls, centerline, inlet, outlet.
    """
    X, Y, pts = make_grid(L, H, nx, ny)
 
    vx_pred = model.predict(pts).reshape(ny, nx)
    vx_true = analytical(pts).reshape(ny, nx)
    err     = np.abs(vx_pred - vx_true)

    # plot
    fig, ax = plt.subplots(figsize=(4, 6), dpi=FIG_DPI)
    pcm = ax.pcolormesh(X, Y, err, cmap=CMAP_ERR, shading="auto")
    cbar = plt.colorbar(pcm, ax=ax, label="|error|")
    
    # format
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
 
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Absolute error  $|\\hat{v}_x - v_x^*|$")
    ax.set_aspect("equal")

    # save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "error_map.png"),
                dpi=FIG_DPI)
    # plt.show()


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


# --- L2 Test Error ---
def evaluate_l2(model, L, H, nx=200, ny=100, verbose=True):
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

data, net = init_data_and_net()

model = dde.Model(data, net)
model.compile(optimizer="adam",
              lr=1e-3,
              metrics=['l2 relative error'])

# Train the model
# Comment out if you want to restore a previous model

loss_history, train_state = model.train(iterations=10000,
                                        display_every=1000,
                                        model_save_path=model_save_prefix)

dde.saveplot(loss_history, train_state, issave=True, 
            isplot=True, output_dir=outputs_dir)


# Restore the model from training
model_path = os.path.join(outputs_dir,
                          [f for f in os.listdir(outputs_dir) if '.pt' in f][0])
model.restore(model_path)

# Call analysis functions
plot_velocity_slice(model, L, H)
plot_velocity_field(model, L, H)
plot_error_map(model, L, H)
plot_loss_curves(loss_history)
evaluate_l2(model, L, H)
# dde.utils.external.plot_loss_history(loss_history, os.path.join(plots_dir, 'loss_curves.png'))