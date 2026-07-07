"""
Simple 2D Poisseuille Flow example of a Navier-Stokes PINN,
but solving the Inverse Problem (estimating NS flow parameters)

Spatial domain: 2D rectangle with length L and height H.

Simplified Navier-Stokes PDEs:
    Continuity: (∂u/∂x) + (∂v/∂y) = 0
    Momentum_x: u(∂u/∂x) + v(∂u/∂y) + (1/rho)(∂p/∂x) - nu(∂2u/∂x2 + ∂2u/∂y2)
    Momentum_y: u(∂v/∂x) + v(∂v/∂y) + (1/rho)(∂p/∂y) - nu(∂2v/∂x2 + ∂2v/∂y2)

    Analytical Soln: u = (1/2µ)•(-dP/dx)•(y)•(H-y)

PINN Model
- Inputs:
    x position
    y position
- Outputs:
    u (x-velocity)
    v (y-velocity)
    p (pressure field)
- Data:
    Interior collocation points (x,y) --> u,v,p --> auto-diff --> PDE loss
    Boundary condition points (x,y) --> u,v,p --> BC Loss
    Interior velocity "measurements" (x,y) --> u,v,p --> Data Loss
- Loss:
    L_pde = residuals from the NS PDEs above
    L_bc = residuals from conditions (u=0 at walls, inlet pressure, outlet pressure)
    L_data = difference between predicted and observed (true) velocity data
    L_total = w1•L_pde + w2•L_bc + w3•L_data
        with w3 weighted heavily

Evan Hackstadt
Rugonyi Lab
"""


# UNFINISHED as of 2026-07-02


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
P1 = 1.0    # inlet pressure
P2 = 0.0    # outlet pressure
geometry = dde.geometry.Rectangle([0, 0], [L, H])


# --- PARAMETER GROUND TRUTH ---
MU_TRUE = 1.0   # viscosity
NU_TRUE = 1.0   # viscosity (µ/rho)
DPDX_TRUE = None                            # TODO

nu = dde.Variable(0.0)     # random initial value


# --- DATA CONSTANTS ---
N_INTERIOR_PTS = 2000   # default 2000, can change. Fed to PDE loss.
N_BOUNDARY_PTS = 200    # default 200, can change. Fed to BC loss.
N_DATA_PTS = 50         # default 50, can change. Fed to Data loss.
N_TEST_PTS = 500        # default 500, should change dependent on above points.


# --- TRAINING CONSTANTS ---
N_ITERATIONS = 10000
ITERATIONS_TO_SAVE = [1, 5, 10, 100, 200, 1000, 10000]   # the iters after which to save + plot


# --- Establish Paths ---
parent_dir = '/Users/evan/Documents/GitHub/pinns-fluid-mechanics/examples/2-inverse-poiseuille/n=2200/'
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

# --- PDE Residual ---
def pde_loss(x, y):
    '''
    x: collocation points (x, y)
    y: model output (u, v, p)
    Returns the residual between the model-predicted values and the governing PDEs.
    '''
    # unpack data
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    
    # compute derivatives using auto-diff
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    dv_x = dde.grad.jacobian(y, x, i=1, y=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)
    
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    
    # compute residuals per Navier-Stokes
    continuity = du_x + dv_y
    x_momentum = u*du_x + v*du_y + dp_x - nu*(du_xx + du_yy)
    y_momentum = u*dv_x + v*dv_y + dp_y - nu*(dv_xx + dv_yy)
    
    # return a list of residuals
    return [continuity, x_momentum, y_momentum]


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