# 2D Stenosis, Physics-Informed Deep Operator Network

Evan Hackstadt
Rugonyi Lab

## Added Complexity from Scenario 5

- Uses a Physics-Informed Deep Operator Network instead of a standard PINN
- No geometry parameters (a, b) as inputs - instead, stenosis SDF --> branch net
- Training data organized as (N_geom, N_sensors) branch array + (M_points, 2) trunk array
- PDE loss computed through trunk net autodiff only (trunk takes spatial coords)

## Spatial domain

- 2D rectangle (L, H_MAX) obstructed by an ellipse on the top wall
- Variable ellipse (a, b)

## Known Values

- L, H_MAX, ellipse params
- Reynold's number (Re) = 100
- Inlet x-velocity profile = Poiseuille parabola, max at y(H/2) = 1.0
- No-slip walls (u=v=0 on walls)
- Outlet pressure = 0.0

## Data Breakdown

- For each training geometry:
  - Evaluate SDF at each n_sensor points
  - Pick query points (x, y)
- Assemble N x M matrix (Cartesian Product) to train on all simultaneously

## PI-DeepONet Model

- Physics-Informed Deep Operator Network
  - learns operator: geometry SDF field --> solution (u,v,p) field
- Architecture
  - Branch Net
    - Input: stenosis SDF (signed distance function) discretized on n_sensors
    - Output: latent representation
  - Trunk Net
    - Input: query points (x, y)
    - Output: latent representation
  - Fusion:
    - inner product + bias --> (u, v, p) at query point
    - repeated for all N_geometries x M_points pairs (Cartesian product)
- Inputs - Branch Net:
  - stenosis SDF (signed distance function), d(x, y), sampled on a fixed grid
- Inputs - Trunk Net:
  - query points (x, y)
- Outputs:
  - solution field (u, v, p) = (x-velocity, y-velocity, pressure)
- Data:
  - Interior collocation points (x,y) --> u,v,p --> auto-diff --> PDE loss
  - Boundary condition points (x,y) --> u,v,p --> BC Loss
  - Labeled points from FEM (x,y) --> u,v,p --> BC Loss
- Loss Terms:
  - L_pde = residuals from the NS PDEs above
  - L_bc = residuals from conditions (inlet u profile, u=0 at walls, outlet pressure) + residuals from labeled data (obs u, obs v, obs p)
- Training:
  - Model trains on multiple pairs of (geometry, FEM solution field)
  - Cartesian Product formulation means we train on all pairs simultaneously

## Input Functions

## Navier-Stokes PDEs

```math
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
```

```math
u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \frac{1}{RE}(\frac{\partial^2u}{\partial x^2} + \frac{\partial ^2u}{\partial y^2}) = 0
```

```math
u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \frac{1}{RE}(\frac{\partial^2v}{\partial x^2} + \frac{\partial ^2v}{\partial y^2}) = 0
```
