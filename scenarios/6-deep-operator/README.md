# 2D Stenosis, Physics-Informed Deep Operator Network

Evan Hackstadt
Rugonyi Lab

## Added Complexity from Scenario 5

- Uses a Physics-Informed Deep Operator Network instead of a standard PINN

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

- x

## PI-DeepONet Model

- Physics-Informed Deep Operator Network
- Inputs - Branch Net Functions:
  - stenosis (encoded by signed distance function to obstruction), d(x, y)
  - inlet velocity BC, u(x, y)
  - no-slip walls x-velocity BC, u(x, y)
  - no-slip walls v-velocity BC, v(x, y)
  - outlet pressure BC, p(x, y)
- Inputs - Trunk Net Points:
  - query point (x, y)
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