# 2D Stenosis, Training Across

Evan Hackstadt
Rugonyi Lab

## Added Complexity from Supervised

- Now performs curriculum learning across multiple geometries
- Validation on held-out test geometries to test generalization

## Spatial domain

- 2D rectangle (L, H_MAX) obstructed by an ellipse on the top wall
- Explicitly defined with CSG Difference

## Known Values

- L, H_MAX, ellipse params
- Reynold's number (Re) = 100
- Inlet x-velocity profile = Poiseuille parabola, max at y(H/2) = 1.0
- Outlet pressure = 0.0

## Data Breakdown

- Interior collocation points - unlabeled - for PDE loss (e.g. 2000)
- Boundary points - unlabeled - for BC loss (e.g. 800)
- **Sparse velocity measurements - labeled** - sampled from FEM (e.g. 25)

## Explicit Navier-Stokes PDEs

```math
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
```

```math
u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \frac{1}{RE}(\frac{\partial^2u}{\partial x^2} + \frac{\partial ^2u}{\partial y^2}) = 0
```

```math
u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \frac{1}{RE}(\frac{\partial^2v}{\partial x^2} + \frac{\partial ^2v}{\partial y^2}) = 0
```

## PINN Model

- Supervised
- Inputs: (x, y, h) = (x-position, y-position, channel height)
- Outputs: (u, v, p) = (x-velocity, y-velocity, pressure)
- Data:
  - Interior collocation points (x,y) --> u,v,p --> auto-diff --> PDE loss
  - Boundary condition points (x,y) --> u,v,p --> BC Loss
  - Labeled points from FEM (x,y) --> u,v,p --> BC Loss
- Loss Terms:
  - L_pde = residuals from the NS PDEs above
  - L_bc = residuals from conditions (inlet u profile, u=0 at walls, outlet pressure) + residuals from labeled data (obs u, obs v, obs p)

- Warm started on each geometry based on weights from previous
