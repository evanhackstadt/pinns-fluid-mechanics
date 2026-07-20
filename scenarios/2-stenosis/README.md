# 2D Parameterized Stenosis

Evan Hackstadt
Rugonyi Lab

## Added Complexity from Poiseuille Flow

- Geometry - variable top wall height defined by ellipse
- Inputs - position and h(x) for eventual training across geometries
- Outputs - predict u, v, and p
- Loss - non-dimensionalization to normalize magnitudes (use Re)

## Spatial domain

- 2D rectangle (L, H_MAX) obstructed by an ellipse on the top wall
- Explicitly defined with CSG Difference

## Known Values

- L, H_MAX, ellipse params
- Reynold's number (Re) = 100
- Inlet pressure = 1.0
- Outlet pressure = 0.0

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

- Unsupervised
- Inputs: (x, y, h) = (x-position, y-position, channel height)
- Outputs: (u, v, p) = (x-velocity, y-velocity, pressure)
- Data:
  - Interior collocation points (x,y) --> u,v,p --> auto-diff --> PDE loss
  - Boundary condition points (x,y) --> u,v,p --> BC Loss
- Loss Terms:
  - L_pde = residuals from the NS PDEs above
  - L_bc = residuals from conditions (u=0 at walls, inlet pressure, outlet pressure)
