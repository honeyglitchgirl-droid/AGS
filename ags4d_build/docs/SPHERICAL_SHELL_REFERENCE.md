# Spherical-shell 3D reference integration

Source used: A. Tilgner, *Spectral Methods for the Simulation of Incompressible Flows in Spherical Shells*, International Journal for Numerical Methods in Fluids 30, 713–724 (1999).

## Incorporated ideas

- Spherical-shell spectral discretization using angular spherical harmonics and radial Chebyshev collocation.
- Solenoidal velocity representation through poloidal/toroidal scalars as a future extension point.
- Radial resolution as an explicit stability/quality diagnostic.
- Semi-implicit time stepping and integrating-factor treatment of the linear diffusive operator.
- Caution around Crank–Nicolson for stiff, widely separated decay modes.

## Implemented now

`ags_sci.fields.spherical_shell3d` provides a deliberately bounded subset that can be independently verified:

- Chebyshev–Lobatto nodes and differentiation matrix.
- Optional monotone radial stretching.
- Linear toroidal diffusion operator on a spherical shell with homogeneous boundary values.
- Eigenvalue-based linear stability diagnostic.
- Exact integrating-factor propagation of the discretized linear diffusion operator.

## Not claimed

This module is **not** a complete spherical-shell Navier–Stokes/MHD solver. In particular, the full spherical-harmonic nonlinear transform machinery, poloidal/toroidal coupled nonlinear equations, rotating/precessing forcing, and influence-matrix boundary treatment remain future experimental plugins.

The external paper's numerical observations are not promoted to AGS ground truth. Any benchmark data must be versioned and independently reproduced before being used as a certification reference.
