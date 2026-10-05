# AGS-Sci Experimental 3D Engine

Version 102.2.5 adds the first validated 3D pseudo-spectral backend while keeping the 2D engine independent.

## Implemented

- periodic Cartesian 3D pseudo-spectral derivatives
- dimensionless-mode dealiasing (spherical or componentwise)
- Leray projection
- divergence, curl, Laplacian
- velocity advection `(u·∇)u`
- incompressible Navier–Stokes RHS with viscosity and optional forcing
- kinetic energy, helicity and enstrophy diagnostics
- strain/rotation tensors
- Q and lambda-2 vortex criteria
- shell-binned kinetic-energy spectrum
- conservative advective CFL timestep estimate
- 5-stage low-storage RK4-family integrator
- Taylor–Green vortex benchmark
- ABC flow benchmark
- deterministic anti-parallel Gaussian vortex-pair seed for reconnection experiments
- conservative 3D physical referee and BKM interval monitor
- optional smooth Hou--Li spectral filter with configurable even order
- rotation-invariant tensor diagnostics for sparse PDE discovery
- normalized divergence gate in addition to an absolute tolerance
- low-storage RK4 order regression and filter calibration tests

## Scientific status

This is an **experimental** 3D backend. Passing numerical gates does not constitute a proof of turbulence properties, global regularity, or finite-time singularity absence.

The BKM monitor reports only whether the tested finite-interval integral is finite. It never emits a "no singularity" proof.

Enstrophy is a diagnostic in 3D, not a universal conservation gate. Energy/helicity gates are interpreted according to the selected PDE and forcing/dissipation configuration.

## Verified smoke benchmark

A 16^3 Taylor–Green viscous step was executed successfully with:

- dt = 1e-3
- viscosity = 0.01
- finite state: yes
- post-step divergence L∞ ≈ 2.05e-15
- energy: 0.125000 → 0.1249925
- runtime ≈ 13 ms in the verification environment

The complete regression suite passed after the implementation.


## v102.2.9 hybrid spectral/refinement additions
The 3D experimental stack now includes a static Fourier–Chebyshev slab foundation (periodic x/y, Chebyshev z) and a periodic negative-Sobolev spectral residual estimator. These are foundations for future free-surface and hybrid operator-learning experiments; they are not a complete free-surface solver or neural operator.
