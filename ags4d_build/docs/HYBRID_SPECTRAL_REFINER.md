# Hybrid Spectral / Spectral-Refiner Integration

AGS-Sci now exposes a numerical foundation inspired by the supplied references:

- Fourier/Chebyshev hybrid field representation remains the numerical authority.
- Restarted GMRES and a bounded p-multigrid-style preconditioner provide a reusable linear-solver layer.
- Negative-Sobolev spectral residuals provide a solver-agnostic a-posteriori refinement metric.
- `STFNO3D` is an optional compact three-axis PyTorch spectral operator; it is not a full trajectory-to-trajectory ST-FNO implementation. It is an accelerator/candidate generator, never a physical referee.

The full free-surface SEM, mixed-stage Poisson formulation, and production p-multigrid implementation are not claimed here; they require additional geometry and boundary-condition work and remain separate future plugins.
