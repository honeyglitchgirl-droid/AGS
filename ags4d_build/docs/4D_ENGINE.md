# AGS-Sci 4D Scalar Field Engine

## Status

**v102 — experimental 4D scientific backend**

AGS previously accepted 4D experiments in the sandbox but did not have a dedicated 4D scientific engine. v102 promotes the existing dimension-generic spectral core into an explicit `FourDScalarFieldEngine` backend.

This is **4D Euclidean/spatial numerical infrastructure**. It is not a claim of a relativistic spacetime solver. Lorentzian tensor operations remain in `ags_sci.fields.geometry` and can be combined with the 4D spatial backend in future work.

## Implemented

- 4D periodic scalar fields
- 4D Fourier gradient
- 4D Laplacian
- zero-mode-fixed periodic Poisson inversion
- spherical/componentwise/Hou-Li spectral filtering through the shared core
- diffusion RHS: `kappa * Laplacian(u)`
- Helmholtz/screened RHS: `kappa * Laplacian(u) - mass2 * u`
- L2, gradient-energy, mean, and max-amplitude diagnostics
- spectral tail diagnostics
- field registry entry: `scalar4d`

## Explicitly not implemented yet

- 4D Navier–Stokes
- 4D turbulence engine
- a four-coordinate spacetime evolution solver
- full Lorentz-covariant discretization
- Einstein/Yang–Mills/Maxwell production solvers
- 4D topology discovery engine

Those should only be added as separate modules after the scalar/vector mathematical foundation is independently validated.

## Example

```python
from ags_sci.fields import FourDScalarFieldEngine

engine = FourDScalarFieldEngine((16, 16, 16, 16))

lap = engine.laplacian(field)
solution = engine.solve_poisson(source)
rhs = engine.diffusion_rhs(field, diffusivity=0.01)
metrics = engine.diagnostics(field)
```

## Design principle

The 4D backend deliberately reuses the proven `SpectralEngineND` numerical core rather than creating a second FFT implementation. This minimizes duplicated numerical logic while making the 4D capability explicit, discoverable, and testable through the field registry.
