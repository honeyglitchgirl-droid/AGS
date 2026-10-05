# v102 Frontier diagnostic notes

## Corrections
- Do not use peak location `x*` as the core width. The diagnostic now uses a local half-height width.
- BKM reciprocal extrapolation is treated as a diagnostic only; threshold crossing is not declared a singularity.
- Self-similarity requires amplitude-rate, width-rate, and profile-collapse consistency simultaneously.
- `a=1` is not classified as globally regular from a finite simulation horizon.
- Sandbox callers must use the multiprocessing `__main__` guard.

## Scientific status
The generalized De Gregorio model is a 1D testbed and is kept separate from the production 2D turbulence engine.
