# AGS-Sci 2-D Turbulence Engine

## Production capability

`ags_sci.fields.turbulence2d.FastRFFTTurbulence2D` is the reusable production solver extracted from the EXP-2D-TURB-002 work.

Implemented:

- real-valued `rfft2/irfft2` Fourier state
- optional multi-worker `scipy.fft` backend with NumPy fallback
- hyperspherical 2/3 spectral truncation
- precomputed spectral multipliers
- vectorized radial shell spectra using `numpy.bincount`
- kinetic energy, enstrophy and palinstrophy diagnostics
- exact viscous energy-balance reference `dE/dt = -2 nu Omega`
- strict (`1e-4`) and hard (`1e-3`) spectral-tail first-passage gates
- log-log power-law fitting
- fixed-power BIC comparison primitive
- adaptive advective CFL timestep control
- deterministic narrowband initialization
- immutable run/checkpoint result records

## Deliberately excluded

The previously proposed integrating-factor RK4 implementation is **not** part of the certified engine. Its staging was compared with the reference classical RK4 implementation and was rejected because it did not reproduce the reference closely enough for scientific certification.

## Numerical boundary

The resolution gate is an admissibility diagnostic, not a proof of continuum convergence. A run that breaches the gate must not be used to support an inertial-range or singularity claim.

## Example use

The standalone example/benchmark remains outside the package runtime. Production code should import:

```python
from ags_sci.fields import FastRFFTTurbulence2D

solver = FastRFFTTurbulence2D(N=512, nu=1e-4, workers=-1)
result = solver.run(tmax=10.0, dt=None)
```

`dt=None` enables adaptive CFL stepping. Passing a positive `dt` preserves a bounded fixed-step mode.
