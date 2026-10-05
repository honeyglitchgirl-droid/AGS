# AGS-Sci v102 Frontier Findings — Generalized De Gregorio

## Corrections

1. The hypothesis that positive advection necessarily arrests blow-up is false for the generalized De Gregorio family. Published analysis already contains finite-time blow-up mechanisms for positive `a`; regularity and domain/symmetry class matter.
2. The proposed `a>0 + odd boundary => self-similar blow-up` should not be hard-coded. Recent analysis includes non-self-similar blow-up mechanisms, so AGS tests both self-similar and non-self-similar indicators.
3. A threshold crossing is not by itself a proof of blow-up. AGS requires RK4, adaptive CFL/stretching step control, de-aliasing, resolution comparison, and spectral-tail diagnostics.

## Numerical finding

For smooth odd periodic data `omega0 = sin(x) + 0.15 sin(3x)`, the sandboxed pseudospectral solver found:

| a | N | time to ||omega||inf >= 100 |
|---:|---:|---:|
| 0.50 | 32 | 2.6162 |
| 0.50 | 64 | 2.5797 |
| 0.50 | 128 | 2.5773 |
| 0.50 | 256 | 2.5769 |
| 0.75 | 64 | 3.9216 |
| 0.75 | 128 | 3.9264 |
| 0.85 | 128 | 5.7162 |
| 0.85 | 256 | 5.6417 |
| 0.90 | 128 | 7.7659 |
| 0.90 | 256 | threshold not reached by t=8 |
| 0.925 | 128 | threshold not reached by t=8 |
| 0.925 | 256 | threshold not reached by t=8 |
| 0.95 | 128 | threshold not reached by t=8 |
| 0.95 | 256 | threshold not reached by t=8 |
| 1.00 | 64 | threshold not reached by t=8 |
| 1.00 | 128 | threshold not reached by t=8 |

These are threshold/convergence observations, not theorem-level blow-up times. The sharp transition still requires longer-horizon and higher-resolution analysis near `a≈0.9`.

## Low-regularity diagnostic

For the intentionally non-C1 odd profile `sign(sin x)|sin x|^0.5`, threshold crossings occurred for positive `a`, but the convergence was not clean enough to certify a blow-up time. This is consistent with the known importance of initial regularity and is retained as a diagnostic rather than a headline result.

## Sandbox hardening

- Fixed nested-function loop-fuel `UnboundLocalError` by injecting `global _ags_fuel` into functions containing loops.
- Added `complex` to the safe builtin set for numerical FFT workloads.
- Added controlled HTTPS `internet_fetch()` capability with HTTPS-only URLs, domain allowlisting, DNS/IP private-target rejection, response-size limits, timeout, redirect target validation, and SHA-256 content hashing.
- The current execution container has DNS/egress disabled, so an external fetch currently fails with a network-resolution error. This is an environment restriction, not a silent fallback to host filesystem/network access.
- Internet remains disabled by default in `ExecutionSandbox`; callers must explicitly set `internet_enabled=True`.

## Scientific next step

Do not fit only `dOmega/dt = c1 Omega + c2 Omega^2`. Add a dynamic-rescaling diagnostic and test:

`||omega||inf ~ C (T-t)^(-1)`

alongside a non-self-similar concentration/cascade diagnostic. Near `a≈0.9`, perform N=128/256/512 runs with a fixed error-controlled time integrator and compare threshold-time extrapolation rather than treating the threshold itself as singularity proof.
