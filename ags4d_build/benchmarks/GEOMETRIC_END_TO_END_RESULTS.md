# AGS-Sci v102.2.0 Geometric End-to-End Benchmark Results

## Suite A — 2D incompressible Navier–Stokes

Configuration: N=(128,128), L=(2π,2π), ν=0.01, spherical spectral cutoff.

| Metric | Result |
|---|---:|
| advection coefficient | -1.0000000000 |
| Laplacian coefficient | +0.0100000000 |
| |u|²u coefficient | 0.0 |
| grad(div u) coefficient | 0.0 |
| initial max |div u| | 2.06e-14 |
| projected max |div u| | 4.39e-13 |
| max spectral tail | 1.32e-29 |
| verdict | PASS |

The manufactured divergence-free field and its projected Navier–Stokes RHS recover the prescribed coefficients exactly to numerical precision. The deliberately supplied unphysical candidate terms are eliminated.

## Suite B — 4D Lorentzian nonlinear Klein–Gordon

Important interpretation: this is a 3+1 spacetime problem: three periodic spatial dimensions at N=32³ with continuous time integration. It is **not** a 4D FFT over a time coordinate. This preserves the numerical-geometry separation of v102.2.

Metric: η=diag(-1,1,1,1), m²=1, λ=0.5.

| Metric | Result |
|---|---:|
| Laplacian coefficient in φ_tt | +1.0000000000 |
| mass coefficient | -1.0000000000 |
| cubic coefficient | -0.5000000000 |
| d'Alembertian identity residual | 4.02e-16 |
| spectral tail | 1.22e-19 |
| referee verdict | ACCEPT |

The identified first-order evolution form is

φ_tt = Δφ − 1·φ − 0.5·φ³,

which is algebraically equivalent to

□φ − φ − 0.5φ³ = 0.

## Test status

19 existing targeted AGS-Sci tests: PASS.
The end-to-end benchmark script itself passes both benchmark suites.

These are manufactured/controlled verification experiments. They validate the implementation and identification path; they do not constitute physical discoveries about Navier–Stokes or nonlinear Klein–Gordon dynamics.
