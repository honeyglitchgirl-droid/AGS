# AGS-Sci v102 — 25 Complex-Systems Domain Primitives

This release adds an executable **foundation layer** corresponding to the 25 domains in the roadmap image.

It is important not to overstate this: these are **validated computational primitives**, not 25 complete research-grade solvers. Each primitive gives AGS a concrete mathematical entry point that can later be expanded into a full domain engine.

Implemented domains:

1. General relativity — Christoffel symbols from a metric and coordinate derivatives.
2. Quantum field theory — Euclidean free-scalar lattice action.
3. Chaos theory — logistic map and finite-time Lyapunov estimate.
4. Fluid dynamics — periodic 1D Burgers RHS.
5. Boltzmann/information entropy — discrete entropy.
6. Seismology — Ricker wavelet.
7. Lotka–Volterra — predator/prey vector field.
8. Plasma physics — electron plasma frequency.
9. Machine learning — linear-regression gradient step.
10. LLM attention — scaled dot-product attention.
11. Quantum computing — single-qubit Pauli expectation.
12. Information theory — mutual information.
13. Optimization — quadratic gradient descent step.
14. Fourier analysis — N-dimensional FFT wrapper.
15. Differential geometry — metric inversion.
16. Topology — Euler characteristic from simplex counts.
17. Number theory — prime sieve.
18. Category theory — finite-function composition.
19. Itô calculus — Euler–Maruyama step.
20. Quantitative finance — Black–Scholes call price.
21. Network science — PageRank iteration.
22. Game theory — pure Nash equilibrium enumeration.
23. String theory — discretized Nambu–Goto length/energy primitive.
24. Plasma confinement — Larmor radius.
25. Computational complexity — empirical log–log scaling exponent.

## Verification

- New domain tests: **26 passed**.
- Full AGS regression suite: **144 passed, 0 failed**.
- Warnings: 2 existing multiprocessing/fork deprecation warnings.

The implementation deliberately avoids claiming that a primitive is equivalent to a complete theory. Full GR, QFT, Yang–Mills, string theory, etc. require substantially larger validated engines and domain-specific verification.
