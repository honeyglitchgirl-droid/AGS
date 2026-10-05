# AGS-Sci v102

## v102 — Capability upgrade

- **Service facade.** Added `ags_sci.AGSService`, a single stable entry point an
  external AI drives: capability discovery, field engines, sandboxed execution,
  law discovery, domain primitives, quantum, and plugin management. Results carry
  explicit provenance and never silently claim scientific authority.
- **Plugin architecture.** Added `ags_sci.plugins` with an `AGSPlugin` interface
  and a fail-closed `PluginRegistry`. Registration rejects duplicate names,
  malformed identifiers, undeclared capabilities, and privileged plugins without
  explicit host opt-in. Plugins cannot reach protected core state.
- **5D field engines.** Added `FiveDScalarFieldEngine` (`scalar5d`) and
  `FiveDVectorFieldEngine` (`vector5d`), extending `SpectralEngineND` and the
  backend registry to five dimensions. The vector engine exposes the 5D curl as a
  rank-2 antisymmetric tensor with 10 independent components, plus Leray
  projection and energy/enstrophy diagnostics. The scalar engine gained an
  exact-in-time integrating-factor diffusion step.
- **Hardened sandbox.** Fixed the worker memory limit, which set `RLIMIT_RSS`
  (unenforced on Linux) and is now `RLIMIT_AS`, applied conditionally so BLAS
  address-space reservations do not abort valid experiments. Fixed the start
  method, which chose `forkserver` against its own documented intent and broke
  non-module callers. Blocked reflection builtins (`getattr`, `type`, `object`,
  ...) and all loop-fuel tampering at the AST boundary.
- **Improved law discovery.** Added `ags_sci.discovery.law.SparseLawDiscovery`
  with relative thresholding, BIC threshold selection, and held-out validation.
  Recovers the harmonic oscillator, logistic growth and Lorenz system cleanly,
  where the previous identifier left ~1e-3 spurious terms. Evidence tiers are
  reported separately from fit quality.
- **Passive self-evolution.** Added `PassiveEvolutionEngine`, which observes
  experiment outcomes and proposes changes only inside a host-declared numeric
  parameter space. It cannot edit source or touch protected paths, and requires a
  genuine improvement before proposing anything.
- **Quantum layer.** Added `ags_sci.quantum`: a statevector simulator, circuit
  builder, and a `QuantumBackend` abstraction with VQE and quantum-walk
  primitives. Documented explicitly as a simulation with no classical speedup.
- **Invention layer.** Added `ags_sci.invention`, which *generates* hypotheses
  rather than fitting a declared library: a typed symbolic grammar with
  description-length scoring, a genetic-programming search reporting a Pareto
  front over complexity and fit, corpus-relative novelty assessment with
  numeric corroboration, and formulation of open problems from unexplained
  residual structure. Every detector is significance-aware, so white noise
  yields no findings. Exposed on `AGSService` as `invent`, `conjecture_sequence`
  and `propose_problems`, and advertised through `capabilities()` with an
  explicit epistemic note. Nothing is asserted as a law of nature.

- **Superior gradient.** Added `SpectralEngineND.gradient_denoised`, a
  Wiener-style spectral multiplier that is *derived* rather than tuned. It
  estimates the noise floor from the highest modes (bias-corrected, since the
  median of an exponential underestimates its mean by `ln 2`) and shrinks modes
  where noise dominates the signal. It is exact on clean band-limited data and
  cuts gradient error by 25-54% under noise. Across 24 trials it was never worse
  than the plain gradient; when the spectral tail still decays it declines to
  shrink and returns output identical to the plain gradient, reporting
  `regime="plain"` instead of silently degrading. Exposed on `AGSService` as
  `gradient`/`denoised_gradient` and delegated by the 4D and 5D engines.

### Bugs found and fixed while building it

- `equivalent()` compared declared variable tuples, so a corpus law declaring no
  variables could never be proven equivalent to a candidate declaring `("x",)`;
  it now compares free symbols by name, and aligns symbols by name before
  subtracting, so `cos(t)` matches `cos(t)` regardless of assumptions.
- `safe_evaluate()` lambdified over the declared variable tuple, silently
  dropping any symbol the tuple did not list. This made every numeric check
  against the default corpus fail. It now evaluates over the expression's actual
  free symbols and broadcasts constant results onto the caller's sample grid.
- `ProblemGenerator`'s parity check gated on `min(x) == -max(x)` instead of the
  grid actually being mirrored, so an even number of uniformly spaced points
  over a symmetric interval manufactured a parity violation; and it reported the
  raw even-part energy fraction, which sits near 0.5 for *any* noise-like
  residual. It now requires a mirrored grid and reports a signed z-score against
  that null. Drift, serial-dependence and spectral checks gained matching
  sampling floors for the same reason.
- `conjecture_sequence` indexed from `n = 0` with no way to say otherwise, so a
  1-based sequence was silently conjectured as `(n+1)**2`; the convention is now
  an explicit `start_index` parameter.

## v102 — 4D scalar-field foundation

- Added an explicit experimental `FourDScalarFieldEngine` backend.
- Added 4D scalar gradient, Laplacian, zero-mode-fixed Poisson inversion, filtering, diffusion/Helmholtz RHS, and diagnostics.
- Registered the backend as `scalar4d` in the field registry.
- Kept 4D Euclidean spectral discretization explicitly separate from Lorentzian geometry.
- Added validation and regression tests for 4D scalar operations.


- Hardened result transport: sandbox results cross the process boundary only as bounded JSON-safe data, preventing arbitrary pickle deserialization in the parent.
- Added parent-side Linux/Android RSS monitoring for one-shot sandbox workers.
- Hardened dimension metadata validation against booleans, non-finite runtimes, and malformed shapes.
- Hardened Active Inference probability normalization against negative/non-finite probabilities and zero-probability KL targets.
- Hardened MCTS simulation/reward/expansion budgets.
- Corrected spectral-operator semantics: linear derivatives/Laplacians/Poisson solves now use the raw spectrum; dealiasing/filtering is explicit for nonlinear/state filtering.
- Made SPSA refinement directions deterministic but independent across iterations.
- Added parameter validation for objective stress-closure regression and scaling searches.

## v102 — Unified 2D/3D Field Architecture

- Added a common field-engine capability contract for 2D and 3D backends.
- Added a pluggable field-backend registry so AGS can select engines by dimension without dimension-specific planning logic.
- Added normalized divergence, energy, and residual diagnostics.
- Added 2D checkpoint/restore/checksum state controls.
- Added bounded CFL admission to the 3D advance path and explicit pressure-projection API.
- Added architecture-level regression tests while preserving the existing 2D certification surface.
- Kept 3D Cartesian, Fourier–Chebyshev, and spherical-shell implementations isolated as experimental backends.

## v102
- Added `docs/SPHERICAL_SHELL_REFERENCE.md` documenting the external spherical-shell method reference and implementation boundaries.
- Added verified spherical-shell spectral building blocks: Chebyshev radial collocation, optional radial stretching, and integrating-factor toroidal diffusion.
- Added regression tests for polynomial differentiation, monotone stretching, dissipativity, and integrating-factor stability.
- This is not a complete spherical-shell Navier-Stokes solver.

## v102 — 3D benchmark calibration and invariant layer
- Added optional Hou--Li smooth spectral filtering to the dimension-agnostic spectral engine.
- Corrected componentwise 2/3 dealiasing to use dimensionless Fourier mode numbers.
- Added rotation-invariant 3D tensor diagnostics for sparse PDE discovery.
- Added normalized divergence gating while retaining an absolute fail-safe tolerance.
- Added low-storage RK4 order and filter regression tests.
- Deliberately did not ingest the image's questionable DNS table as ground truth; external reference data remains versioned input.

## v102 — Experimental 3D backend
- Added periodic 3D pseudo-spectral incompressible-flow backend.
- Added Taylor–Green, ABC, and deterministic vortex-pair seeds.
- Added Leray projection, vortex diagnostics, shell spectrum, CFL estimate, low-storage RK4-family stepping, and conservative 3D referee/BKM monitor.
- Corrected spectral de-aliasing normalization for non-2π physical domains.
- 3D remains experimental; no regularity or singularity proof is claimed.

## v102 — Security / Stability Hardening

- Added fail-closed security policy primitives.
- Hardened AI/plugin input and output boundaries.
- Added protected self-evolution gate.
- Added source/AST/output/artifact egress limits.
- Added security audit and scorecard.
- Fixed sandbox lifecycle cleanup for legacy callers.

## v102 — EXP-2D-TURB-003-B integration

- Added production forced 2-D RK4/OU engine.
- Added RK4-stage enstrophy injection/dissipation audit.
- Added two-tier J_budget/J_stat gate architecture.
- Added exact MT19937 checkpoint/restore.
- Integrated certified sigma_f=5615 calibration record and regression tests.

## v102 — 2-D turbulence engine integration

- Integrated the validated real-FFT 2-D decaying-turbulence solver into `ags_sci.fields`.
- Added SciPy FFT worker support with NumPy fallback.
- Added adaptive advective CFL control and referee-grade tail/energy-balance diagnostics.
- Kept classical RK4; the unvalidated integrating-factor staging is intentionally excluded.
- Added regression tests for Parseval normalization, diffusion decay, CFL bounds, and referee outputs.

# v102

- Hardened Sandbox with AST source firewall, explicit sanitized environment, ephemeral working directory, and loop-fuel enforcement.
- Added adversarial Sandbox regression tests.

# AGS-Sci v76-v102 — Changelog

v76: modular research-kernel registry.
v77: data-quality and leakage guards.
v78: provenance and reproducibility ledger.
v79: statistical uncertainty and multiple-testing tools.
v80: causal DAG and stratified causal estimator.
v81: signal/time-series diagnostics.
v82: numerical methods and solver safety.
v83: optimizer research primitives.
v84: gradient-law research bench; existing velocity branch preserved as candidate only.
v85: representation/token economics.
v86: deep-learning research primitives and bounded architecture invention.
v87: training-dynamics diagnostics.
v88: AI evaluation and red-team safety primitives.
v89-v95: physics, chemistry, biology, earth/environment, astronomy, mathematics, and cross-domain dimensional modules.
v96: active experiment design.
v97: safe self-evolution and capability synthesis.
v98: bug-hunting and metamorphic verification.
v99: evidence graph and bounded orchestration.
v102: hardened warm-worker sandbox, v102 release packaging, and synchronized audit/manifest metadata.

v101: integrated Research OS, full audit, and current version export.

Known issues fixed during implementation:
1. v66 carried a v67 identity string; active cumulative master corrected the identity and created an explicit v67 marker.
2. Multiple legacy top-level VERSION assignments could leave an imported master reporting an old release. Active cumulative master now keeps legacy markers private and exports VERSION=AGS_V101_VERSION.
3. The v75 active __main__ block would coexist with a later v101 block; it is retained only under a historical marker, leaving one active __main__.
4. v78/v99 hashing paths incorrectly assumed `_safe_json` returned text; hashing now serializes strict JSON explicitly.
5. v86 initially allowed fully masked attention rows to create NaNs; active implementation now rejects them.
6. v81 initially accepted constant signals as a dominant-frequency case; active implementation now rejects spectrally empty signals.
7. v83/v97 initially accepted invalid configuration values; active implementation now validates optimizer and evolution-gate settings.
8. v80 causal strata ordering is deterministic even with mixed Python types.

## v102 — Verified Paper Pipeline Remediation
- Restored `DynamicIdentificationResult` as an immutable dataclass.
- Added canonical `identify_stlsq(...)` API.
- Added bounded in-memory sandbox artifact transfer.
- Added BIC/Schwarz-grounded model comparison and Bayes-factor calculation.
- Added cryptographic `ProvenanceBundle` for experiment and metric provenance.
- Added strict `VerifiedPaperCompiler` metric-slot validation.
- Added end-to-end paper generation experiment and remediation regression tests.
- Security boundary remains process/AST defense-in-depth; full host-filesystem isolation requires WASI/MicroVM.

## v102 — Geometric Frontier Engine
- Added `ags_sci.fields` dimension-agnostic spectral operators for D=1..4.
- Added explicit metric-aware contractions and Lorentzian scalar operators.
- Added Abelian 1-form exterior derivative and rank-2 tensor divergence/trace operations.
- Added invariant scalar and vector-advection dictionary builders.
- Added D-dimensional spectral-tail resolution referee.
- Added targeted geometric regression suite (8 tests).
- Explicitly separated periodic spectral discretization from Lorentzian covariance claims.

## v102 geometric end-to-end verification
- Added `benchmarks/geometric_end_to_end.py`.
- Added `benchmarks/GEOMETRIC_END_TO_END_RESULTS.md`.
- Suite A (2D incompressible Navier–Stokes, 128²) recovered advection=-1 and viscosity=0.01; unphysical candidate terms were zeroed.
- Suite B (3+1 Lorentzian nonlinear Klein–Gordon, spatial 32³) recovered the Laplacian, mass, and cubic coefficients; d'Alembertian identity residual was 4.02e-16.
- 19 targeted regression/security/geometric tests passed.
- Explicitly documented that continuous-time 3+1 spacetime is not a 4D FFT.

## v102 — 3D objective discovery and vortex-geometry foundation
- Added Cayley–Hamilton characteristic invariants and residual verification for 3x3 tensor fields.
- Added ten-term Pope-style objective tensor integrity basis for 3D stress-closure experiments.
- Added objective sparse stress-closure regression with independent-validation warning.
- Added bounded self-similar scaling search constrained by alpha + beta = 1.
- Added weighted finite-interval BKM scaling indicators without singularity/regularity claims.
- Added finite-dimensional Poisson Casimir residuals and linear Casimir nullspace discovery.
- Added discrete curvature, torsion, Hasimoto transform, and NLS residual diagnostics for vortex-filament experiments.
- Added regression tests for rotation covariance, closure recovery, Casimir nullspaces, and circular-filament geometry.


## v102 — Hybrid spectral and a-posteriori error foundation
- Added periodic negative Sobolev (H^-1-type) spectral residual estimators using Parseval-compatible Fourier weighting.
- Added a 3D Fourier–Chebyshev slab foundation for periodic x/y and non-periodic z, with derivative, Laplacian, and divergence operators.
- Added regression tests for low-frequency residual weighting and Fourier–Chebyshev polynomial differentiation.
- Kept free-surface dynamics and neural-operator training out of the release until independently verified; the new components are solver-agnostic foundations.
- Incorporated source-derived ideas from the uploaded Spectral-Refiner paper and Melander thesis without treating their empirical results as AGS ground truth.

## v102 — Hybrid Spectral + Solver/Refiner Integration
- Added bounded restarted GMRES.
- Added Chebyshev p-multigrid-style preconditioner foundation.
- Added solver-agnostic spectral H^-1 PDE residual refinement.
- Added optional compact 3D spectral trajectory model (STFNO3D) when PyTorch is available.
- Kept numerical solver/referee authoritative; neural refinement is never a certification source.
