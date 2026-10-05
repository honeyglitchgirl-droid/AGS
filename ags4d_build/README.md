# AGS-Sci v102 — INTEGRATED SCIENTIFIC RESEARCH OS / CLEAN 2D BASELINE

Lineage: v75 → v76 → ... → v102.

This cumulative master preserves the v75 research stack and adds a modular research OS covering scientific method controls, statistics, causal inference, signals, numerical methods, optimization, gradient research, representation economics, AI/ML/DL research, domain packs, active experiment design, safe self-evolution, bug hunting, and evidence orchestration.

Validation completed in this build:
- 52 cumulative self-test functions exercised without failure.
- 100 randomized attention cases and 17 finite-difference controls passed.
- active AST audit: zero direct eval/exec calls, one active `__main__` block.
- v98 bug hunter: zero findings in the active cumulative master.
- current imported `VERSION` is v102. The active numerical research scope is 2D.

Important scientific boundary: passing software tests validates implementation contracts; it does not prove scientific hypotheses or imply state-of-the-art performance.

## Active-Inference Upgrade

The research stack now includes strict causal hypothesis schemas, Bayesian evidence
updates, Expected Free Energy / BOED policy ranking, symbolic verification, a
process-isolated dry-lab runner, adversarial LLM role contracts, and causal MCTS.
See `docs/ACTIVE_INFERENCE.md`.


### EXP-2D-TURB-003-B
The forced 2-D turbulence protocol is integrated in `ags_sci.fields.forced_turbulence2d`, including RK4-consistent stationarity auditing, OU forcing, checkpoint restore, spectral gates, and the immutable `J_stat <= 0.05` criterion. See `docs/EXP_2D_TURB_003_B.md`.

## Experimental 3D backend

The project now includes a separate periodic 3D pseudo-spectral backend. See `docs/3D_ENGINE.md`. 3D is experimental and does not change 2D certification status.


## Driving AGS from an external AI

`ags_sci.AGSService` is the single entry point for an external client:

```python
from ags_sci import AGSService
svc = AGSService()
svc.capabilities()                      # what this build can do
svc.field("scalar5d", (8, 8, 8, 8, 8))  # 5D engines
svc.discover_law(t, X, ["x", "v"])      # sparse law discovery
svc.denoised_gradient("scalar5d", shape, f)  # noise-robust gradient
svc.invent(t, np.cos(t))                # invent candidate laws
svc.conjecture_sequence(seq)            # invent + falsify a closed form
svc.propose_problems(resid, x=x)        # formulate open problems
svc.run_code("result = 1 + 1")          # hardened sandbox
svc.install_plugin(MyPlugin())          # plugin architecture
```

`discover_law` fits a **declared** library of terms. `invent` **generates**
candidates by searching a symbolic grammar, reports novelty only relative to a
declared corpus of known laws, and formulates new questions from structure a fit
leaves unexplained. It produces candidates, problems and conjectures — never
asserted laws of nature.

See `docs/CAPABILITY_UPGRADE.md` for the full capability surface, including the
5D engines, sandbox hardening, plugin system, improved law discovery, passive
self-evolution, the invention layer, and the quantum simulator (a simulation,
not a speedup).

## v102 4D

An experimental `FourDScalarFieldEngine` is now implemented for 4D periodic scalar PDE experiments. It provides gradient, Laplacian, Poisson inversion, spectral filtering, diffusion/Helmholtz RHS operators, and diagnostics. See `docs/4D_ENGINE.md`.
