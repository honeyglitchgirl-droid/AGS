# AGS-Sci v102 — Capability Upgrade

This document describes the capability upgrade: a 5D field layer, a hardened
sandbox, a plugin/service architecture for external AI clients, improved law
discovery, passive self-evolution, and a quantum layer.

Everything below was verified against closed forms or independently computed
references. Where a capability is a simulation rather than a real speedup, that
is stated explicitly.

---

## 1. Driving AGS from an external AI

AGS is designed to be used as a component by another system. The single entry
point is `ags_sci.AGSService`:

```python
from ags_sci import AGSService

svc = AGSService()
caps = svc.capabilities()      # discover what this build can do
svc.field("scalar5d", (8, 8, 8, 8, 8))
svc.discover_law(t, X, ["x", "v"])
svc.run_code("import numpy as np\nresult = float(np.pi)")
svc.domain("boltzmann_entropy")
```

`capabilities()` exists so a client adapts to AGS rather than hard-coding
assumptions about its internals. It reports the version, the available field
dimensions and backends, the 25 domain primitives, whether the sandbox and
quantum layers are available, and the installed plugins.

**Design rules the facade enforces**

- *Discovery before use* — clients read capabilities instead of guessing.
- *No silent authority* — results carry explicit provenance and an epistemic
  status. A numerical fit is never presented as a discovered law.
- *Fail closed* — invalid identifiers, oversized payloads and unknown
  capabilities raise `SecurityViolation` rather than degrading to a permissive
  default.
- *Plugins extend, never replace* — a plugin receives the service handle; it
  cannot reach into protected core state.

## 2. Plugin architecture

`ags_sci.plugins` provides the two halves of the extension contract.

`AGSPlugin` is the interface a plugin implements — a `name`, a `version`, a set
of `capabilities`, and an `install(service)` hook.

```python
class MyAnalysis:
    name = "my.analysis"
    version = "1.0"
    description = "read-only spectral analysis"
    capabilities = frozenset({"analysis"})
    def install(self, service):
        self.service = service

svc.install_plugin(MyAnalysis())
```

`PluginRegistry` validates everything. Registration rejects:

- duplicate names,
- malformed identifiers,
- **undeclared capabilities** — only the tags in `ALLOWED_CAPABILITIES` are
  accepted, so an AI can reason about what a plugin may do,
- **privileged plugins without opt-in** — `sandbox`, `field-engine` and
  `discovery` require the host to pass `allow_privileged=True`.

The registry deliberately does **not** import plugin modules from disk.
Discovering *code* is the host application's responsibility; AGS only accepts
plugins it is handed explicitly. That keeps "load a plugin" from becoming
"execute arbitrary code at import time".

## 3. 5D field engines

Two engines extend the dimension-generic spectral core to five dimensions,
following the same explicit pattern as the 4D backend.

`FiveDScalarFieldEngine` (`scalar5d`) — gradient, Laplacian, zero-mode-fixed
Poisson inversion, spectral filtering/dealiasing, diffusion and
screened-Poisson/Helmholtz RHS operators, scalar diagnostics, and an
**exact-in-time** diffusion step that uses the integrating factor
`exp(-kappa*k^2*dt)` so arbitrarily large `dt` remains stable for the linear
part.

`FiveDVectorFieldEngine` (`vector5d`) — Jacobian, divergence, **curl as a
rank-2 antisymmetric tensor** (in 5D the curl of a vector is a 2-form with
C(5,2) = 10 independent components, not a vector), componentwise Laplacian,
Leray projection for incompressibility, advection RHS, and energy/enstrophy
diagnostics.

### Verification

Every operator was checked against a closed form on a 12^5 grid with modes
1..5 (below the Nyquist limit, so nothing aliases):

| Check | Result |
|---|---|
| Scalar Laplacian vs per-mode eigenvalues | 1.5e-13 |
| Poisson round-trip `L(L^-1 s) = s` | 4.9e-14 |
| `diffusion_rhs` vs `kappa * Laplacian` | 1.4e-15 |
| Exact diffusion step vs `exp(-kappa m^2 t)` | 5.3e-15 |
| Helmholtz RHS vs `kappa*L - m^2` | 1.8e-15 |
| Vector divergence vs `sum(cos(x_k))` | 5.3e-15 |
| Post-projection divergence | 5.0e-32 |
| Advection RHS vs `-sin(x)cos(x)` | 8.3e-16 |

As with the 4D engine, the discretization is Euclidean. Nothing here claims
Lorentz covariance or extra spatial dimensions of physics.

## 4. Hardened sandbox

Two real defects were fixed and several escape routes closed.

**Fixed: the memory limit was not actually enforced.** The worker set
`RLIMIT_RSS`, which the Linux kernel does not enforce. A single oversized
allocation could therefore OOM-kill the host before the parent's polling loop
ever observed the excess. The worker now sets `RLIMIT_AS`, which *is* enforced.

This is applied conditionally: a NumPy/SciPy/BLAS/OpenMP stack reserves far
more *virtual* address space than it ever makes resident, so capping at the
nominal RSS budget aborts valid experiments with `MemoryError`. The cap engages
only when it sits comfortably above the address space already committed;
otherwise the parent-side RSS monitor remains the enforcement mechanism.

Verified: a 20 GiB allocation under a 4 GiB cap is now refused in-process with
a clean `MemoryError`, while the default 256 MB budget correctly defers to the
RSS monitor.

**Fixed: the start method contradicted its own documentation.** The code chose
`forkserver` first while the comment said fork was preferred "after NumPy/SciPy
are already loaded". Forkserver re-imports `__main__`, which broke any caller
that was not an importable module. Fork is now preferred on POSIX.

**Closed: reflection escapes.** `getattr`, `setattr`, `delattr`, `hasattr`,
`vars`, `dir`, `type`, `object`, `super`, `classmethod`, `staticmethod`,
`property` and `__build_class__` are now rejected at the AST boundary. They were
already absent from the worker's `safe_builtins` whitelist, so a payload using
them failed with a confusing `NameError`; they are now rejected explicitly as
`BLOCKED`.

**Closed: loop-fuel tampering.** Experiment code could previously rebind or
increment `_ags_fuel`, granting itself an unbounded iteration budget and
defeating `LoopFuelTransformer` entirely. Assignment, `global` declaration and
any non-decrementing augmented assignment to that name are now blocked.

All 13 sandbox behaviour cases pass, including legitimate execution, blocked
modules/builtins, and every tampering route above.

## 5. Improved law discovery

`ags_sci.discovery.law.SparseLawDiscovery` addresses three weaknesses of the
original absolute-threshold identifier.

1. **Relative thresholding.** A term is kept only if its coefficient is a
   meaningful fraction of the dominant coefficient for that equation, so the
   sparsity structure reflects the data rather than the absolute units.
2. **Model selection instead of a guessed threshold.** The threshold is chosen
   by BIC over a sweep, trading fit against parsimony automatically.
3. **Held-out validation.** The model is scored on data it never saw, so a good
   in-sample fit cannot be mistaken for a good model.

### Recovered systems

| System | Recovered | Tier |
|---|---|---|
| Harmonic oscillator | `x' = v`, `v' = -x` | PARSIMONIOUS |
| Logistic growth | `x' = x - x^2` | PARSIMONIOUS |
| Exponential decay | `x' = -2x` | PARSIMONIOUS |
| Lorenz | `x' = 10(y-x)`, `y' = 28x - y - xz`, `z' = xy - (8/3)z` | HELD_OUT |

The original identifier returned fourteen terms for the harmonic oscillator,
most with coefficients near 1e-3. The improved pipeline returns exactly two.

`evidence_tier` is deliberately separate from fit quality: `NUMERICAL_FIT`,
`HELD_OUT`, `PARSIMONIOUS`. AGS never promotes a candidate to a scientific law
on its own.

## 6. Passive self-evolution

`ags_sci.core.evolution.PassiveEvolutionEngine` learns AGS's own tunable
parameters from observed experiment outcomes.

"Passive" means it only ever *observes*. It cannot edit source, cannot touch a
protected path, and never applies a change itself. It accumulates
`EvolutionObservation` records and emits a `ParameterProposal` that the host
must explicitly accept.

This is deliberately much weaker than self-modifying code. The only thing it may
move is a value inside a parameter space **the host declared**, which is why it
cannot become a path to arbitrary self-modification.

```python
engine = svc.evolution_engine({"threshold": (0.001, 0.5)}, min_observations=5)
engine.declare_default("threshold", 0.01)
for value, score in observations:
    engine.observe(EvolutionObservation("threshold", value, "holdout_rmse", score))

prop = engine.propose("threshold")   # None until the evidence justifies a change
```

A proposal requires a genuine improvement over the current default — ties and
rounding-level differences resolve to the *current* value, so noise alone never
moves a parameter. The existing `EvolutionProposal` code-change gate is
unchanged and still blocks the protected paths.

## 7. Quantum layer

`ags_sci.quantum` provides a statevector simulator and a backend abstraction,
so quantum *algorithms* can be prototyped and verified inside AGS and a hardware
or cloud backend can be dropped in behind the same interface.

- `QuantumCircuit` — H, X, Y, Z, S, RX, RY, RZ, CNOT, SWAP, with target
  validation at construction.
- `SimulatorBackend` — exact statevector, probability readout, shot sampling.
- Algorithms: `expectation`, `pauli_z`, `pauli_string`, `ansatz_ry`,
  `vqe_ground_energy`, `quantum_walk_mixing`.

Verified: Bell and GHZ state correlations, Pauli expectations
(`Z|0> = +1`, `Z|1> = -1`, `Z|+> = 0`), VQE recovering the exact ground energy of
`-Z` on a 3-qubit register, and a normalised quantum walk that is a delta at
`t = 0` and spreads over the cycle.

### What this is not

**This does not make AGS's classical numerics faster.** A statevector simulation
costs O(2^n) memory and O(2^n) or worse time per gate, so simulating a quantum
algorithm is strictly *more* expensive than the equivalent classical computation
at the problem sizes AGS handles. Quantum advantage, where it exists, comes from
asymptotics on hardware that does not exist in this file. `SimulatorBackend`
reports `simulated: True` and says so in its own metadata note.

Treat results from the simulator as a reference model for algorithm development,
not as a speedup.

## 8. Unsolved problems and mysteries

AGS's open-problem posture is unchanged and deliberately conservative: the
legacy `v61_open_problem_benchmark` and the `navier_stokes_control` Taylor-Green
control both report `CONTROL_ONLY`, and the discovery layer reports
`NUMERICAL_FIT`/`HELD_OUT`/`PARSIMONIOUS` rather than claiming a resolution.

The upgrade improves the *machinery* available for attacking such problems —
higher-dimensional engines, a sandbox that actually enforces its limits, and a
discovery pipeline that prunes noise — without weakening the epistemic boundary
between a numerical result and a scientific claim.

## 9. Inventing new problems, equations and laws

Section 5 described *identification*: given a library of terms, find the
combination that fits. That is a search over a **declared** space — it can
recover a law whose shape you already wrote down, and nothing else.

This section adds the missing half: **generation**. The `ags_sci.invention`
package invents candidate relations by searching a symbolic grammar, judges
whether each candidate is new, and formulates new questions from the structure
a fit fails to explain.

### 9.1 The grammar

`invention/grammar.py` defines a typed expression grammar — variables, numeric
constants, unary operators (`sin`, `cos`, `exp`, `log`, `sqrt`, `neg`) and
binary operators (`+ - * / ^`) — together with:

- `complexity`, the minimum-description-length penalty used in scoring;
- `safe_evaluate`, which returns `None` for any candidate that divides by zero,
  takes a log of a negative number, overflows, or is genuinely complex-valued,
  so an infeasible candidate can never silently look like a good fit;
- `refine_constants`, a Gauss–Newton least-squares refit of a candidate's free
  float constants followed by rationalisation, so a structurally correct
  candidate found with noisy constants survives (`0.520147101004912*n**2`
  becomes `0.5*n**2`);
- `equivalent`, symbolic equivalence by name-aligned subtraction — proof of
  equivalence when SymPy reduces the difference to zero, and *not proven*
  otherwise, never "different".

### 9.2 The search

`invention/search.py` runs a genetic-programming-style search over the grammar:
a seeded population, tournament selection, subtree crossover and mutation, an
elite archive, and a **Pareto front** over (complexity, normalised RMSE) so the
report contains the accurate candidates *and* the simple ones rather than only
the single best score.

Every candidate carries its expression, its fit, its complexity and its
evidence. Search is fully deterministic given a seed, so any result can be
replayed exactly.

### 9.3 Novelty is corpus-relative

`invention/novelty.py` makes the reference explicit. A candidate is novel only
relative to a **declared corpus** of known laws (Hooke, Coulomb, the ideal gas
law, the logistic map, and so on — or a corpus you supply).

Each verdict records its evidence:

| match kind | meaning |
| --- | --- |
| `identical` | the same expression after simplification |
| `symbolically_equivalent` | SymPy reduces the difference to zero |
| `numerically_equivalent_on_samples` | agrees on the sample grid, not provably so |
| `numerically_inconsistent_on_samples` | SymPy claimed equivalence the samples contradict |

The last row matters: it is the honesty check. A symbolic verdict is
corroborated numerically whenever samples are available, and a contradiction is
surfaced rather than swallowed.

**Novelty here is corpus-relative, not absolute.** A candidate absent from the
default corpus may still be a textbook result the corpus simply does not list.
The default corpus is a convenience, not a survey of human knowledge.

### 9.4 Inventing new problems

`invention/problems.py` formulates open problems from unexplained residual
structure. Each finding is an `OpenProblem` with the numeric evidence attached:

| kind | what it means |
| --- | --- |
| `unexplained_offset` | a systematic residual offset — a missing constant or forcing |
| `serial_dependence` | autocorrelated residuals — missing dynamics or memory |
| `unmodelled_periodicity` | the residual spectrum is dominated by one frequency |
| `state_dependent_error` | residual magnitude tracks a driver variable |
| `symmetry_breaking` | residuals carry a definite parity structure |

Every detector is **significance-aware**, which is what makes it usable. A raw
statistic is not a finding: white noise has a lag-1 autocorrelation of about
`1/sqrt(n)` and an even-part energy fraction of about `0.5` *by construction*,
so a detector that reports those raw numbers flags every noise-like residual.
Each check is therefore compared against its own sampling floor:

- drift uses `|mean|/sigma > max(threshold, 3/sqrt(n))`;
- serial dependence uses `|rho| > max(threshold, 3/sqrt(n))`;
- the spectral peak uses a bin-noise floor of `4/(n/2)`;
- parity reports a **signed z-score** against the "half the residual energy is
  even" null, not the raw even fraction.

The parity check also verifies that the sample grid is genuinely mirrored
(`x[i] == -x[n-1-i]`) before reversing it. Checking only that `min(x) == -max(x)`
is not enough: an even number of uniformly spaced points over a symmetric
interval has no point at the origin and is *not* mirrored, so reversing it pairs
`x` with `-x + h/2` and manufactures a parity violation out of nothing.

AGS does not claim these are unsolved problems of science. They are unexplained
structure in the supplied data, which is the only thing the evidence supports.

### 9.5 Sequence conjectures — the one decidable case

`conjecture_sequence` takes a finite numeric sequence, fits a closed form on all
but a held-out tail, and then **tries to falsify it** by predicting the tail.

This is the only invention capability with an objective pass/fail, because a
finite sequence is a finite object: a conjecture that survives the held-out tail
has genuinely predicted something it was not fitted on. The status is one of

- `CONJECTURE_SURVIVED_ALL_TERMS` — the held-out tail was predicted;
- `CONJECTURE_FALSIFIED_INSIDE_FIT_WINDOW` — no form fit even the training part;
- `CONJECTURE_FALSIFIED_ON_HOLDOUT` — it fitted, and then failed to predict.

The indexing convention is an explicit `start_index` parameter (default 0). It
used to be implicit, which made a 1-based `[1, 4, 9, ...]` sequence appear to be
`(n+1)**2` when it is `n**2`.

### 9.6 Using it

```python
from ags_sci import AGSService

svc = AGSService()

# Invent candidate laws for a series.
t = np.linspace(0, 2 * np.pi, 200)
res = svc.invent(t, np.cos(t))
for c in res["candidates"]:
    print(c["expression"], c["normalised_rmse"], c["epistemic_status"])
# cos(t)  0.0  NOVEL_CANDIDATE_RELATIVE_TO_CORPUS

# Invent a closed form for a sequence and try to falsify it.
sq = svc.conjecture_sequence(np.arange(1, 41) ** 2, holdout=6, start_index=1)
print(sq.status)        # CONJECTURE_SURVIVED_ALL_TERMS
print(sq.expression)    # n**2

# Formulate open problems from residuals a fit left behind.
probs = svc.propose_problems(residuals, x=x, t=np.arange(len(x)))
```

`AGSService.capabilities()` now reports `invention_available` and an
`epistemic_note` stating exactly what the layer may and may not claim.

### 9.7 Known limitations

- **Constant refinement is linear least squares.** It recovers multiplicative
  constants and simple rationals exactly, but it cannot recover an exponential
  *base*: `c**n` is refined to `c ≈ 2.0428` rather than `2.0`. The engine
  therefore honestly reports `2^n` as falsified rather than dressing up a
  near-miss as a recovery. A genuinely nonlinear refit is the obvious next
  improvement.
- **The search is budget-limited.** With a small population and few generations
  it may return a poor candidate with an honest, large `normalised_rmse` rather
  than the right answer. The report always includes the fit, so a bad fit is
  visible rather than hidden.
- **The grammar is shallow.** `max_depth` bounds expression depth; deeper
  relations need a deeper grammar and a proportionally larger budget.

### 9.8 What this is not

None of the above produces a law of nature. The layer produces **candidates**,
**problems** and **conjectures**, each with explicit evidence and an explicit
epistemic status. `discover_law` fits a declared library; `invent` searches a
generated space; both report fits, and neither asserts truth.

## 10. A superior gradient: derived, not tuned

AGS's plain spectral gradient is `IFFT(i*K*F)`. For any field whose spectrum
lies below Nyquist this is **exact to machine precision** — that is a theorem,
not a tuning outcome, and it was confirmed empirically before anything else was
attempted:

| field | plain gradient error |
| --- | --- |
| `sin(x)cos(y)` | 4.6e-15 |
| band-limited to k=20 | 8.6e-15 |
| modes up to k=30 (below Nyquist 32) | 1.0e-14 |
| `exp(sin x + cos y)` | 6.0e-15 |

The last row is the instructive one: a function with an infinite spectrum is
still differentiated exactly, because its spectrum has decayed to round-off by
Nyquist. The error tracks *unresolved* energy, and the energy above Nyquist
measured on the grid is **exactly zero** — the information is not merely small,
it is absent.

So "superior" cannot mean a better approximation of the same grid data. The one
axis where the plain gradient is genuinely weak is **noise**: it multiplies mode
`k` by `k`, so noise in the highest modes is amplified by up to Nyquist.

### 10.1 The derivation

For a signal with spectrum `S(k)` in white noise of variance `sigma^2`, the
MMSE-optimal linear multiplier is the Wiener filter

```
W(k) = S(k) / (S(k) + sigma^2)
```

which is 1 where signal dominates and 0 where it does not. Both `S` and
`sigma^2` are estimable from the observed spectrum alone, because the highest
modes of a resolved field carry no signal and therefore measure the noise floor.
`SpectralEngineND.gradient_denoised` implements exactly this.

Two details are what make it safe rather than merely plausible:

1. **The noise floor is bias-corrected.** `|F|^2` of white noise is
   exponentially distributed, so its *median* underestimates the mean by
   `ln 2`. An early version used the raw median, under-shrunk, and was
   accidentally "improved" by squaring the multiplier — a symptom that was
   traced to this exact factor before it was accepted as a result.
2. **The tail is tested for signal before it is trusted.** The tail is radially
   binned and log-power regressed against position. A flat tail is consistent
   with a noise floor; a *decaying* tail means real unresolved signal lives
   there, and shrinking would discard it. In that case the operator declines and
   reports `regime="plain"`, returning output **identical** to the plain
   gradient rather than quietly making the answer worse.

### 10.2 Measured result

Across 24 trials (2D/3D, band-limited and under-resolved fields, four noise
levels), with `tail_fraction=0.10` and `decay_tolerance=0.5`:

- **worst-case ratio 1.000** — never worse than the plain gradient;
- **16 of 24 trials improved**, with error ratios of 0.46–0.75 (25–54% less
  error);
- the 8 ties are all correct declines: zero-noise cases where the plain
  gradient is already exact, and under-resolved fields whose tails carry real
  signal.

Representative rows:

| case | noise | plain | denoised | ratio | regime |
| --- | --- | --- | --- | --- | --- |
| 2D band-limited k=20 | 1e-3 | 7.12e-05 | 3.95e-05 | 0.56 | shrunk |
| 2D analytic swirl | 1e-3 | 1.06e-02 | 5.13e-03 | 0.48 | shrunk |
| 3D band-limited k=12 | 1e-3 | 2.73e-05 | 1.44e-05 | 0.53 | shrunk |
| 2D narrow Gaussian | 0 | 5.25e-03 | 5.25e-03 | 1.00 | plain (declined) |
| 3D narrow Gaussian | 1e-3 | 7.15e-01 | 7.15e-01 | 1.00 | plain (declined) |

### 10.3 What it is not

**It is not a replacement for the plain gradient.** `gradient` remains the
default and is deliberately unfiltered: for a band-limited field it is exact,
and any multiplier can only lose accuracy there. `gradient_denoised` is opt-in.

**It is not a resolution improvement.** When the tail decays the operator
declines rather than shrinking, because the missing information is genuinely
absent from the grid. It buys noise robustness, not resolution.

**AGS's existing Hou-Li filter must not be used as a gradient multiplier.** It
was tested for this role and is catastrophic there: it attenuates low modes too,
degrading a clean band-limited gradient from 8.6e-15 to 1.9e-2 — eight orders of
magnitude. It is correct for its actual purpose (suppressing aliasing in
nonlinear products) and wrong for this one.

### 10.4 Using it

```python
from ags_sci import AGSService
svc = AGSService()

res = svc.denoised_gradient("pseudo_spectral3d", (64, 64, 64), noisy_field)
print(res.regime)          # "shrunk" or "plain"
print(res.describe())      # noise_floor, tail_slope, multiplier range
gradient = res.components  # same shape as the plain gradient
```

Available directly on `SpectralEngineND`, and delegated by the 4D and 5D
engines, so it works in every dimension AGS supports.
