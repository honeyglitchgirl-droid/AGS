# Audited mathematics and physics equation reference corpus

## Purpose and hard boundary

This corpus is trusted **reference data**. It is not AGS reasoning, a solver, a
symbolic engine, an experiment, a textbook, or learned/discovered knowledge.
Retrieval is lexical presentation only:

```text
query → inert candidate records → independent human/application checks
```

Formula fields are opaque strings. No corpus component compiles, evaluates,
executes, imports, substitutes into, solves, or automatically selects a formula.
Nothing in this release connects a formula to reasoning, discovery feedback,
experiments, scientific constants, self-modification, or learned knowledge.
Existing AGS algorithms, security boundaries, and software version `102` remain
independent.

## Release 3.0.0

Corpus `3.0.0` uses backward-compatible record schema `1.2.0`. It adds an inert
research tier without changing any of the 1,020 pre-research records.

| Level | Mathematics | Physics | Total |
|---|---:|---:|---:|
| K–12 (immutable v1.0.0 baseline) | 229 | 137 | 366 |
| Undergraduate | 200 | 158 | 358 |
| Graduate/advanced | 154 | 142 | 296 |
| Research (new in v3) | 3,033 | 2,567 | 5,600 |
| **Total** | **3,616** | **3,004** | **6,620** |

There are 100 level/domain/topic JSONL files totaling 3,703,264 JSONL bytes.
The research release deliberately exceeds 5,000 additions so that none of the
requested major domains is empty. It is broad reference coverage, **not** a
claim of exhaustive research coverage.

## Layout and preservation locks

```text
src/ags_sci/knowledge/equations/
├── reference/
│   ├── mathematics/<topic>/equations.jsonl          # unchanged K–12 paths
│   ├── physics/<topic>/equations.jsonl              # unchanged K–12 paths
│   ├── undergraduate/{mathematics,physics}/<topic>/equations.jsonl
│   ├── graduate/{mathematics,physics}/<topic>/equations.jsonl
│   ├── research/{mathematics,physics}/<topic>/equations.jsonl
│   ├── k12_baseline.json
│   ├── pre_research_baseline.json
│   ├── research_coverage.json
│   ├── schema.json
│   ├── sources.json
│   └── manifest.json
└── discovered/{candidate,verified,rejected}/
```

Two independent locks are validated on every corpus audit:

- `k12_baseline.json` protects 27 files and 366 records. Its ordered data digest
  remains `aeb48c1ed39069e7109eabfac485e7a2ba049e795d539a81eafb608b538491cc`.
- `pre_research_baseline.json` protects all 65 v2 data files, 1,020 records,
  286,767 bytes, every path/hash/count, the v2 metadata hashes, and ordered data
  digest `1040af44eb93fab3de307341e6172f4edc1afc6a0aef269c2ae55563bf99de1e`.

Research files live only below `reference/research/`. Existing K–12,
undergraduate, and graduate paths, IDs, records, ordering, and bytes are
unchanged. Discovery lifecycle directories remain physically and logically
separate; the loader has no promotion API and even a verified discovery is never
automatically placed into a trusted reference release.

## Schema and identifiers

All historical fields and records remain valid. Research records use:

- `MATH-RS-<TOPIC>-NNNNNN`
- `PHY-RS-<TOPIC>-NNNNNN`

Every research record requires:

- the original `id`, `domain`, `topic`, and opaque `formula`;
- `source_level: "research"` and at least one registered `source_id`;
- an HTTPS `source_locator`, normally a DLMF permalink, immutable Git commit and
  line range, official course-note locator, or stable PDF page;
- `relation_type`;
- `status`: `established`, `model_dependent`, or `conjectural`; and
- `unit_system`.

Optional `assumptions` and `conventions` preserve compact validity constraints,
normalizations, metric/index choices, and unit qualifications. Unit systems
include `si`, `natural`, `geometrized`, `atomic`, `gaussian_cgs`,
`heaviside_lorentz`, `dimensionless`, `mixed`, `source_defined`, and
`not_applicable`. The build performs no silent unit or convention conversion.

Research relation types include identities, definitions, operator and
commutation relations, transformations, differential systems, conservation and
evolution laws, variational principles, boundary and constitutive relations,
inequalities, recurrences, canonical/symmetry relations, asymptotics, integrals,
and spectral relations. Relations are not forced into a single `equation` label.

The v3 status distribution is:

| Status | Records |
|---|---:|
| `established` | 3,292 |
| `model_dependent` | 2,308 |
| `conjectural` | 0 |

The zero is intentional: schema support does not justify adding a conjecture
solely to populate a category. String/quantum-gravity relations included here
are explicitly `model_dependent`; no speculative relation is called
established.

## Research coverage matrix

`research_coverage.json` is the authoritative machine-readable matrix. A domain
is `covered` at 25 or more distinct relations, `partial` at 1–24, and
`not covered` at zero. Current totals are 26 covered, 9 partial, and zero not
covered.

### Mathematics

| Domain | Records | Sources | Status |
|---|---:|---:|---|
| `algebraic_geometry` | 360 | 1 | covered |
| `algebraic_topology` | 132 | 2 | covered |
| `calculus_of_variations` | 6 | 1 | partial |
| `category_homological_algebra` | 260 | 1 | covered |
| `differential_riemannian_geometry` | 139 | 1 | covered |
| `differential_topology` | 2 | 1 | partial |
| `dynamical_systems` | 50 | 1 | covered |
| `functional_analysis` | 119 | 1 | covered |
| `lie_theory` | 99 | 1 | covered |
| `mathematical_physics` | 299 | 1 | covered |
| `number_theory` | 100 | 1 | covered |
| `numerical_mathematics` | 50 | 1 | covered |
| `operator_theory` | 80 | 2 | covered |
| `optimization_control` | 7 | 1 | partial |
| `partial_differential_equations` | 6 | 1 | partial |
| `probability_stochastic_analysis` | 37 | 2 | covered |
| `representation_theory` | 85 | 2 | covered |
| `special_functions` | 1,197 | 1 | covered |
| `symplectic_complex_geometry` | 5 | 1 | partial |

### Physics

| Domain | Records | Sources | Status |
|---|---:|---:|---|
| `astrophysics` | 300 | 1 | covered |
| `condensed_matter` | 6 | 1 | partial |
| `cosmology` | 90 | 1 | covered |
| `fluid_physics` | 150 | 1 | covered |
| `gauge_theory` | 80 | 1 | covered |
| `general_relativity` | 21 | 1 | partial |
| `many_body_physics` | 400 | 1 | covered |
| `nonlinear_physics` | 50 | 1 | covered |
| `nuclear_physics` | 89 | 1 | covered |
| `plasma_physics` | 300 | 1 | covered |
| `quantum_field_theory` | 470 | 2 | covered |
| `quantum_information_optics` | 176 | 1 | covered |
| `quantum_mechanics` | 80 | 1 | covered |
| `standard_model_qcd` | 330 | 1 | covered |
| `statistical_mechanics` | 16 | 1 | partial |
| `string_quantum_gravity` | 9 | 1 | partial |

Partial means useful but narrow—not absent and not complete. In particular,
differential topology, symplectic/complex geometry, calculus of variations,
optimization/control, PDE, condensed matter, GR, statistical mechanics, and
string/QG should be expanded only when equally reliable formula-level sources
and locators are available.

## Sources and extraction policy

`sources.json` registers 56 real sources. Fifty-five are used by formula
records. The sole orphan is the official 2025 Chinese Ministry curriculum
currency notice, retained because it documents currency review but was never a
formula source. Nineteen sources support research records.

Research inputs include NIST DLMF numbered equations; the Stacks Project;
Frederic Schuller graduate geometry and quantum-theory notes; UC Davis PDE and
calculus-of-variations notes; MIT stochastic and symplectic geometry notes;
Stanford convex optimization; David Tong solid-state and string notes; David
Wiltshire GR notes; University of Toronto QFT materials; Diego Restrepo's
DOI-backed QFT/Standard Model notes; Richard Fitzpatrick's UT Austin plasma
notes; Michigan State stellar-physics notes; MIT nuclear course materials;
Nuclear TALENT many-body materials; Keio/Q-LEAP quantum-communication material;
and Yachay Tech nonlinear-dynamics course material.

For repository sources, `sources.json` records the exact extraction commit and
each formula points to that immutable commit and source lines where practical.
DLMF records retain numbered permalinks and constraints. The corpus contains
formula text and minimal metadata only—no copied exposition, proofs,
derivations, abstracts, exercises, or worked solutions.

`scripts/build_research_equation_corpus.py` documents the deterministic source
snapshots, extraction filters, topic assignment, and canonical seed relations.
It rejects figures, references, unresolved source-defined macros, prose,
short/contextless displays, oversized derivation blocks, malformed delimiters,
and normalized or reversed duplicates. It does not synthesize symbolic
permutations or rename variables to inflate counts.

## Integrity and audit method

The validator performs these network-free checks:

1. strict JSON/JSONL, allowed fields, UTF-8, NFC, and structural LaTeX delimiters;
2. path, level, domain, topic, and ID namespace agreement;
3. source resolution and stable HTTPS locators for every research record;
4. research status, relation type, unit system, assumptions, and conventions;
5. duplicate IDs, records, exact formulas, and whitespace-normalized formulas
   globally across all four levels;
6. total/domain/level/topic/file distributions, bytes, and every SHA-256;
7. exact K–12 and complete pre-research preservation locks;
8. coverage-matrix counts and source sets against actual records; and
9. package-resource inclusion and installed-wheel validation.

The build adds an additional normalized key that removes spacing and decorative
`\left`/`\right` and canonicalizes reversed single equalities. Mathematical
spot checks cover exact identities and DLMF constraints. Physics spot checks
cover dimensional relations where SI is actually declared and verify natural,
source-defined, and model-dependent qualifications elsewhere. These checks
reduce transcription risk but do not prove every relation or guarantee
applicability outside its recorded assumptions.

## Validation and inert retrieval

```bash
python -m ags_sci.knowledge.validate_equations
python -m ags_sci.knowledge.validate_equations --json
```

Research retrieval remains lazy and inert:

```python
from ags_sci.knowledge import EquationCorpus

corpus = EquationCorpus()  # verifies declared resource hashes
records = tuple(corpus.iter_records(
    source_level="research",
    domain="physics",
    topic="general_relativity",
))
candidates = corpus.retrieve(
    "curvature", domain="mathematics", source_level="research", limit=10
)
```

`iter_records()` parses only matching files. `all()` materializes the full corpus
only when explicitly requested. Construction does not build a whole-corpus
formula index. Formula strings remain data throughout loading and retrieval.
