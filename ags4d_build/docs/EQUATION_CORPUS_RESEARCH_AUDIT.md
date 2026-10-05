# Research equation corpus v3.0.0 — final audit

Audit date: 2026-10-05  
Implementation commit: `305ddf0660ca5a314ff2ebdd55c0b47ff0362931`  
AGS-Sci software version: `102` (unchanged)

## Release distributions

| Scope | Mathematics | Physics | Total |
|---|---:|---:|---:|
| K–12 | 229 | 137 | 366 |
| Undergraduate | 200 | 158 | 358 |
| Graduate | 154 | 142 | 296 |
| Research | 3,033 | 2,567 | 5,600 |
| **Overall** | **3,616** | **3,004** | **6,620** |

- Corpus/schema versions: `3.0.0` / `1.2.0`.
- Data files: 100 JSONL files; 35 are research files.
- Manifest-tracked resources: 105 (100 JSONL plus five metadata resources).
- Packaged reference resources including the manifest itself: 106.
- JSONL bytes: 3,703,264 overall; 3,416,497 in the research tier.
- Complete reference resource bytes: 3,784,435.
- Current manifest SHA-256:
  `7328fc60814e846fbcd006d4745986b0447fbf63e6554756f370198d9cb6f524`.

## Research metadata distributions

### Result status

| Status | Records |
|---|---:|
| `established` | 3,292 |
| `model_dependent` | 2,308 |
| `conjectural` | 0 |

No conjecture was added solely to populate a category. Included string and
quantum-gravity relations are qualified as model-dependent.

### Unit/convention system

| System | Records |
|---|---:|
| `not_applicable` | 3,033 |
| `source_defined` | 1,538 |
| `natural` | 979 |
| `dimensionless` | 50 |

Every physics research record has explicit convention metadata. No source unit
system was silently converted.

### Relation type

| Type | Records |
|---|---:|
| `equation` | 2,608 |
| `evolution_equation` | 601 |
| `recurrence` | 502 |
| `integral_relation` | 490 |
| `transformation` | 383 |
| `commutation_relation` | 332 |
| `asymptotic_relation` | 244 |
| `inequality` | 234 |
| `definition` | 191 |
| `differential_system` | 15 |

The schema additionally supports identity, operator, conservation, variational,
boundary, constitutive, canonical, symmetry, spectral, and constraint relations
for future verified records.

## Provenance and source audit

- Registered real sources: 56.
- Sources used by records: 55.
- Sources used by research records: 19.
- Research records with no source: 0.
- Research records with no HTTPS locator: 0.
- Unknown/fake source IDs: 0.
- Overall metadata-only orphan sources: 1,
  `CN-MOE-DAILY-REVISION-2025`. This official currency-review notice is
  intentionally retained and is not represented as a formula source.
- Repository-derived records use pinned commit locators; DLMF records retain
  numbered permalinks and constraints.

## Duplicate and malformed audit

Deterministic corpus validation reported:

| Finding | Count |
|---|---:|
| Duplicate IDs | 0 |
| Exact duplicate formulas | 0 |
| Whitespace-normalized duplicate formulas | 0 |
| Exact duplicate records | 0 |
| Malformed records | 0 |
| Unknown sources | 0 |
| Invalid source locators | 0 |
| Manifest/hash/count discrepancies | 0 |

The research builder also canonicalizes decorative spacing and reversed single
equalities to prevent count inflation. It does not generate rearrangements,
renamed-variable copies, arbitrary permutations, or synthetic nonsense.

## Preservation audit

- K–12: 27 files and 366 records remain byte-for-byte locked.
- K–12 ordered digest:
  `aeb48c1ed39069e7109eabfac485e7a2ba049e795d539a81eafb608b538491cc`.
- Complete pre-research corpus: all 65 files, 1,020 records, paths, IDs,
  ordering, byte sizes, and SHA-256 values match `pre_research_baseline.json`.
- Pre-research manifest SHA-256:
  `f5ac95b289d073321687361f7928865b4fe814b99765a56af43c92887fbc77d3`.
- Pre-research ordered data digest:
  `1040af44eb93fab3de307341e6172f4edc1afc6a0aef269c2ae55563bf99de1e`.
- Pre-research preservation issues: 0.

## Validation outcomes

| Audit | Result |
|---|---|
| `python -m compileall -q src scripts tests` | PASS |
| Corpus validator | PASS: 6,620 records, 100 files, 0 issues |
| Equation-focused tests | PASS: 23 |
| Complete AGS test suite | PASS: 291 |
| Security audit | PASS |
| Wheel content inspection | PASS |
| Isolated wheel installation | PASS |
| Installed-package corpus validation | PASS: 6,620 records, 100 files |
| Installed research loading | PASS: 3,033 mathematics, 2,567 physics |

## Wheel and package audit

- Wheel: `ags_sci-102-py3-none-any.whl`.
- Wheel bytes: 547,090.
- Wheel SHA-256:
  `f6596ff33f76aa5d587bd3957c8b371af0cd3cc8c839fe093165fe71e48016af`.
- Uncompressed wheel contents: 4,206,846 bytes in 180 entries.
- Packaged equation JSONL files: 100, including all 35 research files.
- Packaged reference resources: 106.
- Validation was run from
  `/tmp/ags-install-final-research/lib/python3.11/site-packages`, not the source
  checkout.

## Architecture-modification declaration

No reasoning architecture, discovery architecture, experiment system/result,
learned knowledge, scientific constant, solver, sandbox/security mechanism,
self-modification mechanism, or formula-execution path was modified. The AGS
software version remains `102`. Changes are confined to inert corpus data,
its immutable record/loader/validator API, package-resource declarations,
deterministic corpus build/audit scripts, documentation, and tests.

The loader exposes no solve, execute, apply, learn, or promote operation. Formula
strings remain opaque data. Discovered candidate/verified/rejected areas are not
manifest members and are not loaded by `EquationCorpus`.

## Known gaps

This release does not claim exhaustive research coverage. The machine-readable
matrix records 26 domains as `covered`, nine as `partial`, and zero as
`not covered`. Partial areas are:

- mathematics: calculus of variations, differential topology,
  optimization/control, PDEs, and symplectic/complex geometry;
- physics: condensed matter, general relativity, statistical mechanics, and
  string/quantum gravity.

These areas should be expanded only from equally reliable formula-level sources
with stable locators and convention metadata. Source-dependent custom notation,
contextless displays, unverified expressions, exposition, proofs, derivations,
worked exercises, and all-rights-reserved bulk source material were excluded.
