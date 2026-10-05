# Equation reference corpus

This package is a **read-only, level-aware reference layer**, independent of
AGS-Sci reasoning, invention, discovery, experiments, results, constants, and
self-evolution.

## Trust boundary

- `reference/` contains independently versioned corpus `3.0.0`: 6,620 inert
  mathematics and physics records across K–12, undergraduate, graduate, and
  research scopes, with schema `1.2.0`, a source register, and SHA-256 manifest.
- The 366-record K–12 baseline is byte-locked by `k12_baseline.json`.
- All 1,020 pre-research records and 65 files are independently byte-locked by
  `pre_research_baseline.json`.
- The 5,600 research additions live only below
  `reference/research/{mathematics,physics}/` and use `MATH-RS-*` or `PHY-RS-*`
  identifiers. Every one has source, locator, relation type, status, and unit
  metadata.
- `research_coverage.json` reports per-domain records, source counts, and
  `covered`/`partial`/`not covered` status without claiming exhaustive coverage.
- `discovered/candidate/`, `discovered/verified/`, and `discovered/rejected/`
  are separate lifecycle areas. Nothing automatically promotes discovery into
  `reference/`.
- Formula strings are opaque LaTeX-compatible text. The loader never imports,
  evaluates, executes, solves, substitutes into, or automatically applies one.

## Validate

```bash
python -m ags_sci.knowledge.validate_equations
```

Validation covers strict schema and UTF-8, paths and ID namespaces, sources and
locators, statuses and conventions, LaTeX delimiters, global duplicates,
manifest distributions and hashes, both preservation locks, and the coverage
matrix.

## Retrieve lazily

```python
from ags_sci.knowledge import EquationCorpus

corpus = EquationCorpus()  # verifies the manifest
records = tuple(corpus.iter_records(
    source_level="research",
    domain="mathematics",
    topic="operator_theory",
))
```

`iter_records()` parses only matching files. `all()` materializes all records
only when explicitly requested. Retrieval presents candidates only; applicability
and scientific use require independent checks.

Records contain compact relations and minimal metadata—not lessons,
derivations, proofs, abstracts, examples, exercises, or solutions. See
`docs/EQUATION_CORPUS.md`, `reference/sources.json`, and
`reference/research_coverage.json` for provenance, audit methods, and known gaps.
