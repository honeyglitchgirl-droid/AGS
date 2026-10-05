# AGS-Sci v59.0.0

Focused Autonomous Genesis System for Scientific Discovery.

## Active scope
- symbolic/dimensional expression discovery
- ODE/system identification
- PDE identification
- delay/latent-state diagnostics
- robust numerical differentiation
- uncertainty/evidence comparison
- active experiment design
- O(D) adversarial falsification
- reproducible research capsules
- spawn/forkserver JSON-only process supervision using importable callable references

## Architectural change
Historical discrete solvers and the append-only v58 master are **archived**, not imported into the active runtime. The old v57 fork-based supervisor is also archival-only; v59 uses forkserver/spawn with importable callable references and JSON IPC. This includes Hadamard, BPSW, Diophantine, and unrelated procedural research code.

The archive is retained for reproducibility and historical recovery.

## Scientific boundary
Empirical numerical survival is evidence, not a proof of universal truth. Formal claims require independent proof or experimental evidence appropriate to the claim.

## Quick test
`python -m pytest -q`

## Complete active master
`AGS_Sci_COMPLETE_MASTER_v59.py`
