# Verified Paper Pipeline

AGS-Sci v102 can compile manuscript sections from verified computational artifacts.
The LLM is restricted to narrative and structured proposals; quantitative values are
injected from `ProvenanceBundle` through `VerifiedPaperCompiler` template slots.

## Evidence

Model comparison uses the BIC/Schwarz approximation:

`log BF10 = -0.5 * (BIC(M1) - BIC(M0))`

An explicit null model is required. The compiler must never receive a heuristic
Bayes factor as a verified metric.

## Artifact transfer

Sandbox-created relative files are serialized into `ExecutionResult.artifacts` as
in-memory bytes. Callers can deserialize them with `io.BytesIO` without writing
results to host `/tmp`.

## Security boundary

The warm subprocess sandbox provides AST filtering, constrained builtins, sanitized
environment variables, ephemeral working directories, loop fuel, timeout recycling,
and bounded artifact extraction. This is defense-in-depth, not a complete host
filesystem security boundary: permitted scientific libraries may expose file I/O
APIs. A true zero-host-access guarantee requires WASI/MicroVM capability isolation.
