# AGS-Sci v102.2.3 Security / Stability Scorecard

Target rating: **10/10 for the application-level sandboxed scientific runtime**.

| Domain | Status | Evidence |
|---|---|---|
| Input validation | PASS | bounded JSON-like AI/plugin payloads |
| Code execution boundary | PASS | AST firewall + isolated workers |
| Resource admission | PASS | dimension/cell/field/memory/step/runtime gates |
| Runtime containment | PASS | timeout + worker termination/recycle |
| Network privacy | PASS | disabled by default; constrained opt-in |
| Data egress | PASS | bounded result/stdout/artifact channels |
| Integrity | PASS | canonical SHA-256 provenance |
| Discovery stability | PASS | bounded MCTS simulation budget |
| Self-evolution safety | PASS | protected-path evolution gate |
| Regression safety | PASS | complete automated suite |

This rating is an engineering target for the implemented controls, not a claim of formal security certification.
