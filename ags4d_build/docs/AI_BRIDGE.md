# AGS-Sci v60 — AI-Guided Research Bridge

This release combines the original AGS-Sci v59 scientific discovery engine with the tested AGS AI partnership prototype Sessions 1–3.1.

## Architecture

AI is an untrusted research partner. It can provide knowledge, methodological suggestions, and structured candidate hypotheses. It cannot execute supplied code, directly promote scientific claims, or bypass AGS validation.

The v59 engine remains the independent scientific path for:
- symbolic regression
- ODE identification
- PDE identification
- robust differentiation
- latent-state diagnostics
- evidence and uncertainty
- active experiment design
- adversarial falsification
- reproducible capsules
- restricted process execution

The v60 bridge adds:

`observations -> AI knowledge -> structured hypotheses -> AGS validation -> independent v59 discovery -> controlled experiment`

AI-generated hypotheses remain unverified until evidence is obtained through the AGS research machinery.

## Main integrated file

`AGS_Sci_COMPLETE_MASTER_v60_AI_BRIDGE.py`

Important entry points:
- `V59FocusedResearch`
- `AGSAIPartnership`
- `AGSAIHypothesisSynthesis`
- `AGSAISelfEvolution`
- `V60AIGuidedResearch`
- `v59_self_test()`
- `run_all_tests()`
- `v60_integration_self_test()`

## Verification

- v59 self-test: 7/7
- original packaged pytest suite: 10/10
- AI partnership/hypothesis suite: 26/26
- v60 cross-layer integration: 5/5
- active master static audit: 0 `eval()` calls
- active master static audit: 0 `exec()` calls
- active master static audit: 0 `pickle.load/loads()` calls

The historical v58 artifacts remain archived exactly as supplied by the v59 backup.
