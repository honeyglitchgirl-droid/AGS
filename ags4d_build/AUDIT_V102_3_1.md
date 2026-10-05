# AGS-Sci v102 Hardening / Bug-Hunt Report

## Verification
- Pytest: 111 passed, 2 Python 3.13 multiprocessing deprecation warnings.
- Security audit: PASS.
- Structural audit: PASS.
- Unified full audit: PASS.
- Python compileall: PASS.
- SHA-256 manifest verification: PASS (118 files).

## Bugs/issues found and fixed

1. **Sandbox result transport could cross a process boundary as an arbitrary Python object.**
   The parent previously received `value` through multiprocessing serialization and only checked `repr(value)` after deserialization. This allowed an unsafe object-deserialization boundary and made the size check too late.
   **Fix:** worker results are now converted to bounded JSON-safe primitives/NumPy data and transported as bounded JSON bytes; the parent never unpickles arbitrary experiment result objects.

2. **Sandbox memory policy was not actually enforced for the one-shot source worker.**
   A memory limit was configured, but the previous RLIMIT approach was not reliable for NumPy/SciPy workers.
   **Fix:** parent-side Linux/Android `/proc/<pid>/status` RSS monitoring now terminates workers that exceed the configured memory budget.

3. **Dimension admission accepted unsafe metadata types.**
   Booleans could pass as integers and NaN runtime values could bypass finite comparisons.
   **Fix:** strict integer/finite validation for shape, fields, dtype size, steps, and runtime.

4. **Active-inference distributions accepted negative/non-finite probabilities.**
   This could produce invalid entropy/KL calculations.
   **Fix:** probabilities must be finite and non-negative; KL divergence returns infinity when a positive-mass event has zero posterior support.

5. **MCTS controls were not independently hardened.**
   Direct construction could bypass the research-loop simulation validation, expansion could create excessive child sets, and non-finite rewards could enter the tree.
   **Fix:** simulation limits, expansion-size limits, finite reward validation, and exploration validation are enforced inside MCTS itself.

6. **Spectral derivative semantics silently applied dealiasing/filtering to every linear operator.**
   This could distort derivatives, Laplacians, and Poisson solves rather than restricting nonlinear products/state updates.
   **Fix:** linear spectral operators use the raw Fourier spectrum; filtering is now explicit via `filter_field()` and state/projection paths.

7. **SPSA-like refinement reused the same random direction every iteration.**
   It was deterministic but was not a proper sequence of independent perturbation directions.
   **Fix:** one seeded RNG is created outside the loop, producing reproducible but independent directions.

8. **Objective stress-regression controls lacked finite/non-negative validation.**
   Negative/NaN ridge or threshold values could silently alter the regression.
   **Fix:** explicit validation.

9. **The optional neural refiner was described too strongly as ST-FNO.**
   The implementation is a compact three-axis spectral operator, not the full trajectory-to-trajectory ST-FNO architecture described by the supplied paper.
   **Fix:** documentation and class description now explicitly state this limitation.

## Remaining warning

The compatibility callable sandbox path still uses `fork()` where available, and Python 3.13 reports the standard warning about forking a multithreaded parent. This is not hidden or treated as a test failure. The source-code experiment sandbox uses forkserver/spawn where available.

## Scientific scope

The supplied Spectral-Refiner paper supports using spectral residual/error estimators and solver-guided refinement, but it does not justify treating a neural prediction as a physical authority. The AGS referee therefore remains authoritative. The supplied Melander thesis supports high-order spectral methods, anti-aliasing, pressure/Poisson solver attention, verification/validation, and p-multigrid directions; these are treated as research inputs rather than claims that AGS already implements a complete free-surface spectral-element solver.
