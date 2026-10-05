# 3D Objective Discovery Foundation

Version 102.2.7 implements the testable parts of the supplied 3D discovery reference.

## Objective stress closures

`ags_sci.dynamics.objective3d` provides a ten-term Pope-style tensor integrity basis from symmetric strain `S` and antisymmetric rotation `Omega`. `fit_objective_stress_closure` performs bounded sparse regression over that basis. The result is an empirical closure candidate only; independent trajectories and physical referee gates are required before acceptance.

## Cayley-Hamilton

`cayley_hamilton_invariants` returns the three characteristic invariants `I1`, `I2`, and `I3` and the numerical residual of

`T^3 - I1*T^2 + I2*T - I3*I = 0`.

This is an algebraic identity for a 3x3 tensor and is used as an integrity check, not a physical law-discovery claim.

## Scaling / BKM

`self_similar_scaling_search` is a bounded hypothesis filter for candidate exponents satisfying `alpha + beta = 1`. `bkm_scaling_indicator` computes finite-interval weighted diagnostics. Neither function certifies blow-up, regularity, or singularity absence.

## Lie-Poisson / Casimir foundation

`casimir_residual` evaluates `J grad(C)`. `linear_casimir_basis` computes the numerical nullspace for linear Casimir candidates. A full field-theoretic Lie-Poisson reduction is intentionally not claimed.

## Vortex-filament / Hasimoto foundation

`hasimoto_geometry` computes discrete curvature, torsion, phase, and the complex Hasimoto field for a 3D curve. `hasimoto_nls_residual` evaluates the focusing NLS residual. A low residual is a consistency diagnostic, not proof of a soliton or vortex-filament equivalence.
