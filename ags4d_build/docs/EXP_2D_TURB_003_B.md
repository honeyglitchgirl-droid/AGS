# EXP-2D-TURB-003-B — Certified Forced 2-D Protocol

AGS-Sci v102.2.3 integrates the forced 2-D referee protocol directly into
`ags_sci.fields.forced_turbulence2d`.

## Model

\[
\partial_t\omega+u\cdot\nabla\omega
=-\alpha\omega+\nu\Delta\omega+f_\omega,
\qquad \Delta\psi=-\omega,
\qquad u=(-\psi_y,\psi_x).
\]

The forcing is an OU process with correlation time `tau_f`. Its endpoint is
updated once per accepted deterministic RK4 step and held fixed through all
four RK4 stages. This is an exact discrete OU endpoint update, not an exact
coupled SDE integrator.

## Two-tier audit

Stage quantities are integrated with the same RK4 weights as the state:

\[
I_\epsilon^{(n)}=\frac{\Delta t}{6}
(\epsilon_1+2\epsilon_2+2\epsilon_3+\epsilon_4),
\]

\[
I_D^{(n)}=\frac{\Delta t}{6}(D_1+2D_2+2D_3+D_4).
\]

The numerical-accounting metric is

\[
J_{budget}=\frac{|I_\epsilon-I_D-\Delta\Omega|}{I_\epsilon},
\]

with threshold `1e-3`.

The physical stationarity metric is

\[
J_{stat}=\frac{|I_\epsilon-I_D|}{I_\epsilon},
\]

with immutable threshold `0.05`.

The spectral gates remain `R_tail < 1e-4` for admissibility and `<1e-3` for
safety. The secular drift criterion is `|c1|*DeltaT/<Omega> <= 0.05`.

## Certified calibration record

The completed N=512 Attempt #2 used `sigma_f=5615.00`, seed 42, `nu=1e-4`,
`alpha=0.075`, and the `[20,26]` audit window. The recorded stage-consistent
audit was:

- `J_budget = 2.96415e-12` — PASS
- `J_stat = 0.0473936` — PASS
- spectral tail — PASS

These values certify the protocol execution, not a universal turbulence law.
The measured spectral exponent remains an empirical finite-resolution fit.
