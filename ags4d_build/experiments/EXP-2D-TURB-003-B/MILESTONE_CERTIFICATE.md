# AGS-Sci 2D Hydrodynamics Milestone — EXP-2D-TURB-003-B

**Status: SEALED & ARCHIVED**

Resolution: 512 × 512 (dealiased, k_max = 170.67)  
ν = 1.0e-4  
Rayleigh drag α = 0.075  
OU forcing band [4, 6], σ_f = 5615, seed = 42  
Evaluation window t ∈ [20.0, 26.0], 13 checkpoints at Δt = 0.50.

## Referee verdicts

- Gate S: PASS; max R_tail = 1.359e-6 < 1e-3.
- Gate A: PASS; max R_tail = 1.359e-6 < 1e-4.
- Gate C drift: PASS; normalized drift = 2.0327% ≤ 5%.
- Tier 1 numerical budget: PASS; J_budget = 2.96415e-12 ≤ 1e-3.
- Tier 2 physical balance: PASS; J_stat = 4.73936% ≤ 5%.
- Corrected forward enstrophy flux: PASS; min ΠΩ on [8,45] = 1.858e-4 > 0.
- Corrected kinetic-energy flux: PASS; min ΠE on [8,45] = 5.315e-8 > 0.

Time-averaged spectrum on k = 10..50:

p_free = 5.14948

BIC_free = -183.53  
BIC_p4 = -48.78  
BIC_p3 = +1.53

## Required qualification

The p ≈ 5.15 spectrum is an empirical description of this specific forced, drag-damped, finite-resolution stationary regime. It is **not** a universal turbulence law. The tested range overlaps substantial linear-damping influence:

α/(νk²) = 7.5 at k = 10 and 0.30 at k = 50.

The enstrophy-flux sign convention is locked as:

ΠΩ(k) = + Σ_{k'≤k} TΩ(k')

for the convention TΩ(k) = Re[ω̂* · FFT(u·∇ω)].
