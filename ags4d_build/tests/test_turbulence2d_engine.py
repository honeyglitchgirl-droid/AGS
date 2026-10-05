import numpy as np
import pytest

from ags_sci.fields.turbulence2d import FastRFFTTurbulence2D


def test_rfft_initial_enstrophy_normalized():
    s = FastRFFTTurbulence2D(N=32, workers=1)
    d = s.diagnostics()
    assert np.isclose(d["enstrophy"], 1.0, rtol=0, atol=2e-12)


def test_rfft_parseval_and_tail_are_finite():
    s = FastRFFTTurbulence2D(N=32, workers=1)
    d = s.diagnostics()
    assert np.isfinite(d["kinetic_energy"])
    assert np.isfinite(d["palinstrophy"])
    assert 0 <= d["r_tail"] <= 1.0


def test_rk4_diffusion_has_expected_energy_decay():
    s = FastRFFTTurbulence2D(N=32, nu=1e-2, workers=1)
    e0 = s.diagnostics()["kinetic_energy"]
    s.step_rk4(1e-3)
    e1 = s.diagnostics()["kinetic_energy"]
    assert e1 < e0


def test_adaptive_cfl_is_bounded():
    s = FastRFFTTurbulence2D(N=32, workers=1, dt_min=1e-5, dt_max=2e-3)
    assert 1e-5 <= s.cfl_timestep(0.5) <= 2e-3
    assert s.cfl_timestep(0.0) == 2e-3


def test_small_adaptive_run_returns_referee_fields():
    s = FastRFFTTurbulence2D(N=32, workers=1, dt_max=2e-3)
    r = s.run(tmax=0.01, sample_dt=0.005)
    assert r.n_steps > 0
    assert r.strict_gate_time is not None or r.max_tail < 1e-4
    assert r.hard_gate_time is not None or r.max_tail < 1e-3
    assert r.max_tail >= 0
    assert len(r.checkpoints) >= 2


def test_power_law_fit_rejects_insufficient_data():
    p, r2 = FastRFFTTurbulence2D.power_law_fit(np.arange(3), np.ones(3))
    assert np.isnan(p) and np.isnan(r2)


def test_fixed_power_bic_is_finite_and_penalizes_wrong_exponent():
    s = FastRFFTTurbulence2D(N=64, workers=1)
    k = np.arange(15.0, 61.0)
    E = k ** -4.0
    bic4 = s.fixed_power_bic(k, E, 4.0)
    bic3 = s.fixed_power_bic(k, E, 3.0)
    assert np.isfinite(bic4) and np.isfinite(bic3)
    assert bic4 < bic3


def test_public_field_export():
    from ags_sci.fields import FastRFFTTurbulence2D as PublicSolver
    assert PublicSolver is FastRFFTTurbulence2D
