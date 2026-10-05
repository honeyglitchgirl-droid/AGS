import numpy as np

from ags_sci.fields.forced_turbulence2d import ForcedRFFTTurbulence2D


def test_forced_stage_audit_budget_and_stationarity_small_grid():
    s = ForcedRFFTTurbulence2D(N=32, sigma_f=40.0, dt=0.002, workers=1)
    s.begin_audit_window()
    s.advance(0.02)
    a = s.audit()
    assert np.isfinite(a.J_budget)
    assert np.isfinite(a.J_stat)
    assert a.J_budget < 1e-5
    assert 0 <= a.max_r_tail < 1e-4 or a.max_r_tail < 1e-3


def test_forced_rng_checkpoint_roundtrip(tmp_path):
    p = tmp_path / "forced.npz"
    a = ForcedRFFTTurbulence2D(N=32, sigma_f=20.0, dt=0.002, workers=1)
    a.advance(0.01)
    a.save_checkpoint(p)
    b = ForcedRFFTTurbulence2D.from_checkpoint(p, workers=1)
    assert b.t == a.t
    assert b.step_count == a.step_count
    assert np.array_equal(a.w_hat, b.w_hat)
    assert np.array_equal(a.f_hat, b.f_hat)
    assert np.array_equal(a.rng_state()[1], b.rng_state()[1])
    a.advance(0.01)
    b.advance(0.01)
    assert np.array_equal(a.w_hat, b.w_hat)
    assert np.isclose(a.I_eps, b.I_eps, rtol=0, atol=1e-14)
    assert np.isclose(a.I_diss, b.I_diss, rtol=0, atol=1e-14)


def test_forced_gate_matrix_is_two_tier():
    s = ForcedRFFTTurbulence2D(N=32, sigma_f=20.0, dt=0.002, workers=1)
    s.begin_audit_window()
    s.advance(0.02)
    a = s.audit(drift_fraction=0.0)
    assert a.budget_pass
    assert a.stationarity_pass == (a.J_stat <= 0.05)
    assert a.certified == (a.budget_pass and a.stationarity_pass and a.drift_pass and a.admissibility_pass and a.safety_pass)
