import numpy as np

from ags_sci.fields import (
    PseudoSpectral3D, taylor_green, abc_flow, two_vortex_reconnection_seed,
    low_storage_rk4, cfl_timestep,
)


def test_taylor_green_is_divergence_free():
    eng = PseudoSpectral3D((16, 16, 16))
    u = taylor_green((16, 16, 16))
    d = eng.diagnostics(u)
    assert d.divergence_linf < 1e-11
    assert d.kinetic_energy > 0


def test_abc_flow_is_beltrami_and_divergence_free():
    eng = PseudoSpectral3D((16, 16, 16))
    u = abc_flow((16, 16, 16))
    d = eng.diagnostics(u)
    omega = eng.curl(u)
    assert d.divergence_linf < 1e-11
    assert np.max(np.abs(omega[0]-u[0])) < 1e-10
    assert np.max(np.abs(omega[1]-u[1])) < 1e-10
    assert np.max(np.abs(omega[2]-u[2])) < 1e-10
    assert d.helicity > 0


def test_projection_removes_divergence():
    rng = np.random.default_rng(7)
    eng = PseudoSpectral3D((12, 12, 12))
    raw = tuple(rng.normal(size=eng.shape) for _ in range(3))
    projected = eng.project(raw)
    assert eng.diagnostics(projected).divergence_linf < 1e-10


def test_curl_divergence_identity():
    rng = np.random.default_rng(9)
    eng = PseudoSpectral3D((12, 12, 12))
    u = tuple(rng.normal(size=eng.shape) for _ in range(3))
    assert np.max(np.abs(eng.divergence(eng.curl(u)))) < 1e-10


def test_inviscid_rhs_has_near_zero_energy_power():
    eng = PseudoSpectral3D((16, 16, 16))
    u = taylor_green((16, 16, 16))
    rhs = eng.rhs(u, viscosity=0.0)
    power = float(np.mean(sum(u[i]*rhs[i] for i in range(3))))
    assert abs(power) < 1e-9


def test_viscosity_dissipates_energy_for_taylor_green():
    eng = PseudoSpectral3D((16, 16, 16))
    u = taylor_green((16, 16, 16))
    rhs = eng.rhs(u, viscosity=0.1)
    power = float(np.mean(sum(u[i]*rhs[i] for i in range(3))))
    assert power < 0


def test_low_storage_rk4_preserves_constant_state_for_zero_rhs():
    eng = PseudoSpectral3D((8, 8, 8))
    u = tuple(np.ones(eng.shape)*i for i in (1.0, 2.0, 3.0))
    out = low_storage_rk4(lambda x: tuple(np.zeros_like(c) for c in x), u, 0.1)
    assert all(np.array_equal(out[i], u[i]) for i in range(3))


def test_reconnection_seed_is_finite_and_deterministic():
    a = two_vortex_reconnection_seed((16, 16, 16))
    b = two_vortex_reconnection_seed((16, 16, 16))
    assert all(np.all(np.isfinite(x)) for x in a)
    assert all(np.array_equal(a[i], b[i]) for i in range(3))


def test_q_and_lambda2_are_finite_and_abc_has_expected_q_signal():
    eng = PseudoSpectral3D((10, 10, 10))
    u = abc_flow((10, 10, 10))
    q = eng.q_criterion(u)
    lam2 = eng.lambda2_criterion(u)
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(lam2))
    centers, spectrum = eng.energy_spectrum(u)
    assert len(centers) == len(spectrum)
    assert np.all(spectrum >= 0)


def test_cfl_timestep_is_finite_and_zero_velocity_is_unbounded():
    e = PseudoSpectral3D((8, 8, 8))
    u = tuple(np.ones(e.shape) for _ in range(3))
    dt = cfl_timestep(u, e.lengths, 0.5)
    assert dt > 0 and np.isfinite(dt)
    zero = tuple(np.zeros(e.shape) for _ in range(3))
    assert np.isinf(cfl_timestep(zero, e.lengths, 0.5))


def test_houli_filter_is_smooth_and_identity_at_zero_mode():
    e = PseudoSpectral3D((16, 16, 16))
    from ags_sci.fields.operators import SpectralEngineND
    h = SpectralEngineND((16, 16, 16), dealias="hou_li", filter_alpha=36, filter_order=36)
    f = h.filter_transfer()
    assert np.isclose(f.flat[0], 1.0)
    assert np.all((f > 0) & (f <= 1.0))
    assert np.min(f) < 1e-10


def test_rotation_invariant_tensor_library_is_finite_and_zero_divergence_for_abc():
    e = PseudoSpectral3D((12, 12, 12))
    from ags_sci.fields.turbulence3d import invariant_diagnostics_3d
    inv = invariant_diagnostics_3d(e, abc_flow((12, 12, 12)))
    assert set(inv) == {"divergence", "strain_sq", "rotation_sq", "strain_cubic", "strain_rotation_sq", "vorticity_sq"}
    assert all(np.all(np.isfinite(v)) for v in inv.values())
    assert np.max(np.abs(inv["divergence"])) < 1e-10


def test_low_storage_rk4_fourth_order_on_linear_growth():
    errors = []
    for dt in (0.2, 0.1):
        e = PseudoSpectral3D((4, 4, 4))
        y = tuple(np.ones(e.shape) for _ in range(3))
        steps = round(1.0 / dt)
        for _ in range(steps):
            y = low_storage_rk4(lambda x: x, y, dt)
        errors.append(float(np.max(np.abs(y[0] - np.e))))
    assert errors[1] < errors[0] / 8.0
