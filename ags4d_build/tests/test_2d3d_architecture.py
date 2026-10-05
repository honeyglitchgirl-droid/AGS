import numpy as np
from ags_sci.fields import FastRFFTTurbulence2D, PseudoSpectral3D, taylor_green
from ags_sci.fields.contracts import ExperimentBudget


def test_shared_capability_contract():
    s2 = FastRFFTTurbulence2D(N=32, workers=1)
    s3 = PseudoSpectral3D((8,8,8))
    assert s2.capabilities.dimension == 2
    assert s3.capabilities.dimension == 3
    assert s2.capabilities.supports_adaptive_dt
    assert s3.capabilities.supports_projection


def test_2d_state_checkpoint_roundtrip():
    s = FastRFFTTurbulence2D(N=32, workers=1)
    before = s.state_checksum()
    state = s.snapshot_state()
    s.step_rk4(1e-4)
    assert s.state_checksum() != before
    s.restore_state(state)
    assert s.state_checksum() == before


def test_3d_normalized_divergence_and_pressure_projection():
    s = PseudoSpectral3D((8,8,8))
    rng = np.random.default_rng(4)
    raw = tuple(rng.normal(size=s.shape) for _ in range(3))
    p = s.pressure_projection(raw)
    assert s.normalized_divergence(p) < 1e-10


def test_3d_cfl_admission_rejects_unstable_step():
    s = PseudoSpectral3D((8,8,8))
    u = tuple(np.ones(s.shape) for _ in range(3))
    suggested = __import__('ags_sci.fields', fromlist=['cfl_timestep']).cfl_timestep(u, s.lengths, 0.5)
    try:
        s.advance(u, suggested * 2.0, viscosity=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError('unstable timestep was admitted')


def test_budget_validation():
    ExperimentBudget().validate()
    try:
        ExperimentBudget(max_steps=0).validate()
    except ValueError:
        pass
    else:
        raise AssertionError('invalid budget accepted')


def test_backend_registry_keeps_2d_and_3d_pluggable():
    from ags_sci.fields.registry import default_field_registry
    r = default_field_registry()
    assert [x.name for x in r.by_dimension(2)] == ["turbulence2d"]
    names3 = {x.name for x in r.by_dimension(3)}
    assert {"pseudo_spectral3d", "fourier_chebyshev3d", "spherical_shell3d"} <= names3
    assert r.get("turbulence2d").experimental is False
