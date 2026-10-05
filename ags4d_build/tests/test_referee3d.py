import numpy as np
from ags_sci.fields import PseudoSpectral3D, PhysicalReferee3D, taylor_green


def test_physical_referee_accepts_valid_taylor_green():
    e = PseudoSpectral3D((16,16,16))
    r = PhysicalReferee3D().judge(e, taylor_green((16,16,16)), tail_ratio=1e-8, core_ratio=4)
    assert r.accepted, r.reasons


def test_physical_referee_rejects_divergent_field():
    e = PseudoSpectral3D((12,12,12))
    x = np.linspace(0, 2*np.pi, 12, endpoint=False)
    X = np.meshgrid(x, x, x, indexing='ij')[0]
    u = (np.sin(X), np.zeros_like(X), np.zeros_like(X))
    r = PhysicalReferee3D().judge(e, u)
    assert not r.accepted
    assert 'divergence_gate_failed' in r.reasons


def test_bkm_is_a_monitor_not_a_singularity_proof():
    r = PhysicalReferee3D.bkm_monitor([0, 1, 2], [1, 2, 1])
    assert r.finite
    assert r.criterion_status == 'consistent_with_tested_interval_only'
