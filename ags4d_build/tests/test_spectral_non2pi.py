import numpy as np
from ags_sci.fields import SpectralEngineND


def test_dealiasing_uses_dimensionless_modes_for_non_2pi_domain():
    eng = SpectralEngineND((12, 12, 12), domain_lengths=(3.0, 4.0, 5.0))
    x = np.arange(12) * 3.0 / 12.0
    X = np.meshgrid(x, x, x, indexing='ij')[0]
    # One physical mode: k=2*pi/L*2, well inside the cutoff.
    f = np.sin(4*np.pi*X/3.0)
    g = eng.gradient(f)[0]
    expected = (4*np.pi/3.0)*np.cos(4*np.pi*X/3.0)
    assert np.max(np.abs(g-expected)) < 1e-10
