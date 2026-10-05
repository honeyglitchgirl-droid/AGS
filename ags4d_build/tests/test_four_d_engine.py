import numpy as np
import pytest
from ags_sci.fields import FourDScalarFieldEngine
from ags_sci.fields.registry import default_field_registry


def periodic_grid(shape, lengths):
    return np.meshgrid(*[np.arange(n) * L/n for n,L in zip(shape,lengths)], indexing='ij')


def test_4d_registry_backend():
    r = default_field_registry()
    spec = r.get('scalar4d')
    assert spec.dimension == 4
    assert spec.factory is FourDScalarFieldEngine


def test_4d_gradient_and_laplacian():
    shape=(16,18,20,22); L=(2*np.pi, 3.0, 4.0, 5.0)
    eng=FourDScalarFieldEngine(shape,L,dealias='componentwise')
    x0,x1,x2,x3=periodic_grid(shape,L)
    # integer modes with physical wave numbers 2, 3, 1, 2
    u=np.sin(2*x0 + 3*(2*np.pi/L[1])*x1 + (2*np.pi/L[2])*x2 + 2*(2*np.pi/L[3])*x3)
    g=eng.gradient(u)
    phase=2*x0 + 3*(2*np.pi/L[1])*x1 + (2*np.pi/L[2])*x2 + 2*(2*np.pi/L[3])*x3
    ks=[2.0, 3*(2*np.pi/L[1]), 2*np.pi/L[2], 2*(2*np.pi/L[3])]
    for gi,k in zip(g,ks):
        assert np.max(np.abs(gi-k*np.cos(phase))) < 1e-11
    lap=eng.laplacian(u)
    assert np.max(np.abs(lap + sum(k*k for k in ks)*u)) < 1e-10


def test_4d_poisson_roundtrip():
    shape=(12,14,16,18); L=(2*np.pi,2*np.pi,2*np.pi,2*np.pi)
    eng=FourDScalarFieldEngine(shape,L,dealias='componentwise')
    x=periodic_grid(shape,L)
    u=np.sin(x[0]+2*x[1]-x[2]+3*x[3])
    source=eng.laplacian(u)
    rec=eng.solve_poisson(source)
    # Poisson inverse fixes only the zero mode; u is zero mean.
    assert np.max(np.abs(rec-u)) < 1e-10


def test_4d_diffusion_rhs_sign():
    shape=(8,8,8,8)
    eng=FourDScalarFieldEngine(shape,dealias='componentwise')
    x=periodic_grid(shape,(2*np.pi,)*4)
    u=np.sin(x[0])
    rhs=eng.diffusion_rhs(u,0.25)
    assert np.max(np.abs(rhs + 0.25*u)) < 1e-11


def test_4d_diagnostics_parseval_scale():
    eng=FourDScalarFieldEngine((8,8,8,8))
    x=periodic_grid(eng.shape,eng.lengths)
    u=np.sin(x[0])+2*np.cos(2*x[1])
    d=eng.diagnostics(u)
    assert d.l2 > 0 and d.gradient_energy > 0
    assert np.isfinite(d.max_abs)


def test_4d_helmholtz():
    eng=FourDScalarFieldEngine((8,8,8,8), dealias='componentwise')
    x=periodic_grid(eng.shape,eng.lengths)
    u=np.sin(x[0])
    rhs=eng.helmholtz_rhs(u,2.0,3.0)
    assert np.max(np.abs(rhs + 5.0*u)) < 1e-11


def test_4d_rejects_nonfinite_and_wrong_shape():
    eng=FourDScalarFieldEngine((8,8,8,8))
    with pytest.raises(ValueError): eng.laplacian(np.zeros((8,8,8,7)))
    a=np.zeros(eng.shape); a[0,0,0,0]=np.nan
    with pytest.raises(ValueError): eng.gradient(a)
