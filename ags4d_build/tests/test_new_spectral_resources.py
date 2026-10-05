import numpy as np
from ags_sci.fields import negative_sobolev_norm, spectral_residual_loss, FourierChebyshev3D


def test_negative_sobolev_ignores_constant_mode():
    a=np.ones((16,16,16))
    assert negative_sobolev_norm(a) == 0.0


def test_negative_sobolev_weights_low_frequency_more_than_high_frequency():
    n=32; x=np.arange(n)*2*np.pi/n
    X,Y,Z=np.meshgrid(x,x,x,indexing='ij')
    low=np.sin(X); high=np.sin(10*X)
    assert negative_sobolev_norm(low) > negative_sobolev_norm(high)
    assert np.isfinite(spectral_residual_loss(low))


def test_fourier_chebyshev_derivatives():
    g=FourierChebyshev3D((12,12,17), z_bounds=(-1,1))
    x=np.arange(12)*2*np.pi/12; y=x; z=g.z
    X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
    f=np.sin(2*X)+np.cos(3*Y)+Z**4
    assert np.max(np.abs(g.dx(f)-2*np.cos(2*X))) < 1e-10
    assert np.max(np.abs(g.dy(f)+3*np.sin(3*Y))) < 1e-10
    assert np.max(np.abs(g.dz(f)-4*Z**3)) < 1e-9


def test_hybrid_divergence_of_simple_field():
    g=FourierChebyshev3D((10,10,13))
    x=np.arange(10)*2*np.pi/10; X,Y,Z=np.meshgrid(x,x,g.z,indexing='ij')
    u=(np.sin(Y), np.cos(X), np.zeros_like(Z))
    d=g.divergence(u)
    assert np.max(np.abs(d)) < 1e-10
