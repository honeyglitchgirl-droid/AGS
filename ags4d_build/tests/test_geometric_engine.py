import numpy as np
from ags_sci.fields.operators import SpectralEngineND
from ags_sci.fields.geometry import Metric, contract, exterior_derivative_1form
from ags_sci.dynamics.tensor_stlsq import InvariantLibraryBuilderND
from ags_sci.fields.referee import ResolutionRefereeND

def test_gradient_and_laplacian_2d():
    e=SpectralEngineND((32,32),(2*np.pi,2*np.pi))
    x=np.arange(32)*2*np.pi/32; y=x
    X,Y=np.meshgrid(x,y,indexing="ij")
    f=np.sin(X)+np.cos(2*Y)
    gx,gy=e.gradient(f)
    assert np.max(np.abs(gx-np.cos(X))) < 1e-10
    assert np.max(np.abs(gy+2*np.sin(2*Y))) < 1e-10
    assert np.max(np.abs(e.laplacian(f)+np.sin(X)+4*np.cos(2*Y))) < 1e-9

def test_leray_projection_divergence_free():
    e=SpectralEngineND((16,16))
    rng=np.random.default_rng(4)
    v=[rng.normal(size=(16,16)) for _ in range(2)]
    p=e.leray_project(v)
    assert np.max(np.abs(e.divergence(p))) < 1e-9

def test_lorentzian_contraction():
    eta=Metric(np.diag([-1.,1.,1.,1.]))
    v=np.array([2.,1.,0.,0.])
    assert np.isclose(contract(v,v,eta),-3.0)

def test_exterior_derivative_abelian():
    # A=(0, x, 0) -> F_xy=1
    G=np.zeros((2,2,4,4))
    G[0,1]=1.0
    F=exterior_derivative_1form(np.zeros((2,4,4)),G)
    assert np.allclose(F[0,1],1.0)
    assert np.allclose(F[1,0],-1.0)

def test_invariant_library_2d():
    e=SpectralEngineND((8,8))
    b=InvariantLibraryBuilderND(e)
    phi=np.random.default_rng(0).normal(size=e.shape)
    T,n=b.build_scalar_library(phi)
    assert T.shape==(64,6)
    assert len(n)==6

def test_referee_gate():
    r=ResolutionRefereeND()
    assert not r.judge(2e-3,1.5).accepted
    assert r.judge(1e-5,4.0).accepted


def test_lorentzian_box_signature():
    eta=Metric(np.diag([-1.,1.,1.,1.]))
    H=np.zeros((4,4,2,2)); H[0,0]=2.; H[1,1]=3.; H[2,2]=4.; H[3,3]=5.
    from ags_sci.fields.geometry import box_scalar
    assert np.allclose(box_scalar(H,eta),10.)

def test_rank2_covariant_divergence_constant_metric():
    eta=Metric(np.eye(2))
    T=np.zeros((2,2,3,3))
    G=np.zeros((2,2,2,3,3))
    G[0,0,0]=1.0  # partial_0 T_00
    G[1,1,1]=2.0  # partial_1 T_11
    from ags_sci.fields.geometry import divergence_covariant_rank2
    out=divergence_covariant_rank2(T,G,eta)
    assert np.allclose(out[0],1.0)
    assert np.allclose(out[1],2.0)
