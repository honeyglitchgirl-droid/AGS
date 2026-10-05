import numpy as np
from ags_sci.fields.linear_solvers import gmres, polynomial_p_multigrid_preconditioner
from ags_sci.fields.spectral_refiner import negative_sobolev_norm, pde_residual_loss, refine_gradient_descent


def test_gmres_solves_spd_system():
    A=np.array([[4.,1.,0.],[1.,3.,1.],[0.,1.,2.]])
    b=np.array([1.,2.,0.])
    r=gmres(lambda x:A@x,b,tol=1e-10,maxiter=50,restart=10)
    assert r.converged
    assert np.linalg.norm(A@r.x-b)<1e-8


def test_p_multigrid_preconditioner_reduces_residual():
    A=np.diag([4.,5.,6.,7.])+0.1*np.ones((4,4))
    M=polynomial_p_multigrid_preconditioner(A)
    b=np.array([1.,-1.,2.,0.5])
    assert np.linalg.norm(b-A@M(b)) < np.linalg.norm(b)


def test_hminus_one_and_residual_are_finite():
    x=np.linspace(0,2*np.pi,16,endpoint=False)
    f=np.sin(x)
    assert negative_sobolev_norm(f)<1.0
    assert np.isfinite(pde_residual_loss(f, lambda z:z, (2*np.pi,)))


def test_refiner_never_produces_nonfinite():
    x=np.linspace(0,2*np.pi,8,endpoint=False)
    f=np.sin(x)+0.2*np.sin(3*x)
    r=refine_gradient_descent(f, lambda z:z-np.sin(x), (2*np.pi,), steps=3)
    assert np.all(np.isfinite(r.field))
    assert r.final_loss <= r.initial_loss
