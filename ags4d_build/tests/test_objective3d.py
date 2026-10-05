import numpy as np
from ags_sci.dynamics.objective3d import (
    cayley_hamilton_invariants, pope_integrity_basis, objective_invariant_scalars,
    self_similar_scaling_search, bkm_scaling_indicator, casimir_residual,
    hasimoto_geometry, hasimoto_nls_residual, fit_objective_stress_closure, linear_casimir_basis,
)


def test_cayley_hamilton_residual_is_small():
    rng = np.random.default_rng(4)
    T = rng.normal(size=(3,3,5))
    out = cayley_hamilton_invariants(T)
    assert np.max(np.abs(out["cayley_hamilton_residual"])) < 1e-10


def test_pope_basis_is_finite_and_objective_under_rotation():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(3,3,4)); S = A + np.swapaxes(A,0,1)
    B = rng.normal(size=(3,3,4)); O = B - np.swapaxes(B,0,1)
    Q,_ = np.linalg.qr(rng.normal(size=(3,3)))
    basis = pope_integrity_basis(S,O)
    Sr = np.einsum('ia,ab...,jb->ij...', Q, S, Q)
    Or = np.einsum('ia,ab...,jb->ij...', Q, O, Q)
    br = pope_integrity_basis(Sr,Or)
    for a,b in zip(basis,br):
        expected = np.einsum('ia,ab...,jb->ij...', Q, a, Q)
        assert np.max(np.abs(expected-b)) < 1e-9


def test_scaling_search_is_bounded_and_filters_relation():
    out = self_similar_scaling_search(np.linspace(-1,2,13), np.linspace(-1,2,13))
    assert len(out) == 169
    assert sum(x.admissible for x in out) == 13


def test_bkm_scaling_indicator_finite():
    t=np.linspace(0,1,20); w=np.ones_like(t)
    vals=bkm_scaling_indicator(t,w,(0,1))
    assert vals[0][1] > 0 and vals[1][1] > 0


def test_casimir_residual_zero_for_known_null_gradient():
    J=np.array([[0,1],[-1,0]],float); g=np.array([0,0],float)
    assert np.allclose(casimir_residual(J,g),0)


def test_hasimoto_geometry_circle_is_constant_curvature_and_zero_torsion():
    s=np.linspace(0,2*np.pi,200,endpoint=False)
    pts=np.column_stack([np.cos(s),np.sin(s),np.zeros_like(s)])
    h=hasimoto_geometry(pts,closed=True)
    assert np.std(h['curvature']) < 0.02
    assert np.max(np.abs(h['torsion'])) < 0.02
    assert np.all(np.isfinite(h['hasimoto']))


def test_hasimoto_residual_finite():
    q=np.exp(1j*np.linspace(0,1,32))
    r=hasimoto_nls_residual(q,0.1)
    assert r.shape==q.shape and np.all(np.isfinite(r))


def test_objective_stress_closure_recovers_single_basis_term():
    rng=np.random.default_rng(8)
    A=rng.normal(size=(3,3,20)); S=A+np.swapaxes(A,0,1)
    B=rng.normal(size=(3,3,20)); O=B-np.swapaxes(B,0,1)
    basis=pope_integrity_basis(S,O)
    tau=1.7*basis[2]-0.4*basis[3]
    r=fit_objective_stress_closure(S,O,tau,threshold=1e-6)
    assert r.rmse < 1e-8
    assert set(r.active)=={2,3}
    assert abs(r.coefficients[2]-1.7) < 1e-6
    assert abs(r.coefficients[3]+0.4) < 1e-6


def test_linear_casimir_basis_finds_nullspace():
    J=np.array([[0,1,0],[-1,0,0],[0,0,0]],float)
    C=linear_casimir_basis(J)
    assert C.shape==(1,3)
    assert np.max(np.abs(J@C[0])) < 1e-12
