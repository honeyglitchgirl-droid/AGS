"""Objective 3D tensor and discovery utilities inspired by the supplied research sheet.

These routines are diagnostics/discovery tools, not proofs of closure or singularity.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _tensor3(T):
    a = np.asarray(T, dtype=float)
    if a.shape[0:2] != (3, 3):
        raise ValueError("expected a 3x3 tensor field")
    if not np.all(np.isfinite(a)):
        raise ValueError("tensor contains non-finite values")
    return a


def cayley_hamilton_invariants(T):
    """Return the three scalar coefficients of a 3x3 characteristic polynomial.

    det(lambda I-T)=lambda^3-I1 lambda^2+I2 lambda-I3.
    """
    T = _tensor3(T)
    I = np.eye(3).reshape(3, 3, *([1] * (T.ndim - 2)))
    I1 = np.trace(T, axis1=0, axis2=1)
    T2 = np.einsum("ij...,jk...->ik...", T, T)
    I2 = 0.5 * (I1 * I1 - np.trace(T2, axis1=0, axis2=1))
    I3 = np.linalg.det(np.moveaxis(T, (0, 1), (-2, -1))).reshape(T.shape[2:])
    residual = np.einsum("ij...,jk...->ik...", T2, T) - I1 * T2 + I2 * T - I3 * I
    return {"I1": I1, "I2": I2, "I3": I3, "cayley_hamilton_residual": residual}


def pope_integrity_basis(S, Omega):
    """Construct the standard ten objective tensor basis terms for 3D turbulence.

    S is symmetric strain and Omega is antisymmetric rotation. The returned terms
    are algebraic candidates; no closure coefficients are inferred here.
    """
    S, O = _tensor3(S), _tensor3(Omega)
    if S.shape != O.shape:
        raise ValueError("S and Omega shapes must match")
    I = np.eye(3).reshape(3, 3, *([1] * (S.ndim - 2)))
    S2 = np.einsum("ij...,jk...->ik...", S, S)
    O2 = np.einsum("ij...,jk...->ik...", O, O)
    trS2 = np.trace(S2, axis1=0, axis2=1)
    trO2 = np.trace(O2, axis1=0, axis2=1)
    SO2 = np.einsum("ij...,jk...,ki...->...", S, O, O)
    S2O2 = np.einsum("ij...,ji...->...", S2, O2)
    terms = [
        S,
        np.einsum("ij...,jk...->ik...", S, O) - np.einsum("ij...,jk...->ik...", O, S),
        S2 - I * trS2 / 3.0,
        O2 - I * trO2 / 3.0,
        np.einsum("ij...,jk...,kl...->il...", O, S, S) - np.einsum("ij...,jk...,kl...->il...", S, S, O),
        np.einsum("ij...,jk...->ik...", O2, S) + np.einsum("ij...,jk...->ik...", S, O2) - I * (2.0 * SO2 / 3.0),
        np.einsum("ij...,jk...,kl...,lm...->im...", O, S, O, O) - np.einsum("ij...,jk...,kl...,lm...->im...", O, O, S, O),
        np.einsum("ij...,jk...,kl...,lm...->im...", S, O, S, S) - np.einsum("ij...,jk...,kl...,lm...->im...", S, S, O, S),
        np.einsum("ij...,jk...->ik...", O2, S2) + np.einsum("ij...,jk...->ik...", S2, O2) - I * (2.0 * S2O2 / 3.0),
        np.einsum("ij...,jk...,kl...,lm...,mn...->in...", O, S, S, O, O) - np.einsum("ij...,jk...,kl...,lm...,mn...->in...", O, O, S, S, O),
    ]
    return tuple(terms)


def objective_invariant_scalars(S, Omega):
    """Return scalar invariants useful for an objective regression library."""
    S, O = _tensor3(S), _tensor3(Omega)
    S2 = np.einsum("ij...,jk...->ik...", S, S)
    O2 = np.einsum("ij...,jk...->ik...", O, O)
    return {
        "trS2": np.trace(S2, axis1=0, axis2=1),
        "trO2": np.trace(O2, axis1=0, axis2=1),
        "trS3": np.einsum("ij...,jk...,ki...->...", S, S, S),
        "trSO2": np.einsum("ij...,jk...,ki...->...", S, O, O),
        "trS2O2": np.einsum("ij...,ji...->...", S2, O2),
    }




@dataclass(frozen=True)
class ObjectiveClosureResult:
    coefficients: tuple[float, ...]
    rmse: float
    active: tuple[int, ...]
    basis_count: int
    note: str


def fit_objective_stress_closure(S, Omega, stress, threshold=1e-8, ridge=1e-10):
    """Fit a global sparse stress closure in the ten objective tensor basis.

    The fit is algebraic evidence only. It does not establish a universal
    constitutive law; validation on independent trajectories is required.
    """
    S, O, tau = _tensor3(S), _tensor3(Omega), _tensor3(stress)
    if S.shape != O.shape or S.shape != tau.shape:
        raise ValueError("S, Omega and stress shapes must match")
    if not np.isfinite(float(threshold)) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if not np.isfinite(float(ridge)) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")
    basis = pope_integrity_basis(S, O)
    A = np.column_stack([b.reshape(-1) for b in basis])
    y = tau.reshape(-1)
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0
    As = A / scale
    coef = np.linalg.solve(As.T @ As + float(ridge) * np.eye(As.shape[1]), As.T @ y) / scale
    active = np.abs(coef) >= float(threshold)
    if not np.any(active):
        active[np.argmax(np.abs(coef))] = True
    idx = np.where(active)[0]
    coef_active = np.linalg.lstsq(A[:, idx], y, rcond=None)[0]
    pred = A[:, idx] @ coef_active
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    full = np.zeros(len(basis), dtype=float)
    full[idx] = coef_active
    return ObjectiveClosureResult(tuple(full), rmse, tuple(int(i) for i in idx), len(basis),
                                  "objective integrity-basis regression; independent validation required")


def linear_casimir_basis(poisson_matrix, rtol=1e-10):
    """Return coefficient vectors c for linear Casimir candidates C=c^T x, Jc=0."""
    J = np.asarray(poisson_matrix, float)
    if J.ndim != 2 or J.shape[0] != J.shape[1] or not np.all(np.isfinite(J)):
        raise ValueError("Poisson matrix must be finite and square")
    U, s, Vt = np.linalg.svd(J)
    tol = float(rtol) * max(float(s[0]) if len(s) else 1.0, 1.0)
    rank = int(np.sum(s > tol))
    return Vt[rank:].copy()


@dataclass(frozen=True)
class ScalingCandidate:
    alpha: float
    beta: float
    residual: float
    admissible: bool


def self_similar_scaling_search(alphas, betas, relation_tol=1e-10):
    """Bounded search for alpha,beta satisfying alpha+beta=1.

    This is a hypothesis filter only; it does not establish blow-up.
    """
    relation_tol = float(relation_tol)
    if not np.isfinite(relation_tol) or relation_tol < 0:
        raise ValueError("relation_tol must be finite and non-negative")
    out = []
    for a in alphas:
        for b in betas:
            a, b = float(a), float(b)
            if not np.isfinite(a) or not np.isfinite(b):
                continue
            r = abs(a + b - 1.0)
            out.append(ScalingCandidate(a, b, r, bool(r <= relation_tol)))
    return tuple(out)


def bkm_scaling_indicator(times, omega_sup_norms, exponents=(0.0, 1.0)):
    """Measure weighted BKM growth over a finite interval.

    Returns finite diagnostics only; it never labels a solution singular or regular.
    """
    t = np.asarray(times, float)
    w = np.asarray(omega_sup_norms, float)
    if t.ndim != 1 or w.ndim != 1 or len(t) != len(w) or len(t) < 2:
        raise ValueError("matching 1D arrays with >=2 points required")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(w)) or np.any(np.diff(t) <= 0) or np.any(w < 0):
        raise ValueError("invalid BKM data")
    results = []
    T = float(t[-1])
    for p in exponents:
        p = float(p)
        if not np.isfinite(p) or p < 0:
            raise ValueError("exponents must be finite and non-negative")
        weight = np.maximum(T - t, np.finfo(float).eps) ** p
        val = float(np.trapezoid(w * weight, t)) if hasattr(np, "trapezoid") else float(np.trapz(w * weight, t))
        results.append((p, val))
    return tuple(results)


def casimir_residual(poisson_matrix, gradient):
    """Return J grad(C), the finite-dimensional Casimir residual."""
    J = np.asarray(poisson_matrix, float)
    g = np.asarray(gradient, float)
    if J.ndim != 2 or J.shape[0] != J.shape[1] or g.shape != (J.shape[0],):
        raise ValueError("J must be square and gradient must match its dimension")
    if not np.all(np.isfinite(J)) or not np.all(np.isfinite(g)):
        raise ValueError("non-finite Poisson data")
    return J @ g


def hasimoto_geometry(points, closed=False):
    """Compute discrete curvature/torsion and the Hasimoto phase of a space curve."""
    X = np.asarray(points, float)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 5:
        raise ValueError("points must have shape (n>=5,3)")
    if not np.all(np.isfinite(X)):
        raise ValueError("points contain non-finite values")
    if closed:
        P = np.vstack([X[-2:], X, X[:2]])
    else:
        P = X
    d1 = np.gradient(P, axis=0, edge_order=2)
    d2 = np.gradient(d1, axis=0, edge_order=2)
    d3 = np.gradient(d2, axis=0, edge_order=2)
    speed = np.linalg.norm(d1, axis=1)
    cross = np.cross(d1, d2)
    cn = np.linalg.norm(cross, axis=1)
    kappa = cn / np.maximum(speed**3, 1e-30)
    torsion = np.einsum("ij,ij->i", cross, d3) / np.maximum(cn**2, 1e-30)
    if closed:
        sl = slice(2, -2)
        kappa, torsion, speed = kappa[sl], torsion[sl], speed[sl]
    ds = float(np.median(speed))
    phase = np.concatenate([[0.0], np.cumsum(0.5 * (torsion[1:] + torsion[:-1]) * ds)])
    psi = kappa * np.exp(1j * phase)
    return {"curvature": kappa, "torsion": torsion, "speed": speed, "phase": phase, "hasimoto": psi}


def hasimoto_nls_residual(psi, ds=1.0, psi_t=None):
    """Compute i psi_t + psi_ss + 1/2 |psi|^2 psi for a Hasimoto field.

    A small residual supports a consistency test with NLS, but does not prove
    a vortex-filament solution or soliton stability.
    """
    q = np.asarray(psi, complex)
    h = float(ds)
    if q.ndim != 1 or len(q) < 5 or not np.isfinite(h) or h <= 0:
        raise ValueError("psi must be a 1D complex field with >=5 samples and ds>0")
    if not np.all(np.isfinite(q.real)) or not np.all(np.isfinite(q.imag)):
        raise ValueError("psi contains non-finite values")
    qss = np.gradient(np.gradient(q, h, edge_order=2), h, edge_order=2)
    qt = np.zeros_like(q) if psi_t is None else np.asarray(psi_t, complex)
    if qt.shape != q.shape:
        raise ValueError("psi_t shape mismatch")
    return 1j * qt + qss + 0.5 * np.abs(q) ** 2 * q
