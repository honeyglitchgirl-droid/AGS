"""Bounded iterative linear solvers used by AGS-Sci hybrid spectral backends.

The implementation is intentionally small and deterministic.  It provides a
restartable GMRES solver and a p-multigrid-style polynomial preconditioner for
Chebyshev radial operators.  It is a numerical foundation, not a claim of the
full free-surface solver described in the reference thesis.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

Array = np.ndarray

@dataclass(frozen=True)
class LinearSolveResult:
    x: Array
    converged: bool
    iterations: int
    residual_norm: float
    history: tuple[float, ...]


def _norm(x: Array) -> float:
    return float(np.linalg.norm(np.asarray(x).ravel()))


def gmres(
    matvec: Callable[[Array], Array],
    b: Array,
    x0: Optional[Array] = None,
    tol: float = 1e-10,
    maxiter: int = 100,
    restart: int = 30,
    M: Optional[Callable[[Array], Array]] = None,
) -> LinearSolveResult:
    """Restarted left-preconditioned GMRES with strict finite checks."""
    b = np.asarray(b, dtype=float)
    if not np.all(np.isfinite(b)):
        raise ValueError("b contains non-finite values")
    if tol <= 0 or maxiter < 1 or restart < 1:
        raise ValueError("invalid GMRES controls")
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float, copy=True)
    if x.shape != b.shape or not np.all(np.isfinite(x)):
        raise ValueError("invalid x0")
    pre = M or (lambda v: v)
    bnorm = max(_norm(pre(b)), 1e-300)
    history = []
    iters = 0
    while iters < maxiter:
        r = pre(b - matvec(x))
        beta = _norm(r)
        history.append(beta / bnorm)
        if beta / bnorm <= tol:
            return LinearSolveResult(x, True, iters, beta, tuple(history))
        m = min(restart, maxiter - iters)
        V = np.zeros((m + 1, b.size), dtype=float)
        H = np.zeros((m + 1, m), dtype=float)
        V[0] = r.ravel() / beta
        best_x = x.copy(); best_res = beta
        for j in range(m):
            w = np.asarray(pre(matvec(V[j].reshape(b.shape))), dtype=float).ravel()
            if not np.all(np.isfinite(w)):
                return LinearSolveResult(x, False, iters + j + 1, float("inf"), tuple(history))
            for i in range(j + 1):
                H[i, j] = np.dot(w, V[i])
                w -= H[i, j] * V[i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 0:
                V[j + 1] = w / H[j + 1, j]
            e1 = np.zeros(j + 2); e1[0] = beta
            y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)
            cand = x + np.sum(y[:, None] * V[:j + 1], axis=0).reshape(b.shape)
            res = _norm(pre(b - matvec(cand)))
            history.append(res / bnorm)
            iters += 1
            if res < best_res:
                best_x, best_res = cand.copy(), res
            if res / bnorm <= tol:
                return LinearSolveResult(cand, True, iters, res, tuple(history))
            if iters >= maxiter:
                break
        x = best_x
    return LinearSolveResult(x, False, iters, best_res, tuple(history))


def polynomial_p_multigrid_preconditioner(A: Array, levels: int = 3, smooth_steps: int = 2, omega: float = 0.7):
    """Return a bounded Jacobi + coarse polynomial preconditioner.

    This is a lightweight p-multigrid analogue: lower-order polynomial
    subspaces are used as coarse corrections, with damped Jacobi smoothing.
    It is designed for small Chebyshev radial blocks and verification, not
    as a drop-in replacement for a production SEM multigrid library.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if levels < 1 or smooth_steps < 1 or not (0 < omega < 1):
        raise ValueError("invalid multigrid controls")
    diag = np.diag(A).copy()
    if np.any(np.abs(diag) < 1e-14):
        diag = diag + (np.abs(diag) < 1e-14) * 1.0
    n = A.shape[0]
    # Nested low-order index sets emulate a p-hierarchy without inventing a
    # geometry-specific prolongation operator.
    sizes = [max(2, int(round(n / (2 ** i)))) for i in range(levels)]
    sizes = sorted(set(min(n, s) for s in sizes), reverse=True)
    def apply(v):
        x = np.zeros_like(np.asarray(v, dtype=float))
        rhs = np.asarray(v, dtype=float)
        for _ in range(smooth_steps):
            x += omega * (rhs - A @ x) / diag
        for nc in reversed(sizes[1:]):
            idx = np.linspace(0, n - 1, nc).round().astype(int)
            idx = np.unique(idx)
            Ac = A[np.ix_(idx, idx)]
            rc = (rhs - A @ x)[idx]
            try:
                ec = np.linalg.solve(Ac + 1e-12 * np.eye(len(idx)), rc)
            except np.linalg.LinAlgError:
                ec = np.linalg.lstsq(Ac, rc, rcond=None)[0]
            x[idx] += ec
        for _ in range(smooth_steps):
            x += omega * (rhs - A @ x) / diag
        return x
    return apply
