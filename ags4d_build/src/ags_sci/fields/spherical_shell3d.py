"""Spherical-shell spectral building blocks inspired by Tilgner (1999).

This module intentionally implements only the parts that can be verified in isolation:
Chebyshev radial collocation, optional monotone radial stretching, and exact
integrating-factor propagation for the linear toroidal diffusion operator.  It is
NOT a complete spherical-shell Navier--Stokes solver.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def chebyshev_lobatto(n: int):
    if int(n) != n or n < 3:
        raise ValueError("n must be an integer >= 3")
    n = int(n)
    x = np.cos(np.pi * np.arange(n) / (n - 1))
    # Standard descending Chebyshev-Lobatto differentiation matrix.
    c = np.ones(n); c[[0, -1]] = 2.0
    c *= (-1.0) ** np.arange(n)
    X = x[:, None]
    dX = X - X.T
    D = (c[:, None] / c[None, :]) / (dX + np.eye(n))
    D = D - np.diag(np.sum(D, axis=1))
    return x, D


def stretched_radius(x, r_inner, r_outer, beta=0.0):
    """Map x in [-1,1] monotonically to [r_inner,r_outer].

    beta=0 gives the affine map.  Positive beta concentrates points toward both
    boundaries using the sine mapping described qualitatively in the reference.
    """
    ri, ro = float(r_inner), float(r_outer)
    b = float(beta)
    if not (0 <= ri < ro):
        raise ValueError("require 0 <= r_inner < r_outer")
    if not np.isfinite(b) or b < 0 or b >= 1:
        raise ValueError("beta must satisfy 0 <= beta < 1")
    if b == 0:
        return ri + 0.5 * (x + 1.0) * (ro - ri)
    denom = np.sin(np.pi * b / 2.0)
    if abs(denom) < 1e-14:
        return ri + 0.5 * (x + 1.0) * (ro - ri)
    y = np.sin(np.pi * b * x / 2.0) / denom
    return ri + 0.5 * (ro - ri) * (y + 1.0)


@dataclass(frozen=True)
class SphericalShellResolution:
    radial_points: int
    l_max: int
    max_positive_real_part: float
    stable_linear_discretization: bool


class SphericalShellToroidalDiffusion:
    """Chebyshev collocation of the linear toroidal diffusion operator.

    The interior operator follows the radial operator described in the paper,
    with homogeneous Dirichlet values at both shell boundaries.
    """
    def __init__(self, radial_points=33, l=1, r_inner=0.35, r_outer=1.0,
                 viscosity=1e-3, stretch_beta=0.0):
        n = int(radial_points)
        ell = int(l)
        if n < 5 or ell < 1:
            raise ValueError("radial_points >= 5 and l >= 1 required")
        if not np.isfinite(viscosity) or viscosity < 0:
            raise ValueError("viscosity must be finite and non-negative")
        x, Dx = chebyshev_lobatto(n)
        # Reference nodes are descending. Reverse to increasing radius for clearer API.
        x = x[::-1]
        Dx = -Dx[::-1, ::-1]
        r = stretched_radius(x, r_inner, r_outer, stretch_beta)
        # Chain rule d/dr = (dx/dr) d/dx. Build D_r directly from mapped nodes.
        # For a general mapping, construct the derivative matrix using barycentric
        # weights in r; this avoids assuming an affine map.
        _, Dr = chebyshev_lobatto(n)
        Dr = -Dr[::-1, ::-1]
        # Recompute via barycentric differentiation on the mapped r nodes.
        w = np.ones(n)
        for j in range(n):
            for k in range(n):
                if j != k:
                    w[j] /= (r[j] - r[k])
        R = r[:, None]
        dR = R - R.T
        Dr = (w[None, :] / w[:, None]) / (dR + np.eye(n))
        Dr = Dr - np.diag(np.sum(Dr, axis=1))
        D2 = Dr @ Dr
        Lfull = D2 + np.diag(4.0 / r) @ Dr + np.diag((2.0 - ell * (ell + 1.0)) / r**2)
        self.r = r
        self.Dr = Dr
        self.l = ell
        self.viscosity = float(viscosity)
        self.operator_full = self.viscosity * Lfull
        self.operator = self.operator_full[1:-1, 1:-1]
        vals, vecs = np.linalg.eig(self.operator)
        self.eigenvalues = vals
        self.eigenvectors = vecs
        self.eigenvectors_inv = np.linalg.inv(vecs)
        self.resolution = SphericalShellResolution(
            radial_points=n,
            l_max=ell,
            max_positive_real_part=float(np.max(np.real(vals))),
            stable_linear_discretization=bool(np.max(np.real(vals)) <= 1e-10),
        )

    def derivative(self, values):
        a = np.asarray(values, dtype=float)
        if a.shape != self.r.shape:
            raise ValueError("values must match radial grid")
        return self.Dr @ a

    def rhs(self, interior_values):
        a = np.asarray(interior_values, dtype=float)
        if a.shape != (len(self.r) - 2,):
            raise ValueError("interior_values has wrong shape")
        return self.operator @ a

    def integrating_factor(self, interior_values, dt):
        """Exact linear diffusion propagation for the discretized operator."""
        a = np.asarray(interior_values, dtype=float)
        h = float(dt)
        if a.shape != (len(self.r) - 2,):
            raise ValueError("interior_values has wrong shape")
        if not np.isfinite(h) or h < 0:
            raise ValueError("dt must be finite and non-negative")
        coeff = self.eigenvectors_inv @ a
        evolved = self.eigenvectors @ (np.exp(self.eigenvalues * h) * coeff)
        out = np.real_if_close(evolved, tol=1000)
        return np.asarray(out, dtype=float)

    def decay_rate(self, interior_values):
        a = np.asarray(interior_values, dtype=float)
        if a.shape != (len(self.r) - 2,):
            raise ValueError("interior_values has wrong shape")
        before = float(np.linalg.norm(a))
        after = float(np.linalg.norm(self.integrating_factor(a, 1.0)))
        return after / before if before else 0.0
