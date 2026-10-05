"""Spectral a-posteriori error estimators for periodic fields.

Inspired by the uploaded Spectral-Refiner paper: negative Sobolev residual
functionals can emphasize low-frequency error and are computable directly in
Fourier space. This module is deliberately solver-agnostic and does not train a
neural operator.
"""
from __future__ import annotations
import numpy as np


def negative_sobolev_norm(field, domain_lengths=None, order=1, alpha=0.0):
    """Return a regularized periodic H^{-order}-type norm.

    The zero Fourier mode is omitted for the homogeneous seminorm. For
    ``alpha>0`` the weight is ``(alpha + |k|^2)^(-order)`` for nonzero modes.
    The normalization is the Parseval-equivalent mean-square normalization,
    making values independent of the number of grid points for fixed samples.
    """
    a = np.asarray(field, dtype=float)
    if a.ndim < 1:
        raise ValueError("field must have at least one dimension")
    if not np.all(np.isfinite(a)):
        raise ValueError("field contains non-finite values")
    lengths = tuple(float(x) for x in (domain_lengths or (2*np.pi,) * a.ndim))
    if len(lengths) != a.ndim or any(x <= 0 for x in lengths):
        raise ValueError("domain_lengths must match field dimensions and be positive")
    s = float(order); reg = float(alpha)
    if not np.isfinite(s) or s <= 0 or not np.isfinite(reg) or reg < 0:
        raise ValueError("order must be positive and alpha non-negative")
    modes = [np.fft.fftfreq(n) * n for n in a.shape]
    grids = np.meshgrid(*[2*np.pi*m/L for m, L in zip(modes, lengths)], indexing="ij")
    k2 = sum(g*g for g in grids)
    weight = np.zeros_like(k2, dtype=float)
    if reg > 0:
        weight = (reg + k2) ** (-s)
        # Keep the mean out of the homogeneous residual/seminorm.
        weight[k2 == 0] = 0.0
    else:
        nz = k2 > 0
        weight[nz] = k2[nz] ** (-s)
    hat = np.fft.fftn(a)
    norm2 = float(np.sum(weight * np.abs(hat) ** 2) / np.prod(a.shape) ** 2)
    return float(np.sqrt(max(norm2, 0.0)))


def vector_negative_sobolev_norm(vector_field, domain_lengths=None, order=1, alpha=0.0):
    """Euclidean combination of component H^{-order} norms."""
    if len(vector_field) == 0:
        raise ValueError("vector_field must not be empty")
    vals = [negative_sobolev_norm(v, domain_lengths, order, alpha) for v in vector_field]
    return float(np.sqrt(np.sum(np.square(vals))))


def spectral_residual_loss(residual, domain_lengths=None, alpha=0.0):
    """Convenience loss for PDE residual refinement."""
    return negative_sobolev_norm(residual, domain_lengths, order=1, alpha=alpha)
