"""Dimension-agnostic numerical diagnostics shared by 2D/3D engines."""
from __future__ import annotations
import numpy as np


def l2(a) -> float:
    x = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(x*x)))


def linf(a) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float))))


def normalized_residual(residual, scale, floor=1e-14) -> float:
    r = l2(residual)
    s = max(l2(scale), float(floor))
    return r / s


def normalized_divergence(divergence, velocity, lengths=None, floor=1e-14) -> float:
    """Dimensionless divergence residual.

    The length scale is included when supplied, making the metric comparable
    across grids/domains instead of relying on a fixed absolute tolerance.
    """
    u = tuple(np.asarray(v, dtype=float) for v in velocity)
    L = min(tuple(float(x) for x in lengths)) if lengths is not None else 1.0
    return float(L * l2(divergence) / max(l2(np.concatenate([v.ravel() for v in u])), floor))


def kinetic_energy(velocity) -> float:
    return 0.5 * float(np.mean(sum(np.asarray(v, dtype=float)**2 for v in velocity)))


def energy_rate(velocity, rhs) -> float:
    return float(np.mean(sum(np.asarray(u)*np.asarray(f) for u, f in zip(velocity, rhs))))
