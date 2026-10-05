"""Conservative physical/numerical gates for experimental 3D flow results.

The referee distinguishes diagnostics from proofs. In particular, BKM monitoring
never certifies absence of finite-time singularities; it only reports consistency
with the criterion over the tested interval.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .turbulence3d import PseudoSpectral3D
from .referee import ResolutionRefereeND, ResolutionDecision


@dataclass(frozen=True)
class BKMMonitorResult:
    omega_sup_integral: float
    finite: bool
    criterion_status: str


@dataclass(frozen=True)
class PhysicalGate3DResult:
    accepted: bool
    divergence_linf: float
    energy: float
    helicity: float
    enstrophy: float
    resolution: ResolutionDecision | None
    reasons: tuple[str, ...]


class PhysicalReferee3D:
    """Fail-closed 3D gate; no diagnostic is treated as a mathematical proof."""

    def __init__(self, divergence_linf_tol=1e-8, resolution_referee=None,
                 relative_divergence_tol=1e-7):
        self.divergence_linf_tol = float(divergence_linf_tol)
        self.relative_divergence_tol = float(relative_divergence_tol)
        if self.divergence_linf_tol <= 0 or self.relative_divergence_tol <= 0:
            raise ValueError("divergence tolerances must be positive")
        self.resolution_referee = resolution_referee or ResolutionRefereeND()

    def judge(self, engine: PseudoSpectral3D, u, tail_ratio=None, core_ratio=None):
        d = engine.diagnostics(u)
        reasons = []
        velocity_scale = max(float(np.sqrt(np.mean(sum(c*c for c in u)))), 1e-30)
        relative_div = d.divergence_linf / velocity_scale
        if (not np.isfinite(d.divergence_linf) or d.divergence_linf > self.divergence_linf_tol
                or relative_div > self.relative_divergence_tol):
            reasons.append("divergence_gate_failed")
        for name, value in (("kinetic_energy", d.kinetic_energy), ("helicity", d.helicity), ("enstrophy", d.enstrophy)):
            if not np.isfinite(value):
                reasons.append(f"nonfinite_{name}")
        resolution = None
        if tail_ratio is not None:
            resolution = self.resolution_referee.judge(tail_ratio, core_ratio)
            if not resolution.accepted:
                reasons.extend(resolution.reasons)
        return PhysicalGate3DResult(
            accepted=not reasons,
            divergence_linf=d.divergence_linf,
            energy=d.kinetic_energy,
            helicity=d.helicity,
            enstrophy=d.enstrophy,
            resolution=resolution,
            reasons=tuple(reasons),
        )

    @staticmethod
    def bkm_monitor(times, omega_sup_norms) -> BKMMonitorResult:
        t = np.asarray(times, dtype=float)
        w = np.asarray(omega_sup_norms, dtype=float)
        if t.ndim != 1 or w.ndim != 1 or len(t) != len(w) or len(t) < 2:
            raise ValueError("times and omega_sup_norms require matching 1D arrays with >=2 points")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(w)) or np.any(np.diff(t) <= 0) or np.any(w < 0):
            raise ValueError("invalid BKM data")
        integral = float(np.trapezoid(w, t)) if hasattr(np, "trapezoid") else float(np.trapz(w, t))
        return BKMMonitorResult(
            omega_sup_integral=integral,
            finite=bool(np.isfinite(integral)),
            criterion_status="consistent_with_tested_interval_only" if np.isfinite(integral) else "invalid",
        )
