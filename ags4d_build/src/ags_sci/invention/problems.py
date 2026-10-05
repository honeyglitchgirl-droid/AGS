"""Problem generation: invent new questions from what a model fails to explain.

Fitting a law is only half of research. The other half is noticing that
something is *left over*. This module takes a trajectory and the best candidate
law available for it, then interrogates the residuals for unexplained structure:

- **temporal autocorrelation** — residuals that persist across samples point at
  missing dynamics or memory, not noise;
- **spectral concentration** — a sharp peak in the residual spectrum points at an
  unmodelled oscillation;
- **heteroscedasticity** — residual spread that depends on a variable points at a
  missing multiplicative term;
- **drift** — a non-zero residual mean points at a missing constant or forcing;
- **symmetry breaking** — residuals that are not odd under x -> -x point at a
  missing parity term.

Each finding is emitted as an :class:`OpenProblem` with the numeric evidence
attached, so a human (or another AI) can judge whether it is worth pursuing. AGS
does not claim these are unsolved problems of science; they are *unexplained
structure in the supplied data*, which is the only thing the evidence supports.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OpenProblem:
    """A newly formulated question arising from unexplained residual structure."""

    problem_id: str
    kind: str
    statement: str
    evidence: dict[str, float]
    suggested_investigation: str
    severity: float

    def describe(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "kind": self.kind,
            "statement": self.statement,
            "evidence": dict(self.evidence),
            "suggested_investigation": self.suggested_investigation,
            "severity": self.severity,
        }


def _autocorrelation(residuals: np.ndarray, max_lag: int = 20) -> tuple[float, int]:
    """Return (peak |autocorrelation|, its lag) over 1..max_lag."""
    r = residuals - residuals.mean()
    denom = float(np.sum(r * r))
    if denom <= 0:
        return 0.0, 0
    best, best_lag = 0.0, 0
    for lag in range(1, min(max_lag, len(r) - 1) + 1):
        v = float(np.sum(r[lag:] * r[:-lag]) / denom)
        if abs(v) > abs(best):
            best, best_lag = v, lag
    return best, best_lag


def _spectral_peak(residuals: np.ndarray) -> tuple[float, float]:
    """Return (dominant frequency fraction, peak power fraction)."""
    r = residuals - residuals.mean()
    if len(r) < 4 or float(np.sum(r * r)) <= 0:
        return 0.0, 0.0
    power = np.abs(np.fft.rfft(r)) ** 2
    if len(power) < 3:
        return 0.0, 0.0
    body = power[1:]
    peak = int(np.argmax(body)) + 1
    total = float(np.sum(power))
    return peak / max(1, len(r)), float(power[peak] / total) if total > 0 else 0.0


def _heteroscedasticity(residuals: np.ndarray, driver: np.ndarray) -> float:
    """Correlation between |residual| and the driver variable.

    A large absolute value means the error scale changes with the driver, which a
    purely additive model cannot represent.
    """
    a = np.abs(residuals - residuals.mean())
    b = np.asarray(driver, dtype=float)
    if a.std() <= 0 or b.std() <= 0:
        return 0.0
    return abs(float(np.corrcoef(a, b)[0, 1]))


def _parity_asymmetry(residuals: np.ndarray, x: np.ndarray) -> float:
    """How far residuals are from being an odd function of ``x``.

    Computed by comparing residuals at mirrored sample positions when the sample
    grid is symmetric; returns 0.0 when that is not assessable.
    """
    r = np.asarray(residuals, dtype=float)
    xv = np.asarray(x, dtype=float)
    n = len(r)
    if n < 8 or xv.shape != r.shape:
        return 0.0
    # Mirroring by reversal is only valid when the grid really is symmetric,
    # i.e. x[i] == -x[n-1-i] pointwise. Checking only that min(x) == -max(x)
    # is not enough: an even number of uniformly spaced points over a
    # symmetric interval has no point at the origin and is *not* mirrored.
    if not np.allclose(xv + xv[::-1], 0.0, rtol=1e-6, atol=1e-9):
        return 0.0
    mirrored = r[::-1]
    even_part = 0.5 * (r + mirrored)
    denom = float(np.sum(r * r))
    if denom <= 0:
        return 0.0
    # Under "no parity structure" the even part carries half the residual
    # energy, with relative fluctuations of order 1/sqrt(n). Reporting the raw
    # even fraction therefore flags *every* noise-like residual (it sits near
    # 0.5 by construction), so the finding is expressed as a signed z-score
    # against that null hypothesis instead.
    even_fraction = float(np.sum(even_part * even_part) / denom)
    return float((even_fraction - 0.5) * np.sqrt(n))


class ProblemGenerator:
    """Formulate open problems from residual structure."""

    def __init__(self, autocorr_threshold: float = 0.25,
                 spectral_threshold: float = 0.20,
                 heteroscedasticity_threshold: float = 0.35,
                 drift_threshold: float = 0.05,
                 parity_threshold: float = 3.0):
        self.autocorr_threshold = float(autocorr_threshold)
        self.spectral_threshold = float(spectral_threshold)
        self.heteroscedasticity_threshold = float(heteroscedasticity_threshold)
        self.drift_threshold = float(drift_threshold)
        # ``parity_threshold`` is a z-score against the "half the residual
        # energy is even" null, not a raw energy fraction.
        self.parity_threshold = float(parity_threshold)

    def generate(self, residuals, x=None, t=None, driver=None,
                 base_statement: str = "the fitted relation") -> tuple[OpenProblem, ...]:
        """Return the open problems implied by unexplained residual structure.

        ``x`` is the primary independent variable (used for symmetry checks),
        ``t`` an ordering used for autocorrelation, and ``driver`` an optional
        variable suspected of scaling the error. Any of them may be omitted, in
        which case the checks that need them are skipped.
        """
        r = np.asarray(residuals, dtype=float).reshape(-1)
        if r.size < 8:
            return ()
        problems: list[OpenProblem] = []
        scale = float(np.std(r)) or 1.0

        # 1. Drift: a non-zero mean residual.
        mean_res = float(np.mean(r))
        # 3/sqrt(n) is the sampling floor: a residual mean below it is
        # indistinguishable from the noise itself.
        drift_floor = max(self.drift_threshold, 3.0 / np.sqrt(r.size))
        if abs(mean_res) / scale > drift_floor:
            problems.append(OpenProblem(
                problem_id="P-DRIFT",
                kind="unexplained_offset",
                statement=(f"{base_statement} leaves a systematic offset of "
                           f"{mean_res:.4g} (|mean|/sigma = {abs(mean_res)/scale:.3f}); "
                           "an additive term or forcing is missing"),
                evidence={"mean_residual": mean_res, "residual_sigma": scale,
                          "ratio": abs(mean_res) / scale,
                          "sampling_floor": drift_floor},
                suggested_investigation=("extend the library with a constant or an "
                                         "external forcing term and refit"),
                severity=min(1.0, abs(mean_res) / scale / 2.0),
            ))

        # 2. Temporal autocorrelation: missing dynamics or memory.
        if t is not None:
            ac, lag = _autocorrelation(r)
            # Same sampling floor: rho ~ +/-1/sqrt(n) for white noise.
            ac_floor = max(self.autocorr_threshold, 3.0 / np.sqrt(r.size))
            if abs(ac) > ac_floor:
                problems.append(OpenProblem(
                    problem_id="P-MEMORY",
                    kind="serial_dependence",
                    statement=(f"residuals of {base_statement} are autocorrelated at "
                               f"lag {lag} (rho = {ac:.3f}); the residuals carry "
                               "structure rather than being independent noise"),
                    evidence={"autocorrelation": ac, "lag": float(lag),
                              "sampling_floor": ac_floor},
                    suggested_investigation=("consider delay coordinates, a "
                                             "higher-order derivative, or an omitted "
                                             "state variable"),
                    severity=min(1.0, abs(ac)),
                ))

        # 3. Spectral concentration: an unmodelled oscillation.
        freq, frac = _spectral_peak(r)
        # With n/2 bins, white noise puts ~4/n of the total power in its
        # largest bin; below that floor a "peak" is just bin noise.
        bins = max(1, r.size // 2)
        spectral_floor = max(self.spectral_threshold, 4.0 / bins)
        if frac > spectral_floor:
            problems.append(OpenProblem(
                problem_id="P-OSCILLATION",
                kind="unmodelled_periodicity",
                statement=(f"the residual spectrum of {base_statement} is dominated by "
                           f"a single component at normalised frequency {freq:.4f} "
                           f"carrying {frac:.1%} of the power"),
                evidence={"dominant_frequency": freq, "power_fraction": frac},
                suggested_investigation=("add the corresponding oscillatory term, or "
                                         "test whether the system has an unobserved "
                                         "periodic driver"),
                severity=min(1.0, frac),
            ))

        # 4. Heteroscedasticity: a missing multiplicative structure.
        if driver is not None:
            h = _heteroscedasticity(r, driver)
            if h > self.heteroscedasticity_threshold:
                problems.append(OpenProblem(
                    problem_id="P-SCALE",
                    kind="state_dependent_error",
                    statement=(f"the residual magnitude of {base_statement} correlates "
                               f"with the driver variable (|r| = {h:.3f}); the error "
                               "scale is state-dependent"),
                    evidence={"heteroscedasticity": h},
                    suggested_investigation=("test multiplicative or power-law terms, "
                                             "or model the variance explicitly"),
                    severity=min(1.0, h),
                ))

        # 5. Parity asymmetry: a missing odd/even component.
        if x is not None:
            p = _parity_asymmetry(r, np.asarray(x, dtype=float))
            if abs(p) > self.parity_threshold:
                even_fraction = 0.5 + p / np.sqrt(r.size)
                if p > 0:
                    shape, direction = "even-in-x", "an even"
                else:
                    shape, direction = "odd-in-x", "an odd"
                problems.append(OpenProblem(
                    problem_id="P-PARITY",
                    kind="symmetry_breaking",
                    statement=(f"the residuals of {base_statement} carry a "
                               f"{shape} component far beyond chance "
                               f"(z = {p:.1f}); {direction} term is missing from "
                               "the fitted relation"),
                    evidence={"parity_z": p, "even_fraction": even_fraction,
                              "sample_size": float(r.size)},
                    suggested_investigation=("test odd and even basis functions "
                                             "separately, or check whether the "
                                             "underlying symmetry is genuinely broken"),
                    severity=min(1.0, abs(p) / 10.0),
                ))

        problems.sort(key=lambda p: (-p.severity, p.problem_id))
        return tuple(problems)
