"""Improved sparse law/equation discovery.

:class:`~ags_sci.dynamics.identification.DynamicSystemIdentifier` uses a single
absolute coefficient threshold. That is fine when the caller knows the noise
scale, but with real trajectories it leaves behind terms of order 1e-3 that are
numerically indistinguishable from zero and make the recovered equation
unreadable.

This module adds a layer that fixes the three things that matter for *usable*
law discovery:

1. **Relative thresholding.** A term is kept only if its coefficient is a
   meaningful fraction of the dominant coefficient for that equation, so the
   sparsity structure reflects the data rather than the absolute units.
2. **Model selection instead of a guessed threshold.** The threshold is chosen by
   BIC over a sweep, trading fit against parsimony automatically.
3. **Held-out validation.** The model is scored on data it never saw, so a good
   in-sample fit cannot be mistaken for a good model.

Everything here is numerical evidence. ``evidence_tier`` states exactly how much
support a candidate has; nothing in this module promotes a candidate to a
scientific law.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

#: Evidence tiers, weakest to strongest. A candidate never silently upgrades.
EVIDENCE_TIERS = ("NUMERICAL_FIT", "HELD_OUT", "PARSIMONIOUS")


@dataclass(frozen=True)
class LawCandidate:
    """A sparse ODE model with its fit, its held-out score, and its evidence."""

    equations: tuple[str, ...]
    coefficients: tuple[tuple[tuple[str, float], ...], ...]
    train_rmse: float
    holdout_rmse: float | None
    bic: float
    n_active_terms: int
    evidence_tier: str
    library: tuple[str, ...]
    note: str = "sparse regression on finite-difference derivatives; numerical evidence only"

    def describe(self) -> dict[str, object]:
        return {
            "equations": list(self.equations),
            "coefficients": [
                [{"term": t, "coefficient": c} for t, c in eq] for eq in self.coefficients
            ],
            "train_rmse": self.train_rmse,
            "holdout_rmse": self.holdout_rmse,
            "bic": self.bic,
            "n_active_terms": self.n_active_terms,
            "evidence_tier": self.evidence_tier,
            "library_size": len(self.library),
            "note": self.note,
        }


def build_library(X: np.ndarray, names: tuple[str, ...], degree: int = 2,
                  include_transcendentals: bool = False
                  ) -> tuple[np.ndarray, tuple[str, ...]]:
    """Candidate feature matrix and its labels."""
    X = np.asarray(X, float)
    cols = [np.ones(len(X))]
    labels = ["1"]
    for j, n in enumerate(names):
        cols.append(X[:, j])
        labels.append(n)
    for total in range(2, int(degree) + 1):
        for idx in combinations_with_replacement(range(X.shape[1]), total):
            powers = [0] * X.shape[1]
            for i in idx:
                powers[i] += 1
            cols.append(np.prod([X[:, j] ** powers[j] for j in range(X.shape[1])], axis=0))
            labels.append("*".join(names[j] for j in idx))
    if include_transcendentals:
        for j, n in enumerate(names):
            cols.extend([np.sin(X[:, j]), np.cos(X[:, j])])
            labels.extend([f"sin({n})", f"cos({n})"])
    return np.column_stack(cols), tuple(labels)


def _stlsq(A: np.ndarray, b: np.ndarray, threshold: float, max_iter: int = 30):
    """Sequentially-thresholded least squares with a *relative* cut."""
    active = np.ones(A.shape[1], dtype=bool)
    coef = np.zeros(A.shape[1])
    for _ in range(int(max_iter)):
        if not np.any(active):
            break
        sol = np.linalg.lstsq(A[:, active], b, rcond=None)[0]
        coef[:] = 0.0
        coef[active] = sol
        scale = float(np.max(np.abs(coef))) if np.any(coef) else 0.0
        # Relative cut: keep terms that matter compared with the dominant term.
        new = active & (np.abs(coef) >= threshold * max(scale, 1e-300))
        if np.array_equal(new, active):
            break
        active = new
    if not np.any(active):
        active[int(np.argmax(np.abs(coef)))] = True
        coef = np.zeros(A.shape[1])
        coef[active] = np.linalg.lstsq(A[:, active], b, rcond=None)[0]
    return coef, active


class SparseLawDiscovery:
    """Discover sparse ODEs with automatic threshold selection and holdout scoring."""

    def __init__(self, degree: int = 2, include_transcendentals: bool = False,
                 holdout_fraction: float = 0.25, seed: int = 0,
                 thresholds: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)):
        if int(degree) < 1:
            raise ValueError("degree must be >= 1")
        if not 0.0 <= float(holdout_fraction) < 1.0:
            raise ValueError("holdout_fraction must be in [0,1)")
        self.degree = int(degree)
        self.include_transcendentals = bool(include_transcendentals)
        self.holdout_fraction = float(holdout_fraction)
        self.seed = int(seed)
        self.thresholds = tuple(float(t) for t in thresholds)

    def _derivatives(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Smooth finite-difference derivatives when SciPy is available."""
        from ..dynamics.identification import RobustDifferentiator
        diff = RobustDifferentiator()
        return np.column_stack([diff.derivative(t, X[:, j]) for j in range(X.shape[1])])

    def _split(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if self.holdout_fraction <= 0 or n < 8:
            return np.arange(n), np.array([], dtype=int)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n)
        n_hold = max(1, int(round(self.holdout_fraction * n)))
        hold = np.sort(order[:n_hold])
        train = np.sort(order[n_hold:])
        return train, hold

    def fit(self, t, X, names=None) -> LawCandidate:
        t = np.asarray(t, float)
        X = np.asarray(X, float)
        if t.ndim != 1 or X.ndim != 2 or len(t) != X.shape[0]:
            raise ValueError("t must be 1-D and X must be (len(t), n_states)")
        if len(t) < 8:
            raise ValueError("need at least 8 trajectory points")
        names = tuple(names or [f"x{i}" for i in range(X.shape[1])])
        if len(names) != X.shape[1]:
            raise ValueError("names must match the number of state columns")

        dX = self._derivatives(t, X)
        Theta, labels = build_library(X, names, self.degree, self.include_transcendentals)
        if not np.all(np.isfinite(Theta)) or not np.all(np.isfinite(dX)):
            raise ValueError("library or derivatives contain non-finite values")

        train_idx, hold_idx = self._split(len(t))
        n = len(train_idx)

        best: tuple[float, float, np.ndarray] | None = None
        for thr in self.thresholds:
            train_rmse, active, coef = 0.0, None, None
            per_target = []
            for j in range(X.shape[1]):
                c, a = _stlsq(Theta[train_idx], dX[train_idx, j], thr)
                pred = Theta[train_idx] @ c
                train_rmse += float(np.mean((pred - dX[train_idx, j]) ** 2))
                per_target.append((c, a))
            train_rmse = float(np.sqrt(train_rmse / X.shape[1]))
            n_terms = int(sum(int(np.sum(a)) for _, a in per_target))
            # BIC: n*log(RSS/n) + k*log(n). Lower is better.
            bic = n * np.log(max(train_rmse ** 2, 1e-300)) + n_terms * np.log(max(n, 2))
            if best is None or bic < best[0]:
                best = (float(bic), thr, np.array([a for _, a in per_target]))
        assert best is not None
        _, chosen_thr, active_mask = best

        # Refit on the full data using the selected support.
        equations: list[str] = []
        coefficients: list[tuple[tuple[str, float], ...]] = []
        train_rmse_total = 0.0
        n_terms_total = 0
        for j in range(X.shape[1]):
            active = active_mask[j]
            coef = np.zeros(Theta.shape[1])
            if np.any(active):
                coef[active] = np.linalg.lstsq(Theta[:, active], dX[:, j], rcond=None)[0]
            pred = Theta @ coef
            train_rmse_total += float(np.mean((pred - dX[:, j]) ** 2))
            n_terms_total += int(np.sum(active))
            kept = [(labels[k], float(coef[k])) for k in range(len(labels)) if active[k]]
            coefficients.append(tuple(kept))
            equations.append(
                f"d({names[j]})/dt = " + " + ".join(f"({c:.6g})*({t_})" for t_, c in kept)
                if kept else f"d({names[j]})/dt = 0"
            )
        train_rmse = float(np.sqrt(train_rmse_total / X.shape[1]))

        # Held-out scoring.
        holdout_rmse: float | None = None
        if len(hold_idx):
            errs = []
            for j in range(X.shape[1]):
                active = active_mask[j]
                coef = np.zeros(Theta.shape[1])
                if np.any(active):
                    coef[active] = np.linalg.lstsq(Theta[train_idx][:, active],
                                                   dX[train_idx, j], rcond=None)[0]
                pred = Theta[hold_idx] @ coef
                errs.append(float(np.mean((pred - dX[hold_idx, j]) ** 2)))
            holdout_rmse = float(np.sqrt(sum(errs) / len(errs)))

        # Evidence tier: never stronger than the data supports.
        if holdout_rmse is None:
            tier = "NUMERICAL_FIT"
        elif holdout_rmse <= max(10.0 * train_rmse, 1e-9):
            tier = "PARSIMONIOUS" if n_terms_total <= 2 * X.shape[1] else "HELD_OUT"
        else:
            tier = "NUMERICAL_FIT"

        return LawCandidate(
            equations=tuple(equations),
            coefficients=tuple(coefficients),
            train_rmse=train_rmse,
            holdout_rmse=holdout_rmse,
            bic=float(best[0]),
            n_active_terms=n_terms_total,
            evidence_tier=tier,
            library=labels,
        )
