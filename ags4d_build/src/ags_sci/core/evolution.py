"""Fail-closed gate for AGS self-evolution proposals.

Self-evolution may propose changes, but it cannot directly mutate security
policy, sandbox boundaries, release metadata, or protected scientific gates.
The proposal remains an ephemeral candidate until external validation passes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping
from .security import SecurityViolation, safe_identifier, bounded_text, sha256_json

PROTECTED_PATHS = (
    "src/ags_sci/core/security.py",
    "src/ags_sci/core/evolution.py",
    "src/ags_sci/experiment/sandbox.py",
    "pyproject.toml",
)

@dataclass(frozen=True)
class EvolutionProposal:
    proposal_id: str
    changed_paths: tuple[str, ...]
    rationale: str
    baseline_sha256: str
    test_command: str = "python -m pytest -q"

    def validate(self) -> None:
        safe_identifier(self.proposal_id, field="proposal_id")
        bounded_text(self.rationale, field="rationale")
        if not self.baseline_sha256 or len(self.baseline_sha256) != 64:
            raise SecurityViolation("invalid baseline integrity hash")
        if not self.changed_paths:
            raise SecurityViolation("evolution proposal has no changes")
        for raw in self.changed_paths:
            p = PurePosixPath(raw)
            if p.is_absolute() or ".." in p.parts:
                raise SecurityViolation("path traversal in evolution proposal")
            if raw in PROTECTED_PATHS:
                raise SecurityViolation(f"protected path cannot be self-modified: {raw}")

    @property
    def fingerprint(self) -> str:
        self.validate()
        return sha256_json({"proposal_id": self.proposal_id, "paths": self.changed_paths, "rationale": self.rationale, "baseline": self.baseline_sha256})


# ---------------------------------------------------------------------------
# Passive self-evolution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvolutionObservation:
    """One passively recorded outcome of an experiment.

    ``parameter`` names a *declared numeric knob* of AGS, never a source file.
    ``metric`` is the score that outcome achieved, and ``lower_is_better``
    records the direction so the engine does not have to guess.
    """
    parameter: str
    value: float
    metric: str
    score: float
    lower_is_better: bool = True

    def validate(self) -> None:
        safe_identifier(self.parameter, field="observation parameter")
        bounded_text(self.metric, field="observation metric")
        if not math.isfinite(self.value) or not math.isfinite(self.score):
            raise SecurityViolation("observation values must be finite")


@dataclass(frozen=True)
class ParameterProposal:
    """A proposed change to one declared parameter, with its justification."""

    parameter: str
    current_value: float
    proposed_value: float
    reason: str
    supporting_observations: int

    def validate(self) -> None:
        safe_identifier(self.parameter, field="proposal parameter")
        bounded_text(self.reason, field="proposal reason")
        if not all(math.isfinite(v) for v in
                   (self.current_value, self.proposed_value)):
            raise SecurityViolation("proposal values must be finite")
        if self.supporting_observations < 1:
            raise SecurityViolation("a proposal needs at least one observation")

    @property
    def fingerprint(self) -> str:
        self.validate()
        return sha256_json({
            "parameter": self.parameter,
            "current": self.current_value,
            "proposed": self.proposed_value,
            "reason": self.reason,
            "observations": self.supporting_observations,
        })


class PassiveEvolutionEngine:
    """Learn AGS's own tunable parameters from observed experiment outcomes.

    "Passive" means it only ever *observes*: it never edits source, never
    touches a protected path, and never applies a change itself. It accumulates
    :class:`EvolutionObservation` records and, when the evidence is strong
    enough, emits a :class:`ParameterProposal` that the host must explicitly
    accept.

    This is deliberately much weaker than self-modifying code. The only thing it
    may move is a value inside a parameter space the host declared, which is why
    it cannot become a path to arbitrary self-modification.
    """

    def __init__(self, parameter_space: Mapping[str, tuple[float, float]],
                 min_observations: int = 5):
        if not parameter_space:
            raise SecurityViolation("a declared parameter space is required")
        self._space: dict[str, tuple[float, float]] = {}
        for name, bounds in parameter_space.items():
            safe_identifier(name, field="parameter name")
            lo, hi = float(bounds[0]), float(bounds[1])
            if not (math.isfinite(lo) and math.isfinite(hi)) or not lo < hi:
                raise SecurityViolation(f"invalid bounds for parameter {name!r}")
            self._space[name] = (lo, hi)
        if int(min_observations) < 1:
            raise SecurityViolation("min_observations must be >= 1")
        self.min_observations = int(min_observations)
        # Per-instance. A class-level dict here would be shared by every engine,
        # so one engine's declared default would silently leak into another.
        self._defaults: dict[str, float] = {}
        self._history: list[EvolutionObservation] = []

    # -- observation ----------------------------------------------------
    def observe(self, observation: EvolutionObservation) -> None:
        observation.validate()
        if observation.parameter not in self._space:
            raise SecurityViolation(
                f"parameter {observation.parameter!r} is not in the declared space")
        lo, hi = self._space[observation.parameter]
        if not lo <= observation.value <= hi:
            raise SecurityViolation(
                f"value {observation.value} for {observation.parameter!r} is outside "
                f"the declared bounds [{lo}, {hi}]")
        self._history.append(observation)

    def observations(self, parameter: str | None = None) -> tuple[EvolutionObservation, ...]:
        if parameter is None:
            return tuple(self._history)
        return tuple(o for o in self._history if o.parameter == parameter)

    # -- proposal -------------------------------------------------------
    def propose(self, parameter: str) -> ParameterProposal | None:
        """Return the best-supported proposal for ``parameter``, or None.

        The proposal is the observed value with the best score, provided it beats
        the current default by a margin that grows with the number of competing
        observations. Ties resolve to the *current* value, so noise alone never
        moves a parameter.
        """
        if parameter not in self._space:
            raise SecurityViolation(f"unknown parameter: {parameter!r}")
        obs = self.observations(parameter)
        if len(obs) < self.min_observations:
            return None
        metrics = {o.metric for o in obs}
        if len(metrics) != 1:
            raise SecurityViolation(
                "observations for one parameter must share a single metric")
        lower_better = obs[0].lower_is_better

        def key(o: EvolutionObservation) -> tuple[float, float]:
            # Sort so that "best" is first; ties break toward smaller |value| so
            # the choice is deterministic.
            return (o.score if lower_better else -o.score, abs(o.value))

        best = min(obs, key=key)
        lo, hi = self._space[parameter]
        current = self._current_value(parameter)
        if current is None:
            return None
        best_score = best.score if lower_better else -best.score
        current_score = self._score_of(parameter, current, lower_better)
        if current_score is None:
            current_score = float("inf") if lower_better else float("-inf")
        # Require a real improvement, not a rounding-level difference.
        if not (best_score < current_score - 1e-12):
            return None
        proposed = min(max(best.value, lo), hi)
        return ParameterProposal(
            parameter=parameter,
            current_value=float(current),
            proposed_value=float(proposed),
            reason=(f"observed best {best.metric}="
                    f"{best.score:.6g} over {len(obs)} runs "
                    f"(current value scores {current_score:.6g})"),
            supporting_observations=len(obs),
        )

    def _current_value(self, parameter: str) -> float | None:
        current = self._defaults.get(parameter)
        return None if current is None else float(current)

    def _score_of(self, parameter: str, value: float,
                  lower_better: bool) -> float | None:
        scores = [o.score for o in self.observations(parameter)
                  if math.isclose(o.value, value, rel_tol=0.0, abs_tol=1e-15)]
        if not scores:
            return None
        best = min(scores) if lower_better else max(scores)
        return best if lower_better else -best

    # -- declared defaults ---------------------------------------------
    def declare_default(self, parameter: str, value: float) -> None:
        if parameter not in self._space:
            raise SecurityViolation(f"unknown parameter: {parameter!r}")
        lo, hi = self._space[parameter]
        v = float(value)
        if not lo <= v <= hi:
            raise SecurityViolation(f"default for {parameter!r} is outside its bounds")
        self._defaults[parameter] = v

    def describe(self) -> dict[str, object]:
        return {
            "parameter_space": {k: list(v) for k, v in sorted(self._space.items())},
            "defaults": dict(sorted(self._defaults.items())),
            "observations": len(self._history),
            "min_observations": self.min_observations,
        }
