"""Novelty assessment for invented candidates.

A candidate is only "new" relative to something. This module therefore makes the
reference explicit: a caller supplies (or accepts the default) corpus of known
laws, and a candidate is novel only if it is *not symbolically equivalent* to any
entry in that corpus.

Two consequences are worth stating plainly:

- **Novelty here is corpus-relative, not absolute.** A candidate absent from the
  default corpus may still be a textbook result the corpus simply does not list.
  The default corpus is a convenience, not a survey of human knowledge.
- **Non-equivalence is a proof, equivalence is a proof, and "cannot decide" is
  neither.** SymPy reducing the difference to zero proves equivalence; a nonzero
  reduced difference is strong evidence of difference but is reported as such,
  never as certainty, unless the expressions are structurally identical.
"""
from __future__ import annotations
from dataclasses import dataclass

import sympy as sp

from .grammar import Expression, equivalent, numeric_agreement, simplified


@dataclass(frozen=True)
class KnownLaw:
    name: str
    expression: Expression
    note: str = ""


def _default_corpus() -> tuple[KnownLaw, ...]:
    """A small, explicit set of classic relations used as the novelty baseline."""
    x, y, z, t = sp.symbols("x y z t", real=True)
    F, k, m, a, V, I, R, P, n, T = sp.symbols("F k m a V I R P n T", real=True, positive=True)
    G, q1, q2, r, c, E = sp.symbols("G q1 q2 r c E", real=True, positive=True)
    w = sp.Symbol("w", real=True)

    def law(name, expr, note=""):
        return KnownLaw(name, Expression(sp.sympify(expr), ()), note)

    return (
        law("hooke", -k * x, "restoring force proportional to displacement"),
        law("newton_second", m * a, "force equals mass times acceleration"),
        law("ideal_gas", n * R * T / V, "pressure of an ideal gas"),
        law("ohm", I * R, "voltage equals current times resistance"),
        law("coulomb", q1 * q2 / r ** 2, "electrostatic force"),
        law("gravitation", G * m * m / r ** 2, "gravitational force"),
        law("exponential_decay", sp.exp(-k * t), "first-order decay"),
        law("mass_energy", m * c ** 2, "rest energy"),
        law("kinetic_energy", sp.Rational(1, 2) * m * x ** 2, "kinetic energy"),
        law("harmonic_acceleration", -w ** 2 * x, "simple harmonic motion"),
        law("logistic_growth", x * (1 - x), "logistic per-capita growth"),
        law("wave", sp.sin(w * t), "monochromatic wave"),
        law("pythagorean", sp.sqrt(x ** 2 + y ** 2), "Euclidean norm"),
        law("area_circle", sp.pi * x ** 2, "area of a circle"),
    )


@dataclass(frozen=True)
class NoveltyReport:
    candidate: str
    is_novel: bool
    matched_law: str | None
    matches: tuple[tuple[str, str], ...]
    method: str

    def describe(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "is_novel": self.is_novel,
            "matched_law": self.matched_law,
            "matches": [{"law": n, "kind": k} for n, k in self.matches],
            "method": self.method,
        }


class NoveltyAssessor:
    """Decide whether a candidate is equivalent to a known law."""

    def __init__(self, corpus: tuple[KnownLaw, ...] | None = None):
        self.corpus = corpus if corpus is not None else _default_corpus()

    def assess(self, candidate: Expression,
               sample_values: dict[str, object] | None = None) -> NoveltyReport:
        cand = simplified(candidate)
        matches: list[tuple[str, str]] = []
        matched: str | None = None

        for law in self.corpus:
            known = simplified(law.expression)
            kinds: list[str] = []
            # Structural comparison after simplification catches exact identities.
            if str(cand.sympy_expr) == str(known.sympy_expr):
                kinds.append("identical")
            elif equivalent(cand, known):
                kinds.append("symbolically_equivalent")
            if sample_values:
                # Sampling is recorded even when a symbolic verdict was reached,
                # as corroboration -- and, importantly, as a contradiction when
                # SymPy claims an equivalence the samples do not support.
                agree = numeric_agreement(cand, known, sample_values, tol=1e-7)
                if agree is True:
                    kinds.append("numerically_equivalent_on_samples")
                elif agree is False and kinds:
                    # Only a contradiction when a symbolic verdict was reached:
                    # a law the candidate never matched is not a disagreement.
                    kinds.append("numerically_inconsistent_on_samples")
            for kind in kinds:
                matches.append((law.name, kind))
            if kinds:
                matched = matched or law.name

        return NoveltyReport(
            candidate=str(cand.sympy_expr),
            is_novel=(matched is None),
            matched_law=matched,
            matches=tuple(matches),
            method="symbolic equivalence against a declared corpus",
        )

    def describe_corpus(self) -> tuple[dict[str, str], ...]:
        return tuple({"name": l.name, "expression": str(l.expression.sympy_expr),
                      "note": l.note} for l in self.corpus)
