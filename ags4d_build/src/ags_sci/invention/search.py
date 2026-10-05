"""Abductive search: invent candidate laws by searching the expression grammar.

Identification (as in :mod:`ags_sci.discovery.law`) fits a *fixed* library of
terms. This module instead *generates* candidates by searching a grammar, so the
hypothesis space is not limited to whatever a human wrote down in advance.

The search is a deterministic genetic algorithm:

- **Initialisation** — random expressions bounded by ``max_depth``.
- **Selection** — tournament selection on a description-length fitness
  (normalised RMSE plus a complexity penalty), so a 40-node expression that fits
  marginally better loses to a 3-node expression that fits as well.
- **Variation** — subtree mutation and subtree crossover.
- **Elitism** — the best candidate is always carried forward.

The result is a Pareto front of (complexity, error) candidates, because the
interesting object is the *trade-off*, not a single winner.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

from .grammar import (Expression, ExpressionError, complexity, equivalent,
                      random_expression, refine_constants, safe_evaluate,
                      simplified)


@dataclass(frozen=True)
class Candidate:
    """One invented candidate law with its evidence."""

    expression: Expression
    rmse: float
    complexity: int
    fitness: float
    generation: int

    def describe(self) -> dict[str, object]:
        return {
            "expression": str(self.expression.sympy_expr),
            "variables": list(self.expression.variables),
            "normalised_rmse": self.rmse,
            "complexity": self.complexity,
            "fitness": self.fitness,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class SearchReport:
    candidates: tuple[Candidate, ...]
    generations: int
    population_size: int
    evaluations: int
    infeasible: int
    converged: bool

    def describe(self) -> dict[str, object]:
        return {
            "candidates": [c.describe() for c in self.candidates],
            "generations": self.generations,
            "population_size": self.population_size,
            "evaluations": self.evaluations,
            "infeasible_fraction": (self.infeasible / self.evaluations
                                    if self.evaluations else 0.0),
            "converged": self.converged,
        }


class AbductiveSearch:
    """Deterministic genetic search over the expression grammar."""

    def __init__(self, variables, *, max_depth: int = 3, population_size: int = 120,
                 generations: int = 40, complexity_weight: float = 0.01,
                 tournament_size: int = 4, mutation_rate: float = 0.35,
                 crossover_rate: float = 0.45, elite_fraction: float = 0.1,
                 seed: int = 0):
        if int(population_size) < 4:
            raise ExpressionError("population_size must be >= 4")
        if int(generations) < 1:
            raise ExpressionError("generations must be >= 1")
        if not 0.0 <= mutation_rate + crossover_rate <= 1.0:
            raise ExpressionError("mutation_rate + crossover_rate must be <= 1")
        self.variables = tuple(variables)
        self.max_depth = int(max_depth)
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.complexity_weight = float(complexity_weight)
        self.tournament_size = max(2, int(tournament_size))
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)
        if not 0.0 < float(elite_fraction) < 1.0:
            raise ExpressionError("elite_fraction must be in (0,1)")
        self.elite_fraction = float(elite_fraction)
        self.elite_count = max(1, min(self.population_size - 1,
                                      int(round(self.elite_fraction * self.population_size))))
        self.rng = np.random.default_rng(int(seed))
        self._evaluations = 0
        self._infeasible = 0

    # -- genetic operators -------------------------------------------------
    def _mutate(self, expr: Expression) -> Expression:
        """Replace a random subtree with a freshly generated one."""
        sym = expr.sympy_expr
        # sp.preorder_traversal works on Atoms (e.g. a bare Symbol), whose class
        # does not carry the .preorder_traversal() method in current SymPy.
        nodes = list(sp.preorder_traversal(sym))
        if not nodes:
            return random_expression(self.variables, self.rng, self.max_depth)
        target = nodes[int(self.rng.integers(0, len(nodes)))]
        replacement = random_expression(self.variables, self.rng,
                                        max(1, self.max_depth - 1)).sympy_expr
        try:
            new = sym.xreplace({target: replacement})
            return Expression(sp.sympify(new), expr.variables)
        except Exception:
            return expr

    def _crossover(self, a: Expression, b: Expression) -> Expression:
        """Swap a random subtree of ``a`` for one from ``b``."""
        sa, sb = a.sympy_expr, b.sympy_expr
        nodes_a = list(sp.preorder_traversal(sa))
        nodes_b = list(sp.preorder_traversal(sb))
        if not nodes_a or not nodes_b:
            return a
        ta = nodes_a[int(self.rng.integers(0, len(nodes_a)))]
        tb = nodes_b[int(self.rng.integers(0, len(nodes_b)))]
        try:
            new = sa.xreplace({ta: tb})
            return Expression(sp.sympify(new), a.variables)
        except Exception:
            return a

    def _select(self, scored: list[tuple[float, Expression]]) -> Expression:
        """Tournament selection; lower fitness wins."""
        contenders = [scored[int(self.rng.integers(0, len(scored)))]
                      for _ in range(self.tournament_size)]
        return min(contenders, key=lambda p: p[0])[1]

    # -- scoring -----------------------------------------------------------
    def _score(self, expr: Expression, target: np.ndarray,
               values: dict[str, np.ndarray]) -> tuple[float, float]:
        """Return (fitness, normalised_rmse), counting feasibility."""
        self._evaluations += 1
        got = safe_evaluate(expr, values)
        if got is None:
            self._infeasible += 1
            return math.inf, math.inf
        t = np.asarray(target, dtype=float)
        if got.shape != t.shape:
            self._infeasible += 1
            return math.inf, math.inf
        denom = float(np.max(np.abs(t))) or 1.0
        # Huge but finite intermediate values are expected from random
        # expressions; they simply score badly rather than needing a warning.
        with np.errstate(all="ignore"):
            rmse = float(np.sqrt(np.mean((got - t) ** 2))) / denom
        if not math.isfinite(rmse):
            return math.inf, math.inf
        return rmse + self.complexity_weight * complexity(expr), rmse

    # -- main entry --------------------------------------------------------
    def search(self, target, values: dict[str, np.ndarray]) -> SearchReport:
        """Search for expressions reproducing ``target`` from ``values``.

        ``values`` maps variable name to a sample array; all arrays must share a
        shape. ``target`` is the array the candidate must reproduce.
        """
        target = np.asarray(target, dtype=float)
        shapes = {np.shape(v) for v in values.values()}
        if len(shapes) != 1 or shapes != {target.shape}:
            raise ExpressionError("target and every variable array must share one shape")
        if not np.all(np.isfinite(target)):
            raise ExpressionError("target must be finite")

        population = [random_expression(self.variables, self.rng, self.max_depth)
                      for _ in range(self.population_size)]
        best_overall: tuple[float, Expression, float, int] | None = None
        converged = False
        generation = 0

        for generation in range(1, self.generations + 1):
            scored: list[tuple[float, Expression]] = []
            for expr in population:
                f, rmse = self._score(expr, target, values)
                scored.append((f, expr))
                if f < math.inf and (best_overall is None or f < best_overall[0]):
                    try:
                        refined, _ = refine_constants(expr, target, values)
                        rf, rrmse = self._score(refined, target, values)
                    except Exception:
                        refined, rf, rrmse = expr, f, rmse
                    if rf <= f:
                        best_overall = (rf, refined, rrmse, generation)
                    else:
                        best_overall = (f, expr, rmse, generation)
            scored.sort(key=lambda p: p[0])

            if scored[0][0] < math.inf and generation > 1:
                # Converged when the best fitness stops improving materially.
                prev = getattr(self, "_prev_best", math.inf)
                if abs(prev - scored[0][0]) <= 1e-12:
                    converged = True
                self._prev_best = scored[0][0]

            elites = [e for _, e in scored[:self.elite_count]]
            n_random = max(1, int(0.1 * self.population_size))
            children = list(elites)
            while len(children) < self.population_size - n_random:
                roll = self.rng.random()
                if roll < self.crossover_rate and len(scored) > 1:
                    a = self._select(scored)
                    b = self._select(scored)
                    children.append(self._crossover(a, b))
                elif roll < self.crossover_rate + self.mutation_rate:
                    children.append(self._mutate(self._select(scored)))
                else:
                    children.append(self._select(scored))
            children.extend(random_expression(self.variables, self.rng, self.max_depth)
                            for _ in range(n_random))
            population = children[:self.population_size]

        # Re-score the final population, refining constants before scoring so a
        # structurally-correct candidate with noisy constants is not discarded in
        # favour of a simpler but wrong one.
        final: list[tuple[float, Expression, float]] = []
        for expr in population:
            try:
                refined, _ = refine_constants(expr, target, values)
            except Exception:
                refined = expr
            f, rmse = self._score(refined, target, values)
            if f < math.inf:
                final.append((f, refined, rmse))
        final.sort(key=lambda p: p[0])

        candidates = self._pareto_front(final, generation)
        if best_overall is not None and not any(
                equivalent(c.expression, best_overall[1]) for c in candidates):
            f, expr, rmse, gen = best_overall
            candidates.insert(0, Candidate(expr, rmse, complexity(expr), f, gen))
            candidates.sort(key=lambda c: c.fitness)

        return SearchReport(
            candidates=tuple(candidates[:8]),
            generations=self.generations,
            population_size=self.population_size,
            evaluations=self._evaluations,
            infeasible=self._infeasible,
            converged=converged,
        )

    @staticmethod
    def _pareto_front(final: list[tuple[float, Expression, float]],
                      generation: int) -> list[Candidate]:
        """Non-dominated candidates on (complexity, rmse): lower is better."""
        front: list[Candidate] = []
        seen: set[str] = set()
        for f, expr, rmse in final:
            key = str(simplified(expr).sympy_expr)
            if key in seen:
                continue
            dominated = False
            for f2, expr2, rmse2 in final:
                if expr2 is expr:
                    continue
                if (complexity(expr2) <= complexity(expr)
                        and rmse2 <= rmse
                        and (complexity(expr2) < complexity(expr) or rmse2 < rmse)):
                    dominated = True
                    break
            if dominated:
                continue
            seen.add(key)
            front.append(Candidate(expr, rmse, complexity(expr), f, generation))
            if len(front) >= 8:
                break
        return front
