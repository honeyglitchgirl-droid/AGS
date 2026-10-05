"""The invention engine: generate candidate laws, assess novelty, pose new problems.

This is the entry point for AGS's *generative* capability. It is deliberately
layered so that each claim it makes is backed by a specific, checkable piece of
evidence:

1. :meth:`InventionEngine.invent` searches the expression grammar for candidates
   that reproduce supplied data, then assesses each against a declared corpus of
   known laws. The output is a set of candidates plus an explicit
   corpus-relative novelty verdict.
2. :meth:`InventionEngine.propose_problems` looks at what a fit *fails* to
   explain and formulates open problems from that residual structure.
3. :meth:`InventionEngine.conjecture_sequence` invents a closed form for a
   numeric sequence and then tries to **falsify** it on held-out terms. This is
   the one place where "new" is objectively decidable, because the sequence is a
   finite object and the conjecture makes a checkable prediction.

None of these produce a law of nature. They produce candidates, problems and
conjectures, each with the evidence attached and an explicit epistemic status.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .grammar import Expression, safe_evaluate, simplified
from .novelty import KnownLaw, NoveltyAssessor, NoveltyReport
from .problems import OpenProblem, ProblemGenerator
from .search import AbductiveSearch, SearchReport


@dataclass(frozen=True)
class InventedCandidate:
    """A candidate law with its fit, its novelty verdict, and its epistemic status."""

    expression: str
    variables: tuple[str, ...]
    normalised_rmse: float
    complexity: int
    fitness: float
    novelty: NoveltyReport
    epistemic_status: str

    def describe(self) -> dict[str, object]:
        return {
            "expression": self.expression,
            "variables": list(self.variables),
            "normalised_rmse": self.normalised_rmse,
            "complexity": self.complexity,
            "fitness": self.fitness,
            "novelty": self.novelty.describe(),
            "epistemic_status": self.epistemic_status,
        }


@dataclass(frozen=True)
class SequenceConjecture:
    """A conjectured closed form for a sequence, with its falsification result."""

    expression: str
    verified_terms: int
    total_terms: int
    status: str
    first_failure_index: int | None
    max_abs_error: float

    def describe(self) -> dict[str, object]:
        return {
            "expression": self.expression,
            "verified_terms": self.verified_terms,
            "total_terms": self.total_terms,
            "status": self.status,
            "first_failure_index": self.first_failure_index,
            "max_abs_error": self.max_abs_error,
        }


class InventionEngine:
    """Generate, assess and falsify candidate laws."""

    def __init__(self, corpus: tuple[KnownLaw, ...] | None = None,
                 *, max_depth: int = 3, population_size: int = 120,
                 generations: int = 40, seed: int = 0,
                 problem_generator: ProblemGenerator | None = None):
        self.assessor = NoveltyAssessor(corpus)
        self.max_depth = int(max_depth)
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.seed = int(seed)
        self.problems = problem_generator or ProblemGenerator()

    # -- 1. law invention -------------------------------------------------
    def invent(self, target, values: dict[str, np.ndarray], *,
               complexity_weight: float = 0.01, top_k: int = 5) -> dict[str, object]:
        """Search for candidate laws reproducing ``target`` from ``values``.

        Returns the search report, each candidate's corpus-relative novelty
        verdict, and the generated open problems implied by the best candidate's
        residuals.
        """
        target_arr = np.asarray(target, dtype=float).reshape(-1)
        if not np.all(np.isfinite(target_arr)):
            # A target containing NaN/inf cannot be fitted at all; reporting
            # this as "no feasible candidate" is honest and cheaper than
            # searching for one.
            return {"status": "NO_FEASIBLE_CANDIDATE",
                    "reason": "target contains non-finite values",
                    "candidates": [], "open_problems": []}
        search = AbductiveSearch(
            tuple(values), max_depth=self.max_depth,
            population_size=self.population_size, generations=self.generations,
            complexity_weight=complexity_weight, seed=self.seed,
        )
        report: SearchReport = search.search(target_arr, values)
        if not report.candidates:
            return {
                "status": "NO_FEASIBLE_CANDIDATE",
                "candidates": [],
                "open_problems": [],
                "search": report.describe(),
            }

        best = report.candidates[0]
        residuals = self._residuals(best.expression, target_arr, values)

        enriched: list[InventedCandidate] = []
        for rank, cand in enumerate(report.candidates[:top_k]):
            novelty = self.assessor.assess(cand.expression, values)
            if novelty.is_novel:
                status = "NOVEL_CANDIDATE_RELATIVE_TO_CORPUS"
            elif rank == 0 and cand.rmse <= 1e-6:
                # Best overall fit, but it reproduces a law already in the corpus.
                status = "KNOWN_FORM_GOOD_FIT"
            else:
                status = "KNOWN_FORM"
            enriched.append(InventedCandidate(
                expression=str(simplified(cand.expression).sympy_expr),
                variables=cand.expression.variables,
                normalised_rmse=cand.rmse,
                complexity=cand.complexity,
                fitness=cand.fitness,
                novelty=novelty,
                epistemic_status=status,
            ))

        open_problems = self.problems.generate(
            residuals,
            x=values.get("x"),
            t=values.get("t"),
            driver=values.get("x"),
            base_statement=str(simplified(best.expression).sympy_expr),
        ) if residuals is not None else ()

        return {
            "status": "CANDIDATES_GENERATED",
            "candidates": [c.describe() for c in enriched],
            "open_problems": [p.describe() for p in open_problems],
            "search": report.describe(),
            "novelty_method": self.assessor.assess(best.expression).method,
        }

    # -- 2. problem invention --------------------------------------------
    def propose_problems(self, residuals, x=None, t=None, driver=None,
                         base_statement: str = "the fitted relation"
                         ) -> tuple[OpenProblem, ...]:
        return self.problems.generate(residuals, x=x, t=t, driver=driver,
                                      base_statement=base_statement)

    # -- 3. sequence conjectures -----------------------------------------
    def conjecture_sequence(self, sequence, *, holdout: int = 5,
                            tolerance: float = 1e-8,
                            start_index: int = 0) -> SequenceConjecture:
        """Invent a closed form for ``sequence`` and attempt to falsify it.

        The first ``len(sequence) - holdout`` terms are used to search; the last
        ``holdout`` terms are a prediction the conjecture must survive. This is
        the only capability here with an objective pass/fail, because a finite
        sequence is a finite object.

        ``start_index`` is the value of the independent variable ``n`` assigned to
        ``sequence[0]``. It defaults to 0, so ``[1, 4, 9, 16]`` is modelled as
        ``(n + 1)**2``; pass ``start_index=1`` to model it as ``n**2``.
        """
        seq = np.asarray(sequence, dtype=float).reshape(-1)
        if seq.size < 8:
            raise ValueError("a sequence conjecture needs at least 8 terms")
        if holdout < 1 or holdout >= seq.size - 3:
            raise ValueError("holdout must leave at least 3 fitting terms")
        if not np.all(np.isfinite(seq)):
            raise ValueError("sequence must be finite")

        fit_n = seq.size - int(holdout)
        n = np.arange(int(start_index), int(start_index) + seq.size, dtype=float)
        search = AbductiveSearch(
            ("n",), max_depth=self.max_depth,
            population_size=self.population_size, generations=self.generations,
            seed=self.seed,
        )
        report = search.search(seq[:fit_n], {"n": n[:fit_n]})
        if not report.candidates:
            return SequenceConjecture("", 0, int(seq.size), "NO_FEASIBLE_CANDIDATE",
                                      None, float("inf"))

        best = report.candidates[0]
        expr = simplified(best.expression)
        got = safe_evaluate(expr, {"n": n})
        if got is None:
            return SequenceConjecture(str(expr.sympy_expr), 0, int(seq.size),
                                      "INFEASIBLE_ON_FULL_SEQUENCE", None, float("inf"))

        err = np.abs(got - seq)
        tol = float(tolerance) * max(1.0, float(np.max(np.abs(seq))))
        failures = np.nonzero(err > tol)[0]
        if failures.size == 0:
            return SequenceConjecture(
                str(expr.sympy_expr), int(seq.size), int(seq.size),
                "CONJECTURE_SURVIVED_ALL_TERMS", None, float(err.max()))
        first = int(failures[0])
        verified = first
        status = ("CONJECTURE_FALSIFIED_INSIDE_FIT_WINDOW" if first < fit_n
                  else "CONJECTURE_FALSIFIED_ON_HOLDOUT")
        return SequenceConjecture(
            str(expr.sympy_expr), verified, int(seq.size), status, first,
            float(err.max()))

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _residuals(expr: Expression, target, values):
        got = safe_evaluate(expr, values)
        if got is None:
            return None
        return np.asarray(target, dtype=float) - got
