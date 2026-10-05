"""Generative hypothesis invention for AGS-Sci.

Unlike :mod:`ags_sci.discovery.law`, which *identifies* dynamics from a fixed
library of terms, this package *invents* candidates by searching a symbolic
grammar, assesses their novelty against a declared corpus of known laws, and
formulates new open problems from structure a fit fails to explain.
"""
from .grammar import (Expression, ExpressionError, DEFAULT_VARIABLES, complexity,
                      equivalent, fitness, random_expression, refine_constants,
                      safe_evaluate, simplified, numeric_agreement)
from .search import AbductiveSearch, Candidate, SearchReport
from .novelty import KnownLaw, NoveltyAssessor, NoveltyReport
from .problems import OpenProblem, ProblemGenerator
from .engine import InventionEngine, InventedCandidate, SequenceConjecture

__all__ = [
    "Expression", "ExpressionError", "DEFAULT_VARIABLES", "complexity",
    "equivalent", "fitness", "random_expression", "refine_constants",
    "safe_evaluate", "simplified", "numeric_agreement", "AbductiveSearch",
    "Candidate", "SearchReport", "KnownLaw", "NoveltyAssessor", "NoveltyReport",
    "OpenProblem", "ProblemGenerator", "InventionEngine", "InventedCandidate",
    "SequenceConjecture",
]
