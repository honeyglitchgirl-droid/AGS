"""Read-only scientific reference knowledge for AGS-Sci.

Knowledge data is deliberately separate from experiments, results, symbolic
reasoning, and autonomous discovery.  The equation package exposes retrieval
only; it does not parse, evaluate, or apply formula strings.
"""

from .equations import (
    EquationCorpus,
    EquationRecord,
    EquationSchemaError,
    load_equations,
    retrieve_equations,
)

__all__ = [
    "EquationCorpus",
    "EquationRecord",
    "EquationSchemaError",
    "load_equations",
    "retrieve_equations",
]
