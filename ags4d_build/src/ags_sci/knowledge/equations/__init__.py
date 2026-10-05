"""Immutable, level-aware mathematics and physics equation references."""

from .loader import (
    CorpusIntegrityError,
    EquationCorpus,
    iter_equation_file,
    load_equations,
    retrieve_equations,
)
from .schema import (
    EquationRecord,
    EquationSchemaError,
    LEVEL_TOPICS,
    RELATION_TYPES,
    RESEARCH_STATUSES,
    TOPIC_CODES,
    UNIT_SYSTEMS,
    VALID_TOPICS,
)
from .validation import ValidationIssue, ValidationReport, validate_corpus

__all__ = [
    "CorpusIntegrityError",
    "EquationCorpus",
    "EquationRecord",
    "EquationSchemaError",
    "LEVEL_TOPICS",
    "RELATION_TYPES",
    "RESEARCH_STATUSES",
    "TOPIC_CODES",
    "UNIT_SYSTEMS",
    "VALID_TOPICS",
    "ValidationIssue",
    "ValidationReport",
    "iter_equation_file",
    "load_equations",
    "retrieve_equations",
    "validate_corpus",
]
