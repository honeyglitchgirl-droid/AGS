"""Strict, non-executable schema for equation reference records."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

LEVEL_TOPICS: dict[str, dict[str, frozenset[str]]] = {
    "k12": {
        "mathematics": frozenset({
            "arithmetic", "algebra", "geometry", "trigonometry",
            "coordinate_geometry", "sequences", "functions", "calculus",
            "vectors", "matrices", "probability", "statistics",
            "discrete_mathematics", "number_theory",
        }),
        "physics": frozenset({
            "mechanics", "gravitation", "properties_of_matter",
            "thermodynamics", "waves", "sound", "optics", "electricity",
            "magnetism", "electromagnetism", "modern_physics",
            "atomic_physics", "nuclear_physics",
        }),
    },
    "undergraduate": {
        "mathematics": frozenset({
            "algebra", "linear_algebra", "multivariable_calculus",
            "differential_equations", "real_analysis", "complex_analysis",
            "abstract_algebra", "number_theory", "probability_statistics",
            "numerical_mathematics", "optimization", "discrete_mathematics",
        }),
        "physics": frozenset({
            "classical_mechanics", "electromagnetism", "thermodynamics",
            "statistical_mechanics", "quantum_mechanics",
            "special_relativity", "optics", "fluid_mechanics",
            "mathematical_physics",
        }),
    },
    "graduate": {
        "mathematics": frozenset({
            "measure_theory", "functional_analysis", "advanced_pde",
            "differential_geometry", "riemannian_geometry",
            "differential_topology", "algebraic_topology",
            "algebraic_geometry", "lie_theory", "advanced_probability",
        }),
        "physics": frozenset({
            "general_relativity", "quantum_field_theory", "gauge_theory",
            "advanced_statistical_mechanics", "condensed_matter",
            "plasma_physics", "advanced_optics",
        }),
    },
    "research": {
        "mathematics": frozenset({
            "differential_riemannian_geometry", "differential_topology",
            "algebraic_topology", "algebraic_geometry", "number_theory",
            "functional_analysis", "operator_theory",
            "partial_differential_equations", "calculus_of_variations",
            "dynamical_systems", "probability_stochastic_analysis",
            "lie_theory", "representation_theory",
            "category_homological_algebra", "symplectic_complex_geometry",
            "optimization_control", "numerical_mathematics",
            "mathematical_physics", "special_functions",
        }),
        "physics": frozenset({
            "general_relativity", "quantum_mechanics",
            "quantum_field_theory", "gauge_theory", "standard_model_qcd",
            "statistical_mechanics", "condensed_matter",
            "many_body_physics", "plasma_physics", "fluid_physics",
            "cosmology", "astrophysics", "nuclear_physics",
            "quantum_information_optics", "nonlinear_physics",
            "string_quantum_gravity",
        }),
    },
}

VALID_TOPICS: dict[str, frozenset[str]] = {
    domain: frozenset().union(*(
        level_topics[domain] for level_topics in LEVEL_TOPICS.values()
    ))
    for domain in ("mathematics", "physics")
}

TOPIC_CODES: dict[tuple[str, str], str] = {
    # K-12 and shared topic names retain their original stable codes.
    ("mathematics", "arithmetic"): "ARI",
    ("mathematics", "algebra"): "ALG",
    ("mathematics", "geometry"): "GEO",
    ("mathematics", "trigonometry"): "TRI",
    ("mathematics", "coordinate_geometry"): "COG",
    ("mathematics", "sequences"): "SEQ",
    ("mathematics", "functions"): "FUN",
    ("mathematics", "calculus"): "CALC",
    ("mathematics", "vectors"): "VEC",
    ("mathematics", "matrices"): "MAT",
    ("mathematics", "probability"): "PROB",
    ("mathematics", "statistics"): "STAT",
    ("mathematics", "discrete_mathematics"): "DISC",
    ("mathematics", "number_theory"): "NUM",
    ("physics", "mechanics"): "MEC",
    ("physics", "gravitation"): "GRV",
    ("physics", "properties_of_matter"): "POM",
    ("physics", "thermodynamics"): "THM",
    ("physics", "waves"): "WAV",
    ("physics", "sound"): "SND",
    ("physics", "optics"): "OPT",
    ("physics", "electricity"): "ELE",
    ("physics", "magnetism"): "MAG",
    ("physics", "electromagnetism"): "EM",
    ("physics", "modern_physics"): "MOD",
    ("physics", "atomic_physics"): "ATM",
    ("physics", "nuclear_physics"): "NUC",
    # Undergraduate mathematics.
    ("mathematics", "linear_algebra"): "LA",
    ("mathematics", "multivariable_calculus"): "MVC",
    ("mathematics", "differential_equations"): "DE",
    ("mathematics", "real_analysis"): "RA",
    ("mathematics", "complex_analysis"): "CA",
    ("mathematics", "abstract_algebra"): "AA",
    ("mathematics", "probability_statistics"): "PS",
    ("mathematics", "numerical_mathematics"): "NM",
    ("mathematics", "optimization"): "OPTM",
    # Undergraduate physics.
    ("physics", "classical_mechanics"): "CM",
    ("physics", "statistical_mechanics"): "SM",
    ("physics", "quantum_mechanics"): "QM",
    ("physics", "special_relativity"): "SR",
    ("physics", "fluid_mechanics"): "FL",
    ("physics", "mathematical_physics"): "MP",
    # Graduate mathematics.
    ("mathematics", "measure_theory"): "MT",
    ("mathematics", "functional_analysis"): "FA",
    ("mathematics", "advanced_pde"): "PDE",
    ("mathematics", "differential_geometry"): "DG",
    ("mathematics", "riemannian_geometry"): "RG",
    ("mathematics", "differential_topology"): "DT",
    ("mathematics", "algebraic_topology"): "AT",
    ("mathematics", "algebraic_geometry"): "AG",
    ("mathematics", "lie_theory"): "LIE",
    ("mathematics", "advanced_probability"): "AP",
    # Graduate physics.
    ("physics", "general_relativity"): "GR",
    ("physics", "quantum_field_theory"): "QFT",
    ("physics", "gauge_theory"): "GT",
    ("physics", "advanced_statistical_mechanics"): "ASM",
    ("physics", "condensed_matter"): "CMP",
    ("physics", "plasma_physics"): "PL",
    ("physics", "advanced_optics"): "AOPT",
    # Research mathematics.
    ("mathematics", "differential_riemannian_geometry"): "DRG",
    ("mathematics", "operator_theory"): "OT",
    ("mathematics", "partial_differential_equations"): "PDE",
    ("mathematics", "calculus_of_variations"): "CVAR",
    ("mathematics", "dynamical_systems"): "DS",
    ("mathematics", "probability_stochastic_analysis"): "PSA",
    ("mathematics", "representation_theory"): "REP",
    ("mathematics", "category_homological_algebra"): "CHA",
    ("mathematics", "symplectic_complex_geometry"): "SCG",
    ("mathematics", "optimization_control"): "OC",
    ("mathematics", "mathematical_physics"): "MPH",
    ("mathematics", "special_functions"): "SF",
    # Research physics.
    ("physics", "standard_model_qcd"): "SMQCD",
    ("physics", "many_body_physics"): "MB",
    ("physics", "fluid_physics"): "FLUID",
    ("physics", "cosmology"): "COS",
    ("physics", "astrophysics"): "ASTRO",
    ("physics", "quantum_information_optics"): "QIO",
    ("physics", "nonlinear_physics"): "NLP",
    ("physics", "string_quantum_gravity"): "SQG",
}

RELATION_TYPES = frozenset({
    "identity", "definition", "equation", "operator_relation",
    "transformation", "differential_system", "conservation_law",
    "variational_principle", "boundary_condition", "constitutive_relation",
    "inequality", "recurrence", "canonical_relation", "symmetry_relation",
    "asymptotic_relation", "integral_relation", "spectral_relation",
    "constraint", "commutation_relation", "evolution_equation",
})
RESEARCH_STATUSES = frozenset({"established", "model_dependent", "conjectural"})
UNIT_SYSTEMS = frozenset({
    "not_applicable", "si", "natural", "geometrized", "atomic",
    "gaussian_cgs", "heaviside_lorentz", "dimensionless", "mixed",
    "source_defined",
})

_REQUIRED_FIELDS = frozenset({"id", "domain", "topic", "formula"})
_OPTIONAL_FIELDS = frozenset({
    "subtopic", "aliases", "symbols", "source_level", "curriculum_level",
    "source_ids", "relation_type", "status", "source_locator", "unit_system",
    "assumptions", "conventions",
})
_ALLOWED_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS
_ID_RE = re.compile(r"^(MATH|PHY)-((?:(?:UG|GR|RS)-)?[A-Z0-9]+)-([0-9]{6})$")
_LEVEL_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SOURCE_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9-]{1,95}$")


class EquationSchemaError(ValueError):
    """Raised when an equation record fails the data-only schema."""


def _plain_string(value: Any, field: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise EquationSchemaError(f"{field} must be a string")
    if not value or not value.strip():
        raise EquationSchemaError(f"{field} must not be empty")
    if value != value.strip():
        raise EquationSchemaError(f"{field} must not have surrounding whitespace")
    if len(value) > max_length:
        raise EquationSchemaError(f"{field} exceeds {max_length} characters")
    if unicodedata.normalize("NFC", value) != value:
        raise EquationSchemaError(f"{field} must use NFC Unicode normalization")
    if any(
        ord(char) < 32
        or ord(char) == 127
        or 0xD800 <= ord(char) <= 0xDFFF
        or (ord(char) & 0xFFFF) in {0xFFFE, 0xFFFF}
        for char in value
    ):
        raise EquationSchemaError(f"{field} contains invalid Unicode or control characters")
    return value


def _string_tuple(
    value: Any,
    field: str,
    *,
    max_items: int = 32,
    max_length: int = 128,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise EquationSchemaError(f"{field} must be a JSON array")
    if len(value) > max_items:
        raise EquationSchemaError(f"{field} has more than {max_items} items")
    result = tuple(_plain_string(item, field, max_length=max_length) for item in value)
    if len(set(result)) != len(result):
        raise EquationSchemaError(f"{field} contains duplicates")
    return result


@dataclass(frozen=True, slots=True)
class EquationRecord:
    """An immutable equation record whose formula is always opaque text."""

    id: str
    domain: str
    topic: str
    formula: str
    subtopic: str | None = None
    aliases: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    source_level: str | None = None
    curriculum_level: str | None = None
    source_ids: tuple[str, ...] = ()
    relation_type: str | None = None
    status: str | None = None
    source_locator: str | None = None
    unit_system: str | None = None
    assumptions: tuple[str, ...] = ()
    conventions: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EquationRecord":
        if not isinstance(value, Mapping):
            raise EquationSchemaError("record must be a JSON object")
        keys = set(value)
        missing = _REQUIRED_FIELDS - keys
        if missing:
            raise EquationSchemaError(
                "missing required fields: " + ", ".join(sorted(missing))
            )
        unknown = keys - _ALLOWED_FIELDS
        if unknown:
            raise EquationSchemaError(
                "unknown fields: " + ", ".join(sorted(unknown))
            )

        record_id = _plain_string(value["id"], "id", max_length=96)
        domain = _plain_string(value["domain"], "domain", max_length=32)
        topic = _plain_string(value["topic"], "topic", max_length=64)
        formula = _plain_string(value["formula"], "formula", max_length=4096)

        if domain not in VALID_TOPICS:
            raise EquationSchemaError(f"invalid domain: {domain!r}")
        if topic not in VALID_TOPICS[domain]:
            raise EquationSchemaError(
                f"invalid topic {topic!r} for domain {domain!r}"
            )

        levels: dict[str, str | None] = {}
        for field in ("source_level", "curriculum_level"):
            raw = value.get(field)
            if raw is None:
                levels[field] = None
                continue
            level = _plain_string(raw, field, max_length=64)
            if _LEVEL_RE.fullmatch(level) is None:
                raise EquationSchemaError(f"invalid {field}: {level!r}")
            levels[field] = level

        match = _ID_RE.fullmatch(record_id)
        expected_domain_code = "MATH" if domain == "mathematics" else "PHY"
        expected_topic_code = TOPIC_CODES[(domain, topic)]
        if match is None:
            raise EquationSchemaError("id does not match the stable identifier format")
        namespace = match.group(2)
        if match.group(1) != expected_domain_code or namespace.split("-")[-1] != expected_topic_code:
            raise EquationSchemaError("id prefix does not match domain and topic")
        if namespace.startswith("UG-") and levels["source_level"] != "undergraduate":
            raise EquationSchemaError("UG id namespace requires source_level undergraduate")
        if namespace.startswith("GR-") and levels["source_level"] != "graduate":
            raise EquationSchemaError("GR id namespace requires source_level graduate")
        if namespace.startswith("RS-") and levels["source_level"] != "research":
            raise EquationSchemaError("RS id namespace requires source_level research")
        if levels["source_level"] == "research" and not namespace.startswith("RS-"):
            raise EquationSchemaError("research source_level requires the RS id namespace")

        subtopic_value = value.get("subtopic")
        subtopic = (
            _plain_string(subtopic_value, "subtopic", max_length=96)
            if subtopic_value is not None
            else None
        )
        aliases = _string_tuple(value.get("aliases"), "aliases")
        symbols = _string_tuple(value.get("symbols"), "symbols", max_items=64)
        source_ids = _string_tuple(value.get("source_ids"), "source_ids", max_items=8)
        for source_id in source_ids:
            if _SOURCE_ID_RE.fullmatch(source_id) is None:
                raise EquationSchemaError(f"invalid source id: {source_id!r}")

        research_metadata: dict[str, str | None] = {}
        for field, allowed in (
            ("relation_type", RELATION_TYPES),
            ("status", RESEARCH_STATUSES),
            ("unit_system", UNIT_SYSTEMS),
        ):
            raw = value.get(field)
            if raw is None:
                research_metadata[field] = None
            else:
                parsed = _plain_string(raw, field, max_length=64)
                if parsed not in allowed:
                    raise EquationSchemaError(f"invalid {field}: {parsed!r}")
                research_metadata[field] = parsed
        source_locator_value = value.get("source_locator")
        source_locator = (
            _plain_string(source_locator_value, "source_locator", max_length=1024)
            if source_locator_value is not None
            else None
        )
        assumptions = _string_tuple(
            value.get("assumptions"), "assumptions", max_items=16, max_length=512
        )
        conventions = _string_tuple(
            value.get("conventions"), "conventions", max_items=16, max_length=512
        )
        is_research = levels["source_level"] == "research"
        if is_research:
            missing_research = [
                field for field in (
                    "relation_type", "status", "source_locator", "unit_system"
                )
                if value.get(field) is None
            ]
            if missing_research:
                raise EquationSchemaError(
                    "research record missing metadata: "
                    + ", ".join(missing_research)
                )
            if not source_ids:
                raise EquationSchemaError("research record requires at least one source id")
        elif any(
            value.get(field) is not None
            for field in (
                "relation_type", "status", "source_locator", "unit_system",
                "assumptions", "conventions",
            )
        ):
            raise EquationSchemaError(
                "research metadata is only permitted for source_level research"
            )

        return cls(
            id=record_id,
            domain=domain,
            topic=topic,
            formula=formula,
            subtopic=subtopic,
            aliases=aliases,
            symbols=symbols,
            source_level=levels["source_level"],
            curriculum_level=levels["curriculum_level"],
            source_ids=source_ids,
            relation_type=research_metadata["relation_type"],
            status=research_metadata["status"],
            source_locator=source_locator,
            unit_system=research_metadata["unit_system"],
            assumptions=assumptions,
            conventions=conventions,
        )

    def as_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""
        result: dict[str, object] = {
            "id": self.id,
            "domain": self.domain,
            "topic": self.topic,
            "formula": self.formula,
        }
        for field in (
            "subtopic", "source_level", "curriculum_level", "relation_type",
            "status", "source_locator", "unit_system",
        ):
            value = getattr(self, field)
            if value is not None:
                result[field] = value
        for field in (
            "aliases", "symbols", "source_ids", "assumptions", "conventions"
        ):
            value = getattr(self, field)
            if value:
                result[field] = list(value)
        return result
