"""Research-tier schema, provenance, preservation, and package contracts."""
from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ags_sci.knowledge.equations import (
    EquationCorpus,
    EquationRecord,
    EquationSchemaError,
    LEVEL_TOPICS,
    RELATION_TYPES,
    RESEARCH_STATUSES,
    UNIT_SYSTEMS,
    validate_corpus,
)
from ags_sci.knowledge.equations.loader import default_reference_root


def _root() -> Path:
    root = default_reference_root()
    assert isinstance(root, Path)
    return root


def _json(name: str) -> dict:
    return json.loads((_root() / name).read_text(encoding="utf-8"))


def _research_records() -> tuple[EquationRecord, ...]:
    return tuple(EquationCorpus().iter_records(source_level="research"))


def test_research_schema_is_backward_compatible_and_strict():
    legacy = EquationRecord.from_mapping({
        "id": "MATH-ALG-999999", "domain": "mathematics",
        "topic": "algebra", "formula": "x+y=y+x",
    })
    assert legacy.status is None and legacy.source_locator is None

    value = {
        "id": "MATH-RS-OT-999999", "domain": "mathematics",
        "topic": "operator_theory", "formula": r"T^*T=TT^*",
        "source_level": "research", "source_ids": ["MIT-18102-FA"],
        "relation_type": "operator_relation", "status": "established",
        "source_locator": "https://ocw.mit.edu/", "unit_system": "not_applicable",
        "assumptions": [r"T\in\mathcal{B}(H)"],
    }
    record = EquationRecord.from_mapping(value)
    assert record.status == "established"
    assert record.assumptions == (r"T\in\mathcal{B}(H)",)
    with pytest.raises(FrozenInstanceError):
        record.status = "model_dependent"  # type: ignore[misc]

    for field in ("relation_type", "status", "source_locator", "unit_system"):
        malformed = dict(value)
        malformed.pop(field)
        with pytest.raises(EquationSchemaError, match="missing metadata"):
            EquationRecord.from_mapping(malformed)
    malformed = dict(value, status="certain")
    with pytest.raises(EquationSchemaError, match="invalid status"):
        EquationRecord.from_mapping(malformed)
    malformed = dict(value, id="MATH-GR-OT-999999")
    with pytest.raises(EquationSchemaError):
        EquationRecord.from_mapping(malformed)


def test_research_release_counts_namespaces_and_lazy_scoping():
    corpus = EquationCorpus()
    manifest = corpus.manifest
    assert manifest["corpus_version"] == "3.0.0"
    assert manifest["schema_version"] == "1.2.0"
    assert manifest["record_count"] == 6620
    assert manifest["domain_counts"] == {"mathematics": 3616, "physics": 3004}
    assert manifest["research_record_count"] == 5600
    assert manifest["research_domain_counts"] == {"mathematics": 3033, "physics": 2567}
    assert manifest["level_domain_counts"] == {
        "graduate": {"mathematics": 154, "physics": 142},
        "k12": {"mathematics": 229, "physics": 137},
        "research": {"mathematics": 3033, "physics": 2567},
        "undergraduate": {"mathematics": 200, "physics": 158},
    }

    # Filtering by level loads only matching files; construction itself creates
    # no whole-corpus formula index.
    assert corpus._cache == {}  # noqa: SLF001 - explicit lazy-loading contract
    mathematics = tuple(corpus.iter_records(source_level="research", domain="mathematics"))
    assert len(mathematics) == 3033
    assert all(record.id.startswith("MATH-RS-") for record in mathematics)
    assert all(record.source_level == "research" for record in mathematics)
    assert all("/research/mathematics/" in f"/{path}" for path in corpus._cache)  # noqa: SLF001
    assert corpus._all is None and corpus._index is None  # noqa: SLF001

    physics = tuple(corpus.iter_records(source_level="research", domain="physics"))
    assert len(physics) == 2567
    assert all(record.id.startswith("PHY-RS-") for record in physics)
    assert len({record.id for record in mathematics + physics}) == 5600


def test_research_metadata_status_conventions_and_sources():
    records = _research_records()
    statuses = Counter(record.status for record in records)
    units = Counter(record.unit_system for record in records)
    relation_types = Counter(record.relation_type for record in records)
    assert statuses == {"established": 3292, "model_dependent": 2308}
    assert _json("manifest.json")["research_status_counts"] == {
        "conjectural": 0, "established": 3292, "model_dependent": 2308,
    }
    # Conjectural is supported by schema but no conjecture was added merely to
    # populate the category or inflate the corpus.
    assert RESEARCH_STATUSES == {"established", "model_dependent", "conjectural"}
    assert set(relation_types) <= RELATION_TYPES and len(relation_types) >= 8
    assert set(units) <= UNIT_SYSTEMS
    assert units["not_applicable"] == 3033

    source_map = {source["id"]: source for source in _json("sources.json")["sources"]}
    assert len(source_map) == 56
    assert all(record.source_ids for record in records)
    assert all(set(record.source_ids) <= set(source_map) for record in records)
    assert all(record.source_locator and record.source_locator.startswith("https://") for record in records)
    assert all(record.unit_system == "not_applicable" for record in records if record.domain == "mathematics")
    assert all(record.conventions for record in records if record.domain == "physics")
    assert all(record.unit_system != "not_applicable" for record in records if record.domain == "physics")


def test_requested_research_domains_have_machine_readable_coverage():
    coverage = _json("research_coverage.json")
    assert coverage["claim"].startswith("Broad formula-reference coverage")
    status_counts = Counter()
    for domain, expected_topics in LEVEL_TOPICS["research"].items():
        entries = coverage["domains"][domain]
        assert set(entries) == set(expected_topics)
        for topic, entry in entries.items():
            assert entry["records"] > 0, (domain, topic)
            assert entry["source_count"] == len(entry["source_ids"]) >= 1
            assert entry["status"] in {"covered", "partial", "not covered"}
            assert entry["status"] != "not covered"
            status_counts[entry["status"]] += 1
    assert status_counts == {"covered": 26, "partial": 9}


def test_pre_research_corpus_is_byte_for_byte_preserved():
    baseline = _json("pre_research_baseline.json")
    assert baseline["corpus_version"] == "2.0.0"
    assert baseline["schema_version"] == "1.1.0"
    assert baseline["record_count"] == 1020
    assert baseline["file_count"] == 65
    assert baseline["manifest_sha256"] == (
        "f5ac95b289d073321687361f7928865b4fe814b99765a56af43c92887fbc77d3"
    )
    assert baseline["ordered_data_hash_digest"] == (
        "1040af44eb93fab3de307341e6172f4edc1afc6a0aef269c2ae55563bf99de1e"
    )
    ordered_hashes: list[str] = []
    record_count = 0
    for relative, metadata in sorted(baseline["files"].items()):
        assert not relative.startswith("research/")
        raw = (_root() / relative).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == metadata["sha256"]
        assert len(raw) == metadata["bytes"]
        assert len(raw.splitlines()) == metadata["records"]
        ordered_hashes.append(digest)
        record_count += metadata["records"]
    assert record_count == 1020 and len(ordered_hashes) == 65
    assert hashlib.sha256("".join(ordered_hashes).encode("ascii")).hexdigest() == (
        baseline["ordered_data_hash_digest"]
    )


def test_research_formula_and_provenance_spot_checks():
    corpus = EquationCorpus()
    gamma = corpus.get("MATH-RS-SF-000004")
    assert gamma is not None
    assert gamma.formula == r"\Gamma\left(z\right)=\int_{0}^{\infty}e^{-t}t^{z-1}\mathrm{d}t"
    assert gamma.assumptions == (r"\Re z>0",)
    assert gamma.source_locator == "https://dlmf.nist.gov/5.2.E1"
    assert gamma.relation_type == "integral_relation"

    nambu_goto = next(
        record for record in corpus.iter_records(
            source_level="research", domain="physics", topic="string_quantum_gravity"
        ) if record.subtopic == "nambu_goto_action"
    )
    assert nambu_goto.formula == r"S_{\mathrm{NG}}=-T\int d^2\sigma\sqrt{-\det\gamma}"
    assert nambu_goto.status == "model_dependent"
    assert nambu_goto.unit_system == "natural"
    assert nambu_goto.source_ids == ("TONG-STRING",)

    bloch = next(
        record for record in corpus.iter_records(
            source_level="research", domain="physics", topic="condensed_matter"
        ) if record.subtopic == "bloch_theorem"
    )
    assert bloch.status == "established"
    assert "u_{n\\mathbf{k}}" in bloch.formula


def test_research_reference_remains_isolated_from_discovery_and_execution():
    corpus = EquationCorpus()
    research_paths = [path for path in corpus._data_paths if path.startswith("research/")]  # noqa: SLF001
    assert len(research_paths) == 35
    assert all("discovered" not in path for path in corpus._data_paths)  # noqa: SLF001
    assert not hasattr(EquationCorpus, "solve")
    assert not hasattr(EquationCorpus, "execute")
    assert not hasattr(EquationCorpus, "promote")
    assert not hasattr(EquationCorpus, "apply")


def test_research_global_integrity_duplicates_and_package_resources():
    report = validate_corpus()
    assert report.valid, report.issues
    assert report.record_count == 6620
    assert report.file_count == 100
    assert report.duplicate_ids == 0
    assert report.duplicate_records == 0
    assert report.duplicate_formulas == 0
    assert report.normalized_duplicate_formulas == 0
    assert report.malformed_records == 0

    records = _research_records()
    strong_keys = []
    for record in records:
        normalized = re.sub(r"\\(?:left|right)|\s+", "", record.formula).strip(".,;")
        if normalized.count("=") == 1 and "\\begin" not in normalized:
            normalized = "=".join(sorted(normalized.split("=", 1)))
        strong_keys.append(normalized)
    assert len(strong_keys) == len(set(strong_keys))

    manifest = _json("manifest.json")
    packaged = importlib.resources.files("ags_sci.knowledge.equations.reference")
    research_paths = [path for path in manifest["files"] if path.startswith("research/")]
    assert len(research_paths) == 35
    for relative in research_paths:
        resource = packaged
        for part in relative.split("/"):
            resource = resource.joinpath(part)
        raw = resource.read_bytes()
        metadata = manifest["files"][relative]
        assert len(raw) == metadata["bytes"]
        assert hashlib.sha256(raw).hexdigest() == metadata["sha256"]
