"""Contracts for the immutable mathematics/physics equation reference layer."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ags_sci.knowledge.equations import (
    CorpusIntegrityError,
    EquationCorpus,
    EquationRecord,
    EquationSchemaError,
    iter_equation_file,
    validate_corpus,
)
from ags_sci.knowledge.equations.loader import default_reference_root
from ags_sci.knowledge.equations.validation import latex_delimiter_problem


def _reference_path() -> Path:
    root = default_reference_root()
    assert isinstance(root, Path), "tests run from the source checkout"
    return root


def _copy_reference(tmp_path: Path) -> Path:
    target = tmp_path / "reference"
    shutil.copytree(_reference_path(), target)
    return target


def test_equation_loader():
    corpus = EquationCorpus()
    records = corpus.all()
    assert len(records) == corpus.record_count == 6620
    assert sum(record.domain == "mathematics" for record in records) == 3616
    assert sum(record.domain == "physics" for record in records) == 3004
    assert corpus.get("MATH-ALG-000001") is not None
    with pytest.raises(FrozenInstanceError):
        records[0].formula = "changed"  # type: ignore[misc]


def test_equation_schema():
    record = EquationRecord.from_mapping({
        "id": "MATH-ALG-999999",
        "domain": "mathematics",
        "topic": "algebra",
        "formula": "x+y=y+x",
        "source_level": "undergraduate",
    })
    assert record.source_level == "undergraduate"

    bad_records = (
        {"id": "MATH-ALG-999998", "domain": "chemistry", "topic": "algebra", "formula": "x"},
        {"id": "MATH-ALG-999997", "domain": "mathematics", "topic": "not_a_topic", "formula": "x"},
        {"id": "MATH-ALG-999996", "domain": "mathematics", "topic": "algebra", "formula": ""},
        {"id": "PHY-MEC-999995", "domain": "mathematics", "topic": "algebra", "formula": "x"},
    )
    for value in bad_records:
        with pytest.raises(EquationSchemaError):
            EquationRecord.from_mapping(value)
    assert latex_delimiter_problem(r"x=\frac{a}{b}") is None
    assert latex_delimiter_problem(r"x=\frac{a}{b") is not None
    assert latex_delimiter_problem(r"\begin{cases}x=1") is not None


def test_duplicate_equations(tmp_path: Path):
    root = _copy_reference(tmp_path)
    path = root / "mathematics" / "algebra" / "equations.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    with path.open("a", encoding="utf-8") as stream:
        stream.write(first + "\n")
    report = validate_corpus(root)
    assert not report.valid
    assert report.duplicate_ids == 1
    assert report.duplicate_formulas == 1
    assert report.duplicate_records == 1


def test_equation_data_never_executes(tmp_path: Path):
    marker = tmp_path / "formula-ran"
    payload = f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')"
    path = tmp_path / "equations.jsonl"
    path.write_text(json.dumps({
        "id": "MATH-ALG-999999",
        "domain": "mathematics",
        "topic": "algebra",
        "formula": payload,
    }) + "\n", encoding="utf-8")

    records = tuple(iter_equation_file(path))
    assert records[0].formula == payload
    assert not marker.exists()
    assert not hasattr(EquationCorpus, "apply")


def test_equation_retrieval():
    corpus = EquationCorpus()
    force = corpus.retrieve("F = ma", domain="physics", limit=3)
    assert [record.id for record in force] == ["PHY-MEC-000010"]
    chinese = corpus.retrieve("牛顿第二定律", domain="physics")
    assert chinese and chinese[0].formula == "F = ma"
    calculus = corpus.retrieve(
        "power rule", domain="mathematics", topic="calculus"
    )
    assert calculus and calculus[0].id == "MATH-CALC-000003"
    assert corpus.retrieve("definitely absent query") == ()


def test_discovery_is_not_loaded_or_promoted_into_reference_corpus():
    manifest = _reference_path() / "manifest.json"
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    discovered = (
        _reference_path().parent / "discovered" / "candidate" / "untrusted.jsonl"
    )
    # A candidate-like file outside reference is never declared by, loaded by,
    # or promoted into the trusted manifest.
    assert "discovered/" not in manifest.read_text(encoding="utf-8")
    corpus = EquationCorpus()
    assert all("discovered" not in path for path in corpus._data_paths)  # noqa: SLF001
    assert not hasattr(corpus, "promote")
    assert not hasattr(corpus, "learn")
    assert not discovered.exists()
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before


def test_manifest_hashes():
    report = validate_corpus()
    assert report.valid, report.issues
    assert report.duplicate_ids == 0
    assert report.duplicate_formulas == 0
    assert report.duplicate_records == 0
    assert report.malformed_records == 0

    root = _reference_path()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for relative_path, metadata in manifest["files"].items():
        assert hashlib.sha256((root / relative_path).read_bytes()).hexdigest() == metadata["sha256"]


def test_loader_rejects_manifest_tampering(tmp_path: Path):
    root = _copy_reference(tmp_path)
    path = root / "physics" / "mechanics" / "equations.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(CorpusIntegrityError):
        EquationCorpus(root)
