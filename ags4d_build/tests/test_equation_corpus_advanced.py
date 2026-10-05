"""University-corpus preservation, provenance, and independent spot audits."""
from __future__ import annotations

import hashlib
import importlib.resources
import json
import shutil
from collections import Counter
from pathlib import Path

import sympy as sp

from ags_sci.knowledge import EquationCorpus
from ags_sci.knowledge.equations.loader import default_reference_root
from ags_sci.knowledge.equations.validation import validate_corpus


def _root() -> Path:
    root = default_reference_root()
    assert isinstance(root, Path)
    return root


def _json(name: str) -> dict:
    return json.loads((_root() / name).read_text(encoding="utf-8"))


def test_k12_baseline_is_byte_for_byte_preserved():
    baseline = _json("k12_baseline.json")
    assert baseline["record_count"] == 366
    assert baseline["mathematics_count"] == 229
    assert baseline["physics_count"] == 137
    assert baseline["manifest_sha256"] == (
        "eeed7722a3e15fa4338a581c3dc8ff31f8a61d4b1c72086a7f79e92b8e6c2a11"
    )
    assert baseline["original_metadata_hashes"] == {
        "manifest.json": "eeed7722a3e15fa4338a581c3dc8ff31f8a61d4b1c72086a7f79e92b8e6c2a11",
        "schema.json": "9ecf127ea74505848371722b2a297286917f112d2da81f6c5a094ebb85b0c482",
        "sources.json": "56110d350dd77670e3da1e465bf0a9f465ee60399836434af300538d1fa70c38",
    }

    ordered_hashes = []
    for relative, metadata in sorted(baseline["files"].items()):
        # No baseline file was moved into a level-prefixed hierarchy.
        assert relative.startswith(("mathematics/", "physics/"))
        raw = (_root() / relative).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == metadata["sha256"]
        assert len(raw.splitlines()) == metadata["records"]
        ordered_hashes.append(digest)
    assert len(ordered_hashes) == 27
    digest = hashlib.sha256("".join(ordered_hashes).encode("ascii")).hexdigest()
    assert digest == baseline["ordered_data_hash_digest"]
    assert digest == "aeb48c1ed39069e7109eabfac485e7a2ba049e795d539a81eafb608b538491cc"


def test_k12_lock_is_independent_of_current_manifest(tmp_path: Path):
    target = tmp_path / "reference"
    shutil.copytree(_root(), target)
    relative = "mathematics/algebra/equations.jsonl"
    path = target / relative
    rows = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(rows[0])
    record["formula"] += " "
    rows[0] = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Even recomputing the release manifest cannot bypass the independently
    # packaged v1.0.0 lock.
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = path.read_bytes()
    manifest["files"][relative]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["files"][relative]["bytes"] = len(raw)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = validate_corpus(target)
    assert not report.valid
    assert report.count("k12_preservation") >= 1


def test_cross_level_whitespace_duplicate_is_detected(tmp_path: Path):
    target = tmp_path / "reference"
    shutil.copytree(_root(), target)
    path = target / "undergraduate/mathematics/algebra/equations.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["formula"] = "(a + b)^2 = a^2 + 2ab + b^2"
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    report = validate_corpus(target)
    assert not report.valid
    assert report.normalized_duplicate_formulas == 1


def test_level_hierarchy_ids_counts_and_retrieval():
    corpus = EquationCorpus()
    manifest = corpus.manifest
    assert manifest["corpus_version"] == "3.0.0"
    assert manifest["schema_version"] == "1.2.0"
    assert manifest["record_count"] == 6620
    assert manifest["file_count"] == manifest["topic_count"] == 100
    assert manifest["level_domain_counts"] == {
        "graduate": {"mathematics": 154, "physics": 142},
        "k12": {"mathematics": 229, "physics": 137},
        "research": {"mathematics": 3033, "physics": 2567},
        "undergraduate": {"mathematics": 200, "physics": 158},
    }

    namespace = {
        "undergraduate": {"mathematics": "MATH-UG-", "physics": "PHY-UG-"},
        "graduate": {"mathematics": "MATH-GR-", "physics": "PHY-GR-"},
    }
    for level in ("undergraduate", "graduate"):
        for domain in ("mathematics", "physics"):
            records = tuple(corpus.iter_records(source_level=level, domain=domain))
            assert len(records) == manifest["level_domain_counts"][level][domain]
            assert all(record.id.startswith(namespace[level][domain]) for record in records)
            assert all(record.source_level == level for record in records)

    gr = tuple(corpus.iter_records(
        source_level="graduate", domain="physics", topic="general_relativity"
    ))
    assert len(gr) == 24
    assert {record.topic for record in gr} == {"general_relativity"}
    assert corpus.get("PHY-GR-GR-000009").subtopic == "einstein_equation"


def test_manifest_source_and_resource_integrity():
    report = validate_corpus()
    assert report.valid, report.issues
    assert report.level_domain_counts["undergraduate"] == {
        "mathematics": 200,
        "physics": 158,
    }
    assert report.duplicate_ids == 0
    assert report.duplicate_records == 0
    assert report.duplicate_formulas == 0
    assert report.normalized_duplicate_formulas == 0
    assert report.malformed_records == 0

    manifest = _json("manifest.json")
    sources = _json("sources.json")
    source_map = {source["id"]: source for source in sources["sources"]}
    assert len(source_map) == manifest["source_metadata"]["source_count"] == 56
    assert set(source_map) == set(manifest["source_metadata"]["source_ids"])
    for source in source_map.values():
        assert source["authority"] and source["title"]
        assert source["url"].startswith(("https://", "http://"))

    used_sources: Counter[str] = Counter()
    for record in EquationCorpus().all():
        used_sources.update(record.source_ids)
    assert set(used_sources) <= set(source_map)
    assert used_sources["MIT-1806-LA"] == 22
    assert used_sources["TONG-QFT"] == 23

    packaged = importlib.resources.files("ags_sci.knowledge.equations.reference")
    for relative, metadata in manifest["files"].items():
        resource = packaged
        for part in relative.split("/"):
            resource = resource.joinpath(part)
        raw = resource.read_bytes()
        assert len(raw) == metadata["bytes"]
        assert hashlib.sha256(raw).hexdigest() == metadata["sha256"]


def test_representative_mathematical_identities_with_exact_arithmetic():
    """Independent computations catch signs/order errors in representative data."""
    corpus = EquationCorpus()
    for record_id in (
        "MATH-UG-LA-000004",   # determinant product
        "MATH-UG-MVC-000014", # divergence theorem
        "MATH-UG-NUM-000002", # Chinese remainder theorem
        "MATH-GR-LIE-000002", # Jacobi identity
        "MATH-GR-AT-000004",  # boundary squared
    ):
        assert corpus.get(record_id) is not None

    A = sp.Matrix([[2, 1], [3, 4]])
    B = sp.Matrix([[1, -1], [5, 2]])
    assert (A * B).det() == A.det() * B.det()

    x, y, z = sp.symbols("x y z")
    field = (x**2, y**2, z**2)
    divergence_integral = sum(
        sp.integrate(sp.diff(component, variable), (x, 0, 1), (y, 0, 1), (z, 0, 1))
        for component, variable in zip(field, (x, y, z))
    )
    outward_flux = sp.Integer(1) + sp.Integer(1) + sp.Integer(1)
    assert divergence_integral == outward_flux == 3

    solutions = [n for n in range(15) if n % 3 == 2 and n % 5 == 3]
    assert solutions == [8]

    X = sp.Matrix([[0, 1], [0, 0]])
    Y = sp.Matrix([[0, 0], [1, 0]])
    Z = sp.diag(1, -1)
    bracket = lambda P, Q: P * Q - Q * P
    assert bracket(X, bracket(Y, Z)) + bracket(Y, bracket(Z, X)) + bracket(Z, bracket(X, Y)) == sp.zeros(2)

    # Boundary of oriented [0,1,2], followed by the vertex-edge boundary.
    edge_boundary = sp.Matrix([[-1, -1, 0], [1, 0, -1], [0, 1, 1]])
    triangle_boundary = sp.Matrix([1, -1, 1])
    assert edge_boundary * triangle_boundary == sp.zeros(3, 1)


# SI base dimensions: mass M, length L, time T, electric current I.
Dim = tuple[int, int, int, int]
M: Dim = (1, 0, 0, 0)
L: Dim = (0, 1, 0, 0)
T: Dim = (0, 0, 1, 0)
I: Dim = (0, 0, 0, 1)


def _mul(*dims: Dim) -> Dim:
    return tuple(sum(values) for values in zip(*dims))  # type: ignore[return-value]


def _pow(dim: Dim, exponent: int) -> Dim:
    return tuple(exponent * value for value in dim)  # type: ignore[return-value]


def _div(left: Dim, right: Dim) -> Dim:
    return _mul(left, _pow(right, -1))


def test_representative_physics_dimensional_checks():
    """Check standard SI records; natural-unit records are explicitly excluded."""
    corpus = EquationCorpus()
    energy = _mul(M, _pow(L, 2), _pow(T, -2))
    force_density = _mul(M, _pow(L, -2), _pow(T, -2))

    epsilon_0 = _mul(_pow(I, 2), _pow(T, 4), _pow(M, -1), _pow(L, -3))
    electric_field = _mul(M, L, _pow(T, -3), _pow(I, -1))
    magnetic_field = _mul(M, _pow(T, -2), _pow(I, -1))
    mu_0 = _mul(M, L, _pow(T, -2), _pow(I, -2))
    energy_density = _mul(M, _pow(L, -1), _pow(T, -2))
    assert _mul(epsilon_0, _pow(electric_field, 2)) == energy_density
    assert _div(_pow(magnetic_field, 2), mu_0) == energy_density
    assert "\\varepsilon_0E^2" in corpus.get("PHY-UG-EM-000012").formula

    hbar = _mul(M, _pow(L, 2), _pow(T, -1))
    assert _div(hbar, T) == energy
    assert _div(_pow(hbar, 2), _mul(M, _pow(L, 2))) == energy
    assert corpus.get("PHY-UG-QM-000002").subtopic == "position_wavefunction"

    density = _mul(M, _pow(L, -3))
    acceleration = _mul(L, _pow(T, -2))
    pressure = _mul(M, _pow(L, -1), _pow(T, -2))
    dynamic_viscosity = _mul(M, _pow(L, -1), _pow(T, -1))
    velocity_laplacian = _mul(_pow(L, -1), _pow(T, -1))
    assert _mul(density, acceleration) == force_density
    assert _div(pressure, L) == force_density
    assert _mul(dynamic_viscosity, velocity_laplacian) == force_density
    assert corpus.get("PHY-UG-FL-000006").subtopic == "navier_stokes"

    newton_G = _mul(_pow(L, 3), _pow(M, -1), _pow(T, -2))
    c = _mul(L, _pow(T, -1))
    stress_energy = energy_density
    assert _mul(_div(newton_G, _pow(c, 4)), stress_energy) == _pow(L, -2)
    assert corpus.get("PHY-GR-GR-000009").subtopic == "einstein_equation"

    number_density = _pow(L, -3)
    charge = _mul(I, T)
    plasma_omega_squared = _div(_mul(number_density, _pow(charge, 2)), _mul(epsilon_0, M))
    assert plasma_omega_squared == _pow(T, -2)
    assert "\\text{(SI)}" in corpus.get("PHY-GR-PL-000001").formula

    # QFT/gauge relations explicitly state natural units rather than receiving
    # a misleading SI audit.
    assert corpus.get("PHY-GR-QFT-000001").formula == r"\hbar=c=1"
    assert "\\hbar=c=1" in corpus.get("PHY-GR-GT-000005").formula
