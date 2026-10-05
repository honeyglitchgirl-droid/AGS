#!/usr/bin/env python3
"""Regenerate deterministic checksums and distributions for the equation corpus."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "src/ags_sci/knowledge/equations/reference"


def scope(path: Path) -> tuple[str, str, str]:
    relative = path.relative_to(REFERENCE).parts
    if len(relative) == 3:
        return "k12", relative[0], relative[1]
    if len(relative) == 4 and relative[0] in {"undergraduate", "graduate", "research"}:
        return relative[0], relative[1], relative[2]
    raise ValueError(path)


def main() -> None:
    data_paths = sorted(REFERENCE.rglob("equations.jsonl"))
    metadata_names = [
        "k12_baseline.json", "pre_research_baseline.json", "research_coverage.json",
        "schema.json", "sources.json",
    ]
    files: dict[str, dict[str, int | str]] = {}
    domain_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    level_domain_counts: Counter[tuple[str, str]] = Counter()
    topic_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    used_sources: Counter[str] = Counter()
    research_sources: Counter[str] = Counter()
    data_bytes = 0
    for path in data_paths:
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        relative = path.relative_to(REFERENCE).as_posix()
        level, domain, topic = scope(path)
        files[relative] = {
            "bytes": len(raw), "records": len(rows),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        data_bytes += len(raw)
        domain_counts[domain] += len(rows)
        level_counts[level] += len(rows)
        level_domain_counts[(level, domain)] += len(rows)
        topic_key = f"{domain}/{topic}" if level == "k12" else f"{level}/{domain}/{topic}"
        topic_counts[topic_key] += len(rows)
        for row in rows:
            used_sources.update(row.get("source_ids", []))
            if level == "research":
                status_counts[row["status"]] += 1
                relation_counts[row["relation_type"]] += 1
                unit_counts[row["unit_system"]] += 1
                research_sources.update(row["source_ids"])
    for name in metadata_names:
        path = REFERENCE / name
        raw = path.read_bytes()
        files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    sources = json.loads((REFERENCE / "sources.json").read_text(encoding="utf-8"))
    source_ids = sorted(source["id"] for source in sources["sources"])
    orphan_ids = sorted(set(source_ids) - set(used_sources))
    coverage = json.loads((REFERENCE / "research_coverage.json").read_text(encoding="utf-8"))
    coverage_status_counts = Counter(
        entry["status"]
        for entries in coverage["domains"].values()
        for entry in entries.values()
    )
    level_domains = {
        level: {domain: level_domain_counts[(level, domain)] for domain in ("mathematics", "physics")}
        for level in sorted(level_counts)
    }
    research_domain_counts = level_domains["research"]
    k12_lock = json.loads((REFERENCE / "k12_baseline.json").read_text(encoding="utf-8"))
    pre_lock = json.loads((REFERENCE / "pre_research_baseline.json").read_text(encoding="utf-8"))
    document = {
        "corpus_name": "AGS-Sci Audited Mathematics and Physics Equation Corpus",
        "corpus_version": "3.0.0",
        "schema_version": "1.2.0",
        "generation_date": "2026-10-05",
        "scope": "Additive K-12, undergraduate, graduate, and inert research mathematics/physics trusted reference data.",
        "coverage_status": "Broad, partial research-domain coverage; not exhaustive. See research_coverage.json and docs/EQUATION_CORPUS.md.",
        "inert_data_policy": "Formula strings are opaque LaTeX-compatible text and are never evaluated, imported, substituted, solved, selected for reasoning, or promoted from discovery.",
        "duplicate_policy": "Exact and whitespace-normalized formulas, reversed single equalities, duplicate IDs, and duplicate records are rejected globally; rearrangements and arbitrary symbolic permutations are not generated for count inflation.",
        "hash_algorithm": "sha256",
        "record_count": sum(domain_counts.values()),
        "mathematics_count": domain_counts["mathematics"],
        "physics_count": domain_counts["physics"],
        "domain_counts": dict(sorted(domain_counts.items())),
        "level_counts": dict(sorted(level_counts.items())),
        "level_domain_counts": level_domains,
        "topic_counts": dict(sorted(topic_counts.items())),
        "topic_count": len(topic_counts),
        "file_count": len(data_paths),
        "data_bytes": data_bytes,
        "research_record_count": level_counts["research"],
        "research_domain_counts": research_domain_counts,
        "research_status_counts": {
            status: status_counts[status]
            for status in ("conjectural", "established", "model_dependent")
        },
        "research_relation_type_counts": dict(sorted(relation_counts.items())),
        "research_unit_system_counts": dict(sorted(unit_counts.items())),
        "research_source_count": len(research_sources),
        "k12_baseline": {
            "baseline_name": k12_lock["baseline_name"], "file": "k12_baseline.json",
            "original_corpus_version": k12_lock["corpus_version"],
            "original_manifest_sha256": k12_lock["manifest_sha256"],
            "record_count": k12_lock["record_count"],
            "ordered_data_hash_digest": k12_lock["ordered_data_hash_digest"],
        },
        "pre_research_baseline": {
            "baseline_name": pre_lock["baseline_name"], "file": "pre_research_baseline.json",
            "original_corpus_version": pre_lock["corpus_version"],
            "original_manifest_sha256": pre_lock["manifest_sha256"],
            "record_count": pre_lock["record_count"], "file_count": pre_lock["file_count"],
            "ordered_data_hash_digest": pre_lock["ordered_data_hash_digest"],
        },
        "research_coverage": {
            "file": "research_coverage.json", "coverage_version": coverage["coverage_version"],
            "status_counts": dict(sorted(coverage_status_counts.items())),
        },
        "source_metadata": {
            "file": "sources.json", "metadata_version": sources["metadata_version"],
            "source_count": len(source_ids), "source_ids": source_ids,
            "used_source_count": len(used_sources), "orphan_source_count": len(orphan_ids),
            "orphan_source_ids": orphan_ids,
        },
        "files": dict(sorted(files.items())),
    }
    (REFERENCE / "manifest.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "record_count": document["record_count"], "domain_counts": document["domain_counts"],
        "research_domain_counts": research_domain_counts, "file_count": len(data_paths),
        "data_bytes": data_bytes, "sources": len(source_ids), "orphans": orphan_ids,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
