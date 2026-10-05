"""Deterministic integrity validation for the equation reference corpus."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .loader import default_reference_root
from .schema import (
    EquationRecord,
    EquationSchemaError,
    LEVEL_TOPICS,
    RESEARCH_STATUSES,
)

_K12_RECORD_COUNT = 366
_K12_MANIFEST_SHA256 = "eeed7722a3e15fa4338a581c3dc8ff31f8a61d4b1c72086a7f79e92b8e6c2a11"
_K12_ORDERED_DATA_HASH_DIGEST = "aeb48c1ed39069e7109eabfac485e7a2ba049e795d539a81eafb608b538491cc"
_PRE_RESEARCH_RECORD_COUNT = 1020
_PRE_RESEARCH_FILE_COUNT = 65
_PRE_RESEARCH_ORDERED_DATA_HASH_DIGEST = (
    "1040af44eb93fab3de307341e6172f4edc1afc6a0aef269c2ae55563bf99de1e"
)
_PRE_RESEARCH_MANIFEST_SHA256 = (
    "f5ac95b289d073321687361f7928865b4fe814b99765a56af43c92887fbc77d3"
)


@dataclass(frozen=True, order=True, slots=True)
class ValidationIssue:
    code: str
    location: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    root: str
    record_count: int
    domain_counts: dict[str, int]
    level_counts: dict[str, int]
    level_domain_counts: dict[str, dict[str, int]]
    topic_counts: dict[str, int]
    file_count: int
    total_bytes: int
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def count(self, code: str) -> int:
        return sum(issue.code == code for issue in self.issues)

    @property
    def duplicate_records(self) -> int:
        return self.count("duplicate_record")

    @property
    def duplicate_ids(self) -> int:
        return self.count("duplicate_id")

    @property
    def duplicate_formulas(self) -> int:
        return self.count("duplicate_formula")

    @property
    def normalized_duplicate_formulas(self) -> int:
        """Whitespace-only formula variants that may hide cross-level copies."""
        return self.count("duplicate_formula_normalized")

    @property
    def malformed_records(self) -> int:
        malformed_codes = {
            "json_syntax",
            "schema",
            "empty_formula",
            "invalid_domain",
            "invalid_topic",
            "unicode_integrity",
            "latex_delimiter",
            "path_mismatch",
        }
        return sum(issue.code in malformed_codes for issue in self.issues)

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "root": self.root,
            "record_count": self.record_count,
            "domain_counts": dict(sorted(self.domain_counts.items())),
            "level_counts": dict(sorted(self.level_counts.items())),
            "level_domain_counts": {
                level: dict(sorted(counts.items()))
                for level, counts in sorted(self.level_domain_counts.items())
            },
            "topic_counts": dict(sorted(self.topic_counts.items())),
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "duplicate_records": self.duplicate_records,
            "duplicate_ids": self.duplicate_ids,
            "duplicate_formulas": self.duplicate_formulas,
            "normalized_duplicate_formulas": self.normalized_duplicate_formulas,
            "malformed_records": self.malformed_records,
            "issues": [
                {"code": issue.code, "location": issue.location, "message": issue.message}
                for issue in self.issues
            ],
        }


def _walk_files(root: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    for item in sorted(root.iterdir(), key=lambda child: child.name):
        relative = f"{prefix}/{item.name}" if prefix else item.name
        if item.is_dir():
            yield from _walk_files(item, relative)
        elif item.is_file():
            yield relative, item


def _decode_utf8(raw: bytes, location: str, issues: list[ValidationIssue]) -> str | None:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        issues.append(ValidationIssue(
            "encoding", location, f"invalid UTF-8 at byte {exc.start}"
        ))
        return None
    if text.startswith("\ufeff"):
        issues.append(ValidationIssue(
            "encoding", location, "UTF-8 byte-order mark is not permitted"
        ))
        return None
    return text


def _unicode_problem(value: Any, path: str = "record") -> str | None:
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            return f"{path} is not NFC-normalized"
        for char in value:
            codepoint = ord(char)
            if codepoint < 32 or codepoint == 127:
                return f"{path} contains a control character"
            if 0xD800 <= codepoint <= 0xDFFF:
                return f"{path} contains an invalid Unicode code point"
            if (codepoint & 0xFFFF) in {0xFFFE, 0xFFFF}:
                return f"{path} contains a Unicode noncharacter"
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            problem = _unicode_problem(item, f"{path}[{index}]")
            if problem:
                return problem
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            problem = _unicode_problem(key, f"{path}.key")
            if problem:
                return problem
            problem = _unicode_problem(value[key], f"{path}.{key}")
            if problem:
                return problem
    return None


def latex_delimiter_problem(formula: str) -> str | None:
    """Check structural LaTeX delimiters without attempting to parse math."""
    environments: list[tuple[str, int]] = []
    for match in re.finditer(r"\\(begin|end)\{([A-Za-z*]+)\}", formula):
        action, name = match.groups()
        if action == "begin":
            environments.append((name, match.start()))
        elif not environments or environments[-1][0] != name:
            return f"unmatched \\end{{{name}}} at character {match.start()}"
        else:
            environments.pop()
    if environments:
        name, position = environments[-1]
        return f"unclosed \\begin{{{name}}} from character {position}"

    stack: list[tuple[str, int]] = []
    index = 0
    while index < len(formula):
        if formula[index] == "\\":
            token = formula[index:index + 2]
            if token in {"\\(", "\\["}:
                stack.append((token, index))
                index += 2
                continue
            if token in {"\\)", "\\]"}:
                expected = "\\(" if token == "\\)" else "\\["
                if not stack or stack[-1][0] != expected:
                    return f"unmatched {token} at character {index}"
                stack.pop()
                index += 2
                continue
            # Escaped braces and dollars are literals, not delimiters.
            if token in {"\\{", "\\}", "\\$", "\\\\"}:
                index += 2
                continue
        char = formula[index]
        if char == "{":
            stack.append((char, index))
        elif char == "}":
            if not stack or stack[-1][0] != "{":
                return f"unmatched }} at character {index}"
            stack.pop()
        elif char == "$":
            token = "$$" if formula[index:index + 2] == "$$" else "$"
            if stack and stack[-1][0] == token:
                stack.pop()
            else:
                stack.append((token, index))
            if token == "$$":
                index += 1
        index += 1
    if stack:
        token, position = stack[-1]
        return f"unclosed {token} from character {position}"
    return None


def _issue_code_for_schema(message: str) -> str:
    if message.startswith("formula must not be empty"):
        return "empty_formula"
    if message.startswith("invalid domain"):
        return "invalid_domain"
    if message.startswith("invalid topic"):
        return "invalid_topic"
    if "Unicode" in message:
        return "unicode_integrity"
    return "schema"


def _safe_manifest_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts


def _corpus_path_scope(relative_path: str) -> tuple[str, str, str] | None:
    """Parse legacy K-12 or level-prefixed equation paths."""
    parts = PurePosixPath(relative_path).parts
    if len(parts) == 3 and parts[-1] == "equations.jsonl":
        level, domain, topic = "k12", parts[0], parts[1]
    elif (
        len(parts) == 4
        and parts[0] in {"undergraduate", "graduate", "research"}
        and parts[-1] == "equations.jsonl"
    ):
        level, domain, topic = parts[0], parts[1], parts[2]
    else:
        return None
    if topic not in LEVEL_TOPICS.get(level, {}).get(domain, ()):
        return None
    return level, domain, topic


def _load_json_file(
    file_map: dict[str, Any],
    relative_path: str,
    issues: list[ValidationIssue],
) -> Any | None:
    entry = file_map.get(relative_path)
    if entry is None:
        issues.append(ValidationIssue(
            "missing_file", relative_path, "required corpus file is missing"
        ))
        return None
    text = _decode_utf8(entry.read_bytes(), relative_path, issues)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        issues.append(ValidationIssue(
            "json_syntax", relative_path,
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        ))
        return None


def validate_corpus(root: str | Path | Any | None = None) -> ValidationReport:
    """Validate schema, uniqueness, UTF-8, counts, and SHA-256 deterministically."""
    if root is None:
        corpus_root = default_reference_root()
    elif isinstance(root, (str, Path)):
        corpus_root = Path(root)
    else:
        corpus_root = root

    issues: list[ValidationIssue] = []
    # Keep lightweight Traversable handles, not all corpus bytes. Individual
    # files are decoded one at a time so validation remains practical as later
    # independently versioned corpus releases grow.
    file_map: dict[str, Any] = {}
    total_bytes = 0
    try:
        walked = list(_walk_files(corpus_root))
    except (FileNotFoundError, NotADirectoryError) as exc:
        issue = ValidationIssue("missing_root", str(corpus_root), str(exc))
        return ValidationReport(
            root=str(corpus_root), record_count=0, domain_counts={},
            level_counts={}, level_domain_counts={}, topic_counts={},
            file_count=0, total_bytes=0,
            issues=(issue,),
        )

    for relative_path, item in walked:
        file_map[relative_path] = item
        total_bytes += len(item.read_bytes())

    manifest = _load_json_file(file_map, "manifest.json", issues)
    sources = _load_json_file(file_map, "sources.json", issues)
    schema_document = _load_json_file(file_map, "schema.json", issues)
    k12_baseline = _load_json_file(file_map, "k12_baseline.json", issues)
    pre_research_baseline = _load_json_file(
        file_map, "pre_research_baseline.json", issues
    )
    research_coverage = _load_json_file(file_map, "research_coverage.json", issues)

    source_ids: set[str] = set()
    if sources is not None:
        if not isinstance(sources, dict) or not isinstance(sources.get("sources"), list):
            issues.append(ValidationIssue(
                "source_metadata", "sources.json", "sources must be a JSON array"
            ))
        else:
            for index, source in enumerate(sources["sources"]):
                location = f"sources.json:sources[{index}]"
                if not isinstance(source, dict) or not isinstance(source.get("id"), str):
                    issues.append(ValidationIssue(
                        "source_metadata", location, "source must have a string id"
                    ))
                    continue
                if source["id"] in source_ids:
                    issues.append(ValidationIssue(
                        "source_metadata", location, f"duplicate source id {source['id']!r}"
                    ))
                source_ids.add(source["id"])
                for required in ("title", "authority", "url"):
                    if not isinstance(source.get(required), str) or not source[required].strip():
                        issues.append(ValidationIssue(
                            "source_metadata", location,
                            f"source must have a non-empty {required}",
                        ))
            problem = _unicode_problem(sources, "sources")
            if problem:
                issues.append(ValidationIssue("unicode_integrity", "sources.json", problem))

    if schema_document is not None:
        if not isinstance(schema_document, dict) or schema_document.get("schema_version") != "1.2.0":
            issues.append(ValidationIssue(
                "schema_document", "schema.json", "unsupported or missing schema_version"
            ))

    data_paths = sorted(path for path in file_map if path.endswith(".jsonl"))
    seen_ids: dict[str, str] = {}
    seen_formulas: dict[str, str] = {}
    seen_normalized_formulas: dict[str, str] = {}
    seen_records: dict[str, str] = {}
    domain_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    level_domain_counts: Counter[tuple[str, str]] = Counter()
    topic_counts: Counter[str] = Counter()
    per_file_counts: Counter[str] = Counter()
    research_status_counts: Counter[str] = Counter()
    research_relation_type_counts: Counter[str] = Counter()
    research_unit_system_counts: Counter[str] = Counter()
    used_source_counts: Counter[str] = Counter()
    research_source_counts: Counter[str] = Counter()
    research_topic_sources: dict[tuple[str, str], set[str]] = {}
    record_count = 0

    for relative_path in data_paths:
        scope = _corpus_path_scope(relative_path)
        expected_level: str | None = None
        expected_domain: str | None = None
        expected_topic: str | None = None
        if scope is None:
            issues.append(ValidationIssue(
                "path_mismatch", relative_path,
                "expected a legacy K-12 or <level>/<domain>/<valid-topic>/equations.jsonl path",
            ))
        else:
            expected_level, expected_domain, expected_topic = scope

        text = _decode_utf8(file_map[relative_path].read_bytes(), relative_path, issues)
        if text is None:
            continue
        lines = text.splitlines()
        for line_number, line in enumerate(lines, 1):
            location = f"{relative_path}:{line_number}"
            if not line.strip():
                issues.append(ValidationIssue(
                    "json_syntax", location, "blank lines are not allowed in JSONL"
                ))
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(ValidationIssue(
                    "json_syntax", location, f"invalid JSON: {exc.msg}"
                ))
                continue

            problem = _unicode_problem(value)
            if problem:
                issues.append(ValidationIssue("unicode_integrity", location, problem))
                continue
            try:
                record = EquationRecord.from_mapping(value)
            except EquationSchemaError as exc:
                message = str(exc)
                issues.append(ValidationIssue(
                    _issue_code_for_schema(message), location, message
                ))
                continue

            delimiter_problem = latex_delimiter_problem(record.formula)
            if delimiter_problem:
                issues.append(ValidationIssue(
                    "latex_delimiter", location, delimiter_problem
                ))
                continue
            metadata_delimiter_problem = next((
                (field, problem)
                for field in ("assumptions", "conventions")
                for item in getattr(record, field)
                for problem in (latex_delimiter_problem(item),)
                if problem
            ), None)
            if metadata_delimiter_problem:
                field, problem = metadata_delimiter_problem
                issues.append(ValidationIssue(
                    "latex_delimiter", location,
                    f"{field} metadata: {problem}",
                ))
                continue
            if expected_domain is not None and (
                record.domain != expected_domain
                or record.topic != expected_topic
                or record.source_level != expected_level
            ):
                issues.append(ValidationIssue(
                    "path_mismatch", location,
                    "record level/domain/topic does not match its directory",
                ))
                continue
            if expected_level in {"undergraduate", "graduate", "research"}:
                namespace = {
                    "undergraduate": "UG", "graduate": "GR", "research": "RS"
                }[expected_level]
                if not record.id.startswith(
                    ("MATH-" if record.domain == "mathematics" else "PHY-")
                    + namespace + "-"
                ):
                    issues.append(ValidationIssue(
                        "path_mismatch", location,
                        f"{expected_level} record does not use the {namespace} id namespace",
                    ))
                    continue
            if expected_level == "research" and not (
                isinstance(record.source_locator, str)
                and record.source_locator.startswith("https://")
            ):
                issues.append(ValidationIssue(
                    "source_metadata", location,
                    "research source_locator must be a stable HTTPS locator",
                ))
            unknown_sources = sorted(set(record.source_ids) - source_ids)
            if unknown_sources:
                issues.append(ValidationIssue(
                    "source_metadata", location,
                    "unknown source ids: " + ", ".join(unknown_sources),
                ))

            canonical = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            if record.id in seen_ids:
                issues.append(ValidationIssue(
                    "duplicate_id", location,
                    f"id {record.id!r} first appears at {seen_ids[record.id]}",
                ))
            else:
                seen_ids[record.id] = location
            first_exact_formula = seen_formulas.get(record.formula)
            normalized_formula = re.sub(r"\s+", "", record.formula)
            if (
                first_exact_formula is None
                and normalized_formula in seen_normalized_formulas
            ):
                issues.append(ValidationIssue(
                    "duplicate_formula_normalized", location,
                    "whitespace-normalized formula first appears at "
                    + seen_normalized_formulas[normalized_formula],
                ))
            if first_exact_formula is not None:
                issues.append(ValidationIssue(
                    "duplicate_formula", location,
                    f"formula first appears at {first_exact_formula}",
                ))
            else:
                seen_formulas[record.formula] = location
            seen_normalized_formulas.setdefault(normalized_formula, location)
            if canonical in seen_records:
                issues.append(ValidationIssue(
                    "duplicate_record", location,
                    f"record first appears at {seen_records[canonical]}",
                ))
            else:
                seen_records[canonical] = location

            record_count += 1
            per_file_counts[relative_path] += 1
            domain_counts[record.domain] += 1
            level_key = expected_level or record.source_level or "unknown"
            level_counts[level_key] += 1
            level_domain_counts[(level_key, record.domain)] += 1
            topic_prefix = (
                f"{record.domain}/{record.topic}"
                if level_key == "k12"
                else f"{level_key}/{record.domain}/{record.topic}"
            )
            topic_counts[topic_prefix] += 1
            used_source_counts.update(record.source_ids)
            if expected_level == "research":
                if record.status is not None:
                    research_status_counts[record.status] += 1
                if record.relation_type is not None:
                    research_relation_type_counts[record.relation_type] += 1
                if record.unit_system is not None:
                    research_unit_system_counts[record.unit_system] += 1
                research_source_counts.update(record.source_ids)
                research_topic_sources.setdefault(
                    (record.domain, record.topic), set()
                ).update(record.source_ids)

    if manifest is None:
        manifest = {}
    if not isinstance(manifest, dict):
        issues.append(ValidationIssue(
            "manifest", "manifest.json", "manifest must be a JSON object"
        ))
        manifest = {}

    required_manifest_fields = {
        "corpus_version", "schema_version", "generation_date", "record_count",
        "mathematics_count", "physics_count", "domain_counts", "level_counts",
        "level_domain_counts", "topic_counts", "topic_count", "file_count", "hash_algorithm", "files",
        "source_metadata", "k12_baseline", "pre_research_baseline",
        "research_coverage", "research_record_count", "research_domain_counts",
        "research_status_counts", "research_relation_type_counts",
        "research_unit_system_counts", "research_source_count",
    }
    missing_manifest = sorted(required_manifest_fields - set(manifest))
    if missing_manifest:
        issues.append(ValidationIssue(
            "manifest", "manifest.json",
            "missing fields: " + ", ".join(missing_manifest),
        ))
    if manifest.get("hash_algorithm") != "sha256":
        issues.append(ValidationIssue(
            "manifest", "manifest.json", "hash_algorithm must be sha256"
        ))
    if manifest.get("schema_version") != "1.2.0":
        issues.append(ValidationIssue(
            "manifest", "manifest.json", "unsupported schema_version"
        ))
    generation_date = manifest.get("generation_date")
    try:
        if not isinstance(generation_date, str):
            raise ValueError
        date.fromisoformat(generation_date)
    except ValueError:
        issues.append(ValidationIssue(
            "manifest", "manifest.json", "generation_date must be ISO YYYY-MM-DD"
        ))

    if manifest.get("record_count") != record_count:
        issues.append(ValidationIssue(
            "record_count", "manifest.json",
            f"manifest={manifest.get('record_count')!r}, actual={record_count}",
        ))
    actual_domain_counts = dict(sorted(domain_counts.items()))
    if manifest.get("domain_counts") != actual_domain_counts:
        issues.append(ValidationIssue(
            "record_count", "manifest.json", "domain_counts do not match records"
        ))
    for domain, field in (
        ("mathematics", "mathematics_count"),
        ("physics", "physics_count"),
    ):
        if manifest.get(field) != actual_domain_counts.get(domain, 0):
            issues.append(ValidationIssue(
                "record_count", "manifest.json",
                f"{field} does not match records",
            ))
    actual_level_counts = dict(sorted(level_counts.items()))
    if manifest.get("level_counts") != actual_level_counts:
        issues.append(ValidationIssue(
            "record_count", "manifest.json", "level_counts do not match records"
        ))
    actual_level_domain_counts = {
        level: {
            domain: level_domain_counts.get((level, domain), 0)
            for domain in ("mathematics", "physics")
        }
        for level in sorted(actual_level_counts)
    }
    if manifest.get("level_domain_counts") != actual_level_domain_counts:
        issues.append(ValidationIssue(
            "record_count", "manifest.json",
            "level_domain_counts do not match records",
        ))
    research_level_domains = actual_level_domain_counts.get(
        "research", {"mathematics": 0, "physics": 0}
    )
    research_count = sum(research_level_domains.values())
    research_manifest_values = {
        "research_record_count": research_count,
        "research_domain_counts": research_level_domains,
        "research_status_counts": {
            status: research_status_counts[status]
            for status in sorted(RESEARCH_STATUSES)
        },
        "research_relation_type_counts": dict(
            sorted(research_relation_type_counts.items())
        ),
        "research_unit_system_counts": dict(
            sorted(research_unit_system_counts.items())
        ),
    }
    for field, actual in research_manifest_values.items():
        if manifest.get(field) != actual:
            issues.append(ValidationIssue(
                "record_count", "manifest.json", f"{field} does not match records"
            ))
    actual_topic_counts = dict(sorted(topic_counts.items()))
    if manifest.get("topic_counts") != actual_topic_counts:
        issues.append(ValidationIssue(
            "record_count", "manifest.json", "topic_counts do not match records"
        ))
    if manifest.get("topic_count") != len(actual_topic_counts):
        issues.append(ValidationIssue(
            "record_count", "manifest.json", "topic_count does not match records"
        ))
    if manifest.get("file_count") != len(data_paths):
        issues.append(ValidationIssue(
            "record_count", "manifest.json", "file_count does not match data files"
        ))

    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        issues.append(ValidationIssue(
            "manifest", "manifest.json", "files must be an object"
        ))
        manifest_files = {}

    tracked_actual = set(data_paths) | {
        "sources.json", "schema.json", "k12_baseline.json",
        "pre_research_baseline.json", "research_coverage.json",
    }
    tracked_manifest = set(manifest_files)
    for missing in sorted(tracked_actual - tracked_manifest):
        issues.append(ValidationIssue(
            "checksum", missing, "file is not tracked by manifest"
        ))
    for extra in sorted(tracked_manifest - tracked_actual):
        issues.append(ValidationIssue(
            "checksum", extra, "manifest tracks a missing or unsupported file"
        ))
    for relative_path in sorted(tracked_actual & tracked_manifest):
        if not _safe_manifest_path(relative_path):
            issues.append(ValidationIssue(
                "checksum", relative_path, "unsafe manifest path"
            ))
            continue
        metadata = manifest_files[relative_path]
        if not isinstance(metadata, dict):
            issues.append(ValidationIssue(
                "checksum", relative_path, "file metadata must be an object"
            ))
            continue
        raw_file = file_map[relative_path].read_bytes()
        actual_hash = hashlib.sha256(raw_file).hexdigest()
        if metadata.get("sha256") != actual_hash:
            issues.append(ValidationIssue(
                "checksum", relative_path,
                f"SHA-256 mismatch: expected {metadata.get('sha256')!r}, actual {actual_hash}",
            ))
        if metadata.get("bytes") != len(raw_file):
            issues.append(ValidationIssue(
                "checksum", relative_path,
                f"byte count mismatch: manifest={metadata.get('bytes')!r}, actual={len(raw_file)}",
            ))
        if relative_path.endswith(".jsonl") and metadata.get("records") != per_file_counts[relative_path]:
            issues.append(ValidationIssue(
                "record_count", relative_path,
                f"manifest={metadata.get('records')!r}, actual={per_file_counts[relative_path]}",
            ))

    source_metadata = manifest.get("source_metadata")
    if not isinstance(source_metadata, dict) or source_metadata.get("file") != "sources.json":
        issues.append(ValidationIssue(
            "source_metadata", "manifest.json",
            "source_metadata must point to sources.json",
        ))
    elif sorted(source_metadata.get("source_ids", [])) != sorted(source_ids):
        issues.append(ValidationIssue(
            "source_metadata", "manifest.json",
            "manifest source_ids do not match sources.json",
        ))
    else:
        orphan_source_ids = sorted(source_ids - set(used_source_counts))
        if (
            source_metadata.get("source_count") != len(source_ids)
            or source_metadata.get("used_source_count") != len(used_source_counts)
            or source_metadata.get("orphan_source_count") != len(orphan_source_ids)
            or source_metadata.get("orphan_source_ids") != orphan_source_ids
            or manifest.get("research_source_count") != len(research_source_counts)
        ):
            issues.append(ValidationIssue(
                "source_metadata", "manifest.json",
                "source usage, orphan, or research-source counts do not match records",
            ))

    # The captured v1.0.0 K-12 lock is independently packaged and checked on
    # every validation run. University additions must use level directories;
    # therefore every legacy path and byte hash remains provably immutable.
    if not isinstance(k12_baseline, dict) or not isinstance(k12_baseline.get("files"), dict):
        issues.append(ValidationIssue(
            "k12_preservation", "k12_baseline.json", "invalid baseline lock",
        ))
    else:
        baseline_files = k12_baseline["files"]
        legacy_paths = {
            path for path in data_paths
            if (_corpus_path_scope(path) or (None,))[0] == "k12"
        }
        if legacy_paths != set(baseline_files):
            issues.append(ValidationIssue(
                "k12_preservation", "k12_baseline.json",
                "legacy data-file paths differ from the captured baseline",
            ))
        baseline_hashes: list[str] = []
        baseline_records = 0
        for path in sorted(baseline_files):
            metadata = baseline_files[path]
            if path not in file_map or not isinstance(metadata, dict):
                continue
            actual_hash = hashlib.sha256(file_map[path].read_bytes()).hexdigest()
            expected_hash = metadata.get("sha256")
            if actual_hash != expected_hash:
                issues.append(ValidationIssue(
                    "k12_preservation", path,
                    f"baseline SHA-256 mismatch: expected {expected_hash!r}, actual {actual_hash}",
                ))
            if isinstance(expected_hash, str):
                baseline_hashes.append(expected_hash)
            if isinstance(metadata.get("records"), int):
                baseline_records += metadata["records"]
        ordered_digest = hashlib.sha256("".join(baseline_hashes).encode("ascii")).hexdigest()
        if (
            ordered_digest != k12_baseline.get("ordered_data_hash_digest")
            or ordered_digest != _K12_ORDERED_DATA_HASH_DIGEST
            or k12_baseline.get("manifest_sha256") != _K12_MANIFEST_SHA256
        ):
            issues.append(ValidationIssue(
                "k12_preservation", "k12_baseline.json",
                "captured manifest or ordered data-hash digest does not match the v1.0.0 lock",
            ))
        if (
            baseline_records != k12_baseline.get("record_count")
            or baseline_records != _K12_RECORD_COUNT
            or actual_level_counts.get("k12") != k12_baseline.get("record_count")
        ):
            issues.append(ValidationIssue(
                "k12_preservation", "k12_baseline.json",
                "K-12 record count differs from the captured baseline",
            ))
        baseline_metadata = manifest.get("k12_baseline")
        if (
            not isinstance(baseline_metadata, dict)
            or baseline_metadata.get("file") != "k12_baseline.json"
            or baseline_metadata.get("original_manifest_sha256")
            != k12_baseline.get("manifest_sha256")
            or baseline_metadata.get("ordered_data_hash_digest") != ordered_digest
        ):
            issues.append(ValidationIssue(
                "k12_preservation", "manifest.json",
                "manifest K-12 baseline metadata does not match the lock",
            ))

    # Research additions are independently checked against a lock covering all
    # 65 pre-research data files. Metadata is allowed to evolve, but no existing
    # path, record, identifier, byte, or ordering may change.
    if not isinstance(pre_research_baseline, dict) or not isinstance(
        pre_research_baseline.get("files"), dict
    ):
        issues.append(ValidationIssue(
            "pre_research_preservation", "pre_research_baseline.json",
            "invalid pre-research baseline lock",
        ))
    else:
        locked_files = pre_research_baseline["files"]
        nonresearch_paths = {
            path for path in data_paths if not path.startswith("research/")
        }
        if nonresearch_paths != set(locked_files):
            issues.append(ValidationIssue(
                "pre_research_preservation", "pre_research_baseline.json",
                "pre-research data-file paths differ from the captured baseline",
            ))
        ordered_hashes: list[str] = []
        locked_records = 0
        for path in sorted(locked_files):
            metadata = locked_files[path]
            if path not in file_map or not isinstance(metadata, dict):
                continue
            actual_raw = file_map[path].read_bytes()
            actual_hash = hashlib.sha256(actual_raw).hexdigest()
            expected_hash = metadata.get("sha256")
            if actual_hash != expected_hash or len(actual_raw) != metadata.get("bytes"):
                issues.append(ValidationIssue(
                    "pre_research_preservation", path,
                    "pre-research byte hash or size differs from baseline",
                ))
            if isinstance(expected_hash, str):
                ordered_hashes.append(expected_hash)
            if isinstance(metadata.get("records"), int):
                locked_records += metadata["records"]
        ordered_digest = hashlib.sha256(
            "".join(ordered_hashes).encode("ascii")
        ).hexdigest()
        if (
            len(locked_files) != _PRE_RESEARCH_FILE_COUNT
            or locked_records != _PRE_RESEARCH_RECORD_COUNT
            or ordered_digest != _PRE_RESEARCH_ORDERED_DATA_HASH_DIGEST
            or pre_research_baseline.get("ordered_data_hash_digest")
            != _PRE_RESEARCH_ORDERED_DATA_HASH_DIGEST
            or pre_research_baseline.get("manifest_sha256")
            != _PRE_RESEARCH_MANIFEST_SHA256
        ):
            issues.append(ValidationIssue(
                "pre_research_preservation", "pre_research_baseline.json",
                "pre-research count or digest does not match the v2.0.0 lock",
            ))
        lock_metadata = manifest.get("pre_research_baseline")
        if (
            not isinstance(lock_metadata, dict)
            or lock_metadata.get("file") != "pre_research_baseline.json"
            or lock_metadata.get("record_count") != _PRE_RESEARCH_RECORD_COUNT
            or lock_metadata.get("ordered_data_hash_digest")
            != _PRE_RESEARCH_ORDERED_DATA_HASH_DIGEST
        ):
            issues.append(ValidationIssue(
                "pre_research_preservation", "manifest.json",
                "manifest pre-research lock metadata is invalid",
            ))

    expected_coverage_topics = {
        domain: set(topics)
        for domain, topics in LEVEL_TOPICS["research"].items()
    }
    coverage_domains = (
        research_coverage.get("domains")
        if isinstance(research_coverage, dict)
        else None
    )
    if not isinstance(coverage_domains, dict):
        issues.append(ValidationIssue(
            "research_coverage", "research_coverage.json",
            "coverage matrix must contain a domains object",
        ))
    else:
        for domain, expected_topics in expected_coverage_topics.items():
            entries = coverage_domains.get(domain)
            if not isinstance(entries, dict) or set(entries) != expected_topics:
                issues.append(ValidationIssue(
                    "research_coverage", "research_coverage.json",
                    f"coverage topics do not match research schema for {domain}",
                ))
                continue
            for topic, entry in entries.items():
                actual_records = actual_topic_counts.get(
                    f"research/{domain}/{topic}", 0
                )
                actual_sources = research_topic_sources.get((domain, topic), set())
                if not isinstance(entry, dict) or (
                    entry.get("records") != actual_records
                    or entry.get("source_count") != len(actual_sources)
                    or entry.get("source_ids") != sorted(actual_sources)
                    or entry.get("status") not in {"covered", "partial", "not covered"}
                ):
                    issues.append(ValidationIssue(
                        "research_coverage", f"research_coverage.json:{domain}/{topic}",
                        "coverage counts, sources, or status do not match records",
                    ))
                if actual_records == 0 or entry.get("status") == "not covered":
                    issues.append(ValidationIssue(
                        "research_coverage", f"research_coverage.json:{domain}/{topic}",
                        "requested major research domain is completely uncovered",
                    ))
        coverage_metadata = manifest.get("research_coverage")
        if not isinstance(coverage_metadata, dict) or coverage_metadata.get("file") != "research_coverage.json":
            issues.append(ValidationIssue(
                "research_coverage", "manifest.json",
                "manifest research_coverage metadata is invalid",
            ))

    return ValidationReport(
        root=str(corpus_root),
        record_count=record_count,
        domain_counts=actual_domain_counts,
        level_counts=actual_level_counts,
        level_domain_counts=actual_level_domain_counts,
        topic_counts=actual_topic_counts,
        file_count=len(data_paths),
        total_bytes=total_bytes,
        issues=tuple(sorted(issues)),
    )
