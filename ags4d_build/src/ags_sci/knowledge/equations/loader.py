"""Lazy, read-only loading and retrieval for trusted reference equations."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .schema import EquationRecord, EquationSchemaError

_REFERENCE_PACKAGE = "ags_sci.knowledge.equations.reference"
_MAX_RECORD_BYTES = 16 * 1024
_TOKEN_RE = re.compile(r"[\w\\]+", flags=re.UNICODE)


class CorpusIntegrityError(RuntimeError):
    """Raised when packaged corpus data is unavailable or corrupt."""


def default_reference_root() -> Path:
    """Return the installed/source reference directory.

    Wheels are installed as ordinary files, and returning a concrete path keeps
    the existing public API usable by audit and mirroring tools.
    """
    return Path(__file__).resolve().parent / "reference"


def _path_scope(path: str) -> tuple[str, str, str]:
    """Return (level, domain, topic) without changing legacy K-12 paths."""
    parts = path.split("/")
    if len(parts) == 3 and parts[-1] == "equations.jsonl":
        return "k12", parts[0], parts[1]
    if (
        len(parts) == 4
        and parts[0] in {"undergraduate", "graduate", "research"}
        and parts[-1] == "equations.jsonl"
    ):
        return parts[0], parts[1], parts[2]
    raise CorpusIntegrityError(f"invalid corpus path in manifest: {path}")


def _join(root: Any, relative_path: str) -> Any:
    item = root
    for part in relative_path.split("/"):
        item = item.joinpath(part)
    return item


def _read_manifest(root: Any) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        manifest = json.loads(root.joinpath("manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusIntegrityError("unable to read equation manifest") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        raise CorpusIntegrityError("equation manifest files must be an object")
    paths = tuple(files)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise CorpusIntegrityError("equation manifest paths must be unique and sorted")
    data_paths: list[str] = []
    for path, metadata in files.items():
        if not isinstance(path, str) or not isinstance(metadata, dict):
            raise CorpusIntegrityError("invalid equation manifest file entry")
        digest = metadata.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise CorpusIntegrityError(f"invalid checksum metadata for {path}")
        try:
            raw = _join(root, path).read_bytes()
        except OSError as exc:
            raise CorpusIntegrityError(f"unable to read {path}") from exc
        if hashlib.sha256(raw).hexdigest() != digest:
            raise CorpusIntegrityError(f"checksum mismatch for {path}")
        if path.endswith(".jsonl"):
            _path_scope(path)
            records = metadata.get("records")
            if not isinstance(records, int) or records < 0:
                raise CorpusIntegrityError(f"invalid record count for {path}")
            data_paths.append(path)
    return manifest, tuple(data_paths)


def iter_equation_file(path: str | Path | Any) -> Iterator[EquationRecord]:
    """Parse one inert JSONL file into immutable records without executing data."""
    try:
        raw = path.read_bytes() if hasattr(path, "read_bytes") else Path(path).read_bytes()
    except OSError as exc:
        raise CorpusIntegrityError(f"unable to read equation file: {path}") from exc
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if len(raw_line) > _MAX_RECORD_BYTES:
            raise CorpusIntegrityError(f"record exceeds size limit at line {line_number}")
        try:
            line = raw_line.decode("utf-8", errors="strict")
            record = EquationRecord.from_mapping(json.loads(line))
        except (UnicodeError, json.JSONDecodeError, EquationSchemaError) as exc:
            raise CorpusIntegrityError(
                f"invalid equation record at line {line_number}: {exc}"
            ) from exc
        yield record


class EquationCorpus:
    """Thread-safe reader for immutable, data-only equation references.

    Construction verifies every declared file checksum. Records are parsed only
    when selected, and formula strings remain opaque text at all times.
    """

    def __init__(self, root: str | Path | Any | None = None) -> None:
        self._root = default_reference_root() if root is None else (
            Path(root) if isinstance(root, str) else root
        )
        self._manifest, self._data_paths = _read_manifest(self._root)
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[EquationRecord, ...]] = {}
        self._all: tuple[EquationRecord, ...] | None = None
        self._index: dict[str, EquationRecord] | None = None

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible copy of manifest metadata."""
        return json.loads(json.dumps(self._manifest))

    @property
    def record_count(self) -> int:
        value = self._manifest.get("record_count")
        if not isinstance(value, int):
            raise CorpusIntegrityError("manifest record_count is invalid")
        return value

    def _load_path(self, path: str) -> tuple[EquationRecord, ...]:
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(path)
            if cached is not None:
                return cached
            records = tuple(iter_equation_file(_join(self._root, path)))
            expected = self._manifest["files"][path]["records"]
            if len(records) != expected:
                raise CorpusIntegrityError(f"record count mismatch for {path}")
            self._cache[path] = records
            return records

    def iter_records(
        self,
        *,
        domain: str | None = None,
        topic: str | None = None,
        source_level: str | None = None,
    ) -> Iterator[EquationRecord]:
        """Iterate only files matching the requested level/domain/topic scope."""
        for path in self._data_paths:
            level, path_domain, path_topic = _path_scope(path)
            if domain is not None and path_domain != domain:
                continue
            if topic is not None and path_topic != topic:
                continue
            if source_level is not None and level != source_level:
                continue
            yield from self._load_path(path)

    def all(self) -> tuple[EquationRecord, ...]:
        """Return all records as an immutable tuple, caching the parsed corpus."""
        if self._all is None:
            with self._lock:
                if self._all is None:
                    records = tuple(self.iter_records())
                    if len(records) != self.record_count:
                        raise CorpusIntegrityError("total record count does not match manifest")
                    self._all = records
        return self._all

    def _ensure_index(self) -> dict[str, EquationRecord]:
        if self._index is None:
            with self._lock:
                if self._index is None:
                    index: dict[str, EquationRecord] = {}
                    for record in self.all():
                        if record.id in index:
                            raise CorpusIntegrityError(f"duplicate equation id: {record.id}")
                        index[record.id] = record
                    self._index = index
        return self._index

    def get(self, record_id: str) -> EquationRecord | None:
        """Find a trusted equation by its stable identifier."""
        return self._ensure_index().get(record_id)

    def retrieve(
        self,
        query: str,
        *,
        domain: str | None = None,
        topic: str | None = None,
        source_level: str | None = None,
        limit: int = 10,
    ) -> tuple[EquationRecord, ...]:
        """Perform deterministic text lookup; never interpret formula semantics."""
        if not isinstance(query, str) or not query.strip() or limit <= 0:
            return ()
        needle = query.casefold().strip()
        query_tokens = set(_TOKEN_RE.findall(needle))
        ranked: list[tuple[int, str, EquationRecord]] = []
        for record in self.iter_records(
            domain=domain, topic=topic, source_level=source_level
        ):
            aliases = tuple(alias.casefold() for alias in record.aliases)
            formula = record.formula.casefold()
            subtopic = (record.subtopic or "").casefold()
            symbols = tuple(symbol.casefold() for symbol in record.symbols)
            searchable = " ".join((formula, subtopic, *aliases, *symbols))
            searchable_tokens = set(_TOKEN_RE.findall(searchable))
            if needle == formula:
                score = 1000
            elif needle in aliases:
                score = 900
            elif needle and needle in searchable:
                score = 700 + min(len(needle), 100)
            elif query_tokens and query_tokens <= searchable_tokens:
                score = 500 + len(query_tokens)
            elif query_tokens:
                overlap = len(query_tokens & searchable_tokens)
                if overlap == 0:
                    continue
                score = overlap
            else:
                continue
            ranked.append((-score, record.id, record))
        ranked.sort(key=lambda item: (item[0], item[1]))
        if ranked and -ranked[0][0] >= 900:
            best = ranked[0][0]
            ranked = [item for item in ranked if item[0] == best]
        return tuple(item[2] for item in ranked[:limit])

    def topics(
        self,
        domain: str | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Return unique (domain, topic) pairs for backward compatibility."""
        result = {
            (path_domain, path_topic)
            for path in self._data_paths
            for _, path_domain, path_topic in (_path_scope(path),)
            if domain is None or path_domain == domain
        }
        return tuple(sorted(result))

    def level_topics(self) -> tuple[tuple[str, str, str], ...]:
        """Return every distinct (level, domain, topic) corpus scope."""
        return tuple(sorted({_path_scope(path) for path in self._data_paths}))

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._all = None
            self._index = None


_DEFAULT_CORPUS: EquationCorpus | None = None
_DEFAULT_LOCK = threading.Lock()


def _default_corpus() -> EquationCorpus:
    global _DEFAULT_CORPUS
    if _DEFAULT_CORPUS is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_CORPUS is None:
                _DEFAULT_CORPUS = EquationCorpus()
    return _DEFAULT_CORPUS


def load_equations(
    *,
    domain: str | None = None,
    topic: str | None = None,
    source_level: str | None = None,
) -> tuple[EquationRecord, ...]:
    """Load trusted references with optional inert metadata filters."""
    return tuple(_default_corpus().iter_records(
        domain=domain, topic=topic, source_level=source_level
    ))


def retrieve_equations(
    query: str,
    *,
    domain: str | None = None,
    topic: str | None = None,
    source_level: str | None = None,
    limit: int = 10,
) -> tuple[EquationRecord, ...]:
    """Search trusted references as opaque text."""
    return _default_corpus().retrieve(
        query,
        domain=domain,
        topic=topic,
        source_level=source_level,
        limit=limit,
    )
