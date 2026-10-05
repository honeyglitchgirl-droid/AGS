"""AGS-Sci security, integrity, and stability policy primitives.

This module defines fail-closed limits for untrusted AI input, research
orchestration, provenance, and persistent knowledge.  It deliberately keeps
policy separate from scientific algorithms so plugins cannot silently weaken
security controls.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SecurityPolicy:
    max_text_chars: int = 20_000
    max_payload_keys: int = 128
    max_collection_items: int = 2_000
    max_string_chars: int = 8_000
    max_mcts_simulations: int = 10_000
    max_recursion_depth: int = 64
    max_source_chars: int = 200_000
    max_ast_nodes: int = 50_000
    max_stdout_chars: int = 100_000
    max_artifact_bytes: int = 8 * 1024 * 1024
    max_artifacts: int = 64
    max_result_bytes: int = 8 * 1024 * 1024

    def validate(self) -> None:
        for name in ("max_text_chars", "max_payload_keys", "max_collection_items",
                     "max_string_chars", "max_mcts_simulations", "max_recursion_depth",
                     "max_source_chars", "max_ast_nodes", "max_stdout_chars",
                     "max_artifact_bytes", "max_artifacts", "max_result_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"invalid security policy: {name}")


DEFAULT_SECURITY_POLICY = SecurityPolicy()


class SecurityViolation(ValueError):
    """Raised when an input crosses an AGS security policy boundary."""


def bounded_text(value: str, *, field: str = "text", policy: SecurityPolicy = DEFAULT_SECURITY_POLICY) -> str:
    if not isinstance(value, str):
        raise SecurityViolation(f"{field} must be text")
    if len(value) > policy.max_text_chars:
        raise SecurityViolation(f"{field} exceeds {policy.max_text_chars} characters")
    return value


def validate_payload(value: Any, *, policy: SecurityPolicy = DEFAULT_SECURITY_POLICY,
                    _depth: int = 0) -> Any:
    """Validate JSON-like AI/plugin data; reject executable/non-JSON objects."""
    if _depth > policy.max_recursion_depth:
        raise SecurityViolation("payload nesting exceeds security limit")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SecurityViolation("non-finite numeric payload")
        return value
    if isinstance(value, str):
        return bounded_text(value, field="payload string", policy=policy)
    if isinstance(value, (list, tuple)):
        if len(value) > policy.max_collection_items:
            raise SecurityViolation("payload collection exceeds item limit")
        return [validate_payload(x, policy=policy, _depth=_depth + 1) for x in value]
    if isinstance(value, Mapping):
        if len(value) > policy.max_payload_keys:
            raise SecurityViolation("payload object exceeds key limit")
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str) or len(k) > 256:
                raise SecurityViolation("invalid payload key")
            out[k] = validate_payload(v, policy=policy, _depth=_depth + 1)
        return out
    raise SecurityViolation(f"unsupported payload type: {type(value).__name__}")


def validate_simulations(simulations: int, *, policy: SecurityPolicy = DEFAULT_SECURITY_POLICY) -> int:
    if isinstance(simulations, bool) or not isinstance(simulations, int):
        raise SecurityViolation("simulations must be an integer")
    if simulations <= 0 or simulations > policy.max_mcts_simulations:
        raise SecurityViolation(f"simulations must be in 1..{policy.max_mcts_simulations}")
    return simulations


def canonical_json(value: Any) -> bytes:
    safe = validate_payload(value)
    return json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def finite_number(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SecurityViolation(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise SecurityViolation(f"{field} must be finite")
    return result


def safe_identifier(value: str, *, field: str = "identifier") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        raise SecurityViolation(f"invalid {field}")
    return value
