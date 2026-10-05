"""Fail-closed gate for AGS self-evolution proposals.

Self-evolution may propose changes, but it cannot directly mutate security
policy, sandbox boundaries, release metadata, or protected scientific gates.
The proposal remains an ephemeral candidate until external validation passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from .security import SecurityViolation, safe_identifier, bounded_text, sha256_json

PROTECTED_PATHS = (
    "src/ags_sci/core/security.py",
    "src/ags_sci/core/evolution.py",
    "src/ags_sci/experiment/sandbox.py",
    "pyproject.toml",
)

@dataclass(frozen=True)
class EvolutionProposal:
    proposal_id: str
    changed_paths: tuple[str, ...]
    rationale: str
    baseline_sha256: str
    test_command: str = "python -m pytest -q"

    def validate(self) -> None:
        safe_identifier(self.proposal_id, field="proposal_id")
        bounded_text(self.rationale, field="rationale")
        if not self.baseline_sha256 or len(self.baseline_sha256) != 64:
            raise SecurityViolation("invalid baseline integrity hash")
        if not self.changed_paths:
            raise SecurityViolation("evolution proposal has no changes")
        for raw in self.changed_paths:
            p = PurePosixPath(raw)
            if p.is_absolute() or ".." in p.parts:
                raise SecurityViolation("path traversal in evolution proposal")
            if raw in PROTECTED_PATHS:
                raise SecurityViolation(f"protected path cannot be self-modified: {raw}")

    @property
    def fingerprint(self) -> str:
        self.validate()
        return sha256_json({"proposal_id": self.proposal_id, "paths": self.changed_paths, "rationale": self.rationale, "baseline": self.baseline_sha256})
