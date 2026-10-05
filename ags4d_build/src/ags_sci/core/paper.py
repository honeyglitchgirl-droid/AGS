"""Strict scientific manuscript compilation from verified AGS artifacts."""
from __future__ import annotations
import re
from dataclasses import asdict
from typing import Mapping
from .protocol import ProvenanceBundle

_SLOT = re.compile(r"\{\{\s*([a-zA-Z_][\w.]*)\s*\}\}")
_METRIC_LITERAL = re.compile(r"(?i)\b(?:bayes\s*factor|log\s*bayes\s*factor|rmse|bic|active\s+terms?)\b[^\n]{0,80}?\b(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?)\b")

class PaperCompilationError(ValueError):
    pass

class VerifiedPaperCompiler:
    """Compile prose only when quantitative claims are injected from provenance."""
    def __init__(self, provenance: ProvenanceBundle):
        self.provenance = provenance
        self.facts = {f"metrics.{k}": v for k, v in asdict(provenance).items()
                      if isinstance(v, (int, float))}

    def compile_section(self, template: str) -> str:
        if _METRIC_LITERAL.search(template):
            raise PaperCompilationError("quantitative metric must use a verified {{ metrics.* }} slot")
        slots = list(_SLOT.finditer(template))
        rendered = template
        for m in slots:
            key=m.group(1)
            if key not in self.facts:
                raise PaperCompilationError(f"unverified or unknown template slot: {key}")
            rendered=rendered.replace(m.group(0), str(self.facts[key]))
        if "{{" in rendered or "}}" in rendered:
            raise PaperCompilationError("unresolved template slot")
        return rendered
