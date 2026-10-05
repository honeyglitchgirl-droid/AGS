"""Dimension-agnostic resolution gates."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class ResolutionDecision:
    accepted: bool
    tail_ratio: float
    core_ratio: float|None
    reasons: tuple[str,...]

class ResolutionRefereeND:
    def __init__(self, tail_reject=1e-3, tail_strict=1e-4, min_core_cells=3.0):
        self.tail_reject=float(tail_reject)
        self.tail_strict=float(tail_strict)
        self.min_core_cells=float(min_core_cells)

    def judge(self, tail_ratio, core_ratio=None):
        r=float(tail_ratio); reasons=[]
        if r>self.tail_reject: reasons.append("spectral_tail_rejected")
        if core_ratio is not None and float(core_ratio)<self.min_core_cells:
            reasons.append("core_underresolved")
        return ResolutionDecision(not reasons,r,None if core_ratio is None else float(core_ratio),tuple(reasons))
