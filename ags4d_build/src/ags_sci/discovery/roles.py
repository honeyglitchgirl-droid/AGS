"""Adversarial triad contracts: proposer, skeptic, methodologist.

Adapters may implement these roles; this module contains only orchestration contracts.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from ags_sci.core.protocol import HypothesisProposal, Critique, MethodProtocol

class Proposer(Protocol):
    def propose(self, problem: str) -> HypothesisProposal: ...
class Skeptic(Protocol):
    def critique(self, proposal: HypothesisProposal) -> Critique: ...
class Methodologist(Protocol):
    def design(self, proposal: HypothesisProposal, critique: Critique) -> MethodProtocol: ...

@dataclass(frozen=True)
class TriadResult:
    proposal: HypothesisProposal
    critique: Critique
    protocol: MethodProtocol | None

class AdversarialTriad:
    def __init__(self, proposer: Proposer, skeptic: Skeptic, methodologist: Methodologist):
        self.proposer=proposer; self.skeptic=skeptic; self.methodologist=methodologist
    def run(self, problem: str) -> TriadResult:
        p=self.proposer.propose(problem)
        c=self.skeptic.critique(p)
        return TriadResult(p,c,self.methodologist.design(p,c) if c.survives else None)
