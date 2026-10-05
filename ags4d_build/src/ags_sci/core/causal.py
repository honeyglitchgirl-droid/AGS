"""Strict causal and experimental contracts for AGS-Sci.

This module is deliberately deterministic and side-effect free. LLMs may propose
instances of these schemas, but validation and graph invariants are enforced here.
"""
from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class HypothesisStatus(str, Enum):
    PROPOSED = "PROPOSED"
    RUNNING = "RUNNING"
    SUPPORTED = "SUPPORTED"
    FALSIFIED = "FALSIFIED"

class CausalMechanism(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    mechanism_type: str = Field(..., min_length=1)
    equation: Optional[str] = None
    priors: Dict[str, Tuple[float, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def no_self_edge(self):
        if self.source == self.target:
            raise ValueError("causal self-edge is not allowed")
        for name, (mu, sigma) in self.priors.items():
            if sigma <= 0:
                raise ValueError(f"prior std must be > 0 for {name}")
        return self

class ExperimentProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")
    control_variables: Dict[str, float] = Field(default_factory=dict)
    interventions: Dict[str, float] = Field(default_factory=dict)
    sample_size: int = Field(default=100, ge=10)
    measured_variables: List[str] = Field(default_factory=list, min_length=1)
    falsification_threshold: float = Field(default=0.05, gt=0, lt=1)

class EpistemicHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    causal_dag: List[CausalMechanism] = Field(default_factory=list)
    protocol: ExperimentProtocol
    prior_log_odds: float = 0.0
    posterior_log_odds: Optional[float] = None
    status: HypothesisStatus = HypothesisStatus.PROPOSED

    @model_validator(mode="after")
    def validate_dag(self):
        nodes = set()
        edges = []
        for edge in self.causal_dag:
            nodes.update((edge.source, edge.target)); edges.append((edge.source, edge.target))
        # Kahn topological check: reject directed cycles.
        indeg = {n: 0 for n in nodes}
        adj = {n: [] for n in nodes}
        for a, b in edges:
            indeg[b] += 1; adj[a].append(b)
        q = [n for n, d in indeg.items() if d == 0]; seen = 0
        while q:
            n = q.pop(); seen += 1
            for b in adj[n]:
                indeg[b] -= 1
                if indeg[b] == 0: q.append(b)
        if seen != len(nodes):
            raise ValueError("causal_dag must be acyclic")
        return self
