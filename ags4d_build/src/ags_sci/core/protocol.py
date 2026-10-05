"""Machine-readable LLM research protocol."""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, ConfigDict, Field
from .causal import EpistemicHypothesis
from .security import finite_number, safe_identifier

class HypothesisProposal(BaseModel):
    model_config=ConfigDict(extra="forbid")
    hypothesis: EpistemicHypothesis
    assumptions: List[str]=Field(default_factory=list)
    measurable_variables: List[str]=Field(default_factory=list)
    confounders: List[str]=Field(default_factory=list)
    confidence: float=Field(ge=0,le=1)

class Critique(BaseModel):
    model_config=ConfigDict(extra="forbid")
    hypothesis_id: str
    confounders: List[str]=Field(default_factory=list)
    selection_biases: List[str]=Field(default_factory=list)
    statistical_failures: List[str]=Field(default_factory=list)
    survives: bool
    rationale: str=Field(min_length=1)

class MethodProtocol(BaseModel):
    model_config=ConfigDict(extra="forbid")
    hypothesis_id: str
    minimal_steps: List[str]=Field(min_length=1)
    controls: List[str]=Field(default_factory=list)
    interventions: List[str]=Field(default_factory=list)
    measurements: List[str]=Field(min_length=1)
    stopping_rule: str=Field(min_length=1)


from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json

@dataclass(frozen=True)
class ProvenanceBundle:
    experiment_id: str
    random_seed: int
    simulation_code_sha256: str
    input_data_sha256: str
    dictionary_spec_sha256: str
    ags_version: str
    timestamp_utc: str
    measured_rmse: float
    bic_score: float
    log_bayes_factor_10: float
    active_terms_count: int

    @staticmethod
    def sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def create(cls, experiment_id, random_seed, simulation_code, input_data, dictionary_spec, ags_version, measured_rmse, bic_score, log_bayes_factor_10, active_terms_count):
        def digest(x):
            if isinstance(x, bytes): return cls.sha256_bytes(x)
            if isinstance(x, str): return cls.sha256_bytes(x.encode())
            return cls.sha256_bytes(json.dumps(x, sort_keys=True, default=str).encode())
        return cls(experiment_id,int(random_seed),digest(simulation_code),digest(input_data),digest(dictionary_spec),str(ags_version),datetime.now(timezone.utc).isoformat(),float(measured_rmse),float(bic_score),float(log_bayes_factor_10),int(active_terms_count))
