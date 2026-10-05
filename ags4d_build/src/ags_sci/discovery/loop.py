"""Active-inference research orchestration over validated causal hypotheses."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable
from ags_sci.core import EpistemicHypothesis
from ags_sci.epistemic import PolicyEvaluation, BOED
from ags_sci.search import CausalMCTS
from ags_sci.core.security import validate_simulations

@dataclass(frozen=True)
class ResearchDecision:
    selected_policy: str
    reason: str
    expected_free_energy: float | None = None

class ResearchLoop:
    def __init__(self, triad, experiment_runner: Callable, evidence_update: Callable):
        self.triad=triad; self.experiment_runner=experiment_runner; self.evidence_update=evidence_update
    def choose_policy(self, policies: Iterable[PolicyEvaluation]) -> ResearchDecision:
        ranked=BOED.rank(list(policies))
        if not ranked: raise ValueError("no policies available")
        p=ranked[0]
        return ResearchDecision(p.policy,"minimum expected free energy",p.expected_free_energy)
    def search(self, root_state, expand, evaluate, simulations=100):
        simulations = validate_simulations(simulations)
        return CausalMCTS(root_state,expand,evaluate,simulations=simulations).run()
