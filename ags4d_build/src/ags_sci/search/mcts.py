"""Causal hypothesis-tree search with UCT and Bayesian/epistemic rewards."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Callable, Generic, Optional, TypeVar
from ags_sci.core.security import validate_simulations, DEFAULT_SECURITY_POLICY, finite_number

T=TypeVar("T")

@dataclass
class MCTSNode(Generic[T]):
    state: T
    parent: Optional["MCTSNode[T]"] = None
    action: Optional[str] = None
    children: list["MCTSNode[T]"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    log_bayes_evidence: float = 0.0
    unexpanded_actions: list[str] = field(default_factory=list)
    @property
    def mean_reward(self): return self.value_sum/self.visits if self.visits else 0.0
    def uct(self, exploration: float = math.sqrt(2.0)):
        if self.visits == 0: return float('inf')
        parent_n=max(1,self.parent.visits if self.parent else self.visits)
        return self.mean_reward + exploration*math.sqrt(math.log(parent_n)/self.visits)

class CausalMCTS:
    def __init__(self, root_state:T, expand:Callable[[T],list[tuple[str,T]]], evaluate:Callable[[T],float], simulations:int=100, exploration:float=1.414):
        self.root=MCTSNode(root_state); self.expand=expand; self.evaluate=evaluate
        self.simulations=validate_simulations(simulations)
        self.exploration=finite_number(exploration, field="exploration")
        if self.exploration < 0:
            raise ValueError("exploration must be non-negative")
    def _select(self,node):
        while node.children and not node.unexpanded_actions:
            node=max(node.children,key=lambda c:c.uct(self.exploration))
        return node
    def _expand(self,node):
        if not node.unexpanded_actions:
            pairs=self.expand(node.state)
            if len(pairs) > DEFAULT_SECURITY_POLICY.max_collection_items:
                raise ValueError("MCTS expansion exceeds security collection limit")
            node.unexpanded_actions=[a for a,_ in pairs]
            for a,s in pairs: node.children.append(MCTSNode(s,node,a))
        if node.children:
            child=next((c for c in node.children if c.visits==0),node.children[0])
            return child
        return node
    def run(self):
        for _ in range(self.simulations):
            node=self._select(self.root); node=self._expand(node)
            reward=finite_number(self.evaluate(node.state), field="MCTS reward")
            while node:
                node.visits+=1; node.value_sum+=reward; node=node.parent
        return max(self.root.children,key=lambda c:c.mean_reward) if self.root.children else self.root
