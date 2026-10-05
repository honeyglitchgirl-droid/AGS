"""Minimal structural causal model representation and intervention semantics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping
from .causal import CausalMechanism

@dataclass(frozen=True)
class StructuralCausalModel:
    mechanisms: tuple[CausalMechanism,...]
    functions: Mapping[str,Callable[[Mapping[str,float]],float]]

    def intervention(self, values: Mapping[str,float]) -> dict[str,float]:
        """Apply do(X=x) and propagate deterministic structural mechanisms in DAG order."""
        nodes=set(); parents={}
        for m in self.mechanisms:
            nodes.update((m.source,m.target)); parents.setdefault(m.target,[]).append(m.source)
        order=[]; indeg={n:0 for n in nodes}; adj={n:[] for n in nodes}
        for m in self.mechanisms:
            indeg[m.target]+=1; adj[m.source].append(m.target)
        q=[n for n,d in indeg.items() if d==0]
        while q:
            n=q.pop(); order.append(n)
            for b in adj[n]:
                indeg[b]-=1
                if indeg[b]==0:q.append(b)
        state=dict(values)
        for n in order:
            if n in values: continue
            fn=self.functions.get(n)
            if fn is not None: state[n]=float(fn(state))
        return state
