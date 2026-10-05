"""Active inference and Bayesian optimal experimental design.

The implementation accepts discrete predictive state/observation distributions.
EFE is decomposed into epistemic information gain and pragmatic preference value.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

def _norm(xs: Mapping[str,float]) -> dict[str,float]:
    if not xs: raise ValueError("distribution cannot be empty")
    vals = {}
    for k, v in xs.items():
        if not isinstance(k, str):
            raise ValueError("distribution keys must be strings")
        p = float(v)
        if not math.isfinite(p) or p < 0:
            raise ValueError("distribution probabilities must be finite and non-negative")
        vals[k] = p
    total=sum(vals.values())
    if not math.isfinite(total) or total <= 0: raise ValueError("distribution mass must be finite and positive")
    return {k: v/total for k,v in vals.items()}

def entropy(q: Mapping[str,float]) -> float:
    q=_norm(q)
    return -sum(p*math.log(p) for p in q.values() if p>0)

def information_gain(prior: Mapping[str,float], posterior_by_obs: Mapping[str,Mapping[str,float]], obs_probs: Mapping[str,float]) -> float:
    prior=_norm(prior); po=_norm(obs_probs)
    return sum(po[o] * kl_divergence(_norm(posterior_by_obs[o]), prior) for o in po)

def kl_divergence(p: Mapping[str,float], q: Mapping[str,float]) -> float:
    p=_norm(p); q=_norm(q)
    terms = []
    for x, px in p.items():
        if px <= 0:
            continue
        qx = q.get(x, 0.0)
        if qx <= 0:
            return float("inf")
        terms.append(px * math.log(px / qx))
    return float(sum(terms))

def pragmatic_value(obs_probs: Mapping[str,float], preferences: Mapping[str,float]) -> float:
    p=_norm(obs_probs)
    return sum(p[o]*float(preferences.get(o, 0.0)) for o in p)

def expected_free_energy(prior_state: Mapping[str,float], posterior_by_obs: Mapping[str,Mapping[str,float]], obs_probs: Mapping[str,float], preferences: Mapping[str,float]) -> float:
    """G = - information gain - expected log preference, represented here by supplied utility."""
    return -information_gain(prior_state, posterior_by_obs, obs_probs) - pragmatic_value(obs_probs, preferences)

@dataclass(frozen=True)
class PolicyEvaluation:
    policy: str
    expected_free_energy: float
    information_gain: float
    pragmatic_value: float

class BOED:
    @staticmethod
    def rank(policies: Sequence[PolicyEvaluation]) -> list[PolicyEvaluation]:
        return sorted(policies, key=lambda x: x.expected_free_energy)
