"""Deterministic Bayesian primitives used by the epistemic layer."""
from __future__ import annotations
import math
from typing import Iterable, Sequence

def logsumexp(values: Iterable[float]) -> float:
    xs = list(values)
    if not xs: return float('-inf')
    m = max(xs)
    if math.isinf(m): return m
    return m + math.log(sum(math.exp(x-m) for x in xs))

def bernoulli_log_evidence(successes: int, trials: int, p: float) -> float:
    if trials < 0 or not 0 <= successes <= trials or not 0 < p < 1:
        raise ValueError("invalid Bernoulli evidence")
    return successes*math.log(p) + (trials-successes)*math.log1p(-p)

def bayes_factor_from_log_evidence(log_e1: float, log_e0: float) -> float:
    d = float(log_e1)-float(log_e0)
    return math.exp(d) if d < 709 else float('inf')

def posterior_log_odds(prior_log_odds: float, log_bayes_factor: float) -> float:
    return float(prior_log_odds) + float(log_bayes_factor)

def gaussian_log_likelihood(y: Sequence[float], mean: Sequence[float], sigma: float) -> float:
    if sigma <= 0 or len(y) != len(mean): raise ValueError("invalid Gaussian likelihood")
    c = -math.log(sigma*math.sqrt(2*math.pi))
    return sum(c - ((a-b)/sigma)**2/2 for a,b in zip(y,mean))


from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ModelEvidenceComparison:
    model_name: str
    null_name: str
    sse_model: float
    sse_null: float
    k_model: int
    k_null: int
    bic_model: float
    bic_null: float
    log_bayes_factor_10: float
    bayes_factor_10: float

def compute_grounded_bayes_factor(y_true: np.ndarray, y_pred_model: np.ndarray,
                                  y_pred_null: np.ndarray, k_model: int, k_null: int) -> ModelEvidenceComparison:
    """Compute a BIC/Schwarz-approximated BF10 for nested/comparable predictive models."""
    y=np.asarray(y_true,float).reshape(-1); m=np.asarray(y_pred_model,float).reshape(-1); n=np.asarray(y_pred_null,float).reshape(-1)
    if len(y)==0 or len(m)!=len(y) or len(n)!=len(y): raise ValueError("prediction lengths must match")
    if k_model<0 or k_null<0: raise ValueError("parameter counts must be nonnegative")
    N=len(y); eps=np.finfo(float).tiny
    sse_m1=float(np.sum((y-m)**2)); sse_m0=float(np.sum((y-n)**2))
    bic_m1=N*np.log(max(sse_m1/N,eps))+int(k_model)*np.log(N)
    bic_m0=N*np.log(max(sse_m0/N,eps))+int(k_null)*np.log(N)
    log_bf=-0.5*(bic_m1-bic_m0)
    bf=float(np.exp(np.clip(log_bf,-700,700)))
    return ModelEvidenceComparison("Identified SCM/STLSQ","Null baseline",sse_m1,sse_m0,int(k_model),int(k_null),bic_m1,bic_m0,log_bf,bf)
