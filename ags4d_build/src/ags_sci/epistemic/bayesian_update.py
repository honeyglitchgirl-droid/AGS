"""Likelihood-ratio evidence updates; no heuristic weights."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
from ags_sci.core.bayes import gaussian_log_likelihood, posterior_log_odds

@dataclass(frozen=True)
class EvidenceUpdate:
    log_likelihood_h1: float
    log_likelihood_h0: float
    log_bayes_factor: float
    posterior_log_odds: float

class BayesianUpdater:
    def gaussian(self, observations: Sequence[float], prediction_h1: Sequence[float], prediction_h0: Sequence[float], sigma: float, prior_log_odds: float = 0.0) -> EvidenceUpdate:
        l1=gaussian_log_likelihood(observations,prediction_h1,sigma)
        l0=gaussian_log_likelihood(observations,prediction_h0,sigma)
        return EvidenceUpdate(l1,l0,l1-l0,posterior_log_odds(prior_log_odds,l1-l0))
