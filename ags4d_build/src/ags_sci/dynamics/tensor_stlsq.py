"""Geometry-aware candidate dictionaries for sparse PDE identification."""
from __future__ import annotations
import numpy as np
from ags_sci.fields.operators import SpectralEngineND
from ags_sci.fields.geometry import Metric, contract, tensor_trace

class InvariantLibraryBuilderND:
    """Build scalar invariants without component-monomial explosion.

    The default invariants are Euclidean unless a metric is supplied. A Lorentzian
    metric is therefore explicit rather than inferred from D=4.
    """
    def __init__(self, engine: SpectralEngineND, metric: Metric|None=None):
        self.engine=engine
        self.ndim=engine.ndim
        self.metric=metric or Metric(np.eye(self.ndim))

    def build_scalar_library(self, phi):
        phi=np.asarray(phi)
        if phi.shape!=self.engine.shape: raise ValueError("phi shape mismatch")
        grad=self.engine.gradient(phi)
        grad_norm=contract(np.asarray(grad),np.asarray(grad),self.metric)
        lap=self.engine.laplacian(phi)
        terms=[phi,phi**2,phi**3,grad_norm,lap,self.engine.laplacian(lap)]
        names=["phi","phi^2","phi^3","grad(phi)^2","Delta(phi)","Delta^2(phi)"]
        return np.column_stack([x.reshape(-1) for x in terms]),names

    def build_vector_advection(self,u):
        if len(u)!=self.ndim: raise ValueError("wrong vector dimension")
        out=[np.zeros_like(u[0],dtype=float) for _ in range(self.ndim)]
        for j in range(self.ndim):
            gu=self.engine.gradient(u[j])
            for i in range(self.ndim): out[j]+=u[i]*gu[i]
        return out

    def gauge_field_strength(self,A):
        """Return F=dA for an Abelian 1-form A."""
        if len(A)!=self.ndim: raise ValueError("A dimension mismatch")
        G=np.asarray([self.engine.gradient(a) for a in A])
        # G[nu,mu,...] = partial_mu A_nu.  Therefore
        # F[mu,nu] = partial_mu A_nu - partial_nu A_mu.
        return G.swapaxes(0,1) - G

    def tensor_trace(self,T):
        return tensor_trace(T,self.metric)
