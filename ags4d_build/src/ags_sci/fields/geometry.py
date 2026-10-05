"""Metric-aware tensor contractions and differential forms.

These routines provide the algebraic/geometric layer. A numerical discretization
still supplies coordinate derivatives; coordinate covariance is enforced in the
contraction layer rather than asserted from array shapes.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Metric:
    """Constant non-singular metric in a chosen coordinate basis."""
    components: np.ndarray

    def __post_init__(self):
        g=np.asarray(self.components,dtype=float)
        if g.ndim!=2 or g.shape[0]!=g.shape[1]:
            raise ValueError("metric must be square")
        if abs(np.linalg.det(g)) < 1e-14:
            raise ValueError("metric must be non-singular")
        object.__setattr__(self,"components",g)
        object.__setattr__(self,"inverse",np.linalg.inv(g))

    @property
    def dimension(self): return self.components.shape[0]

def lower_index(v, metric: Metric):
    v=np.asarray(v)
    return np.einsum("ij,j...->i...",metric.components,v)

def raise_index(v, metric: Metric):
    v=np.asarray(v)
    return np.einsum("ij,j...->i...",metric.inverse,v)

def contract(a,b,metric: Metric):
    """Metric contraction g(a,b), supporting trailing field axes."""
    aa=np.asarray(a); bb=np.asarray(b)
    return np.einsum("ij,i...,j...->...",metric.components,aa,bb)

def differential_form_derivative(form, derivative_components):
    """Exterior derivative dA.

    `form` has leading axis corresponding to form indices. `derivative_components`
    has axes [derivative_index, form_index, ...]. Returns antisymmetrized d(form).
    """
    A=np.asarray(form)
    dA=np.asarray(derivative_components)
    if A.ndim < 1 or dA.shape[0] != A.shape[0] + 0:
        # dA[mu,nu,...] for a 1-form; generic q-form support is intentionally
        # restricted to the robust 1-form -> 2-form operation.
        raise ValueError("currently supports a 1-form represented by leading component axis")
    if A.shape[0] != dA.shape[1]:
        raise ValueError("derivative tensor shape incompatible with form")
    return dA - np.swapaxes(dA,0,1)

def exterior_derivative_1form(A, gradients):
    """F_{mu nu}=partial_mu A_nu-partial_nu A_mu."""
    G=np.asarray(gradients)
    A=np.asarray(A)
    if G.shape[0]!=A.shape[0] or G.shape[1]!=A.shape[0]:
        raise ValueError("gradients must have shape (D,D,...) for a 1-form")
    return G-np.swapaxes(G,0,1)

def tensor_trace(T, metric: Metric):
    T=np.asarray(T)
    if T.shape[0]!=metric.dimension or T.shape[1]!=metric.dimension:
        raise ValueError("tensor does not match metric dimension")
    return np.einsum("ij,ij...->...",metric.inverse,T)

def tensor_double_contract(A,B,metric: Metric):
    return np.einsum("ia,jb,ij...,ab...->...",metric.inverse,metric.inverse,A,B)


def metric_gradient_norm(gradient_components, metric: Metric):
    """g^{mu nu} partial_mu phi partial_nu phi for a scalar gradient."""
    ginv=metric.inverse
    return np.einsum("ij,i...,j...->...",ginv,np.asarray(gradient_components),np.asarray(gradient_components))

def box_scalar(gradient_components_of_gradient, metric: Metric):
    """Constant-metric d'Alembert/Laplace-Beltrami operator on a scalar.

    Input has shape (mu,nu,...), where the first derivative index is the
    coordinate differentiated by the second gradient operation. For a constant
    metric this is g^{mu nu} partial_mu partial_nu phi.
    """
    H=np.asarray(gradient_components_of_gradient)
    if H.shape[0]!=metric.dimension or H.shape[1]!=metric.dimension:
        raise ValueError("Hessian does not match metric dimension")
    return np.einsum("ij,ij...->...",metric.inverse,H)

def divergence_covariant_rank2(T, gradient_of_rows, metric: Metric):
    """Constant-metric covariant divergence of a covariant rank-2 tensor.

    For constant coordinates (zero Christoffel symbols),
    (nabla^mu T_{mu nu}) = g^{mu alpha} partial_alpha T_{mu nu}.
    ``gradient_of_rows[mu][alpha]`` is partial_alpha T_{mu nu}.
    """
    T=np.asarray(T)
    D=metric.dimension
    if T.shape[:2]!=(D,D): raise ValueError("T must have leading shape (D,D)")
    out=[]
    for nu in range(D):
        val=0.0
        for mu in range(D):
            for alpha in range(D):
                val = val + metric.inverse[mu,alpha]*gradient_of_rows[mu][alpha,nu]
        out.append(val)
    return np.asarray(out)
