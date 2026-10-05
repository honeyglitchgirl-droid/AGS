"""Small, executable mathematical primitives for the 25-domain roadmap.

This module is deliberately a *foundation layer*: each domain has a deterministic,
unit-testable primitive, not a claim of a complete research-grade solver.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
import numpy as np

DOMAINS = (
    "general_relativity", "quantum_field_theory", "chaos", "fluid_dynamics",
    "boltzmann_entropy", "seismology", "lotka_volterra", "plasma_physics",
    "machine_learning", "llm_attention", "quantum_computing", "information_theory",
    "optimization", "fourier_transform", "differential_geometry", "topology",
    "number_theory", "category_theory", "ito_calculus", "quantitative_finance",
    "network_science", "game_theory", "string_theory", "plasma_confinement",
    "computational_complexity",
)


def _finite(a):
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite input")
    return a

# 1. General relativity: Christoffel symbols from a metric and its coordinate derivatives.
#    `metric_derivatives[i, j, k]` is d(metric[j, k])/d(coordinate i); i.e. the
#    derivative index comes FIRST. Passing derivative-last instead silently
#    yields wrong geometry, so the convention is asserted here.
def christoffel_symbols(metric, metric_derivatives):
    g = _finite(metric)
    dg = _finite(metric_derivatives)
    if g.shape != (4, 4) or dg.shape != (4, 4, 4):
        raise ValueError("metric=(4,4), metric_derivatives=(4,4,4)")
    gi = np.linalg.inv(g)
    G = np.zeros((4,4,4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                G[a,b,c] = 0.5*sum(gi[a,d]*(dg[b,c,d] + dg[c,b,d] - dg[d,b,c]) for d in range(4))
    return G

# 2. QFT: Euclidean free scalar-field action on a periodic lattice.
def scalar_field_action(phi, mass=1.0, spacing=1.0):
    p = _finite(phi)
    if mass < 0 or spacing <= 0: raise ValueError("mass>=0, spacing>0")
    grads = [np.roll(p,-1,axis=i)-p for i in range(p.ndim)]
    return float(0.5*sum(np.sum(g*g) for g in grads)*spacing**(p.ndim-2) + 0.5*mass**2*np.sum(p*p)*spacing**p.ndim)

# 3. Chaos: logistic map and finite-time Lyapunov estimate.
def logistic_map(x, r, steps=1000):
    x=float(x); r=float(r)
    if not 0 < x < 1: raise ValueError("x must be in (0,1)")
    vals=[]
    for _ in range(int(steps)):
        x=r*x*(1-x); vals.append(x)
    return np.asarray(vals)

def logistic_lyapunov(x, r, steps=5000, discard=500):
    vals=logistic_map(x,r,steps)
    z=vals[int(discard):]
    if len(z)==0: raise ValueError("discard too large")
    return float(np.mean(np.log(np.maximum(np.abs(r*(1-2*z)),1e-300))))

# 4. Fluid dynamics: 1D inviscid Burgers RHS.
def burgers_rhs(u, dx, viscosity=0.0):
    u=_finite(u)
    if dx<=0 or viscosity<0: raise ValueError("dx>0, viscosity>=0")
    ux=(np.roll(u,-1)-np.roll(u,1))/(2*dx)
    rhs=-u*ux
    if viscosity: rhs += viscosity*(np.roll(u,-1)-2*u+np.roll(u,1))/dx**2
    return rhs

# 5. Boltzmann/Shannon entropy of a normalized discrete distribution.
def entropy(p):
    p=_finite(p)
    if np.any(p<0) or p.sum() <= 0: raise ValueError("invalid probabilities")
    p=p/p.sum(); nz=p>0
    return float(-np.sum(p[nz]*np.log(p[nz])))

# 6. Seismology: Ricker wavelet.
def ricker(t, frequency):
    t=_finite(t); f=float(frequency)
    if f<=0: raise ValueError("frequency>0")
    a=(pi*f*t)**2
    return (1-2*a)*np.exp(-a)

# 7. Lotka-Volterra.
def lotka_volterra(state, alpha=1.0,beta=0.1,delta=0.1,gamma=1.0):
    x,y=_finite(state)
    return np.array([alpha*x-beta*x*y, delta*x*y-gamma*y], dtype=float)

# 8. Plasma: cold-plasma electron plasma frequency.
def plasma_frequency(n_e, charge=1.602176634e-19, mass=9.1093837139e-31, eps0=8.8541878128e-12):
    if n_e<0: raise ValueError("density>=0")
    return float(np.sqrt(n_e*charge**2/(mass*eps0)))

# 9. Machine learning: one linear-regression gradient step.
def linear_regression_step(X, y, w, lr=1e-2):
    X=_finite(X); y=_finite(y); w=_finite(w)
    if X.ndim!=2 or y.shape!=(X.shape[0],) or w.shape!=(X.shape[1],): raise ValueError("shape mismatch")
    grad=2*X.T@(X@w-y)/X.shape[0]
    return w-lr*grad

# 10. LLM attention: scaled dot-product attention.
def scaled_dot_product_attention(Q,K,V,mask=None):
    Q=_finite(Q); K=_finite(K); V=_finite(V)
    if Q.ndim!=2 or K.ndim!=2 or V.ndim!=2 or Q.shape[1]!=K.shape[1] or K.shape[0]!=V.shape[0]: raise ValueError("shape mismatch")
    logits=Q@K.T/sqrt(Q.shape[1])
    if mask is not None:
        mask=np.asarray(mask,dtype=bool)
        if mask.shape!=logits.shape: raise ValueError("mask shape must match the score matrix")
        # A query row that attends to nothing would subtract -inf from -inf and
        # silently return NaN, so reject it explicitly instead.
        if not np.all(mask.any(axis=1)): raise ValueError("each query row must attend to at least one key")
        logits=np.where(mask, logits, -np.inf)
    logits=logits-np.max(logits,axis=1,keepdims=True)
    w=np.exp(logits); w/=np.sum(w,axis=1,keepdims=True)
    return w@V, w

# 11. Quantum computing: single-qubit state normalization and expectation.
def qubit_expectation(state, pauli):
    z=np.asarray(state,dtype=complex)
    if z.shape!=(2,): raise ValueError("state must have shape (2,)")
    n=np.vdot(z,z).real
    if n<=0: raise ValueError("zero state")
    z=z/np.sqrt(n)
    P={"X":np.array([[0,1],[1,0]],complex),"Y":np.array([[0,-1j],[1j,0]],complex),"Z":np.diag([1,-1])}.get(pauli)
    if P is None: raise ValueError("pauli must be X/Y/Z")
    return float(np.real(np.vdot(z,P@z)))

# 12. Information theory: mutual information from a joint probability table.
def mutual_information(joint):
    p=_finite(joint)
    if p.ndim!=2 or np.any(p<0) or p.sum()<=0: raise ValueError("invalid joint distribution")
    p=p/p.sum(); px=p.sum(axis=1); py=p.sum(axis=0); total=0.0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p[i,j]>0: total += p[i,j]*log(p[i,j]/(px[i]*py[j]))
    return float(total)

# 13. Optimization: Armijo-free gradient descent primitive.
def gradient_descent_quadratic(A,b,x,lr=1e-2):
    A=_finite(A); b=_finite(b); x=_finite(x)
    if A.shape[0]!=A.shape[1] or b.shape!=x.shape or A.shape[0]!=x.size: raise ValueError("shape mismatch")
    return x-lr*(A@x-b)

# 14. Fourier transform wrapper.
def fft_transform(x):
    return np.fft.fftn(_finite(x))

# 15. Differential geometry: inverse of a (4,4) metric.
def metric_inverse(metric):
    g=_finite(metric)
    if g.shape!=(4,4): raise ValueError("4x4 metric required")
    return np.linalg.inv(g)

# 16. Topology: Euler characteristic of a finite simplicial complex from simplex counts.
def euler_characteristic(simplex_counts):
    c=[int(v) for v in simplex_counts]
    if any(v<0 for v in c): raise ValueError("counts must be nonnegative")
    return int(sum((1 if k%2==0 else -1)*v for k,v in enumerate(c)))

# 17. Number theory: deterministic sieve.
def primes_upto(n):
    n=int(n)
    if n<0: raise ValueError("n>=0")
    sieve=np.ones(n+1,dtype=bool); sieve[:2]=False
    for p in range(2,int(np.sqrt(n))+1):
        if sieve[p]: sieve[p*p:n+1:p]=False
    return np.flatnonzero(sieve).tolist()

# 18. Category theory: finite-function composition.
def compose_functions(f,g):
    return lambda x: f(g(x))

# 19. Ito calculus: Euler-Maruyama one step.
def ito_euler_step(x, drift, diffusion, dt, dW):
    if dt<0: raise ValueError("dt>=0")
    return float(x + drift(x)*dt + diffusion(x)*dW)

# 20. Quant finance: Black-Scholes call price.
def black_scholes_call(S,K,T,r,sigma):
    if min(S,K,T,sigma)<=0: raise ValueError("S,K,T,sigma must be positive")
    d1=(log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*sqrt(T)); d2=d1-sigma*sqrt(T)
    N=lambda z: 0.5*(1+erf(z/sqrt(2)))
    return float(S*N(d1)-K*exp(-r*T)*N(d2))

# 21. Network science: PageRank iteration.
def pagerank(adjacency, damping=0.85, iterations=100):
    A=_finite(adjacency)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or not 0<damping<1: raise ValueError("invalid graph")
    n=A.shape[0]; out=A.sum(axis=1); M=np.zeros_like(A)
    for i in range(n): M[i]=A[i]/out[i] if out[i]>0 else 1/n
    p=np.ones(n)/n
    for _ in range(iterations): p=(1-damping)/n+damping*(M.T@p)
    return p/p.sum()

# 22. Game theory: pure Nash equilibria in a bimatrix game.
def pure_nash(payoff_a,payoff_b):
    A=_finite(payoff_a); B=_finite(payoff_b)
    if A.shape!=B.shape or A.ndim!=2: raise ValueError("matching payoff matrices required")
    best_a=np.isclose(A,A.max(axis=0,keepdims=True)); best_b=np.isclose(B,B.max(axis=1,keepdims=True))
    return [(i,j) for i in range(A.shape[0]) for j in range(A.shape[1]) if best_a[i,j] and best_b[i,j]]

# 23. String theory toy primitive: Nambu-Goto energy of a discretized string.
def string_energy(points, tension=1.0):
    P=_finite(points)
    if P.ndim!=2 or P.shape[0]<2 or tension<0: raise ValueError("points and tension invalid")
    return float(tension*np.sum(np.linalg.norm(np.diff(P,axis=0),axis=1)))

# 24. Plasma confinement: Larmor radius.
def larmor_radius(mass, velocity_perp, charge, magnetic_field):
    if min(mass,velocity_perp,abs(charge),magnetic_field)<=0: raise ValueError("positive physical parameters required")
    return float(mass*velocity_perp/(abs(charge)*magnetic_field))

# 25. Computational complexity: empirical log-log scaling exponent.
def scaling_exponent(sizes,times):
    n=_finite(sizes); t=_finite(times)
    if n.shape!=t.shape or np.any(n<=0) or np.any(t<=0): raise ValueError("positive matching arrays required")
    return float(np.polyfit(np.log(n),np.log(t),1)[0])

@dataclass(frozen=True)
class DomainPrimitive:
    name: str
    callable_name: str

# Explicit domain -> implementing primitive mapping. This used to be derived from
# DOMAINS itself, which advertised 24 of 25 callables that do not exist in this
# module (only "lotka_volterra" happened to match). Keeping the mapping explicit
# makes the registry safe to introspect.
_PRIMITIVES: tuple[tuple[str, str], ...] = (
    ("general_relativity", "christoffel_symbols"),
    ("quantum_field_theory", "scalar_field_action"),
    ("chaos", "logistic_map"),
    ("fluid_dynamics", "burgers_rhs"),
    ("boltzmann_entropy", "entropy"),
    ("seismology", "ricker"),
    ("lotka_volterra", "lotka_volterra"),
    ("plasma_physics", "plasma_frequency"),
    ("machine_learning", "linear_regression_step"),
    ("llm_attention", "scaled_dot_product_attention"),
    ("quantum_computing", "qubit_expectation"),
    ("information_theory", "mutual_information"),
    ("optimization", "gradient_descent_quadratic"),
    ("fourier_transform", "fft_transform"),
    ("differential_geometry", "metric_inverse"),
    ("topology", "euler_characteristic"),
    ("number_theory", "primes_upto"),
    ("category_theory", "compose_functions"),
    ("ito_calculus", "ito_euler_step"),
    ("quantitative_finance", "black_scholes_call"),
    ("network_science", "pagerank"),
    ("game_theory", "pure_nash"),
    ("string_theory", "string_energy"),
    ("plasma_confinement", "larmor_radius"),
    ("computational_complexity", "scaling_exponent"),
)

REGISTRY = tuple(DomainPrimitive(name, fn) for name, fn in _PRIMITIVES)

def resolve(domain: str):
    """Return the implementing callable for a domain, or raise KeyError."""
    for entry in REGISTRY:
        if entry.name == domain:
            fn = globals().get(entry.callable_name)
            if not callable(fn):
                raise AttributeError(f"primitive {entry.callable_name!r} for {domain!r} is not callable")
            return fn
    raise KeyError(f"unknown domain: {domain!r}")
