from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations_with_replacement
import numpy as np

try:  # SciPy is an optional dependency; fall back to local polynomial fitting.
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - exercised only when SciPy is absent
    savgol_filter = None

@dataclass(frozen=True)
class DynamicIdentificationResult:
    target: str
    equations: tuple
    rmse: tuple
    library_terms: tuple
    noise_estimate: float
    hidden_state_used: bool
    score: float
    note: str

class DynamicSystemIdentifier:
    """Sparse identification for ODEs with polynomial + transcendental libraries."""
    def __init__(self, degree=2, threshold=1e-8, ridge=1e-10, include_transcendentals=True):
        self.degree=int(degree); self.threshold=float(threshold); self.ridge=float(ridge); self.include_transcendentals=bool(include_transcendentals)

    @staticmethod
    def derivatives(t, X, smoothing_window=0):
        t=np.asarray(t,float); X=np.asarray(X,float)
        if len(t)<5: raise ValueError('need >=5 trajectory points')
        return np.column_stack([np.gradient(X[:,j],t,edge_order=2) for j in range(X.shape[1])])

    def _features(self,X,names):
        cols=[np.ones(len(X))]; labels=['1']
        for j,n in enumerate(names): cols.append(X[:,j]); labels.append(n)
        for total in range(2,self.degree+1):
            for idx in combinations_with_replacement(range(X.shape[1]),total):
                p=[0]*X.shape[1]
                for i in idx:p[i]+=1
                cols.append(np.prod([X[:,j]**p[j] for j in range(X.shape[1])],axis=0)); labels.append('*'.join(names[j] for j in idx))
        if self.include_transcendentals:
            for j,n in enumerate(names):
                cols.extend([np.sin(X[:,j]),np.cos(X[:,j]),np.exp(-np.clip(np.abs(X[:,j]),0,50)),np.tanh(X[:,j])])
                labels.extend([f'sin({n})',f'cos({n})',f'exp(-abs({n}))',f'tanh({n})'])
        return np.column_stack(cols),labels

    def _stlsq(self,A,b,threshold):
        active=np.ones(A.shape[1],dtype=bool)
        coef=np.zeros(A.shape[1],float)
        for _ in range(8):
            if not np.any(active): break
            c=np.linalg.lstsq(A[:,active],b,rcond=None)[0]
            coef[:]=0.0; coef[active]=c
            new=active & (np.abs(coef)>=threshold)
            if np.array_equal(new,active): break
            active=new
        if not np.any(active): active[np.argmax(np.abs(coef))]=True
        c=np.linalg.lstsq(A[:,active],b,rcond=None)[0]
        coef[:]=0.0; coef[active]=c
        return coef,active

    def fit(self,t,X,names=None):
        t=np.asarray(t,float); X=np.asarray(X,float); names=tuple(names or [f'x{i}' for i in range(X.shape[1])])
        dX=self.derivatives(t,X); Theta,labels=self._features(X,names)
        # Normalize columns for stable sparse thresholding.
        scale=np.linalg.norm(Theta,axis=0); scale[scale==0]=1
        A=Theta/scale
        equations=[]; rms=[]
        for j in range(X.shape[1]):
            coef=np.linalg.solve(A.T@A+self.ridge*np.eye(A.shape[1]),A.T@dX[:,j])/scale
            coef,active=self._stlsq(Theta,dX[:,j],self.threshold)
            active=np.where(np.abs(coef)>self.threshold)[0]
            if not len(active): active=np.array([int(np.argmax(np.abs(coef)))])
            c=coef[active]
            pred=Theta[:,active]@c; err=float(np.sqrt(np.mean((pred-dX[:,j])**2)))
            terms=[]
            for k,cc in zip(active,c): terms.append(f'({cc:.12g})*({labels[k]})')
            equations.append(f'd({names[j]})/dt = ' + ' + '.join(terms)); rms.append(err)
        noise=float(np.median(np.abs(dX-np.median(dX,axis=0))/1.4826))
        return DynamicIdentificationResult('dx/dt',tuple(equations),tuple(rms),tuple(labels),noise,False,float(np.mean(rms)),'finite-difference derivative + sparse regression; numerical evidence only')

    def fit_with_delays(self,t,X,names=None,delays=2):
        X=np.asarray(X,float); names=tuple(names or [f'x{i}' for i in range(X.shape[1])]); d=int(delays)
        if d<1 or len(X)<=d+5: raise ValueError('insufficient trajectory for delay embedding')
        Z=np.column_stack([X[d-k:len(X)-k] for k in range(d+1)])
        zn=tuple(f'{n}(t-{k})' for k in range(d+1) for n in names)
        tt=np.asarray(t)[d:]
        result=self.fit(tt,Z,zn)
        return DynamicIdentificationResult(result.target,result.equations,result.rmse,result.library_terms,result.noise_estimate,True,result.score,'delay-coordinate embedding for partially observed/hidden-state dynamics')

def identify_stlsq(
    Theta: np.ndarray,
    X_dot: np.ndarray,
    threshold: float = 0.1,
    max_iter: int = 20,
) -> np.ndarray:
    """Canonical functional STLSQ coefficient identifier."""
    A = np.asarray(Theta, dtype=float)
    B = np.asarray(X_dot, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[0]:
        raise ValueError("Theta and X_dot must be 2-D arrays with matching rows")
    Xi = np.linalg.lstsq(A, B, rcond=None)[0]
    for _ in range(int(max_iter)):
        previous = Xi.copy()
        small = np.abs(Xi) < float(threshold)
        Xi[small] = 0.0
        for k in range(B.shape[1]):
            active = ~small[:, k]
            if np.any(active):
                Xi[active, k] = np.linalg.lstsq(A[:, active], B[:, k], rcond=None)[0]
        if np.array_equal(previous != 0.0, Xi != 0.0):
            break
    return Xi


@dataclass(frozen=True)
class PDEIdentificationResult:
    equation: str
    rmse: float
    terms: tuple
    grid_points: int
    note: str

class PDEIdentifier:
    """Sparse PDE identification from a scalar field u(x,t)."""
    def __init__(self, include_nonlinear=True, threshold=1e-8): self.include_nonlinear=bool(include_nonlinear); self.threshold=float(threshold)
    def fit(self,x,t,U):
        x=np.asarray(x,float); t=np.asarray(t,float); U=np.asarray(U,float)
        if U.shape!=(len(t),len(x)): raise ValueError('U must have shape (time,space)')
        ut=np.gradient(U,t,axis=0,edge_order=2); ux=np.gradient(U,x,axis=1,edge_order=2); uxx=np.gradient(ux,x,axis=1,edge_order=2)
        cols=[np.ones(U.size),U.ravel(),ux.ravel(),uxx.ravel()]; labels=['1','u','u_x','u_xx']
        if self.include_nonlinear:
            cols += [(U*ux).ravel(),(U**2).ravel(),(U**3).ravel()]; labels += ['u*u_x','u^2','u^3']
        A=np.column_stack(cols); b=ut.ravel(); scale=np.linalg.norm(A,axis=0); scale[scale==0]=1
        c=np.linalg.solve((A/scale).T@(A/scale)+1e-10*np.eye(A.shape[1]),(A/scale).T@b)/scale
        active=np.ones(A.shape[1],dtype=bool)
        for _ in range(8):
            cc=np.linalg.lstsq(A[:,active],b,rcond=None)[0]
            full=np.zeros(A.shape[1]); full[active]=cc
            new=active & (np.abs(full)>=self.threshold)
            if np.array_equal(new,active): break
            active=new
        if not np.any(active): active[np.argmax(np.abs(c))]=True
        idx=np.where(active)[0]
        coef=np.linalg.lstsq(A[:,idx],b,rcond=None)[0]; pred=A[:,idx]@coef; rmse=float(np.sqrt(np.mean((pred-b)**2)))
        terms=tuple(f'({cc:.12g})*({labels[k]})' for k,cc in zip(idx,coef))
        return PDEIdentificationResult('u_t = '+' + '.join(terms),rmse,terms,U.size,'finite-difference spatial/temporal derivatives + sparse regression; no universal proof')

class RobustDifferentiator:
    """Derivative estimation robust to noise and nonuniform sampling."""
    def __init__(self, window=9, polyorder=3, smoothing=True):
        self.window=int(window); self.polyorder=int(polyorder); self.smoothing=bool(smoothing)
        if self.window < 5 or self.window % 2 == 0: raise ValueError('window must be odd and >=5')
        if self.polyorder < 1 or self.polyorder >= self.window: raise ValueError('invalid polyorder')

    @staticmethod
    def _local_poly(t, y, window, order):
        t=np.asarray(t,float); y=np.asarray(y,float); n=len(t); out=np.empty(n)
        half=window//2
        for i in range(n):
            lo=max(0,i-half); hi=min(n,lo+window); lo=max(0,hi-window)
            tt=t[lo:hi]-t[i]
            deg=min(order,len(tt)-1)
            if deg < 1: out[i]=0.0; continue
            # Scale the local coordinate to avoid poorly conditioned Vandermonde systems.
            scale=max(float(np.max(np.abs(tt))),1e-15)
            z=tt/scale
            coef=np.polynomial.polynomial.polyfit(z,y[lo:hi],deg)
            out[i]=coef[1]/scale
        return out

    def derivative(self,t,y):
        t=np.asarray(t,float); y=np.asarray(y,float)
        if t.ndim!=1 or y.ndim!=1 or len(t)!=len(y) or len(t)<self.window: raise ValueError('invalid derivative data')
        dt=np.diff(t)
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or np.any(dt<=0): raise ValueError('t must be finite and strictly increasing')
        uniform=bool(np.max(dt)-np.min(dt) <= max(1e-10,1e-5*float(np.median(dt))))
        if self.smoothing and savgol_filter is not None and uniform:
            return savgol_filter(y,self.window,self.polyorder,deriv=1,delta=float(np.median(dt)),mode='interp')
        return self._local_poly(t,y,self.window,self.polyorder)

@dataclass(frozen=True)
class LatentStateResult:
    delays: int
    embedding_dimension: int
    reconstruction_rank: int
    explained_variance: float
    condition_number: float
    warning: str

class LatentStateDiagnostics:
    """Delay-embedding diagnostics; does not claim unique latent-state recovery."""
    def analyze(self, x, delays=3, components=None):
        x=np.asarray(x,float).reshape(-1)
        d=int(delays)
        if d<1 or len(x)<=d+5: raise ValueError('insufficient scalar trajectory')
        Z=np.column_stack([x[d-k:len(x)-k] for k in range(d+1)])
        U,s,Vt=np.linalg.svd(Z-Z.mean(0),full_matrices=False)
        if components is None: components=min(Z.shape)
        k=max(1,min(int(components),len(s)))
        ev=float(np.sum(s[:k]**2)/max(np.sum(s**2),1e-30))
        rank=int(np.sum(s > max(s[0],1e-30)*1e-8)) if len(s) else 0
        cond=float(s[0]/max(s[-1],1e-30)) if len(s) else float('inf')
        warning='delay embedding is diagnostic; uniqueness requires additional dynamical assumptions'
        return LatentStateResult(d,d+1,rank,ev,cond,warning)