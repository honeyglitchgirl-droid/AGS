from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ags_sci.fields.operators import SpectralEngineND
from ags_sci.fields.geometry import Metric, box_scalar
from ags_sci.fields.referee import ResolutionRefereeND
from ags_sci.dynamics.identification import identify_stlsq


def stlsq(A,b,thr=1e-6):
    X=identify_stlsq(np.asarray(A),np.asarray(b).reshape(-1,1),threshold=thr)
    return X[:,0]


def benchmark_ns(N=128, nu=.01):
    eng=SpectralEngineND((N,N),(2*np.pi,2*np.pi),dealias='spherical')
    x=np.arange(N)*2*np.pi/N; y=x
    X,Y=np.meshgrid(x,y,indexing='ij')
    psi=(np.sin(X)+.7*np.cos(2*X-Y)+.45*np.sin(Y+X)+.2*np.cos(3*Y-2*X))
    u=eng.gradient(psi)
    # u = (psi_y, -psi_x) so divergence-free
    u=[u[1],-u[0]]
    adv=eng.leray_project([u[0]*eng.gradient(u[0])[0]+u[1]*eng.gradient(u[0])[1],
                           u[0]*eng.gradient(u[1])[0]+u[1]*eng.gradient(u[1])[1]])
    lap=[eng.laplacian(c) for c in u]
    rhs=eng.leray_project([-adv[0]+nu*lap[0],-adv[1]+nu*lap[1]])
    bad=[(u[0]**2+u[1]**2)*u[0],(u[0]**2+u[1]**2)*u[1]]
    graddiv=eng.gradient(eng.divergence(u))
    # Stack vector components into one regression problem.
    A=np.column_stack([np.concatenate([adv[0].ravel(),adv[1].ravel()]),np.concatenate([lap[0].ravel(),lap[1].ravel()]),np.concatenate([bad[0].ravel(),bad[1].ravel()]),np.concatenate([graddiv[0].ravel(),graddiv[1].ravel()])])
    b=np.concatenate([rhs[0].ravel(),rhs[1].ravel()])
    c=stlsq(A,b,thr=1e-5)
    div=np.max(np.abs(eng.divergence(u)))
    proj=eng.leray_project([np.sin(X)+.1*np.cos(Y),np.cos(Y)+.2*np.sin(X)])
    divp=np.max(np.abs(eng.divergence(proj)))
    tail=max(eng.spectral_tail_ratio(u[0]),eng.spectral_tail_ratio(u[1]))
    # Expected first two coefficients are -1 and +nu.
    return {'N':N,'coeff_adv':float(c[0]),'coeff_lap':float(c[1]),'coeff_speed2u':float(c[2]),'coeff_graddiv':float(c[3]),'div_initial':float(div),'div_projected':float(divp),'tail':float(tail),'pass':abs(c[0]+1)<2e-3 and abs(c[1]-nu)<2e-4 and abs(c[2])<2e-4 and abs(c[3])<2e-4 and divp<1e-10}



if __name__=='__main__':
    t=time.perf_counter(); a=benchmark_ns();
    print('2D_NS',a)
    print('TOTAL_SECONDS',time.perf_counter()-t)
