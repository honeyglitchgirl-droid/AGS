import io, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ags_sci.experiment.sandbox import ExecutionSandbox

PAYLOAD = r'''
import numpy as np

def hilbert(q):
    n=len(q); k=np.fft.fftfreq(n)*n
    return np.real(np.fft.ifft((-1j*np.sign(k))*np.fft.fft(q)))

def spectral_derivative(q):
    n=len(q); k=np.fft.fftfreq(n)*n
    return np.real(np.fft.ifft((1j*k)*np.fft.fft(q)))

def velocity(q):
    # u_x = H(q), mean(u)=0; odd q -> even Hq -> odd u and u(0)=0.
    n=len(q); k=np.fft.fftfreq(n)*n
    h=hilbert(q)
    uh=np.fft.fft(h)
    ik=1j*k
    uh[0]=0
    nz=(k!=0)
    uh[nz]=uh[nz]/ik[nz]
    return np.real(np.fft.ifft(uh))

def rhs(q,a):
    u=velocity(q); qx=spectral_derivative(q); h=hilbert(q)
    r=-a*u*qx + q*h
    # 2/3 de-aliasing of the nonlinear RHS
    n=len(q); kk=np.fft.fftfreq(n)*n; rh=np.fft.fft(r); rh[np.abs(kk)>n/3]=0
    return np.real(np.fft.ifft(rh))

def integrate(n,a,amp,alpha=1.0,tmax=8.0,dtmax=2e-3,threshold=1e3):
    # Odd profile. alpha=1 is smooth; alpha<1 intentionally introduces a
    # low-regularity cusp through sign(sin x)|sin x|^alpha.
    x=2*np.pi*np.arange(n)/n-np.pi
    base=np.sign(np.sin(x))*np.abs(np.sin(x))**alpha
    q=amp*base + 0.15*np.sin(3*x)
    t=0.0; peak=0.0; samples=[]; crossed=False
    step=0
    while t<tmax and step<200000:
        u=velocity(q); h=hilbert(q)
        umax=max(1e-12,float(np.max(np.abs(u))))
        hmax=max(1e-12,float(np.max(np.abs(h))))
        dx=2*np.pi/n
        dt=min(dtmax,0.25*dx/umax,0.1/hmax)
        if t+dt>tmax: dt=tmax-t
        k1=rhs(q,a); k2=rhs(q+0.5*dt*k1,a); k3=rhs(q+0.5*dt*k2,a); k4=rhs(q+dt*k3,a)
        q=q+dt*(k1+2*k2+2*k3+k4)/6; t+=dt; step+=1
        m=float(np.max(np.abs(q))); peak=max(peak,m)
        if step%20==0 or m>=threshold:
            qh=np.fft.fft(q); high=np.sum(np.abs(qh[np.abs(np.fft.fftfreq(n)*n)>n/3])**2); total=np.sum(np.abs(qh)**2)
            samples.append((t,m,0.5*float(np.mean(q*q)),float(np.max(np.abs(hilbert(q)))),high/max(total,1e-300)))
        if m>=threshold:
            crossed=True; break
    return {'n':n,'a':a,'alpha':alpha,'t':t,'peak':peak,'crossed':crossed,'steps':step,'samples':np.array(samples)}

configs=[]
for a in [1.0]:
    for alpha in [1.0]:
        configs.append((32,a,1.0,alpha))
rows=[]
for cfg in configs:
    r=integrate(*cfg,tmax=2.5,dtmax=0.003,threshold=100)
    s=r['samples']
    rows.append([r['n'],r['a'],r['alpha'],r['t'],r['peak'],int(r['crossed']),r['steps'],s[-1,2],s[-1,3],s[-1,4]])
# convergence on the most promising combinations
conv=[]
for a,alpha in [(0.5,0.5),(1.0,0.5)]:
    for n in [64,128,256]:
        r=integrate(n,a,1.0,alpha,tmax=3.0,dtmax=0.003,threshold=50)
        conv.append([n,a,alpha,r['t'],r['peak'],int(r['crossed']),r['steps']])
np.savez('gdg_frontier.npz',rows=np.array(rows),conv=np.array(conv))
print('GDG_FRONTIER_DONE')
'''

def main():
    with ExecutionSandbox(timeout_seconds=45, loop_fuel=200_000_000) as sb:
        r=sb.execute(PAYLOAD)
    print('sandbox:',r.status,'time_ms=',round(r.execution_time_ms,2),'error=',r.error)
    if r.status!='SUCCESS': return 1
    data=np.load(io.BytesIO(r.artifacts['gdg_frontier.npz']))
    print('coarse rows:')
    for row in data['rows']: print(' '.join(f'{x:.8g}' for x in row))
    print('convergence rows:')
    for row in data['conv']: print(' '.join(f'{x:.8g}' for x in row))
    return 0

if __name__=='__main__':
    import numpy as np
    raise SystemExit(main())
