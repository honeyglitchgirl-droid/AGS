import io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ags_sci.experiment.sandbox import ExecutionSandbox

PAYLOAD=r'''
import numpy as np

def H(q):
 n=len(q); k=np.fft.fftfreq(n)*n; return np.real(np.fft.ifft(-1j*np.sign(k)*np.fft.fft(q)))
def DX(q):
 n=len(q); k=np.fft.fftfreq(n)*n; return np.real(np.fft.ifft(1j*k*np.fft.fft(q)))
def U(q):
 n=len(q); k=np.fft.fftfreq(n)*n; hh=np.fft.fft(H(q)); uh=np.zeros(n,dtype=complex); nz=k!=0; uh[nz]=hh[nz]/(1j*k[nz]); return np.real(np.fft.ifft(uh))
def R(q,a): return -a*U(q)*DX(q)+q*H(q)

def core_metrics(q,x):
 n=len(q); m=float(np.max(q)); j=int(np.argmax(q));
 # circular distance from positive peak; half-height contiguous width
 inds=np.arange(n); d=((inds-j+n//2)%n)-n//2; order=np.argsort(np.abs(d)); half=np.abs(d[q[order]]>=m*0.5) if False else None
 # locate nearest crossings by walking left/right from peak
 jj=j; dl=0
 while dl<n//2 and q[(j-dl)%n] >= 0.5*m: dl+=1
 dr=0
 while dr<n//2 and q[(j+dr)%n] >= 0.5*m: dr+=1
 width=(dl+dr)*(2*np.pi/n)
 return m, x[j], width

def run(n,a,tmax=10.0,threshold=500.0):
 x=2*np.pi*np.arange(n)/n-np.pi; q=np.sin(x)+.15*np.sin(3*x)
 t=0.; step=0; rec=[]; crossed=False
 while t<tmax and step<400000:
  u=U(q); h=H(q); umax=max(1e-12,float(np.max(np.abs(u)))); hmax=max(1e-12,float(np.max(np.abs(h))))
  dx=2*np.pi/n; dt=min(.002,.18*dx/umax,.08/hmax)
  if t+dt>tmax: dt=tmax-t
  k1=R(q,a); k2=R(q+.5*dt*k1,a); k3=R(q+.5*dt*k2,a); k4=R(q+dt*k3,a); q += dt*(k1+2*k2+2*k3+k4)/6; t+=dt; step+=1
  if step%10==0 or np.max(q)>=threshold:
   m,xp,w=core_metrics(q,x); y=1.0/max(m,1e-300)
   rec.append((t,m,y,xp,w,.5*np.mean(q*q)))
  if np.max(q)>=threshold: crossed=True; break
 rec=np.array(rec)
 # sliding least-squares line Y(t)=b+s*t; T=-b/s, robustly from last windows
 Ts=[]
 for W in [10,20,40,80]:
  if len(rec)>=W:
   tt=rec[-W:,0]; yy=rec[-W:,2]; A=np.column_stack([tt,np.ones(W)]); s,b=np.linalg.lstsq(A,yy,rcond=None)[0]
   if s<0: Ts.append((W,-b/s,s))
 # local effective exponent p from dM/dt ~ M^p using log slopes over last 100 points
 p=np.nan
 if len(rec)>=40:
  tt=rec[:,0]; mm=rec[:,1]; lm=np.log(mm); ldt=np.log(np.maximum(np.gradient(mm,tt),1e-300));
  lo=max(1,len(mm)-100); hi=len(mm)-2
  p=float(np.polyfit(lm[lo:hi],ldt[lo:hi],1)[0])
 return np.array([n,a,t,float(np.max(q)),int(crossed),step,len(rec),rec[-1,3],rec[-1,4],rec[-1,5],Ts[-1][1] if Ts else np.nan,p]),rec

outs=[]
for a in [0.85,0.88,0.90,0.92]:
 for n in [128,256]:
  r,rec=run(n,a)
  outs.append(r)
  np.savez('diag_%g_%d.npz'%(a,n),rec=rec)
  print('ROW',*r)
np.savez('summary.npz',rows=np.array(outs))
'''
def main():
    with ExecutionSandbox(timeout_seconds=120,loop_fuel=500_000_000) as sb:
        r=sb.execute(PAYLOAD)
    print('STATUS',r.status,'MS',r.execution_time_ms,'ERR',r.error)
    if r.status=='SUCCESS':
        data=np.load(io.BytesIO(r.artifacts['summary.npz']))
        print(data['rows'])
        for k,v in r.artifacts.items(): print('ART',k,len(v))

if __name__=='__main__':
    main()
