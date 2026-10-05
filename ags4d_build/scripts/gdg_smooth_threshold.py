import io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ags_sci.experiment.sandbox import ExecutionSandbox
PAYLOAD=r'''
import numpy as np
def H(q):
 n=len(q); k=np.fft.fftfreq(n)*n; return np.real(np.fft.ifft(-1j*np.sign(k)*np.fft.fft(q)))
def dx(q):
 n=len(q); k=np.fft.fftfreq(n)*n; return np.real(np.fft.ifft(1j*k*np.fft.fft(q)))
def U(q):
 n=len(q); k=np.fft.fftfreq(n)*n; hh=np.fft.fft(H(q)); uh=np.zeros(n,dtype=complex); nz=k!=0; uh[nz]=hh[nz]/(1j*k[nz]); return np.real(np.fft.ifft(uh))
def R(q,a):
 r=-a*U(q)*dx(q)+q*H(q); n=len(q); k=np.fft.fftfreq(n)*n; rh=np.fft.fft(r); rh[np.abs(k)>n/3]=0; return np.real(np.fft.ifft(rh))
def run(n,a):
 x=2*np.pi*np.arange(n)/n-np.pi; q=np.sin(x)+.15*np.sin(3*x); t=0.; peak=0.; threshold=100.; step=0
 while t<8 and step<200000:
  u=U(q); h=H(q); umax=max(1e-12,float(np.max(np.abs(u)))); hmax=max(1e-12,float(np.max(np.abs(h))))
  dt=min(.002,.2*(2*np.pi/n)/umax,.05/hmax)
  if t+dt>8: dt=8-t
  k1=R(q,a); k2=R(q+dt*k1/2,a); k3=R(q+dt*k2/2,a); k4=R(q+dt*k3,a); q += dt*(k1+2*k2+2*k3+k4)/6; t+=dt; step+=1
  m=float(np.max(np.abs(q))); peak=max(peak,m)
  if not np.isfinite(m): return [n,a,t,peak,step,0]
  if m>=threshold: return [n,a,t,peak,step,1]
 return [n,a,t,peak,step,0]
rows=[[n,a,*run(n,a)[2:]] for a in [.85,.9,.925,.95] for n in [128,256]]
np.savez('gdg_smooth_threshold.npz',rows=np.array(rows)); print(rows)
'''
def main():
 with ExecutionSandbox(timeout_seconds=35,loop_fuel=120_000_000) as sb: r=sb.execute(PAYLOAD)
 print(r.status,r.execution_time_ms,r.error)
 if r.artifacts.get('gdg_smooth_threshold.npz'):
  import numpy as np; print(np.load(io.BytesIO(r.artifacts['gdg_smooth_threshold.npz']))['rows'])
if __name__=='__main__': main()
