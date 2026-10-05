import io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ags_sci.experiment.sandbox import ExecutionSandbox

def main():
 payload='''import numpy as np\ndef one(N,dt,T):\n global _ags_fuel\n L=2*np.pi; x=np.linspace(0,L,N,endpoint=False); dx=L/N; k=np.fft.fftfreq(N,d=dx/(2*np.pi)); hf=-1j*np.sign(k); w=.5*np.sin(x)+.4*np.sin(2*x); steps=int(T/dt); O=[]\n def rhs(q):\n  qh=np.fft.fft(q); H=np.real(np.fft.ifft(hf*qh)); return q*H\n for s in range(steps):\n  k1=rhs(w); k2=rhs(w+.5*dt*k1); k3=rhs(w+.5*dt*k2); k4=rhs(w+dt*k3); w=w+dt*(k1+2*k2+2*k3+k4)/6\n  if s%10==0: O.append(.5*np.mean(w*w))\n t=np.arange(len(O))*dt*10; Od=np.gradient(O,np.median(np.diff(t))); O=np.array(O);\n mask=(O>0)&(Od>0); p=np.polyfit(np.log(O[mask]),np.log(Od[mask]),1)[0]\n T0=O[:,None]; c0=np.linalg.lstsq(T0,Od,rcond=None)[0]; T1=np.column_stack([O,O**2]); c1=np.linalg.lstsq(T1,Od,rcond=None)[0]; s0=np.sum((Od-T0@c0)**2); s1=np.sum((Od-T1@c1)**2); n=len(O); bic0=n*np.log(s0/n)+np.log(n); bic1=n*np.log(s1/n)+2*np.log(n); return [N,dt,T,O[-1],p,c1[1],bic0,bic1,.5*(bic0-bic1)]\nrows=[one(64,.001,4.0),one(128,.0005,4.0),one(256,.00025,4.0)]\nnp.savez('rateconv.npz',rows=np.array(rows)); print('DONE')\n'''
 with ExecutionSandbox(timeout_seconds=25,loop_fuel=40_000_000) as s:
  r=s.execute(payload); print(r.status,r.execution_time_ms,r.error); b=r.artifacts.get('rateconv.npz')
 with io.BytesIO(b) as f:
  import numpy as np; a=np.load(f)['rows']
 for x in a: print('N dt T Oend p c2 BIC0 BIC1 lnBF',*x)
if __name__=='__main__': main()
