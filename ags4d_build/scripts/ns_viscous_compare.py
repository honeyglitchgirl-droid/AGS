import io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ags_sci.experiment.sandbox import ExecutionSandbox

def main():
 payload='''import numpy as np\ndef run(nu):\n global _ags_fuel\n N=128; dt=.0005; n_steps=10000; L=2*np.pi; x=np.linspace(0,L,N,endpoint=False); dx=L/N; k=np.fft.fftfreq(N,d=dx/(2*np.pi)); ik=1j*k; ksq=-(k*k); hf=-1j*np.sign(k); w=.5*np.sin(x)+.4*np.sin(2*x); vals=[]\n def rhs(q):\n  qh=np.fft.fft(q); H=np.real(np.fft.ifft(hf*qh)); qxx=np.real(np.fft.ifft(ksq*qh)); return q*H+nu*qxx\n for s in range(n_steps):\n  k1=rhs(w); k2=rhs(w+.5*dt*k1); k3=rhs(w+.5*dt*k2); k4=rhs(w+dt*k3); w=w+dt*(k1+2*k2+2*k3+k4)/6\n  if s%1000==0: vals.append([s*dt,float(np.max(np.abs(w))),float(.5*np.mean(w*w))])\n  if not np.all(np.isfinite(w)): break\n return np.array(vals)\nall=[]\nfor nu in [0.0,0.0001,0.001,0.005,0.01,0.05]:\n a=run(nu); all.append(a); print('NU',nu,'final',a[-1], 'peak',a[:,1].max())\n# pad not needed; save each under keys
np.savez('visc.npz',**{f'nu{i}':a for i,a in enumerate(all)})\n'''
 with ExecutionSandbox(timeout_seconds=25,loop_fuel=30_000_000) as s:
  r=s.execute(payload); print(r.status,r.execution_time_ms,r.error); b=r.artifacts.get('visc.npz')
 with io.BytesIO(b) as f:
  import numpy as np; d=np.load(f)
  for i,nu in enumerate([0,.0001,.001,.005,.01,.05]):
   a=d[f'nu{i}']; print('nu',nu,'last',a[-1].tolist(),'peak',float(a[:,1].max()))
if __name__=='__main__': main()
