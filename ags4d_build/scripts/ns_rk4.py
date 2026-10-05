import io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ags_sci.experiment.sandbox import ExecutionSandbox

def main():
 payload='''import numpy as np\ndef run(N,dt,n_steps,a,nu):
 global _ags_fuel\n L=2*np.pi; x=np.linspace(0,L,N,endpoint=False); dx=L/N; k=np.fft.fftfreq(N,d=dx/(2*np.pi)); ik=1j*k; ksq=-(k*k); hf=-1j*np.sign(k); w=2*np.sin(x)+0.8*np.sin(2*x)\n def rhs(q):\n  qh=np.fft.fft(q); H=np.real(np.fft.ifft(hf*qh)); Hh=np.fft.fft(H); uh=np.zeros_like(qh); nz=k!=0; uh[nz]=Hh[nz]/ik[nz]; u=np.real(np.fft.ifft(uh)); qx=np.real(np.fft.ifft(ik*qh)); qxx=np.real(np.fft.ifft(ksq*qh)); return q*H-a*u*qx+nu*qxx\n mx=[]; en=[]\n for s in range(n_steps):\n  k1=rhs(w); k2=rhs(w+0.5*dt*k1); k3=rhs(w+0.5*dt*k2); k4=rhs(w+dt*k3); w=w+dt*(k1+2*k2+2*k3+k4)/6\n  if s%max(1,n_steps//100)==0: mx.append(np.max(np.abs(w))); en.append(.5*np.mean(w*w))\n  if not np.all(np.isfinite(w)): break\n return [N,dt,n_steps,a,nu,len(mx),mx[-1],max(mx),en[-1],max(en)]\nrows=[]\nfor cfg in [(64,0.001,2000,4,0.0),(128,0.0005,4000,4,0.0),(128,0.0005,4000,4,0.005),(128,0.0005,4000,0.5,0.005)]: rows.append(run(*cfg))\nnp.savez('rk4.npz',rows=np.array(rows)); print('RK4_DONE')\n'''
 with ExecutionSandbox(timeout_seconds=30,loop_fuel=20_000_000) as s:
  r=s.execute(payload); print(r.status,r.execution_time_ms,r.error); b=r.artifacts.get('rk4.npz')
 with io.BytesIO(b) as f:
  import numpy as np; a=np.load(f)['rows']
 for x in a: print('N dt steps a nu samples final_max peak_max final_O peak_O',*x)
if __name__=='__main__': main()
