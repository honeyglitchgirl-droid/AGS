import io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ags_sci.experiment.sandbox import ExecutionSandbox

def main():
 payload='''import numpy as np\nnp.random.seed(42)\nconfigs=[(64,0.004,750),(128,0.002,1500),(256,0.001,3000)]\nrows=[]\nfor N,dt,n_steps in configs:\n L=2*np.pi; x=np.linspace(0,L,N,endpoint=False); dx=L/N; k=np.fft.fftfreq(N,d=dx/(2*np.pi)); ik=1j*k; ksq=-(k*k); hf=-1j*np.sign(k); omega=2*np.sin(x)+0.8*np.sin(2*x); mx=[]; en=[]\n for step in range(n_steps):\n  oh=np.fft.fft(omega); H=np.real(np.fft.ifft(hf*oh)); Hh=np.fft.fft(H); uh=np.zeros_like(oh); nz=k!=0; uh[nz]=Hh[nz]/ik[nz]; u=np.real(np.fft.ifft(uh)); ox=np.real(np.fft.ifft(ik*oh)); diff=np.real(np.fft.ifft(ksq*oh)); omega=omega+dt*(omega*H-4.0*u*ox+0.0*diff)\n  if step%max(1,n_steps//100)==0: mx.append(np.max(np.abs(omega))); en.append(.5*np.mean(omega**2))\n  if not np.all(np.isfinite(omega)): break\n rows.append((N,dt,n_steps,len(mx),float(mx[-1]),float(max(mx)),float(en[-1]),float(max(en))))\nnp.savez('conv.npz',rows=np.array(rows))\nprint('CONVERGENCE_DONE')\n'''
 with ExecutionSandbox(timeout_seconds=20,loop_fuel=10_000_000) as s:
  r=s.execute(payload); print(r.status,r.execution_time_ms,r.error); b=r.artifacts.get('conv.npz')
 with io.BytesIO(b) as f:
  import numpy as np; a=np.load(f)['rows']
 for x in a: print('N dt steps samples final_max peak_max final_O peak_O',*x)
if __name__=='__main__': main()
