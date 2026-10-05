import io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ags_sci.experiment.sandbox import ExecutionSandbox

def main():
 payload='''import numpy as np\nnp.random.seed(42)\nN=128; L=2*np.pi; x=np.linspace(0,L,N,endpoint=False); dx=L/N; dt=0.002; n_steps=1000\nk=np.fft.fftfreq(N,d=dx/(2*np.pi)); ik=1j*k; ksq=-(k*k); hf=-1j*np.sign(k)\nresults=[]\nfor a_param in [0.5,1.0,2.0,4.0]:\n  for nu in [0.0,0.001,0.005,0.02]:\n    omega=2*np.sin(x)+0.8*np.sin(2*x); mx0=np.max(np.abs(omega)); om=[]\n    for step in range(n_steps):\n      oh=np.fft.fft(omega); H=np.real(np.fft.ifft(hf*oh)); stretch=omega*H; Hh=np.fft.fft(H); uh=np.zeros_like(oh); nz=k!=0; uh[nz]=Hh[nz]/ik[nz]; u=np.real(np.fft.ifft(uh)); ox=np.real(np.fft.ifft(ik*oh)); adv=u*ox; diff=nu*np.real(np.fft.ifft(ksq*oh)); omega=omega+dt*(stretch-a_param*adv+diff)\n      if not np.all(np.isfinite(omega)): break\n      if step%20==0: om.append(np.max(np.abs(omega)))\n    om=np.array(om); results.append((a_param,nu,float(om[0]),float(np.max(om)),float(om[-1]) if len(om) else np.nan,len(om)))\nnp.savez('sweep.npz', results=np.array(results))\nprint('SWEEP_DONE')\n'''
 with ExecutionSandbox(timeout_seconds=15,loop_fuel=2_000_000) as s:
  r=s.execute(payload); print(r.status,r.execution_time_ms,r.error); b=r.artifacts.get('sweep.npz')
 with io.BytesIO(b) as f:
  import numpy as np; a=np.load(f)['results']
 print('a nu initial max final n')
 for row in a: print(*row)
if __name__=='__main__': main()
