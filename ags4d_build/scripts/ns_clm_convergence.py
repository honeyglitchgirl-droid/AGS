import io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ags_sci.experiment.sandbox import ExecutionSandbox

def main():
 payload='''import numpy as np\ndef run(N,dt,n_steps,amp1,amp2):\n global _ags_fuel\n L=2*np.pi; x=np.linspace(0,L,N,endpoint=False); dx=L/N; k=np.fft.fftfreq(N,d=dx/(2*np.pi)); ik=1j*k; hf=-1j*np.sign(k); w=amp1*np.sin(x)+amp2*np.sin(2*x)\n def rhs(q):\n  qh=np.fft.fft(q); H=np.real(np.fft.ifft(hf*qh)); return q*H\n threshold=1e4; peak=0\n for s in range(n_steps):\n  k1=rhs(w); k2=rhs(w+.5*dt*k1); k3=rhs(w+.5*dt*k2); k4=rhs(w+dt*k3); w=w+dt*(k1+2*k2+2*k3+k4)/6\n  peak=max(peak,float(np.max(np.abs(w))))\n  if peak>=threshold: return [N,dt,s*dt,peak,.5*np.mean(w*w)]\n return [N,dt,n_steps*dt,peak,.5*np.mean(w*w)]\nrows=[]\nfor cfg in [(64,.001,10000,.5,.4),(128,.0005,20000,.5,.4),(256,.00025,40000,.5,.4)]: rows.append(run(*cfg))\nnp.savez('clmconv.npz',rows=np.array(rows)); print('CLM_CONVERGENCE_DONE')\n'''
 with ExecutionSandbox(timeout_seconds=40,loop_fuel=80_000_000) as s:
  r=s.execute(payload); print(r.status,r.execution_time_ms,r.error); b=r.artifacts.get('clmconv.npz')
 with io.BytesIO(b) as f:
  import numpy as np; a=np.load(f)['rows']
 for x in a: print('N dt threshold_time peak enstrophy',*x)
if __name__=='__main__': main()
