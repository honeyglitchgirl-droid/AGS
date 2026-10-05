import io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ags_sci.experiment.sandbox import ExecutionSandbox

def main():
 payload='''import numpy as np\ndef run(nu):\n global _ags_fuel\n N=128; dt=.0005; n_steps=16000; L=2*np.pi; x=np.linspace(0,L,N,endpoint=False); dx=L/N; k=np.fft.fftfreq(N,d=dx/(2*np.pi)); ik=1j*k; ksq=-(k*k); hf=-1j*np.sign(k); w=.5*np.sin(x)+.4*np.sin(2*x)\n def rhs(q):\n  qh=np.fft.fft(q); H=np.real(np.fft.ifft(hf*qh)); qxx=np.real(np.fft.ifft(ksq*qh)); return q*H+nu*qxx\n peak=0\n for s in range(n_steps):\n  k1=rhs(w); k2=rhs(w+.5*dt*k1); k3=rhs(w+.5*dt*k2); k4=rhs(w+dt*k3); w=w+dt*(k1+2*k2+2*k3+k4)/6\n  peak=max(peak,float(np.max(np.abs(w))))\n  if peak>1000: return [nu,s*dt,peak,.5*np.mean(w*w)]\n return [nu,n_steps*dt,peak,.5*np.mean(w*w)]\nrows=[run(nu) for nu in [0,.0001,.001,.005,.01,.05]]\nnp.savez('vt.npz',rows=np.array(rows)); print('DONE')\n'''
 with ExecutionSandbox(timeout_seconds=35,loop_fuel=50_000_000) as s:
  r=s.execute(payload); print(r.status,r.execution_time_ms,r.error); b=r.artifacts.get('vt.npz')
 with io.BytesIO(b) as f:
  import numpy as np; a=np.load(f)['rows']
 for x in a: print('nu time peak O',*x)
if __name__=='__main__': main()
