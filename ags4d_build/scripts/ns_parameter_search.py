import io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ags_sci.experiment.sandbox import ExecutionSandbox

def main():
 payload='''import numpy as np\ndef run(a,nu,amp1,amp2):\n global _ags_fuel\n N=64; dt=0.001; n_steps=5000; L=2*np.pi; x=np.linspace(0,L,N,endpoint=False); dx=L/N; k=np.fft.fftfreq(N,d=dx/(2*np.pi)); ik=1j*k; ksq=-(k*k); hf=-1j*np.sign(k); w=amp1*np.sin(x)+amp2*np.sin(2*x)\n def rhs(q):\n  qh=np.fft.fft(q); H=np.real(np.fft.ifft(hf*qh)); Hh=np.fft.fft(H); uh=np.zeros_like(qh); nz=k!=0; uh[nz]=Hh[nz]/ik[nz]; u=np.real(np.fft.ifft(uh)); qx=np.real(np.fft.ifft(ik*qh)); qxx=np.real(np.fft.ifft(ksq*qh)); return q*H-a*u*qx+nu*qxx\n peak=0\n for s in range(n_steps):\n  k1=rhs(w); k2=rhs(w+.5*dt*k1); k3=rhs(w+.5*dt*k2); k4=rhs(w+dt*k3); w=w+dt*(k1+2*k2+2*k3+k4)/6\n  peak=max(peak,float(np.max(np.abs(w))))\n  if not np.all(np.isfinite(w)) or peak>1e4: return [a,nu,amp1,amp2,s*dt,peak,.5*np.mean(w*w)]\n return [a,nu,amp1,amp2,n_steps*dt,peak,.5*np.mean(w*w)]\nrows=[]\nfor a in [-1.0,-0.5,0.0,0.25,0.5,1.0,2.0]:\n for nu in [0.0,0.001]: rows.append(run(a,nu,2.0,0.8))\nfor amp1 in [0.5,1.0,2.0,4.0]: rows.append(run(0.0,0.0,amp1,0.4))\nnp.savez('param.npz',rows=np.array(rows)); print('SEARCH_DONE')\n'''
 with ExecutionSandbox(timeout_seconds=30,loop_fuel=50_000_000) as s:
  r=s.execute(payload); print(r.status,r.execution_time_ms,r.error); b=r.artifacts.get('param.npz')
 with io.BytesIO(b) as f:
  import numpy as np; a=np.load(f)['rows']
 for x in a: print('a nu amp1 amp2 time peak enstrophy',*x)
if __name__=='__main__': main()
