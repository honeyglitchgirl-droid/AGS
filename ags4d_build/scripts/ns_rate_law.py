import io,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ags_sci.experiment.sandbox import ExecutionSandbox
from ags_sci.core.bayes import compute_grounded_bayes_factor

def main():
 payload='''import numpy as np\nglobal _ags_fuel\nN=128; dt=.0005; n_steps=9000; L=2*np.pi; x=np.linspace(0,L,N,endpoint=False); dx=L/N; k=np.fft.fftfreq(N,d=dx/(2*np.pi)); hf=-1j*np.sign(k); w=.5*np.sin(x)+.4*np.sin(2*x)\ndef rhs(q):\n qh=np.fft.fft(q); H=np.real(np.fft.ifft(hf*qh)); return q*H\nt=[]; O=[]; M=[]\nfor s in range(n_steps):\n k1=rhs(w); k2=rhs(w+.5*dt*k1); k3=rhs(w+.5*dt*k2); k4=rhs(w+dt*k3); w=w+dt*(k1+2*k2+2*k3+k4)/6\n if s%10==0: t.append(s*dt); O.append(.5*np.mean(w*w)); M.append(np.max(np.abs(w)))\n if np.max(np.abs(w))>500: break\nnp.savez('rate.npz',t=np.array(t),Omega=np.array(O),M=np.array(M))\nprint('RATE_DONE')\n'''
 with ExecutionSandbox(timeout_seconds=20,loop_fuel=30_000_000) as s:
  r=s.execute(payload); print('sandbox',r.status,r.execution_time_ms,r.error); b=r.artifacts.get('rate.npz')
 with io.BytesIO(b) as f:
  d=np.load(f); t=d['t']; O=d['Omega']; M=d['M']
 print('trajectory',len(t),'t_end',t[-1],'Omega0',O[0],'Omega_end',O[-1],'M_end',M[-1])
 dt=float(np.median(np.diff(t))); Od=np.gradient(O,dt)
 T0=O[:,None]; c0=np.linalg.lstsq(T0,Od,rcond=None)[0]; p0=T0@c0
 T1=np.column_stack([O,O**2]); c1=np.linalg.lstsq(T1,Od,rcond=None)[0]; p1=T1@c1
 bf=compute_grounded_bayes_factor(Od,p1,p0,k_model=2,k_null=1)
 print('M0 c',c0[0]); print('M1 c1,c2',c1); print('BIC0',bf.bic_null,'BIC1',bf.bic_model,'dBIC',bf.bic_model-bf.bic_null,'lnBF',bf.log_bayes_factor_10)
 # fit power law Od vs O positive region
 mask=(O>0)&(Od>0); slope=np.polyfit(np.log(O[mask]),np.log(Od[mask]),1)[0]; print('power exponent dOmega/dt ~ Omega^p:',slope)
if __name__=='__main__': main()
