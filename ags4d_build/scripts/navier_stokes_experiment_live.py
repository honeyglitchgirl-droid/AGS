import io, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ags_sci.core.bayes import compute_grounded_bayes_factor
from ags_sci.dynamics.identification import DynamicSystemIdentifier
from ags_sci.experiment.sandbox import ExecutionSandbox

def main():
    sim_payload='''import numpy as np\nnp.random.seed(42)\nN=256\nL=2.0*np.pi\nx=np.linspace(0,L,N,endpoint=False)\ndx=L/N\ndt=0.001\nn_steps=1500\na_param=0.5\nnu=0.005\nomega=2.0*np.sin(x)+0.8*np.sin(2*x)\nk=np.fft.fftfreq(N,d=dx/(2.0*np.pi))\nik=1j*k\nk_sq=-(k**2)\nsgn_k=np.sign(k)\nhilbert_filter=-1j*sgn_k\nt_hist=[]\nenstrophy_hist=[]\nomega_max_hist=[]\nt_curr=0.0\nfor step in range(n_steps):\n    omega_hat=np.fft.fft(omega)\n    H_omega=np.real(np.fft.ifft(hilbert_filter*omega_hat))\n    stretching=omega*H_omega\n    H_omega_hat=np.fft.fft(H_omega)\n    u_hat=np.zeros_like(omega_hat)\n    nonzero=k!=0\n    u_hat[nonzero]=H_omega_hat[nonzero]/(ik[nonzero])\n    u=np.real(np.fft.ifft(u_hat))\n    omega_x=np.real(np.fft.ifft(ik*omega_hat))\n    advection=u*omega_x\n    diffusion=nu*np.real(np.fft.ifft(k_sq*omega_hat))\n    domega_dt=stretching-a_param*advection+diffusion\n    omega=omega+dt*domega_dt\n    t_curr+=dt\n    if step%5==0:\n        t_hist.append(t_curr)\n        enstrophy_hist.append(0.5*np.mean(omega**2))\n        omega_max_hist.append(np.max(np.abs(omega)))\nt_arr=np.array(t_hist)\nOmega_arr=np.array(enstrophy_hist).reshape(-1,1)\nnp.savez('ns_enstrophy.npz',t=t_arr,Omega=Omega_arr,omega_max=np.array(omega_max_hist))\nprint('SIMULATION_AND_ENSTROPHY_TRACING_SUCCESS')\n'''
    with ExecutionSandbox(timeout_seconds=8.0) as sandbox:
        r=sandbox.execute(sim_payload)
        print('SANDBOX',r.status,r.execution_time_ms,r.error)
        if r.status!='SUCCESS': return 2
        b=r.artifacts['ns_enstrophy.npz']
    with io.BytesIO(b) as buf:
        d=np.load(buf); t=d['t']; O=d['Omega'][:,0]; om=d['omega_max']
    print('points',len(t),'O0',O[0],'Omax',O.max(),'omax',om.max(),'Oend',O[-1])
    ident=DynamicSystemIdentifier(degree=2,threshold=0.05,include_transcendentals=False).fit(t,O[:,None],names=['Omega'])
    print('EQ',ident.equations[0]); print('RMSE',ident.rmse[0],'active',ident.score)
    dt=float(np.median(np.diff(t))); Od=np.gradient(O,dt)
    # proper M0 c*Omega and M1 c1*Omega+c2*Omega^2
    T0=O[:,None]; c0=np.linalg.lstsq(T0,Od,rcond=None)[0]; p0=T0@c0
    T1=np.column_stack([O,O**2]); c1=np.linalg.lstsq(T1,Od,rcond=None)[0]; p1=T1@c1
    bf=compute_grounded_bayes_factor(Od,p1,p0,k_model=2,k_null=1)
    print('M0',c0,'M1',c1)
    print('BIC0',bf.bic_null,'BIC1',bf.bic_model,'dBIC',bf.bic_model-bf.bic_null,'lnBF',bf.log_bayes_factor_10,'BF',bf.bayes_factor_10)
    return 0
if __name__=='__main__': raise SystemExit(main())
