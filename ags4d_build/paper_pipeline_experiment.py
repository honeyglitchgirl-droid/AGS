from __future__ import annotations
import io, json, hashlib
from pathlib import Path
import numpy as np
from ags_sci.dynamics.identification import identify_stlsq
from ags_sci.core.bayes import compute_grounded_bayes_factor
from ags_sci.core.protocol import ProvenanceBundle
from ags_sci.experiment.sandbox import ExecutionSandbox

SIM = '''import numpy as np\ndt=0.01\nt=np.arange(0,8,dt)\nx=np.zeros_like(t); y=np.zeros_like(t); z=np.zeros_like(t)\nx[0],y[0],z[0]=-8.,8.,27.\nfor i in range(len(t)-1):\n dx=10*(y[i]-x[i]); dy=x[i]*(28-z[i])-y[i]; dz=x[i]*y[i]-(8/3)*z[i]\n x[i+1]=x[i]+dx*dt; y[i+1]=y[i]+dy*dt; z[i+1]=z[i]+dz*dt\nnp.savez("lorenz.npz", t=t, X=np.stack([x,y,z],axis=1))\nprint("SIMULATION_COMPLETE")'''

def main():
    with ExecutionSandbox(timeout_seconds=3.0, loop_fuel=2_000_000) as sb:
        r=sb.execute(SIM)
        assert r.status == 'SUCCESS', r
        assert 'lorenz.npz' in (r.artifacts or {})
        with io.BytesIO(r.artifacts['lorenz.npz']) as buf:
            d=np.load(buf); t=d['t']; X=d['X']
        dt=float(np.median(np.diff(t))); Xdot=np.gradient(X,dt,axis=0)
        Theta=np.column_stack([np.ones(len(X)),X[:,0],X[:,1],X[:,2],X[:,0]**2,X[:,0]*X[:,1],X[:,0]*X[:,2],X[:,1]**2,X[:,1]*X[:,2],X[:,2]**2])
        Xi=identify_stlsq(Theta,Xdot,threshold=0.1)
        pred=Theta@Xi; rmse=float(np.sqrt(np.mean((Xdot-pred)**2)))
        null=np.zeros_like(Xdot)
        comp=compute_grounded_bayes_factor(Xdot,pred,null,int(np.count_nonzero(Xi)),0)
        prov=ProvenanceBundle.create('lorenz-v102',42,SIM,r.artifacts['lorenz.npz'],['1','x','y','z','x^2','xy','xz','y^2','yz','z^2'],'102',rmse,comp.bic_model,comp.log_bayes_factor_10,int(np.count_nonzero(Xi)))
        paper=Path('paper_artifacts'); paper.mkdir(exist_ok=True)
        (paper/'provenance.json').write_text(json.dumps(prov.__dict__,indent=2),encoding='utf-8')
        print('PAPER_PIPELINE_PASS')
        print('artifact_bytes=',len(r.artifacts['lorenz.npz']))
        print('active_terms=',int(np.count_nonzero(Xi)))
        print('rmse=',rmse)
        print('bic_model=',comp.bic_model)
        print('bic_null=',comp.bic_null)
        print('log_bf10=',comp.log_bayes_factor_10)
        print('bf10=',comp.bayes_factor_10)
        print('provenance_sha=',hashlib.sha256((paper/'provenance.json').read_bytes()).hexdigest())
if __name__=='__main__': main()
