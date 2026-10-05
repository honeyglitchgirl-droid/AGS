"""Run a bounded AGS experiment and compile only verified quantitative facts."""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
from ags_sci.core.bayes import compute_grounded_bayes_factor
from ags_sci.core.paper import VerifiedPaperCompiler
from ags_sci.core.protocol import ProvenanceBundle
from ags_sci.dynamics.identification import identify_stlsq
from ags_sci.experiment.sandbox import ExecutionSandbox

SIM="""import numpy as np\ndt=0.01\nt=np.arange(0,8,dt)\nx=np.zeros_like(t); y=np.zeros_like(t); z=np.zeros_like(t)\nx[0],y[0],z[0]=-8.,8.,27.\nfor i in range(len(t)-1):\n dx=10*(y[i]-x[i]); dy=x[i]*(28-z[i])-y[i]; dz=x[i]*y[i]-(8/3)*z[i]\n x[i+1]=x[i]+dx*dt; y[i+1]=y[i]+dy*dt; z[i+1]=z[i]+dz*dt\nnp.savez('lorenz.npz',t=t,X=np.stack([x,y,z],axis=1))\nprint('SIMULATION_COMPLETE')"""

def run(topic="Chaotic Dynamics Identification"):
    with ExecutionSandbox(timeout_seconds=3, loop_fuel=2_000_000) as sandbox:
        result=sandbox.execute(SIM)
        if result.status!="SUCCESS": raise RuntimeError(result.error)
        blob=result.artifacts["lorenz.npz"]
    with io.BytesIO(blob) as buf:
        d=np.load(buf); t=d["t"]; X=d["X"]
    dt=float(np.median(np.diff(t))); Xdot=np.gradient(X,dt,axis=0)
    Theta=np.column_stack([np.ones(len(X)),X[:,0],X[:,1],X[:,2],X[:,0]**2,X[:,0]*X[:,1],X[:,0]*X[:,2],X[:,1]**2,X[:,1]*X[:,2],X[:,2]**2])
    Xi=identify_stlsq(Theta,Xdot,threshold=.1); pred=Theta@Xi
    rmse=float(np.sqrt(np.mean((Xdot-pred)**2)))
    cmp=compute_grounded_bayes_factor(Xdot,pred,np.zeros_like(Xdot),int(np.count_nonzero(Xi)),0)
    prov=ProvenanceBundle.create("lorenz-v102",42,SIM,blob,["1","x","y","z","x²","xy","xz","y²","yz","z²"],"102.2.3",rmse,cmp.bic_model,cmp.log_bayes_factor_10,int(np.count_nonzero(Xi)))
    compiler=VerifiedPaperCompiler(prov)
    manuscript=compiler.compile_section(f"""# Automated Discovery of {topic}\n\nRMSE: {{{{ metrics.measured_rmse }}}}\nBIC: {{{{ metrics.bic_score }}}}\nlog BF10: {{{{ metrics.log_bayes_factor_10 }}}}\nActive terms: {{{{ metrics.active_terms_count }}}}\n\nNumerical parameters were mechanically estimated from observed trajectory data using sparse regression over a fixed basis dictionary.""")
    out=Path("paper_artifacts"); out.mkdir(exist_ok=True); (out/"manuscript.md").write_text(manuscript); (out/"provenance.json").write_text(__import__('json').dumps(prov.__dict__,indent=2))
    return prov

if __name__=='__main__':
    p=run(); print("VERIFIED_PAPER_PIPELINE_PASS"); print(p)
