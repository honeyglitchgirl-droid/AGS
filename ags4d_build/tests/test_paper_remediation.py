import io
import numpy as np
import pytest
from ags_sci.core.bayes import compute_grounded_bayes_factor
from ags_sci.core.paper import PaperCompilationError, VerifiedPaperCompiler
from ags_sci.core.protocol import ProvenanceBundle
from ags_sci.dynamics.identification import DynamicIdentificationResult, DynamicSystemIdentifier, identify_stlsq
from ags_sci.experiment.sandbox import ExecutionSandbox

def test_identification_result_and_wrapper():
    t=np.linspace(0,5,501); x=np.exp(-0.5*t)[:,None]
    result=DynamicSystemIdentifier(degree=1, threshold=.01, include_transcendentals=False).fit(t,x,["x"])
    assert isinstance(result, DynamicIdentificationResult)
    Theta=np.column_stack([np.ones(len(t)),x[:,0]])
    Xi=identify_stlsq(Theta,np.gradient(x[:,0],t)[:,None],threshold=.01)
    assert Xi.shape==(2,1)
    assert abs(Xi[1,0]+.5)<.01

def test_sandbox_artifact_channel():
    code="""import numpy as np\nnp.savez('result.npz', x=np.arange(4))"""
    with ExecutionSandbox(timeout_seconds=2) as sb:
        r=sb.execute(code)
    assert r.status=="SUCCESS"
    with io.BytesIO(r.artifacts["result.npz"]) as b:
        assert np.array_equal(np.load(b)["x"],np.arange(4))

def test_grounded_bayes_factor_has_explicit_null():
    y=np.array([0.,1.,2.,3.]); m=y.copy(); n=np.zeros_like(y)
    c=compute_grounded_bayes_factor(y,m,n,1,0)
    assert c.sse_model==0
    assert c.bic_model < c.bic_null
    assert c.log_bayes_factor_10 > 0

def test_paper_compiler_rejects_direct_metric():
    p=ProvenanceBundle.create("x",1,"code",b"data",["x"],"102.1.0",.1,2.,3.,4)
    c=VerifiedPaperCompiler(p)
    with pytest.raises(PaperCompilationError):
        c.compile_section("The RMSE was 0.12.")
    out=c.compile_section("RMSE={{ metrics.measured_rmse }}")
    assert "0.1" in out
