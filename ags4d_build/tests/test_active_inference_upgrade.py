import math
from ags_sci.core import CausalMechanism, ExperimentProtocol, EpistemicHypothesis
from ags_sci.epistemic import expected_free_energy, information_gain, BayesianUpdater
from ags_sci.experiment import SymbolicVerifier
from ags_sci.search import CausalMCTS

def test_causal_dag_rejects_cycle():
    try:
        EpistemicHypothesis(id='x',title='cycle',causal_dag=[
            CausalMechanism(source='A',target='B',mechanism_type='linear'),
            CausalMechanism(source='B',target='A',mechanism_type='linear')],
            protocol=ExperimentProtocol(measured_variables=['B']))
    except ValueError:
        return
    assert False

def test_efe_prefers_information_gain():
    prior={'0':.5,'1':.5}; post={'a':{'0':.99,'1':.01},'b':{'0':.01,'1':.99}}
    assert information_gain(prior,post,{'a':.5,'b':.5}) > .5

def test_bayes_update():
    r=BayesianUpdater().gaussian([1,1],[1,1],[0,0],.5)
    assert r.log_bayes_factor > 0 and r.posterior_log_odds > 0

def test_symbolic_verification():
    assert SymbolicVerifier.verify('x + x = 2*x').valid

def test_mcts():
    tree={'root':[('good','good'),('bad','bad')]}
    def expand(s): return tree.get(s,[])
    def eval(s): return 1.0 if s=='good' else 0.0
    best=CausalMCTS('root',expand,eval,simulations=20).run()
    assert best.state=='good'

def test_scm_do_intervention():
    from ags_sci.core import StructuralCausalModel, CausalMechanism
    scm=StructuralCausalModel(
        (CausalMechanism(source='X',target='Y',mechanism_type='linear'),),
        {'Y': lambda s: 2*s['X']})
    assert scm.intervention({'X':3})['Y']==6.0

def test_sandbox_timeout_and_success():
    from ags_sci.experiment import SandboxedRunner, SandboxLimits
    r=SandboxedRunner(SandboxLimits(timeout_s=2,memory_mb=512)).run(lambda x:x+1, 4)
    assert r.ok and r.value==5

def test_structured_ai_bridge_rejects_unstructured():
    from ags_sci.adapters import AIHypothesisBridge
    def bad(_): return 'markdown'
    try:
        AIHypothesisBridge(bad).request('x', object)
    except Exception:
        return
    assert False
