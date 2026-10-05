import math
import numpy as np
import pytest
from ags_sci.core.security import SecurityViolation
from ags_sci.epistemic.active_inference import entropy, kl_divergence
from ags_sci.search import CausalMCTS
from ags_sci.experiment.dimensions import ExperimentSpec, validate_spec
from ags_sci.experiment.sandbox import SandboxedRunner
from ags_sci.fields import SpectralEngineND


def test_active_inference_rejects_negative_and_nonfinite_probabilities():
    with pytest.raises(ValueError): entropy({'a': 1.0, 'b': -0.1})
    with pytest.raises(ValueError): entropy({'a': float('nan')})
    assert math.isinf(kl_divergence({'a': 1.0}, {'b': 1.0}))


def test_mcts_rejects_unbounded_controls_and_nonfinite_reward():
    with pytest.raises(SecurityViolation):
        CausalMCTS(0, lambda s: [], lambda s: 0.0, simulations=100_001)
    with pytest.raises(SecurityViolation):
        CausalMCTS(0, lambda s: [], lambda s: float('nan')) .run()


def test_dimension_spec_rejects_bool_and_nonfinite_runtime():
    with pytest.raises(ValueError):
        validate_spec(ExperimentSpec(3, (8, 8, 8), fields=True))
    with pytest.raises(ValueError):
        validate_spec(ExperimentSpec(3, (8, 8, 8), runtime_s=float('nan')))


def test_sandbox_does_not_unpickle_arbitrary_result_objects():
    code = '''\nclass Evil:\n    def __reduce__(self):\n        return (print, ("SHOULD_NOT_EXECUTE_IN_PARENT",))\nresult = Evil()\n'''
    r = SandboxedRunner().run_source(code)
    assert not r.ok
    assert 'SHOULD_NOT_EXECUTE_IN_PARENT' not in (r.error or '')


def test_spectral_derivative_is_not_silently_dealiased():
    eng = SpectralEngineND((12,), dealias='componentwise')
    x = np.arange(12) * 2*np.pi/12
    # Highest retained/non-retained mode: derivative should reflect the input,
    # while explicit filter_field is responsible for truncation.
    f = np.sin(5*x)
    d = eng.gradient(f)[0]
    assert np.max(np.abs(d - 5*np.cos(5*x))) < 1e-10
    filtered = eng.filter_field(f)
    assert np.max(np.abs(filtered)) < 1e-12
