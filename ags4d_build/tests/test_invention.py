"""Tests for the invention layer: generative law search, novelty assessment,
problem generation, and sequence conjectures.

Each test checks a candidate against an independently known answer. Where a
capability is expected to *fail* (a wrong conjecture, white-noise residuals),
the test asserts that it fails honestly rather than overclaiming.
"""
import numpy as np
import pytest
import sympy as sp

from ags_sci.invention import (AbductiveSearch, Candidate, Expression,
                               InventionEngine, KnownLaw, NoveltyAssessor,
                               NoveltyReport, OpenProblem, ProblemGenerator,
                               SequenceConjecture, complexity, equivalent,
                               random_expression, refine_constants,
                               safe_evaluate, simplified)


@pytest.fixture(scope="module")
def engine():
    return InventionEngine(max_depth=3, population_size=100, generations=25, seed=7)


def _predicts(expression: str, indices, values, tol: float = 1e-9) -> bool:
    """True when ``expression`` reproduces ``values`` at ``indices``.

    This is the semantic claim a sequence conjecture actually makes, so it is
    what the tests check -- not the particular spelling the search settled on.
    """
    fn = sp.lambdify(sp.Symbol("n"), sp.sympify(expression), modules=["numpy"])
    return bool(np.allclose(fn(np.asarray(indices, dtype=float)),
                            np.asarray(values, dtype=float),
                            rtol=tol, atol=tol))


# ==========================================================================
# Grammar
# ==========================================================================
def test_random_expression_is_deterministic_and_bounded():
    a = random_expression(("x",), np.random.default_rng(3), max_depth=3)
    b = random_expression(("x",), np.random.default_rng(3), max_depth=3)
    assert str(a.sympy_expr) == str(b.sympy_expr)
    assert a.variables == ("x",)


def test_random_expression_rejects_bad_input():
    with pytest.raises(ValueError):
        random_expression((), np.random.default_rng(0))
    with pytest.raises(ValueError):
        random_expression(("x",), np.random.default_rng(0), max_depth=-1)


def test_safe_evaluate_rejects_infeasible_expressions():
    # log of a negative number and division by zero must both be infeasible,
    # not silently NaN.
    assert safe_evaluate(Expression(sp.log(sp.Symbol("x")), ("x",)),
                         {"x": np.array([-1.0, -2.0])}) is None
    assert safe_evaluate(Expression(1 / sp.Symbol("x"), ("x",)),
                         {"x": np.array([0.0, 1.0])}) is None


def test_safe_evaluate_rejects_complex_valued_candidates():
    # (-x)**0.5 is complex for x > 0; discarding the imaginary part would let a
    # complex candidate appear to fit real data.
    got = safe_evaluate(Expression((-sp.Symbol("x")) ** 0.5, ("x",)),
                        {"x": np.array([1.0, 4.0])})
    assert got is None


def test_safe_evaluate_accepts_negligible_imaginary_part():
    # (x**2)**0.5 is |x| with a zero imaginary part on the reals.
    got = safe_evaluate(Expression((sp.Symbol("x") ** 2) ** 0.5, ("x",)),
                        {"x": np.array([1.0, 2.0, 3.0])})
    assert got is not None and np.allclose(got, [1.0, 2.0, 3.0])


def test_equivalence_is_symbolic():
    x = sp.Symbol("x", real=True)
    a = Expression((x + 1) ** 2, ("x",))
    b = Expression(x ** 2 + 2 * x + 1, ("x",))
    assert equivalent(a, b)
    c = Expression(x ** 2, ("x",))
    assert not equivalent(a, c)


def test_equivalence_requires_matching_variables():
    x, y = sp.Symbol("x", real=True), sp.Symbol("y", real=True)
    assert not equivalent(Expression(x, ("x",)), Expression(y, ("y",)))


def test_complexity_counts_operations():
    x = sp.Symbol("x", real=True)
    assert complexity(Expression(x, ("x",))) == 1
    assert complexity(Expression(x + 1, ("x",))) == 2
    assert complexity(Expression((x + 1) ** 2, ("x",))) == 3


# ==========================================================================
# Constant refinement
# ==========================================================================
def test_refine_constants_recovers_exact_rational():
    x = np.arange(1, 21, dtype=float)
    noisy = Expression(sp.sympify("0.520147101004912*x**2"), ("x",))
    refined, improvement = refine_constants(noisy, 0.5 * x ** 2, {"x": x})
    assert str(refined.sympy_expr) == "0.5*x**2"
    assert improvement > 0


def test_refine_constants_recovers_multiple_constants():
    x = np.arange(1, 21, dtype=float)
    noisy = Expression(sp.sympify("0.31*x**2 + 0.49*x"), ("x",))
    refined, _ = refine_constants(noisy, 0.5 * x ** 2 + 0.5 * x, {"x": x})
    assert str(refined.sympy_expr) == "0.5*x**2 + 0.5*x"


def test_refine_constants_is_a_noop_without_constants():
    x = np.arange(1, 11, dtype=float)
    e = Expression(sp.sympify("x**2"), ("x",))
    refined, improvement = refine_constants(e, x ** 2, {"x": x})
    assert refined.sympy_expr == e.sympy_expr and improvement == 0.0


# ==========================================================================
# Abductive search
# ==========================================================================
def test_search_recovers_exact_cosine():
    t = np.linspace(0, 2 * np.pi, 120)
    rep = AbductiveSearch(("t",), max_depth=3, population_size=100,
                          generations=25, seed=1).search(np.cos(t), {"t": t})
    assert rep.candidates
    best = rep.candidates[0]
    assert equivalent(best.expression, Expression(sp.cos(sp.Symbol("t", real=True)),
                                                  ("t",)))
    assert best.rmse < 1e-9


def test_search_recovers_polynomial():
    x = np.linspace(-2, 2, 80)
    rep = AbductiveSearch(("x",), max_depth=3, population_size=120,
                          generations=30, seed=2).search(x ** 2 + 1, {"x": x})
    best = rep.candidates[0]
    assert equivalent(best.expression, Expression(sp.Symbol("x", real=True) ** 2 + 1,
                                                  ("x",)))


def test_search_reports_a_pareto_front():
    x = np.linspace(0.1, 3, 100)
    rep = AbductiveSearch(("x",), max_depth=2, population_size=80,
                          generations=15, seed=3).search(np.sin(x), {"x": x})
    fits = [c.fitness for c in rep.candidates]
    assert fits == sorted(fits), "candidates must be ordered by fitness"
    assert all(c.fitness < np.inf for c in rep.candidates)
    # No candidate on the front may be dominated on both axes by another.
    for i, a in enumerate(rep.candidates):
        for j, b in enumerate(rep.candidates):
            if i == j:
                continue
            dominated = (b.complexity <= a.complexity and b.rmse <= a.rmse
                         and (b.complexity < a.complexity or b.rmse < a.rmse))
            assert not dominated, f"{a.expression.sympy_expr} dominated by {b.expression.sympy_expr}"


def test_search_validates_input_shapes():
    s = AbductiveSearch(("x",), population_size=8, generations=2)
    with pytest.raises(ValueError):
        s.search(np.zeros(5), {"x": np.zeros(6)})
    with pytest.raises(ValueError):
        s.search(np.array([np.nan, 1.0]), {"x": np.zeros(2)})


def test_search_is_deterministic():
    t = np.linspace(0, 3, 100)
    a = AbductiveSearch(("t",), population_size=60, generations=12, seed=5).search(
        np.sin(t), {"t": t})
    b = AbductiveSearch(("t",), population_size=60, generations=12, seed=5).search(
        np.sin(t), {"t": t})
    assert [str(c.expression.sympy_expr) for c in a.candidates] == \
           [str(c.expression.sympy_expr) for c in b.candidates]


# ==========================================================================
# Novelty
# ==========================================================================
def test_novelty_recognises_a_known_law():
    x = sp.Symbol("x", real=True)
    k = sp.Symbol("k", real=True, positive=True)
    assessor = NoveltyAssessor()
    rep = assessor.assess(Expression(-k * x, ("x",)))
    assert not rep.is_novel
    assert rep.matched_law == "hooke"


def test_novelty_flags_a_genuinely_new_form():
    t = sp.Symbol("t", real=True)
    assessor = NoveltyAssessor()
    # tanh(t)*cos(t) is not in the default corpus.
    rep = assessor.assess(Expression(sp.tanh(t) * sp.cos(t), ("t",)))
    assert rep.is_novel
    assert rep.matched_law is None
    assert rep.matches == ()


def test_novelty_equivalence_is_detected_after_simplification():
    x = sp.Symbol("x", real=True)
    k = sp.Symbol("k", real=True, positive=True)
    assessor = NoveltyAssessor()
    # -1*x*k is symbolically the same as -k*x.
    rep = assessor.assess(Expression(-1 * x * k, ("x",)))
    assert not rep.is_novel and rep.matched_law == "hooke"


def test_novelty_uses_numeric_evidence_when_supplied():
    x = sp.Symbol("x", real=True)
    k = sp.Symbol("k", real=True, positive=True)
    assessor = NoveltyAssessor()
    xs = np.linspace(-2, 2, 40)
    # A candidate that agrees with a known law only to numerical tolerance is
    # not symbolically identical to it, so sampling is the only evidence that
    # can convict it. (``-k*x`` with k=2 is *literally* the corpus entry for
    # Hooke's law, which would be caught by the syntactic path instead.)
    rep = assessor.assess(Expression(sp.Float("3.14159265358979") * x ** 2, ("x",)),
                          {"x": xs})
    assert not rep.is_novel, "numeric agreement must convict a known form"
    assert any(kind == "numerically_equivalent_on_samples" for _, kind in rep.matches)
    assert not any(kind in {"identical", "symbolically_equivalent"}
                   for _, kind in rep.matches)
    # ... and a candidate that matches nothing must not be convicted.
    away = assessor.assess(Expression(x ** 2 + x, ("x",)), {"x": xs})
    assert away.is_novel


def test_custom_corpus_changes_the_verdict():
    x = sp.Symbol("x", real=True)
    corpus = (KnownLaw("my_law", Expression(2 * x, ("x",))),)
    assessor = NoveltyAssessor(corpus)
    assert not assessor.assess(Expression(2 * x, ("x",))).is_novel
    assert assessor.assess(Expression(3 * x, ("x",))).is_novel


# ==========================================================================
# Problem generation
# ==========================================================================
def test_problems_detected_in_structured_residuals():
    x = np.linspace(-3, 3, 300)
    resid = 0.4 * np.cos(3 * x)          # a clean unmodelled oscillation
    probs = ProblemGenerator().generate(resid, x=x, t=np.arange(len(x)))
    kinds = {p.kind for p in probs}
    assert "unmodelled_periodicity" in kinds
    assert all(isinstance(p, OpenProblem) for p in probs)
    assert all(0.0 <= p.severity <= 1.0 for p in probs)


def test_no_problems_from_white_noise():
    rng = np.random.default_rng(0)
    resid = rng.normal(0, 1, 2000)
    probs = ProblemGenerator().generate(resid, x=np.linspace(-1, 1, 2000),
                                        t=np.arange(2000))
    assert probs == ()


def test_problems_detect_drift():
    resid = np.full(400, 5.0) + np.random.default_rng(0).normal(0, 0.1, 400)
    probs = ProblemGenerator().generate(resid)
    assert any(p.kind == "unexplained_offset" for p in probs)


def test_problems_detect_serial_dependence():
    # A slow random walk has strongly autocorrelated residuals.
    rng = np.random.default_rng(1)
    resid = np.cumsum(rng.normal(0, 1, 800))
    probs = ProblemGenerator().generate(resid, t=np.arange(len(resid)))
    assert any(p.kind == "serial_dependence" for p in probs)


def test_problems_skip_checks_whose_inputs_are_missing():
    rng = np.random.default_rng(2)
    resid = np.cumsum(rng.normal(0, 1, 500))
    # No t and no x supplied: temporal and parity checks cannot run.
    probs = ProblemGenerator().generate(resid)
    assert all(p.kind not in {"serial_dependence", "symmetry_breaking"} for p in probs)


def test_problems_require_enough_samples():
    assert ProblemGenerator().generate(np.zeros(4)) == ()


# ==========================================================================
# Engine: invention
# ==========================================================================
def test_engine_invents_cosine_from_data(engine):
    t = np.linspace(0, 2 * np.pi, 150)
    res = engine.invent(np.cos(t), {"t": t})
    assert res["status"] == "CANDIDATES_GENERATED"
    top = res["candidates"][0]
    assert equivalent(
        Expression(sp.sympify(top["expression"]), ("t",)),
        Expression(sp.cos(sp.Symbol("t", real=True)), ("t",)))
    assert top["normalised_rmse"] < 1e-9
    assert top["epistemic_status"] == "NOVEL_CANDIDATE_RELATIVE_TO_CORPUS"


def test_engine_reports_novelty_method(engine):
    t = np.linspace(0, 2, 60)
    res = engine.invent(np.cos(t), {"t": t})
    assert "corpus" in res["novelty_method"]


def test_engine_generates_problems_from_residuals(engine):
    x = np.linspace(-3, 3, 300)
    res = engine.invent(np.sin(x), {"x": x})
    # The best candidate is sin(x) itself, so residuals are ~0 and no problems
    # should be invented.
    assert res["open_problems"] == []


def test_engine_handles_infeasible_target(engine):
    res = engine.invent(np.array([np.nan] * 20), {"x": np.linspace(0, 1, 20)})
    assert res["status"] == "NO_FEASIBLE_CANDIDATE"


# ==========================================================================
# Engine: sequence conjectures (objective falsification)
# ==========================================================================
def test_conjecture_recovers_squares(engine):
    n = np.arange(1, 41, dtype=float)
    c = engine.conjecture_sequence(n ** 2, holdout=6, start_index=1)
    assert c.status == "CONJECTURE_SURVIVED_ALL_TERMS"
    # The recovered closed form must reproduce the sequence, whatever spelling
    # the search happened to settle on ("n**2.0" is as correct as "n**2").
    assert _predicts(c.expression, n, n ** 2)
    assert c.first_failure_index is None


def test_conjecture_recovers_cubes(engine):
    n = np.arange(1, 41, dtype=float)
    c = engine.conjecture_sequence(n ** 3, holdout=6, start_index=1)
    assert c.status == "CONJECTURE_SURVIVED_ALL_TERMS"
    assert _predicts(c.expression, n, n ** 3)


def test_conjecture_recovers_powers_of_two(engine):
    n = np.arange(1, 41, dtype=float)
    c = engine.conjecture_sequence(2.0 ** n, holdout=6, start_index=1)
    assert c.status == "CONJECTURE_SURVIVED_ALL_TERMS"
    assert _predicts(c.expression, n, 2.0 ** n)


def test_conjecture_start_index_is_explicit(engine):
    # 0-based: [1,4,9,...] is (n+1)**2. 1-based: the same data is n**2.
    seq = np.arange(1, 11, dtype=float) ** 2
    zero = engine.conjecture_sequence(seq, holdout=3, start_index=0)
    one = engine.conjecture_sequence(seq, holdout=3, start_index=1)
    assert zero.status == "CONJECTURE_SURVIVED_ALL_TERMS"
    assert one.status == "CONJECTURE_SURVIVED_ALL_TERMS"
    assert "(n + 1.0)**2.0" in zero.expression or "(n + 1)**2" in zero.expression
    assert one.expression.replace(" ", "") in ("n**2.0", "n**2")


def test_conjecture_is_falsified_when_wrong(engine):
    # Perfectly regular for 19 terms, then broken. The conjecture must not
    # survive, and must report where it broke.
    bad = np.array([float(i ** 2) if i < 20 else float(i ** 2) + 3.0
                    for i in range(1, 41)])
    c = engine.conjecture_sequence(bad, holdout=6, start_index=1)
    assert c.status != "CONJECTURE_SURVIVED_ALL_TERMS"
    assert c.first_failure_index is not None


def test_conjecture_falsified_on_holdout(engine):
    # Regular inside the fit window, broken only in the held-out tail.
    seq = np.arange(1, 31, dtype=float) ** 2
    seq[-3:] += 100.0
    c = engine.conjecture_sequence(seq, holdout=5, start_index=1)
    assert c.status == "CONJECTURE_FALSIFIED_ON_HOLDOUT"
    assert c.first_failure_index >= len(seq) - 5


def test_conjecture_validates_input(engine):
    with pytest.raises(ValueError):
        engine.conjecture_sequence(np.arange(4))
    with pytest.raises(ValueError):
        engine.conjecture_sequence(np.arange(10) ** 2, holdout=9)
    with pytest.raises(ValueError):
        engine.conjecture_sequence(np.array([np.nan] * 10))


# ==========================================================================
# Service integration
# ==========================================================================
def test_service_exposes_invention():
    from ags_sci import AGSService
    svc = AGSService()
    t = np.linspace(0, 2 * np.pi, 120)
    res = svc.invent(t, np.cos(t))
    assert res["status"] == "CANDIDATES_GENERATED"
    assert res["candidates"][0]["normalised_rmse"] < 1e-9
    assert "novelty" in res["candidates"][0]


def test_service_conjecture_sequence():
    from ags_sci import AGSService
    n = np.arange(1, 31, dtype=float)
    c = AGSService().conjecture_sequence(n ** 2, start_index=1)
    assert c.status == "CONJECTURE_SURVIVED_ALL_TERMS"
