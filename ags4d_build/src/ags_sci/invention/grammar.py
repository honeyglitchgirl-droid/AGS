"""Symbolic expression grammar for abductive hypothesis generation.

This module is the substrate for *inventing* candidate laws rather than
identifying them from a fixed library. It provides:

- a typed expression grammar (variables, constants, unary and binary operators),
- a complexity measure used for minimum-description-length scoring,
- safe numeric evaluation that never propagates a NaN into a fitness score,
- symbolic simplification and equivalence testing via SymPy.

Everything is deterministic given a seeded generator, so a search can be
replayed exactly.
"""
from __future__ import annotations
import math
import operator
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import sympy as sp

#: Variable names available to generated expressions.
DEFAULT_VARIABLES = ("x", "y", "z", "t")

#: Unary operators. Each entry is (name, sympy fn, numpy-safe fn or None).
UNARY: tuple[tuple[str, Callable, Callable | None], ...] = (
    ("sin", sp.sin, np.sin),
    ("cos", sp.cos, np.cos),
    ("exp", sp.exp, np.exp),
    ("log", sp.log, np.log),
    ("sqrt", sp.sqrt, np.sqrt),
    ("neg", lambda e: -e, lambda a: -a),
)

#: Binary operators.
BINARY: tuple[tuple[str, Callable, Callable], ...] = (
    ("+", operator.add, lambda a, b: a + b),
    ("-", operator.sub, lambda a, b: a - b),
    ("*", operator.mul, lambda a, b: a * b),
    ("/", operator.truediv, lambda a, b: a / b),
    ("^", lambda a, b: a ** b, lambda a, b: a ** b),
)

_UNARY_BY_NAME = {n: (s, f) for n, s, f in UNARY}
_BINARY_BY_NAME = {n: (s, f) for n, s, f in BINARY}


class ExpressionError(ValueError):
    """Raised when an expression cannot be built, evaluated or simplified."""


@dataclass(frozen=True)
class Expression:
    """A symbolic expression together with its variable set."""

    sympy_expr: sp.Expr
    variables: tuple[str, ...]

    @property
    def complexity(self) -> int:
        """Node count, used as the description-length term in model selection."""
        return int(sp.count_ops(self.sympy_expr)) + 1

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return str(self.sympy_expr)


def make_symbols(names: Sequence[str]) -> dict[str, sp.Symbol]:
    return {n: sp.Symbol(n, real=True) for n in names}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def random_expression(variables: Sequence[str], rng: np.random.Generator,
                      max_depth: int = 3, constants: Sequence[float] = (1.0, 2.0, -1.0)
                      ) -> Expression:
    """Generate a random expression bounded by ``max_depth``.

    ``max_depth`` counts operator nesting, so depth 0 yields a leaf (a variable
    or a constant). Bounding depth bounds both the search space and the risk of
    generating expressions that are expensive to evaluate.
    """
    if not variables:
        raise ExpressionError("at least one variable is required")
    if int(max_depth) < 0:
        raise ExpressionError("max_depth must be >= 0")
    syms = make_symbols(variables)
    pool = list(constants) + [float(rng.integers(-3, 4))]

    def leaf() -> sp.Expr:
        if rng.random() < 0.65:
            return syms[variables[int(rng.integers(0, len(variables)))]]
        return sp.Float(float(pool[int(rng.integers(0, len(pool)))]))

    def build(depth: int) -> sp.Expr:
        if depth <= 0:
            return leaf()
        roll = rng.random()
        if roll < 0.30:                      # leaf
            return leaf()
        if roll < 0.62:                      # unary
            name = UNARY[int(rng.integers(0, len(UNARY)))][0]
            fn = _UNARY_BY_NAME[name][0]
            try:
                return fn(build(depth - 1))
            except Exception:
                return leaf()
        name = BINARY[int(rng.integers(0, len(BINARY)))][0]
        fn = _BINARY_BY_NAME[name][0]
        try:
            return fn(build(depth - 1), build(depth - 1))
        except Exception:
            return leaf()

    expr = build(int(max_depth))
    return Expression(sp.sympify(expr), tuple(variables))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def safe_evaluate(expr: Expression, values: dict[str, np.ndarray]) -> np.ndarray | None:
    """Numerically evaluate ``expr``, returning None on any failure.

    A generated expression may divide by zero, take a log of a negative number,
    or overflow. Such candidates are simply infeasible; returning None keeps that
    decision local instead of letting a NaN silently poison a fitness score.
    """
    try:
        # Evaluate over the expression's *actual* free symbols, not the declared
        # variable tuple. A corpus law such as ``pi*x**2`` legitimately declares
        # no variables, and lambdifying over the declared tuple alone would
        # silently drop ``x`` and make every numeric check against it fail.
        free = sorted(expr.sympy_expr.free_symbols, key=lambda sy: sy.name)
        names = tuple(sy.name for sy in free)
        missing = set(names) - set(values)
        if missing:
            raise ExpressionError(f"missing values for {sorted(missing)}")
        args = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        # Lambdify once per call; expressions are small and this keeps the
        # evaluation vectorised.
        fn = sp.lambdify(free, expr.sympy_expr, modules=["numpy"])
        # Randomly generated expressions routinely divide by zero, take logs of
        # negatives or overflow. Those candidates are simply infeasible, so the
        # warnings they raise are expected noise rather than a problem to report.
        with np.errstate(all="ignore"):
            raw = np.asarray(fn(*[args[n] for n in names]))
        # A generated expression can be complex-valued (e.g. (-x)**0.5). Casting
        # to real would silently discard the imaginary part and let a complex
        # candidate appear to fit real data, so only accept it when the imaginary
        # part is genuinely negligible; otherwise the candidate is infeasible.
        if np.iscomplexobj(raw):
            imag = np.abs(np.imag(raw))
            real_scale = np.abs(np.real(raw))
            if np.any(imag > 1e-9 * np.maximum(real_scale, 1.0)):
                return None
            raw = np.real(raw)
        out = np.asarray(raw, dtype=float)
        # A constant expression lambdifies to a scalar; broadcast it against the
        # caller's sample grid so callers always get one value per sample.
        ref = names[0] if names else next(iter(args), None)
        if ref is None:
            return out
        out = np.broadcast_to(out, np.shape(args[ref]))
        if not np.all(np.isfinite(out)):
            return None
        return out
    except Exception:
        return None


def complexity(expr: Expression) -> int:
    return expr.complexity


def simplified(expr: Expression) -> Expression:
    """Symbolically simplify, falling back to the original on failure."""
    try:
        return Expression(sp.simplify(expr.sympy_expr), expr.variables)
    except Exception:
        return expr


def equivalent(a: Expression, b: Expression) -> bool:
    """True when ``a`` and ``b`` are symbolically identical.

    Equivalence is checked by simplifying the difference. This is a *proof of
    equivalence* when SymPy can reduce the difference to zero; failure to
    simplify is reported as "not proven equivalent", never as "different".
    """
    # Compare the free symbols *by name*, not the declared variable tuples: a
    # corpus law legitimately declares no variables while a candidate declares
    # ("x",), and gating on the declared tuples made symbolic equivalence
    # unreachable against the default corpus.
    if {s.name for s in a.sympy_expr.free_symbols} != {s.name for s in b.sympy_expr.free_symbols}:
        return False
    try:
        # Symbols are matched by *name*, because the same name may carry
        # different assumptions (e.g. real=True) in different expressions and
        # a naive subtraction would then compare two distinct symbols.
        subs = {sa: sb for sa, sb in zip(
            sorted(a.sympy_expr.free_symbols, key=lambda s: s.name),
            sorted(b.sympy_expr.free_symbols, key=lambda s: s.name))}
        if any(sa.name != sb.name for sa, sb in subs.items()):
            return False
        diff = sp.simplify(sp.expand(a.sympy_expr.subs(subs) - b.sympy_expr))
        return bool(diff == 0)
    except Exception:
        return False


def numeric_agreement(a: Expression, b: Expression,
                      values: dict[str, np.ndarray], tol: float = 1e-9) -> bool | None:
    """Sample-based agreement check; None when either expression is infeasible."""
    va = safe_evaluate(a, values)
    vb = safe_evaluate(b, values)
    if va is None or vb is None:
        return None
    scale = max(1.0, float(np.max(np.abs(vb))))
    return bool(np.max(np.abs(va - vb)) <= tol * scale)


def fitness(expr: Expression, target: np.ndarray,
            values: dict[str, np.ndarray], complexity_weight: float = 0.01) -> float:
    """Description-length score: normalised RMSE plus a complexity penalty.

    Lower is better. Infeasible expressions get +inf so they are never selected.
    """
    got = safe_evaluate(expr, values)
    if got is None:
        return math.inf
    t = np.asarray(target, dtype=float)
    if got.shape != t.shape:
        return math.inf
    denom = float(np.max(np.abs(t))) or 1.0
    # Generated candidates overflow routinely (e.g. 10**n); that is an
    # infeasible candidate, not something worth a warning.
    with np.errstate(all="ignore"):
        rmse = float(np.sqrt(np.mean((got - t) ** 2))) / denom
    return rmse + float(complexity_weight) * expr.complexity


# ---------------------------------------------------------------------------
# Constant refinement
# ---------------------------------------------------------------------------
def _floats(expr: Expression) -> list:
    """The floating-point constants appearing in ``expr``, in a stable order."""
    return sorted(expr.sympy_expr.atoms(sp.Float), key=lambda a: float(a))


def _evaluate_with(expr: Expression, values: dict[str, np.ndarray],
                   constants: Sequence[float]) -> np.ndarray | None:
    """Evaluate ``expr`` with its float constants replaced by ``constants``."""
    floats = _floats(expr)
    if len(floats) != len(constants):
        return None
    sub = dict(zip(floats, [float(c) for c in constants]))
    try:
        replaced = sp.sympify(expr.sympy_expr.subs(sub))
        # Lambdify over the replaced expression's free symbols, for the same
        # reason as in ``safe_evaluate``: the declared variable tuple may be
        # narrower than the symbols actually referenced.
        free = sorted(replaced.free_symbols, key=lambda sy: sy.name)
        names = tuple(sy.name for sy in free)
        missing = set(names) - set(values)
        if missing:
            return None
        fn = sp.lambdify(free, replaced, modules=["numpy"])
        args = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        with np.errstate(all="ignore"):
            raw = np.asarray(fn(*[args[n] for n in names]))
        if np.iscomplexobj(raw):
            if np.any(np.abs(np.imag(raw)) > 1e-9 * np.maximum(np.abs(np.real(raw)), 1.0)):
                return None
            raw = np.real(raw)
        out = np.asarray(raw, dtype=float)
        # A constant expression lambdifies to a scalar; broadcast it against the
        # caller's sample grid so callers always get one value per sample.
        ref = names[0] if names else next(iter(args), None)
        if ref is None:
            return out
        out = np.broadcast_to(out, np.shape(args[ref]))
        return out if np.all(np.isfinite(out)) else None
    except Exception:
        return None


def _rationalise(value: float, max_denominator: int = 12,
                 tolerance: float = 1e-9) -> sp.Rational | None:
    """Return a simple rational equal to ``value`` within tolerance, else None."""
    frac = sp.nsimplify(value, rational=True, tolerance=sp.Float(tolerance, 20))
    if isinstance(frac, sp.Rational) and abs(float(frac) - value) <= tolerance:
        if 1 <= frac.q <= max_denominator:
            return frac
    return None


def refine_constants(expr: Expression, target: np.ndarray,
                     values: dict[str, np.ndarray], *,
                     iterations: int = 12, step: float = 1e-6,
                     rationalise: bool = True) -> tuple[Expression, float]:
    """Refit the free float constants of ``expr`` by Gauss-Newton least squares.

    Symbolic search over an unconstrained grammar routinely lands on structures
    that are right but whose numeric constants are noisy, e.g. ``0.5201*n**2``
    instead of ``n**2/2``. Refitting the constants closes that gap and often
    converts a near-miss into an exact recovery.

    Returns the refined expression and the improvement in normalised RMSE. The
    original expression is returned unchanged when refinement does not help.
    """
    target = np.asarray(target, dtype=float)
    floats = _floats(expr)
    if not floats:
        return expr, 0.0
    original_rmse = fitness(expr, target, values)
    denom = float(np.max(np.abs(target))) or 1.0

    def rmse(vals: Sequence[float]) -> float:
        got = _evaluate_with(expr, values, vals)
        if got is None or got.shape != target.shape:
            return math.inf
        with np.errstate(all="ignore"):
            return float(np.sqrt(np.mean((got - target) ** 2))) / denom

    params = [float(f) for f in floats]
    best_rmse = rmse(params)
    for _ in range(int(iterations)):
        J = np.zeros((target.size, len(params)))
        for i in range(len(params)):
            up = list(params); dn = list(params)
            h = max(abs(params[i]) * step, step)
            up[i] += h; dn[i] -= h
            fu = _evaluate_with(expr, values, up)
            fd = _evaluate_with(expr, values, dn)
            if fu is None or fd is None:
                continue
            J[:, i] = (fu - fd) / (2 * h)
        residual = _evaluate_with(expr, values, params)
        if residual is None:
            break
        try:
            delta, *_ = np.linalg.lstsq(J, -(residual - target), rcond=None)
        except Exception:
            break
        if not np.all(np.isfinite(delta)):
            break
        trial = [pp + float(d) for pp, d in zip(params, delta)]
        trial_rmse = rmse(trial)
        if trial_rmse < best_rmse - 1e-15:
            params, best_rmse = trial, trial_rmse
        else:
            break

    if rationalise:
        # Prefer an exact simple rational when one is within tolerance; this is
        # what turns 0.520147101004912 into 1/2.
        subs = {}
        for original, value in zip(floats, params):
            rat = _rationalise(value)
            subs[original] = rat if rat is not None else sp.Float(value, 15)
        candidate = Expression(sp.sympify(expr.sympy_expr.subs(subs)), expr.variables)
        cand_rmse = fitness(candidate, target, values)
        if cand_rmse <= best_rmse + 1e-12:
            return candidate, max(0.0, original_rmse - cand_rmse)
    refined = Expression(sp.sympify(expr.sympy_expr.subs(
        dict(zip(floats, [sp.Float(v, 15) for v in params])))), expr.variables)
    return refined, max(0.0, original_rmse - best_rmse)
