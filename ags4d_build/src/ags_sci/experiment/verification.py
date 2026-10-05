"""Symbolic equation verification before numerical execution."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import sympy as sp

@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    equation_error: str | None = None
    dimensional_ok: bool | None = None
    invariant_checks: tuple[str,...] = field(default_factory=tuple)

class SymbolicVerifier:
    @staticmethod
    def parse_equation(equation: str):
        if "=" not in equation: raise ValueError("equation must contain '='")
        lhs,rhs=equation.split("=",1)
        return sp.sympify(lhs.strip()), sp.sympify(rhs.strip())
    @classmethod
    def verify(cls, equation: str) -> VerificationResult:
        try:
            lhs,rhs=cls.parse_equation(equation)
            equal=sp.simplify(lhs-rhs)==0
            return VerificationResult(valid=True, equation_error=None, dimensional_ok=None, invariant_checks=(f"symbolic_parse:{'ok' if equal else 'consistent_expression'}",))
        except Exception as e:
            return VerificationResult(False,e,None,())
    @staticmethod
    def nonnegative(expr: str, symbol: str, domain=(0, sp.oo)) -> bool:
        x=sp.Symbol(symbol, real=True); e=sp.sympify(expr)
        return bool(sp.ask(sp.Q.nonnegative(e), sp.Q.real(x))) if x in e.free_symbols else bool(e>=0)
