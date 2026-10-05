"""Untrusted AI boundary: structured outputs only; no statistical authority."""
from __future__ import annotations
from typing import Any, Callable, Type, TypeVar
from pydantic import BaseModel, ConfigDict
from ags_sci.core import HypothesisProposal, Critique, MethodProtocol
from ags_sci.core.security import DEFAULT_SECURITY_POLICY, bounded_text, validate_payload

T=TypeVar('T', bound=BaseModel)
class StructuredAIRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    task: str
    schema_name: str
    payload: dict[str,Any] = {}

class StructuredAIResponse(BaseModel):
    model_config=ConfigDict(extra='forbid')
    schema_name: str
    content: dict[str,Any]

class AIHypothesisBridge:
    """Adapter around an external callable. It cannot mutate AGS state directly."""
    def __init__(self, call: Callable[[StructuredAIRequest], Any]):
        if not callable(call):
            raise TypeError("AI bridge callable is required")
        self.call=call
    def request(self, task: str, schema: Type[T], payload: dict[str,Any]|None=None) -> T:
        bounded_text(task, field="AI task", policy=DEFAULT_SECURITY_POLICY)
        bounded_text(schema.__name__, field="AI schema", policy=DEFAULT_SECURITY_POLICY)
        safe_payload = validate_payload(payload or {}, policy=DEFAULT_SECURITY_POLICY)
        req=StructuredAIRequest(task=task,schema_name=schema.__name__,payload=safe_payload)
        raw=self.call(req)
        if isinstance(raw, BaseModel): raw=raw.model_dump()
        if not isinstance(raw,dict): raise TypeError('AI bridge must return an object')
        # Bound the untrusted response before schema validation to prevent
        # oversized model output from reaching deeper application logic.
        safe_raw = validate_payload(raw, policy=DEFAULT_SECURITY_POLICY)
        return schema.model_validate(safe_raw)
    def propose(self, problem: str) -> HypothesisProposal:
        return self.request(problem,HypothesisProposal)
    def critique(self, proposal: HypothesisProposal) -> Critique:
        return self.request('red-team:'+proposal.hypothesis.title,Critique,{'proposal':proposal.model_dump()})
    def method(self, proposal: HypothesisProposal, critique: Critique) -> MethodProtocol:
        return self.request('methodology:'+proposal.hypothesis.title,MethodProtocol,{'proposal':proposal.model_dump(),'critique':critique.model_dump()})
