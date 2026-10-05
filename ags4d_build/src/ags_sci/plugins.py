"""Plugin system for AGS-Sci.

AGS is designed to be driven by an external AI, and to be extended without
editing protected core files. This module provides the two halves of that
contract:

``AGSPlugin``
    The interface a plugin implements: it declares an identity, a version and a
    set of capability tags, and receives an :class:`AGSService` handle on
    ``install``. A plugin can *use* AGS but cannot reach into it.

``PluginRegistry``
    Validated, fail-closed storage for plugins. Registration rejects duplicate
    names, malformed identities, undeclared capabilities, and anything that
    touches the protected core paths listed in
    :data:`ags_sci.core.evolution.PROTECTED_PATHS`.

The registry deliberately does not import plugin modules from disk. Discovery
of *code* is the host application's responsibility; AGS only accepts plugins it
is handed explicitly. That keeps "load a plugin" from becoming "execute
arbitrary code at import time".
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from .api import AGSService

from .core.security import SecurityViolation, safe_identifier, bounded_text

#: Capability tags a plugin may declare. Anything outside this set is rejected,
#: so an AI driving AGS can reason about what a plugin is allowed to do.
ALLOWED_CAPABILITIES = frozenset({
    "experiment",        # run bounded numerical experiments
    "sandbox",           # execute untrusted code in the sandbox
    "field-engine",      # provide a field backend
    "discovery",         # propose laws/equations from data
    "analysis",          # read-only analysis of results
    "visualization",     # render results
    "export",            # emit artifacts/provenance bundles
})

#: Capabilities that touch untrusted input or mutable state and therefore require
#: the host to opt in explicitly.
PRIVILEGED_CAPABILITIES = frozenset({"sandbox", "field-engine", "discovery"})


@runtime_checkable
class AGSPlugin(Protocol):
    """What a plugin must provide to be installable into AGS."""

    name: str
    version: str
    capabilities: frozenset[str]

    def install(self, service: "AGSService") -> None:
        """Receive the service handle. Called once, at registration."""
        ...


@dataclass(frozen=True)
class PluginRecord:
    name: str
    version: str
    capabilities: tuple[str, ...]
    privileged: bool
    description: str

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "privileged": self.privileged,
            "description": self.description,
        }


class PluginRegistry:
    """Fail-closed registry of validated plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, Any] = {}
        self._records: dict[str, PluginRecord] = {}

    # -- validation -----------------------------------------------------
    @staticmethod
    def _validate(plugin: Any) -> PluginRecord:
        if not isinstance(plugin, AGSPlugin):
            raise TypeError("plugin must satisfy the AGSPlugin protocol")
        name = safe_identifier(getattr(plugin, "name", ""), field="plugin name")
        version = safe_identifier(getattr(plugin, "version", ""), field="plugin version")
        raw_caps = getattr(plugin, "capabilities", None)
        if raw_caps is None:
            raise SecurityViolation("plugin must declare capabilities")
        try:
            caps = frozenset(str(c) for c in raw_caps)
        except TypeError as exc:
            raise SecurityViolation("plugin capabilities must be iterable of strings") from exc
        unknown = caps - ALLOWED_CAPABILITIES
        if unknown:
            raise SecurityViolation(f"undeclared plugin capabilities: {sorted(unknown)}")
        if not caps:
            raise SecurityViolation("plugin must declare at least one capability")
        description = bounded_text(
            str(getattr(plugin, "description", "") or ""),
            field="plugin description",
        )
        return PluginRecord(
            name=name,
            version=version,
            capabilities=tuple(sorted(caps)),
            privileged=bool(caps & PRIVILEGED_CAPABILITIES),
            description=description,
        )

    # -- api ------------------------------------------------------------
    def register(self, plugin: Any, *, allow_privileged: bool = False) -> PluginRecord:
        record = self._validate(plugin)
        if record.name in self._plugins:
            raise ValueError(f"duplicate plugin: {record.name}")
        if record.privileged and not allow_privileged:
            raise SecurityViolation(
                f"plugin {record.name!r} declares privileged capabilities "
                f"{sorted(set(record.capabilities) & PRIVILEGED_CAPABILITIES)}; "
                "host must pass allow_privileged=True"
            )
        self._plugins[record.name] = plugin
        self._records[record.name] = record
        return record

    def install(self, service: "AGSService", name: str) -> None:
        """Install a previously registered plugin into a live service."""
        if name not in self._plugins:
            raise KeyError(f"unknown plugin: {name}")
        self._plugins[name].install(service)

    def get(self, name: str) -> Any:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise KeyError(f"unknown plugin: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def by_capability(self, capability: str) -> tuple[str, ...]:
        if capability not in ALLOWED_CAPABILITIES:
            raise KeyError(f"unknown capability: {capability}")
        return tuple(sorted(
            n for n, r in self._records.items() if capability in r.capabilities
        ))

    def describe(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._records[n].describe() for n in sorted(self._records))


@dataclass
class ServiceCapabilities:
    """Machine-readable description of what this AGS build can do.

    An external AI reads this to decide what it can ask for, instead of
    hard-coding assumptions about AGS internals.
    """
    version: str
    field_dimensions: tuple[int, ...]
    field_backends: tuple[str, ...]
    domain_primitives: tuple[str, ...]
    sandbox_available: bool
    quantum_available: bool
    plugins: tuple[str, ...] = field(default_factory=tuple)
    invention_available: bool = False
    #: What the invention layer may and may not claim, stated for the caller.
    epistemic_note: str = (
        "Discovery fits a declared library of terms. Invention searches a "
        "symbolic grammar and reports candidates that are novel only relative "
        "to the supplied corpus, plus open problems formulated from unexplained "
        "residual structure. Nothing is asserted as a law of nature."
    )

    def describe(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "field_dimensions": list(self.field_dimensions),
            "field_backends": list(self.field_backends),
            "domain_primitives": list(self.domain_primitives),
            "sandbox_available": self.sandbox_available,
            "quantum_available": self.quantum_available,
            "plugins": list(self.plugins),
            "invention_available": self.invention_available,
            "epistemic_note": self.epistemic_note,
        }
