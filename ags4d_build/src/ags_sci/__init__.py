"""AGS-Sci v102 — causally grounded autonomous scientific discovery engine."""
__version__ = "102"

from .api import AGSService, DiscoveredLaw
from .plugins import AGSPlugin, PluginRegistry, ServiceCapabilities, ALLOWED_CAPABILITIES
from .quantum import (QuantumCircuit, SimulatorBackend, ansatz_ry, expectation,
                      pauli_z, quantum_walk_mixing, vqe_ground_energy)
from .core.evolution import (PassiveEvolutionEngine, EvolutionObservation,
                             ParameterProposal, EvolutionProposal)

__all__ = ["__version__", "AGSService", "DiscoveredLaw", "AGSPlugin", "PluginRegistry",
           "ServiceCapabilities", "ALLOWED_CAPABILITIES", "QuantumCircuit",
           "SimulatorBackend", "ansatz_ry", "expectation", "pauli_z",
           "quantum_walk_mixing", "vqe_ground_energy", "PassiveEvolutionEngine",
           "EvolutionObservation", "ParameterProposal", "EvolutionProposal"]
