"""Quantum computation layer for AGS-Sci.

Scope and honesty
-----------------
This module provides a **statevector simulator** and a backend abstraction. It
exists so that quantum *algorithms* can be prototyped and verified inside AGS,
and so that a hardware or cloud backend can be dropped in behind the same
interface.

It does **not** make AGS's classical numerics faster. A statevector simulation
costs O(2^n) memory and O(2^n) or worse time per gate, so simulating a quantum
algorithm is strictly *more* expensive than the equivalent classical computation
for the problem sizes AGS handles. Quantum advantage, where it exists, comes
from asymptotics on hardware that does not exist in this file. Treat results
from :class:`SimulatorBackend` as a reference model for algorithm development,
not as a speedup.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------
@runtime_checkable
class QuantumBackend(Protocol):
    """What AGS needs from a quantum execution backend."""

    name: str
    max_qubits: int

    def run(self, circuit: "QuantumCircuit", shots: int = 1024) -> dict[str, int]:
        """Execute ``circuit`` and return a bitstring histogram."""
        ...


@dataclass(frozen=True)
class SimulatorMetadata:
    name: str
    simulated: bool
    max_qubits: int
    note: str


class SimulatorBackend:
    """Exact statevector simulator with an optional shot-sampling readout."""

    def __init__(self, max_qubits: int = 20, seed: int = 0):
        if not 1 <= int(max_qubits) <= 28:
            raise ValueError("max_qubits must be in 1..28 (2^28 complex128 ~ 4 GiB)")
        self.name = "statevector-simulator"
        self.max_qubits = int(max_qubits)
        self._rng = np.random.default_rng(int(seed))

    def metadata(self) -> SimulatorMetadata:
        return SimulatorMetadata(
            name=self.name,
            simulated=True,
            max_qubits=self.max_qubits,
            note="Classical simulation. No quantum speedup; cost grows as 2^n.",
        )

    def statevector(self, circuit: "QuantumCircuit") -> np.ndarray:
        n = circuit.n_qubits
        if n > self.max_qubits:
            raise ValueError(f"circuit needs {n} qubits; simulator allows {self.max_qubits}")
        psi = np.zeros(2 ** n, dtype=complex)
        psi[0] = 1.0
        for gate in circuit.gates:
            psi = gate.apply(psi, n)
        return psi

    def probabilities(self, circuit: "QuantumCircuit") -> np.ndarray:
        psi = self.statevector(circuit)
        return np.abs(psi) ** 2

    def run(self, circuit: "QuantumCircuit", shots: int = 1024) -> dict[str, int]:
        if int(shots) < 1:
            raise ValueError("shots must be >= 1")
        p = self.probabilities(circuit)
        # Zero-probability outcomes are never sampled.
        idx = self._rng.choice(len(p), size=int(shots), p=p)
        labels = [format(i, f"0{circuit.n_qubits}b") for i in idx]
        out: dict[str, int] = {}
        for lab in labels:
            out[lab] = out.get(lab, 0) + 1
        return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Gate:
    """A named single/two-qubit gate acting on explicit qubit indices."""

    name: str
    targets: tuple[int, ...]
    matrix: np.ndarray
    parameter: float | None = None

    def apply(self, psi: np.ndarray, n: int) -> np.ndarray:
        if any(not 0 <= t < n for t in self.targets):
            raise ValueError(f"gate {self.name} target outside register")
        k = len(self.targets)
        if self.matrix.shape != (2 ** k, 2 ** k):
            raise ValueError(f"gate {self.name} matrix does not match its targets")
        psi = psi.reshape([2] * n)
        # Move the acted-on axes to the front, apply, then move them back.
        rest = [i for i in range(n) if i not in self.targets]
        psi = np.transpose(psi, list(self.targets) + rest)
        psi = psi.reshape(2 ** k, -1)
        psi = self.matrix @ psi
        psi = psi.reshape([2] * k + [2] * (n - k))
        psi = np.transpose(psi, np.argsort(list(self.targets) + rest))
        return psi.reshape(2 ** n)


_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_CNOT = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=complex)
_SWAP = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------
class QuantumCircuit:
    """A minimal, explicit gate-list circuit."""

    def __init__(self, n_qubits: int):
        if int(n_qubits) < 1:
            raise ValueError("a circuit needs at least one qubit")
        self.n_qubits = int(n_qubits)
        self.gates: list[Gate] = []

    def _add(self, gate: Gate) -> "QuantumCircuit":
        # Validate targets when the gate is added, not when it is applied, so an
        # out-of-range circuit is rejected at the point of construction rather
        # than surfacing later as a confusing reshape error.
        for t in gate.targets:
            if not 0 <= t < self.n_qubits:
                raise ValueError(
                    f"gate {gate.name} target {t} outside register of {self.n_qubits}")
        self.gates.append(gate)
        return self

    def h(self, q: int) -> "QuantumCircuit":
        return self._add(Gate("H", (q,), _H))

    def x(self, q: int) -> "QuantumCircuit":
        return self._add(Gate("X", (q,), _X))

    def y(self, q: int) -> "QuantumCircuit":
        return self._add(Gate("Y", (q,), _Y))

    def z(self, q: int) -> "QuantumCircuit":
        return self._add(Gate("Z", (q,), _Z))

    def s(self, q: int) -> "QuantumCircuit":
        return self._add(Gate("S", (q,), _S))

    def rx(self, q: int, theta: float) -> "QuantumCircuit":
        return self._add(Gate("RX", (q,), rx(float(theta)), float(theta)))

    def ry(self, q: int, theta: float) -> "QuantumCircuit":
        return self._add(Gate("RY", (q,), ry(float(theta)), float(theta)))

    def rz(self, q: int, theta: float) -> "QuantumCircuit":
        return self._add(Gate("RZ", (q,), rz(float(theta)), float(theta)))

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        if control == target:
            raise ValueError("CNOT control and target must differ")
        return self._add(Gate("CNOT", (control, target), _CNOT))

    def swap(self, a: int, b: int) -> "QuantumCircuit":
        if a == b:
            raise ValueError("SWAP operands must differ")
        return self._add(Gate("SWAP", (a, b), _SWAP))

    def depth(self) -> int:
        return len(self.gates)

    def describe(self) -> dict[str, object]:
        return {"n_qubits": self.n_qubits, "depth": self.depth(),
                "gates": [g.name for g in self.gates]}


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------
def expectation(circuit: QuantumCircuit, observable: np.ndarray,
                backend: QuantumBackend | None = None) -> float:
    """Return <psi|O|psi> for a Hermitian observable ``observable``."""
    backend = backend or SimulatorBackend(max_qubits=max(1, circuit.n_qubits))
    if not isinstance(backend, SimulatorBackend):
        raise TypeError("expectation() currently requires the simulator backend")
    psi = backend.statevector(circuit)
    if observable.shape != (len(psi), len(psi)):
        raise ValueError("observable dimension does not match the register")
    if not np.allclose(observable, observable.conj().T, atol=1e-12):
        raise ValueError("observable must be Hermitian")
    return float(np.real(np.vdot(psi, observable @ psi)))


def pauli_z(n_qubits: int, qubit: int) -> np.ndarray:
    """Full-register Z acting on one qubit, identity elsewhere."""
    if not 0 <= qubit < n_qubits:
        raise ValueError("qubit out of range")
    ops: list[np.ndarray] = [_I2] * n_qubits
    ops[qubit] = _Z
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def pauli_string(n_qubits: int, spec: Mapping) -> np.ndarray:
    """Tensor product of Paulis, e.g. {0: 'X', 1: 'Z'}."""
    table = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}
    ops: list[np.ndarray] = []
    for q in range(n_qubits):
        label = spec.get(q, "I")
        if label not in table:
            raise ValueError(f"unknown Pauli {label!r}")
        ops.append(table[label])
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def ansatz_ry(n_qubits: int, params: Sequence[float]) -> QuantumCircuit:
    """Hardware-efficient Ry ansatz with a linear CNOT entangler."""
    if len(params) != n_qubits:
        raise ValueError("one parameter per qubit is required")
    c = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        c.ry(q, params[q])
    for q in range(n_qubits - 1):
        c.cnot(q, q + 1)
    return c


def vqe_ground_energy(hamiltonian: np.ndarray, n_qubits: int,
                      restarts: int = 4, seed: int = 0,
                      maxiter: int = 200) -> dict[str, object]:
    """Variational quantum eigensolver over the Ry ansatz.

    Uses a Nelder-Mead-free coordinate search so the result is deterministic and
    dependency-light. Returns the best energy found plus the parameters; it is a
    *bounded variational* estimate, not a proof of the ground state.
    """
    if hamiltonian.shape != (2 ** n_qubits, 2 ** n_qubits):
        raise ValueError("hamiltonian dimension does not match the register")
    rng = np.random.default_rng(int(seed))
    best: dict[str, object] | None = None
    scale = float(np.max(np.abs(hamiltonian))) or 1.0
    for _ in range(int(restarts)):
        params = rng.uniform(-np.pi, np.pi, size=n_qubits)
        energy = expectation(ansatz_ry(n_qubits, params), hamiltonian)
        step = 0.5
        for _ in range(int(maxiter)):
            improved = False
            for q in range(n_qubits):
                for sign in (+1.0, -1.0):
                    trial = params.copy()
                    trial[q] += sign * step
                    trial[q] = (trial[q] + np.pi) % (2 * np.pi) - np.pi
                    e = expectation(ansatz_ry(n_qubits, trial), hamiltonian)
                    if e < energy - 1e-15:
                        params, energy, improved = trial, e, True
                        break
                if improved:
                    break
            if not improved:
                step *= 0.5
                if step < 1e-9:
                    break
        if best is None or energy < best["energy"]:  # type: ignore[index]
            best = {"energy": float(energy), "parameters": params.tolist(), "n_qubits": n_qubits}
    assert best is not None
    best["hamiltonian_scale"] = scale
    return best


def quantum_walk_mixing(n_sites: int, steps: int) -> np.ndarray:
    """Continuous-time quantum walk on a cycle; returns the position distribution.

    Provided as a quantum-algorithm primitive for AGS experiments. This is a
    simulation of the walk, not a hardware execution.
    """
    n = int(n_sites)
    if n < 2:
        raise ValueError("a cycle needs at least 2 sites")
    k = int(steps)
    if k < 0:
        raise ValueError("steps must be non-negative")
    # The cycle adjacency is circulant, so it is diagonalised by the unitary DFT:
    # A = F^dagger diag(lambda) F, and A^k = F^dagger diag(lambda^k) F.
    j = np.arange(n)
    eig = 2 * np.cos(2 * np.pi * j / n)
    F = np.exp(2j * np.pi * np.outer(j, j) / n) / np.sqrt(n)
    # Continuous-time evolution exp(-i A k). The phase factor must be a length-n
    # vector: shaping it as a column would broadcast against (F @ psi0) into an
    # (n, n) matrix and silently return unnormalised probabilities.
    phase = np.exp(-1j * eig * k)
    psi0 = np.zeros(n, dtype=complex)
    psi0[0] = 1.0
    psi = F.conj().T @ (phase * (F @ psi0))
    prob = np.abs(psi) ** 2
    # Guard against drift so callers always receive a distribution.
    total = float(prob.sum())
    if total <= 0:
        raise ValueError("quantum walk produced an empty distribution")
    return prob / total

