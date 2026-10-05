"""Experimental 3D pseudo-spectral incompressible-flow backend.

This module is deliberately conservative: periodic Cartesian domains only, explicit
resource checks, divergence diagnostics, Leray projection, and benchmark initial
conditions.  It is an experimental backend, not a claim of Navier--Stokes proof.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .operators import SpectralEngineND
from .contracts import FieldCapabilities, validate_state_arrays
from .diagnostics import normalized_divergence, kinetic_energy, energy_rate


@dataclass(frozen=True)
class VectorDiagnostics3D:
    divergence_linf: float
    divergence_l2: float
    kinetic_energy: float
    helicity: float
    enstrophy: float


@dataclass(frozen=True)
class Simulation3DResult:
    state: tuple[np.ndarray, np.ndarray, np.ndarray]
    time: float
    steps: int
    diagnostics: VectorDiagnostics3D


def _check_vector(u, shape):
    if len(u) != 3:
        raise ValueError("3D vector field requires exactly three components")
    out = tuple(np.asarray(x, dtype=float) for x in u)
    if any(x.shape != shape for x in out):
        raise ValueError("all vector components must match engine shape")
    if any(not np.all(np.isfinite(x)) for x in out):
        raise ValueError("vector field contains non-finite values")
    return out


class PseudoSpectral3D:
    """Periodic 3D incompressible pseudo-spectral Navier--Stokes utilities."""

    def __init__(self, shape=(16, 16, 16), domain_lengths=None, dealias="spherical"):
        if len(tuple(shape)) != 3:
            raise ValueError("PseudoSpectral3D requires a 3D shape")
        self.engine = SpectralEngineND(tuple(shape), domain_lengths, dealias=dealias)
        self.shape = self.engine.shape
        self.lengths = self.engine.lengths
        self.capabilities = FieldCapabilities(3, "cartesian-periodic", "pseudo-spectral", True, True, True, True)

    def validate_state(self, u):
        return validate_state_arrays(u, self.shape, 3)

    def normalized_divergence(self, u):
        u = self.validate_state(u)
        return normalized_divergence(self.divergence(u), u, self.lengths)

    def vorticity(self, u):
        return self.curl(self.validate_state(u))

    def energy_rate(self, u, rhs):
        return energy_rate(self.validate_state(u), self.validate_state(rhs))

    def viscous_dissipation_rate(self, u, viscosity):
        u = self.validate_state(u)
        nu = float(viscosity)
        if nu < 0 or not np.isfinite(nu):
            raise ValueError("viscosity must be finite and non-negative")
        omega = self.curl(u)
        return -nu * float(np.mean(sum(w*w for w in omega)))

    def pressure_projection(self, u):
        """Alias emphasizing that the Leray projection is a pressure solve."""
        return self.project(u)

    def gradient(self, field):
        return self.engine.gradient(field)

    def divergence(self, u):
        u = _check_vector(u, self.shape)
        return self.engine.divergence(u)

    def curl(self, u):
        u = _check_vector(u, self.shape)
        du = [self.engine.gradient(c) for c in u]
        # curl u = (dw/dy-dv/dz, du/dz-dw/dx, dv/dx-du/dy)
        return (
            du[2][1] - du[1][2],
            du[0][2] - du[2][0],
            du[1][0] - du[0][1],
        )

    def laplacian(self, u):
        u = _check_vector(u, self.shape)
        return tuple(self.engine.laplacian(c) for c in u)

    def project(self, u):
        """Return the L2-orthogonal Leray projection onto divergence-free fields."""
        u = _check_vector(u, self.shape)
        uh = [np.fft.fftn(c) for c in u]
        k = self.engine.K_grids
        k2 = self.engine.K_sq
        dot = sum(k[i] * uh[i] for i in range(3))
        out = []
        for i in range(3):
            ph = uh[i].copy()
            nz = k2 > 0
            ph[nz] -= k[i][nz] * dot[nz] / k2[nz]
            # Keep the mean mode unchanged; no projection direction exists at k=0.
            ph *= self.engine.dealias_mask
            out.append(np.real(np.fft.ifftn(ph)))
        return tuple(out)

    def nonlinear_advection(self, u):
        """Compute (u . grad)u in physical space, with spectral derivatives."""
        u = _check_vector(u, self.shape)
        grads = [self.engine.gradient(c) for c in u]
        return tuple(sum(u[i] * grads[j][i] for i in range(3)) for j in range(3))

    def rhs(self, u, viscosity=0.0, forcing=None):
        u = _check_vector(u, self.shape)
        nu = float(viscosity)
        if not np.isfinite(nu) or nu < 0:
            raise ValueError("viscosity must be finite and non-negative")
        adv = self.nonlinear_advection(u)
        lap = self.laplacian(u)
        raw = tuple(-adv[i] + nu * lap[i] for i in range(3))
        if forcing is not None:
            f = _check_vector(forcing, self.shape)
            raw = tuple(raw[i] + f[i] for i in range(3))
        return self.project(raw)

    def advance(self, u, dt, viscosity=0.0, forcing=None, cfl=0.5):
        """Single bounded RK step with optional CFL admission check."""
        u = self.validate_state(u)
        h = float(dt)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("dt must be positive and finite")
        suggested = cfl_timestep(u, self.lengths, cfl)
        if np.isfinite(suggested) and h > suggested * (1.0 + 1e-12):
            raise ValueError(f"dt exceeds CFL suggestion: {h} > {suggested}")
        out = low_storage_rk4(lambda x: self.rhs(x, viscosity=viscosity, forcing=forcing), u, h)
        return self.project(out)

    def diagnostics(self, u) -> VectorDiagnostics3D:
        u = _check_vector(u, self.shape)
        div = self.divergence(u)
        omega = self.curl(u)
        energy = 0.5 * float(np.mean(sum(c*c for c in u)))
        helicity = float(np.mean(sum(u[i] * omega[i] for i in range(3))))
        enstrophy = 0.5 * float(np.mean(sum(c*c for c in omega)))
        return VectorDiagnostics3D(
            float(np.max(np.abs(div))), float(np.sqrt(np.mean(div*div))),
            energy, helicity, enstrophy,
        )

    def spectral_energy(self, u):
        u = _check_vector(u, self.shape)
        n = np.prod(self.shape)
        return sum(np.abs(np.fft.fftn(c))**2 for c in u) / (2.0 * n*n)


    def strain_tensor(self, u):
        """Return velocity-gradient strain tensor S_ij=(d_j u_i+d_i u_j)/2."""
        u = _check_vector(u, self.shape)
        grads = [self.engine.gradient(c) for c in u]
        G = np.stack([np.stack(grads[i], axis=0) for i in range(3)], axis=0)
        return 0.5 * (G + np.swapaxes(G, 0, 1))

    def rotation_tensor(self, u):
        u = _check_vector(u, self.shape)
        grads = [self.engine.gradient(c) for c in u]
        G = np.stack([np.stack(grads[i], axis=0) for i in range(3)], axis=0)
        return 0.5 * (G - np.swapaxes(G, 0, 1))

    def q_criterion(self, u):
        S = self.strain_tensor(u)
        O = self.rotation_tensor(u)
        s2 = np.sum(S*S, axis=(0,1))
        o2 = np.sum(O*O, axis=(0,1))
        return 0.5 * (o2 - s2)

    def lambda2_criterion(self, u):
        """Return the middle eigenvalue of S^2+Omega^2 at every grid cell."""
        S = self.strain_tensor(u)
        O = self.rotation_tensor(u)
        M = np.einsum('ij...,jk...->ik...', S, S) + np.einsum('ij...,jk...->ik...', O, O)
        flat = np.moveaxis(M, (0,1), (-2,-1)).reshape(-1, 3, 3)
        vals = np.linalg.eigvalsh(flat)
        return vals[:, 1].reshape(self.shape)

    def energy_spectrum(self, u, bins=None):
        """Shell-binned kinetic-energy spectrum using integer Fourier modes."""
        u = _check_vector(u, self.shape)
        n = float(np.prod(self.shape))
        power = sum(np.abs(np.fft.fftn(c))**2 for c in u) / (2.0*n*n)
        kmag = np.sqrt(sum(M*M for M in self.engine.mode_grids))
        if bins is None:
            bins = np.arange(0, int(np.max(kmag))+2, dtype=float)
        bins = np.asarray(bins, dtype=float)
        if bins.ndim != 1 or len(bins) < 2 or np.any(np.diff(bins) <= 0):
            raise ValueError('bins must be a strictly increasing 1D array with >=2 entries')
        spectrum = np.zeros(len(bins)-1, dtype=float)
        idx = np.digitize(kmag.ravel(), bins) - 1
        p = power.ravel()
        valid = (idx >= 0) & (idx < len(spectrum))
        np.add.at(spectrum, idx[valid], p[valid])
        centers = 0.5*(bins[:-1]+bins[1:])
        return centers, spectrum


# Classical 5-stage, 4th-order low-storage RK coefficients (Carpenter--Kennedy family).
CK54_A = np.array([
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
])
CK54_B = np.array([
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
])


def low_storage_rk4(rhs, state, dt, stages_a=CK54_A, stages_b=CK54_B):
    """Advance a tuple of arrays with a two-register-style 5-stage RK scheme."""
    if len(state) == 0 or len(state) != 3:
        raise ValueError("3D state must have three components")
    h = float(dt)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("dt must be positive and finite")
    y = tuple(np.array(x, dtype=float, copy=True) for x in state)
    residual = tuple(np.zeros_like(x) for x in y)
    for a, b in zip(stages_a, stages_b):
        f = rhs(y)
        residual = tuple(a * residual[i] + h * f[i] for i in range(3))
        y = tuple(y[i] + b * residual[i] for i in range(3))
    return y


def cfl_timestep(u, domain_lengths, cfl=0.5):
    """Return a conservative advective CFL timestep for a Cartesian 3D grid."""
    if len(u) != 3:
        raise ValueError("u must have three components")
    shape = np.asarray(u[0]).shape
    if len(shape) != 3 or any(np.asarray(c).shape != shape for c in u):
        raise ValueError("u components must share a 3D shape")
    L = tuple(float(x) for x in domain_lengths)
    if len(L) != 3 or any(x <= 0 for x in L):
        raise ValueError("domain_lengths must contain three positive lengths")
    c = float(cfl)
    if not np.isfinite(c) or c <= 0:
        raise ValueError("cfl must be positive and finite")
    umax = max(float(np.max(np.abs(c))) for c in u)
    if umax == 0.0:
        return float("inf")
    dx = min(L[i]/shape[i] for i in range(3))
    return c * dx / umax


def taylor_green(shape=(16, 16, 16), domain_lengths=None):
    """Taylor--Green vortex initial condition on a periodic box."""
    shape = tuple(int(x) for x in shape)
    if len(shape) != 3:
        raise ValueError("Taylor-Green requires a 3D shape")
    L = tuple(domain_lengths or (2*np.pi,)*3)
    if len(L) != 3:
        raise ValueError("domain_lengths must have length 3")
    axes = [np.linspace(0, L[i], shape[i], endpoint=False) for i in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    X = 2*np.pi*x/L[0]; Y = 2*np.pi*y/L[1]; Z = 2*np.pi*z/L[2]
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    return (u, v, w)


def abc_flow(shape=(16, 16, 16), A=1.0, B=1.0, C=1.0, domain_lengths=None):
    """Arnold--Beltrami--Childress flow with periodic 2pi-scale coordinates."""
    shape = tuple(int(x) for x in shape)
    if len(shape) != 3:
        raise ValueError("ABC flow requires a 3D shape")
    L = tuple(domain_lengths or (2*np.pi,)*3)
    axes = [np.linspace(0, L[i], shape[i], endpoint=False) for i in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    X = 2*np.pi*x/L[0]; Y = 2*np.pi*y/L[1]; Z = 2*np.pi*z/L[2]
    u = A*np.sin(Z) + C*np.cos(Y)
    v = B*np.sin(X) + A*np.cos(Z)
    w = C*np.sin(Y) + B*np.cos(X)
    return (u, v, w)


def two_vortex_reconnection_seed(shape=(32, 32, 32), separation=1.5, core=0.25):
    """Deterministic anti-parallel Gaussian vortex-pair seed.

    This is an initial-condition generator, not a claim that the subsequent
    dynamics reproduce a canonical reconnection experiment at arbitrary resolution.
    """
    shape = tuple(int(x) for x in shape)
    if len(shape) != 3 or any(n < 8 for n in shape):
        raise ValueError("reconnection seed requires a 3D grid with >=8 cells/axis")
    axes = [np.linspace(-np.pi, np.pi, n, endpoint=False) for n in shape]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    r1 = (y-separation/2)**2 + z**2
    r2 = (y+separation/2)**2 + z**2
    envelope = np.exp(-r1/(2*core**2)) - np.exp(-r2/(2*core**2))
    # Stream-function-like seed for predominantly x-directed, opposite vortices.
    u = np.zeros_like(x)
    v = -z * envelope
    w = y * envelope
    return (u, v, w)


def taylor_green_dns_reference():
    """Small immutable reference trajectory used only for regression calibration.

    These values are intentionally normalized diagnostics, not a claim of a
    universal DNS solution. Larger reference tables should be supplied as
    external resources and versioned separately.
    """
    return {
        0.0: {"kinetic_energy": 0.125, "enstrophy": 0.75},
    }


def anti_parallel_vortex_seed(shape=(32, 32, 32), separation=1.5, core=0.25):
    """Alias with explicit naming for the anti-parallel reconnection benchmark."""
    return two_vortex_reconnection_seed(shape, separation=separation, core=core)


def invariant_diagnostics_3d(engine, u):
    """Return rotation-invariant scalar diagnostics for tensor-SINDy libraries."""
    u = _check_vector(u, engine.shape)
    S = engine.strain_tensor(u)
    O = engine.rotation_tensor(u)
    omega = engine.curl(u)
    I1 = np.trace(S, axis1=0, axis2=1)
    S2 = np.einsum('ij...,ji...->...', S, S)
    O2 = np.einsum('ij...,ji...->...', O, O)
    S3 = np.einsum('ij...,jk...,ki...->...', S, S, S)
    SO2 = np.einsum('ij...,jk...,ki...->...', S, O, O)
    W2 = np.sum(np.asarray(omega)**2, axis=0)
    return {
        "divergence": I1,
        "strain_sq": S2,
        "rotation_sq": O2,
        "strain_cubic": S3,
        "strain_rotation_sq": SO2,
        "vorticity_sq": W2,
    }
