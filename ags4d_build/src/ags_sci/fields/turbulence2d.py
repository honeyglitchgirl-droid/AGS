"""Production 2-D decaying turbulence engine for AGS-Sci.

This module contains the reusable solver and referee diagnostics extracted from
EXP-2D-TURB-002.  It deliberately uses classical RK4; the previously explored
integrating-factor RK4 staging is not included because its validation did not
match the reference RK4 integrator closely enough for certification.

Equation
--------
    omega_t + u . grad(omega) = nu * Delta(omega)
    Delta(psi) = -omega
    u = (-psi_y, psi_x)

The solver uses a real-valued rFFT state, hyperspherical 2/3 truncation,
optional scipy.fft workers, and adaptive advective CFL control.  All numerical
claims remain subject to the resolution referee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import numpy as np
from .contracts import FieldCapabilities

try:  # SciPy is optional at package-install time.
    from scipy import fft as _scipy_fft
except Exception:  # pragma: no cover - exercised only without SciPy
    _scipy_fft = None


@dataclass(frozen=True)
class TurbulenceCheckpoint:
    t: float
    kinetic_energy: float
    enstrophy: float
    palinstrophy: float
    r_tail: float
    dE_dt_measured: float
    dE_dt_exact: float
    balance_relative_error: float
    spectral_p: float
    spectral_r2: float


@dataclass(frozen=True)
class TurbulenceRunResult:
    experiment: str
    solver: str
    N: int
    nu: float
    seed: int
    tmax: float
    n_steps: int
    min_dt: float
    max_dt: float
    strict_gate_time: float | None
    hard_gate_time: float | None
    max_tail: float
    max_balance_relative_error: float
    palinstrophy_peak_time: float
    palinstrophy_peak: float
    checkpoints: tuple[TurbulenceCheckpoint, ...]
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["checkpoints"] = [asdict(c) for c in self.checkpoints]
        return d


class FastRFFTTurbulence2D:
    """High-throughput 2-D vorticity solver with referee-grade diagnostics."""

    def __init__(
        self,
        N: int = 512,
        nu: float = 1e-4,
        seed: int = 42,
        k_low: float = 8.0,
        k_high: float = 14.0,
        workers: int = -1,
        cfl: float = 0.40,
        dt_min: float = 1e-6,
        dt_max: float = 0.01,
    ) -> None:
        if N < 16 or N % 2:
            raise ValueError("N must be an even integer >= 16")
        if nu < 0:
            raise ValueError("nu must be non-negative")
        if cfl <= 0 or dt_min <= 0 or dt_max <= 0 or dt_min > dt_max:
            raise ValueError("invalid CFL or timestep bounds")
        if _scipy_fft is None and workers not in (None, 1):
            # We silently fall back to NumPy only when the caller explicitly
            # requested the portable single-worker path.
            workers = 1

        self.N = int(N)
        self.nu = float(nu)
        self.seed = int(seed)
        self.workers = workers
        self.L = 2.0 * np.pi
        self.dx = self.L / self.N
        self.cfl = float(cfl)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

        kx = np.fft.fftfreq(self.N, d=self.dx / (2.0 * np.pi))
        ky = np.fft.rfftfreq(self.N, d=self.dx / (2.0 * np.pi))
        self.KX = kx[:, None]
        self.KY = ky[None, :]
        self.K_sq = self.KX**2 + self.KY**2
        self.K = np.sqrt(self.K_sq)

        self.k_cut = self.N / 3.0
        self.dealias_mask = self.K_sq < self.k_cut**2

        self.inv_K_sq = np.zeros_like(self.K_sq, dtype=np.float64)
        nz = self.K_sq > 0
        self.inv_K_sq[nz] = 1.0 / self.K_sq[nz]

        self.u_mult = 1j * self.KY * self.inv_K_sq
        self.v_mult = -1j * self.KX * self.inv_K_sq
        self.dx_mult = 1j * self.KX
        self.dy_mult = 1j * self.KY
        self.diff_mult = -self.nu * self.K_sq

        self.tail_mask = (
            (self.K_sq >= (0.8 * self.k_cut) ** 2) & self.dealias_mask
        )

        self.k_max_shell = int(self.k_cut)
        self.k_bin = np.rint(self.K).astype(np.int64)
        self.k_bin_flat = self.k_bin.ravel()
        self.k_bins = np.arange(1, self.k_max_shell, dtype=np.int64)

        self.rfft_weight = np.ones_like(self.K_sq, dtype=np.float64)
        if self.rfft_weight.shape[1] > 2:
            self.rfft_weight[:, 1:-1] = 2.0

        self._rng = np.random.default_rng(self.seed)
        self.w_hat = self._initial_condition(k_low, k_high)
        self.shape = (self.N, self.N)
        self.lengths = (self.L, self.L)
        self.capabilities = FieldCapabilities(2, "cartesian-periodic", "vorticity-pseudo-spectral", True, True, True, False)

    def snapshot_state(self) -> np.ndarray:
        return np.array(self.w_hat, copy=True)

    def restore_state(self, state: np.ndarray) -> None:
        a = np.asarray(state)
        if a.shape != self.w_hat.shape or not np.all(np.isfinite(a)):
            raise ValueError("invalid 2D spectral state")
        self.w_hat = np.array(a, dtype=np.complex128, copy=True) * self.dealias_mask

    def state_checksum(self) -> str:
        import hashlib
        return hashlib.sha256(np.ascontiguousarray(self.w_hat).view(np.uint8)).hexdigest()

    def rhs(self) -> np.ndarray:
        return self.rhs_fourier(self.w_hat)[0]

    def _rfft2(self, a: np.ndarray) -> np.ndarray:
        if _scipy_fft is not None:
            return _scipy_fft.rfft2(a, workers=self.workers)
        return np.fft.rfft2(a)

    def _irfft2(self, a: np.ndarray) -> np.ndarray:
        if _scipy_fft is not None:
            return _scipy_fft.irfft2(a, s=(self.N, self.N), workers=self.workers)
        return np.fft.irfft2(a, s=(self.N, self.N))

    def _initial_condition(self, k_low: float, k_high: float) -> np.ndarray:
        noise = self._rng.standard_normal((self.N, self.N))
        noise_hat = self._rfft2(noise)
        band = (self.K >= k_low) & (self.K <= k_high)
        envelope = np.zeros_like(self.K, dtype=np.float64)
        envelope[band] = 1.0 / (1.0 + (self.K[band] - 10.0) ** 2)
        w_hat = noise_hat * envelope * self.dealias_mask
        omega = self._irfft2(w_hat)
        norm = math.sqrt(0.5 * float(np.mean(omega * omega)))
        if not np.isfinite(norm) or norm == 0.0:
            raise RuntimeError("initial-condition normalization failed")
        return (w_hat / norm) * self.dealias_mask

    def nonlinear_advection_hat(self, w_hat: np.ndarray) -> tuple[np.ndarray, float]:
        """Return P_dealias[F(u.grad(omega))] and peak velocity."""
        u = self._irfft2(self.u_mult * w_hat)
        v = self._irfft2(self.v_mult * w_hat)
        wx = self._irfft2(self.dx_mult * w_hat)
        wy = self._irfft2(self.dy_mult * w_hat)
        u_max = float(np.max(np.hypot(u, v)))
        adv_hat = self._rfft2(u * wx + v * wy)
        adv_hat *= self.dealias_mask
        return adv_hat, u_max

    def rhs_fourier(self, w_hat: np.ndarray) -> tuple[np.ndarray, float]:
        adv_hat, u_max = self.nonlinear_advection_hat(w_hat)
        return -adv_hat + self.diff_mult * w_hat, u_max

    def cfl_timestep(self, u_max: float, dt: float | None = None) -> float:
        """Advective CFL timestep, clipped to configured bounds."""
        if not np.isfinite(u_max) or u_max <= 0:
            candidate = self.dt_max
        else:
            candidate = self.cfl / (self.k_cut * u_max)
        if dt is not None:
            candidate = min(candidate, float(dt))
        return float(np.clip(candidate, self.dt_min, self.dt_max))

    def step_rk4(self, dt: float) -> float:
        k1, umax = self.rhs_fourier(self.w_hat)
        k2, _ = self.rhs_fourier(self.w_hat + 0.5 * dt * k1)
        k3, _ = self.rhs_fourier(self.w_hat + 0.5 * dt * k2)
        k4, _ = self.rhs_fourier(self.w_hat + dt * k3)
        self.w_hat += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.w_hat *= self.dealias_mask
        return umax

    def _weighted_modes(self, values: np.ndarray) -> np.ndarray:
        return values * self.rfft_weight

    def diagnostics(self, fit_kmin: float = 15.0, fit_kmax: float = 60.0) -> dict[str, Any]:
        weighted_amp2 = self._weighted_modes(np.abs(self.w_hat) ** 2)
        norm = float(self.N ** 4)
        enstrophy = 0.5 * float(np.sum(weighted_amp2)) / norm
        energy_modes = weighted_amp2 * self.inv_K_sq
        kinetic_energy = 0.5 * float(np.sum(energy_modes)) / norm
        palinstrophy = 0.5 * float(np.sum(weighted_amp2 * self.K_sq)) / norm
        tail = 0.5 * float(np.sum(weighted_amp2[self.tail_mask])) / norm
        r_tail = tail / enstrophy if enstrophy > 0 else 0.0
        shell_values = (0.5 * energy_modes / norm).ravel()
        shell_sums = np.bincount(
            self.k_bin_flat, weights=shell_values, minlength=self.k_max_shell + 1
        )
        E1d = shell_sums[1:self.k_max_shell].copy()
        p, r2 = self.power_law_fit(self.k_bins, E1d, fit_kmin, fit_kmax)
        return {
            "kinetic_energy": kinetic_energy,
            "enstrophy": enstrophy,
            "palinstrophy": palinstrophy,
            "r_tail": float(r_tail),
            "k_bins": self.k_bins.copy(),
            "E_1d": E1d,
            "spectral_p": p,
            "spectral_r2": r2,
        }

    @staticmethod
    def power_law_fit(k: np.ndarray, E: np.ndarray, kmin: float = 15.0,
                      kmax: float = 60.0) -> tuple[float, float]:
        mask = np.isfinite(k) & np.isfinite(E) & (k >= kmin) & (k <= kmax) & (E > 0)
        if np.count_nonzero(mask) < 4:
            return float("nan"), float("nan")
        x = np.log(k[mask].astype(np.float64))
        y = np.log(E[mask].astype(np.float64))
        slope, _ = np.polyfit(x, y, 1)
        yhat = slope * x + np.polyfit(x, y, 1)[1]
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return float(-slope), float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    @staticmethod
    def fixed_power_bic(k: np.ndarray, E: np.ndarray, exponent: float,
                        kmin: float = 15.0, kmax: float = 60.0) -> float:
        mask = np.isfinite(k) & np.isfinite(E) & (k >= kmin) & (k <= kmax) & (E > 0)
        if np.count_nonzero(mask) < 4:
            return float("nan")
        x = np.log(k[mask].astype(float))
        y = np.log(E[mask].astype(float))
        residual = y - (np.mean(y + exponent * x) - exponent * x)
        sse = max(float(np.sum(residual ** 2)), np.finfo(float).tiny)
        n = residual.size
        return float(n * np.log(sse / n) + 1.0 * np.log(n))

    def run(
        self,
        tmax: float = 10.0,
        dt: float | None = None,
        sample_dt: float = 0.2,
        tail_strict: float = 1e-4,
        tail_hard: float = 1e-3,
        fit_kmin: float = 15.0,
        fit_kmax: float = 60.0,
        verbose: bool = False,
    ) -> TurbulenceRunResult:
        if tmax <= 0 or sample_dt <= 0:
            raise ValueError("tmax and sample_dt must be positive")
        requested_dt = None if dt is None else float(dt)
        if requested_dt is not None and requested_dt <= 0:
            raise ValueError("dt must be positive")

        checkpoints: list[TurbulenceCheckpoint] = []
        strict_gate_time = None
        hard_gate_time = None
        max_tail = 0.0
        max_balance = 0.0
        peak_p = -float("inf")
        peak_t = 0.0
        t = 0.0
        step_count = 0
        min_dt_seen = float("inf")
        max_dt_seen = 0.0
        last_sample_t = None
        last_energy = None
        next_sample = 0.0
        start = time.perf_counter()

        while t < tmax - 1e-14:
            # One cheap velocity evaluation is intentionally avoided: RK4's
            # first stage already supplies u_max for the chosen state. For a
            # requested fixed dt we preserve it exactly; adaptive mode derives
            # dt from the previous accepted state's CFL estimate and clips the
            # final step to tmax.
            if requested_dt is None:
                _, u0 = self.nonlinear_advection_hat(self.w_hat)
                step_dt = self.cfl_timestep(u0)
            else:
                step_dt = min(requested_dt, self.dt_max)
                step_dt = max(step_dt, self.dt_min)
            step_dt = min(step_dt, tmax - t)
            if step_dt < self.dt_min and tmax - t > 0:
                raise RuntimeError("CFL timestep fell below dt_min")

            self.step_rk4(step_dt)
            t += step_dt
            step_count += 1
            min_dt_seen = min(min_dt_seen, step_dt)
            max_dt_seen = max(max_dt_seen, step_dt)

            if t + 1e-12 < next_sample and t < tmax - 1e-14:
                continue
            next_sample += sample_dt

            d = self.diagnostics(fit_kmin, fit_kmax)
            E = d["kinetic_energy"]
            Omega = d["enstrophy"]
            exact = -2.0 * self.nu * Omega
            if last_energy is None or last_sample_t is None:
                measured = float("nan")
                balance = float("nan")
            else:
                measured = (E - last_energy) / (t - last_sample_t)
                balance = abs(measured - exact) / max(abs(exact), 1e-15)
                if np.isfinite(balance):
                    max_balance = max(max_balance, balance)
            if d["r_tail"] >= tail_strict and strict_gate_time is None:
                strict_gate_time = float(t)
            if d["r_tail"] >= tail_hard and hard_gate_time is None:
                hard_gate_time = float(t)
            max_tail = max(max_tail, d["r_tail"])
            if d["palinstrophy"] > peak_p:
                peak_p = d["palinstrophy"]
                peak_t = t

            checkpoints.append(TurbulenceCheckpoint(
                t=float(t), kinetic_energy=float(E), enstrophy=float(Omega),
                palinstrophy=float(d["palinstrophy"]), r_tail=float(d["r_tail"]),
                dE_dt_measured=float(measured), dE_dt_exact=float(exact),
                balance_relative_error=float(balance),
                spectral_p=float(d["spectral_p"]), spectral_r2=float(d["spectral_r2"]),
            ))
            last_energy, last_sample_t = E, t
            if verbose:
                print(f"t={t:.4f} E={E:.6e} Omega={Omega:.6e} Rtail={d['r_tail']:.3e}")

        elapsed = time.perf_counter() - start
        return TurbulenceRunResult(
            experiment="EXP-2D-TURB-002",
            solver="FastRFFTTurbulence2D-RK4",
            N=self.N, nu=self.nu, seed=self.seed, tmax=float(tmax),
            n_steps=step_count, min_dt=float(min_dt_seen), max_dt=float(max_dt_seen),
            strict_gate_time=strict_gate_time, hard_gate_time=hard_gate_time,
            max_tail=float(max_tail), max_balance_relative_error=float(max_balance),
            palinstrophy_peak_time=float(peak_t), palinstrophy_peak=float(peak_p),
            checkpoints=tuple(checkpoints), elapsed_seconds=float(elapsed),
        )
