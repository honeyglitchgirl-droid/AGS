"""Forced 2-D turbulence referee engine and integrated enstrophy audit.

Implements the EXP-2D-TURB-003-B protocol inside AGS-Sci:

    omega_t + u.grad(omega) = -alpha*omega + nu*Delta(omega) + f
    Delta(psi) = -omega,  u = (-psi_y, psi_x)

The OU forcing is advanced once per accepted deterministic RK4 step and then
held fixed across the four RK4 stages. Enstrophy injection and dissipation are
integrated with the same RK4 weights as the state update. This produces two
separate audit quantities:

    J_budget = |Ieps - ID - DeltaOmega| / Ieps
    J_stat   = |Ieps - ID| / Ieps

J_budget tests numerical/accounting consistency; J_stat is the physical
stationarity gate and remains immutable at 0.05 for the 003-B protocol.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    from scipy import fft as _scipy_fft
except Exception:  # pragma: no cover
    _scipy_fft = None


@dataclass(frozen=True)
class StationarityAudit:
    experiment: str
    N: int
    nu: float
    alpha: float
    sigma_f: float
    tau_f: float
    t_start: float
    t_end: float
    omega_start: float
    omega_end: float
    delta_omega: float
    I_eps: float
    I_diss: float
    J_budget: float
    J_stat: float
    max_r_tail: float
    drift_fraction: float
    budget_pass: bool
    stationarity_pass: bool
    drift_pass: bool
    admissibility_pass: bool
    safety_pass: bool

    @property
    def certified(self) -> bool:
        return (
            self.budget_pass
            and self.stationarity_pass
            and self.drift_pass
            and self.admissibility_pass
            and self.safety_pass
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"certified": self.certified}


@dataclass(frozen=True)
class ForcedCheckpoint:
    t: float
    step_count: int
    enstrophy: float
    palinstrophy: float
    r_tail: float
    I_eps: float
    I_diss: float


class ForcedRFFTTurbulence2D:
    """Production forced 2-D vorticity solver with stage-consistent audit."""

    EXPERIMENT = "EXP-2D-TURB-003-B"
    DT_DEFAULT = 0.0035
    TAIL_STRICT = 1e-4
    TAIL_HARD = 1e-3
    J_BUDGET_MAX = 1e-3
    J_STAT_MAX = 0.05
    DRIFT_MAX = 0.05

    def __init__(
        self,
        N: int = 512,
        nu: float = 1e-4,
        alpha: float = 0.075,
        sigma_f: float = 5615.0,
        tau_f: float = 1.0,
        seed: int = 42,
        dt: float = DT_DEFAULT,
        workers: int = -1,
    ) -> None:
        if N < 16 or N % 2:
            raise ValueError("N must be an even integer >= 16")
        if nu < 0 or alpha < 0 or sigma_f < 0 or tau_f <= 0 or dt <= 0:
            raise ValueError("invalid physical or timestep parameter")
        if _scipy_fft is None and workers not in (None, 1):
            workers = 1
        self.N = int(N)
        self.nu = float(nu)
        self.alpha = float(alpha)
        self.sigma_f = float(sigma_f)
        self.tau_f = float(tau_f)
        self.seed = int(seed)
        self.dt = float(dt)
        self.workers = workers
        self.L = 2.0 * np.pi
        self.dx = self.L / self.N

        kx = np.fft.fftfreq(self.N, d=self.dx / (2.0 * np.pi))
        ky = np.fft.rfftfreq(self.N, d=self.dx / (2.0 * np.pi))
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.K_sq = self.KX**2 + self.KY**2
        self.K = np.sqrt(self.K_sq)
        self.k_cut = self.N / 3.0
        self.dealias = self.K_sq < self.k_cut**2
        nz = self.K_sq > 0
        self.inv_K_sq = np.zeros_like(self.K_sq)
        self.inv_K_sq[nz] = 1.0 / self.K_sq[nz]

        # Locked production chirality: Delta psi=-omega, u=(-psi_y,psi_x).
        self.u_mult = -1j * self.KY * self.inv_K_sq
        self.v_mult = 1j * self.KX * self.inv_K_sq
        self.dx_mult = 1j * self.KX
        self.dy_mult = 1j * self.KY

        self.weights = np.full(self.K_sq.shape, 2.0, dtype=np.float64)
        self.weights[:, 0] = 1.0
        self.weights[:, -1] = 1.0
        self.forcing_mask = (self.K >= 4.0) & (self.K <= 6.0) & self.dealias
        self.tail_mask = (self.K >= 0.8 * self.k_cut) & self.dealias

        # Legacy MT19937 is used because the 003-B checkpoints serialize its
        # five-part numpy.random state exactly.
        self.rng = np.random.RandomState(self.seed)
        self.w_hat = np.zeros_like(self.K_sq, dtype=np.complex128)
        self.f_hat = np.zeros_like(self.K_sq, dtype=np.complex128)
        self.noise_buf = np.zeros_like(self.K_sq, dtype=np.complex128)
        self.w_stage = np.zeros_like(self.K_sq, dtype=np.complex128)
        self.k1 = np.zeros_like(self.K_sq, dtype=np.complex128)
        self.k2 = np.zeros_like(self.K_sq, dtype=np.complex128)
        self.k3 = np.zeros_like(self.K_sq, dtype=np.complex128)
        self.k4 = np.zeros_like(self.K_sq, dtype=np.complex128)
        self.t = 0.0
        self.step_count = 0
        self.I_eps = 0.0
        self.I_diss = 0.0
        self.window_t0 = 0.0
        self.window_omega0 = 0.0
        self.window_omega_history: list[tuple[float, float]] = []
        self.max_r_tail = 0.0

    def _rfft2(self, x: np.ndarray) -> np.ndarray:
        if _scipy_fft is not None:
            return _scipy_fft.rfft2(x, workers=self.workers)
        return np.fft.rfft2(x)

    def _irfft2(self, x: np.ndarray) -> np.ndarray:
        if _scipy_fft is not None:
            return _scipy_fft.irfft2(x, s=(self.N, self.N), workers=self.workers)
        return np.fft.irfft2(x, s=(self.N, self.N))

    def fill_noise(self) -> None:
        self.noise_buf[:] = (
            self.rng.randn(*self.K_sq.shape)
            + 1j * self.rng.randn(*self.K_sq.shape)
        ) / np.sqrt(2.0)
        # Hermitian symmetry on ky=0 is sufficient for the rFFT half-plane;
        # the production forcing ring does not touch ky=N/2.
        pos = np.arange(1, self.N // 2)
        neg = self.N - pos
        self.noise_buf[neg, 0] = np.conj(self.noise_buf[pos, 0])
        self.noise_buf[0, 0] = np.real(self.noise_buf[0, 0])
        self.noise_buf[self.N // 2, 0] = np.real(self.noise_buf[self.N // 2, 0])
        self.noise_buf *= self.forcing_mask

    def _rates(self, w: np.ndarray) -> tuple[float, float]:
        modes = np.abs(w) ** 2 * self.weights
        norm = float(self.N ** 4)
        omega = 0.5 * float(np.sum(modes)) / norm
        palinstrophy = 0.5 * float(np.sum(modes * self.K_sq)) / norm
        return omega, 2.0 * self.alpha * omega + 2.0 * self.nu * palinstrophy

    def _eval(self, w: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        u = self._irfft2(self.u_mult * w)
        v = self._irfft2(self.v_mult * w)
        wx = self._irfft2(self.dx_mult * w)
        wy = self._irfft2(self.dy_mult * w)
        adv_hat = self._rfft2(u * wx + v * wy)
        adv_hat *= self.dealias
        rhs = -adv_hat - (self.alpha + self.nu * self.K_sq) * w + self.f_hat
        omega, diss = self._rates(w)
        norm = float(self.N ** 4)
        eps = float(np.sum(self.weights * np.real(self.f_hat * np.conj(w)) * self.forcing_mask) / norm)
        return rhs, eps, diss, omega

    def diagnostics(self) -> dict[str, float]:
        omega, diss = self._rates(self.w_hat)
        modes = np.abs(self.w_hat) ** 2 * self.weights
        norm = float(self.N ** 4)
        pal = 0.5 * float(np.sum(modes * self.K_sq)) / norm
        tail = 0.5 * float(np.sum(modes[self.tail_mask])) / norm
        rtail = tail / omega if omega > 0 else 0.0
        self.max_r_tail = max(self.max_r_tail, rtail)
        return {"enstrophy": omega, "palinstrophy": pal, "dissipation": diss, "r_tail": rtail}

    def _advance_one(self, dt: float) -> None:
        phi = math.exp(-dt / self.tau_f)
        self.fill_noise()
        self.f_hat[:] = phi * self.f_hat + self.sigma_f * math.sqrt(1.0 - phi * phi) * self.noise_buf

        k1, e1, d1, o1 = self._eval(self.w_hat)
        self.w_stage[:] = self.w_hat + 0.5 * dt * k1
        k2, e2, d2, _ = self._eval(self.w_stage)
        self.w_stage[:] = self.w_hat + 0.5 * dt * k2
        k3, e3, d3, _ = self._eval(self.w_stage)
        self.w_stage[:] = self.w_hat + dt * k3
        k4, e4, d4, _ = self._eval(self.w_stage)

        self.I_eps += dt * (e1 + 2.0 * e2 + 2.0 * e3 + e4) / 6.0
        self.I_diss += dt * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
        self.w_hat += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        self.w_hat *= self.dealias
        self.t += dt
        self.step_count += 1
        self.window_omega_history.append((self.t, o1))

    def advance(self, t_target: float, checkpoint_every: float | None = None) -> tuple[ForcedCheckpoint, ...]:
        if t_target < self.t - 1e-12:
            raise ValueError("t_target must not precede current time")
        out: list[ForcedCheckpoint] = []
        next_cp = self.t + checkpoint_every if checkpoint_every else float("inf")
        while self.t < t_target - 1e-12:
            dt = min(self.dt, t_target - self.t)
            self._advance_one(dt)
            d = self.diagnostics()
            if checkpoint_every and self.t + 1e-12 >= next_cp:
                out.append(ForcedCheckpoint(self.t, self.step_count, d["enstrophy"], d["palinstrophy"], d["r_tail"], self.I_eps, self.I_diss))
                next_cp += checkpoint_every
        return tuple(out)

    def begin_audit_window(self) -> None:
        d = self.diagnostics()
        self.window_t0 = self.t
        self.window_omega0 = d["enstrophy"]
        self.I_eps = 0.0
        self.I_diss = 0.0
        self.window_omega_history = [(self.t, d["enstrophy"])]
        self.max_r_tail = d["r_tail"]

    def audit(self, drift_fraction: float | None = None) -> StationarityAudit:
        d = self.diagnostics()
        delta = d["enstrophy"] - self.window_omega0
        j_budget = abs(self.I_eps - self.I_diss - delta) / max(abs(self.I_eps), np.finfo(float).tiny)
        j_stat = abs(self.I_eps - self.I_diss) / max(abs(self.I_eps), np.finfo(float).tiny)
        if drift_fraction is None and len(self.window_omega_history) >= 2:
            ts = np.asarray([x[0] for x in self.window_omega_history], dtype=float)
            os = np.asarray([x[1] for x in self.window_omega_history], dtype=float)
            slope = float(np.polyfit(ts, os, 1)[0]) if len(ts) >= 2 else 0.0
            mean_o = float(np.mean(os))
            drift_fraction = abs(slope) * (self.t - self.window_t0) / max(mean_o, np.finfo(float).tiny)
        drift_fraction = float(0.0 if drift_fraction is None else drift_fraction)
        return StationarityAudit(
            experiment=self.EXPERIMENT, N=self.N, nu=self.nu, alpha=self.alpha,
            sigma_f=self.sigma_f, tau_f=self.tau_f, t_start=self.window_t0,
            t_end=self.t, omega_start=self.window_omega0, omega_end=d["enstrophy"],
            delta_omega=delta, I_eps=self.I_eps, I_diss=self.I_diss,
            J_budget=float(j_budget), J_stat=float(j_stat), max_r_tail=float(self.max_r_tail),
            drift_fraction=drift_fraction, budget_pass=j_budget <= self.J_BUDGET_MAX,
            stationarity_pass=j_stat <= self.J_STAT_MAX, drift_pass=drift_fraction <= self.DRIFT_MAX,
            admissibility_pass=self.max_r_tail < self.TAIL_STRICT,
            safety_pass=self.max_r_tail < self.TAIL_HARD,
        )

    def rng_state(self) -> tuple[Any, ...]:
        return self.rng.get_state()

    def save_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = self.rng_state()
        cfg = {
            "experiment": self.EXPERIMENT, "N": self.N, "nu": self.nu,
            "alpha": self.alpha, "sigma_f": self.sigma_f, "tau_f": self.tau_f,
            "seed": self.seed, "dt": self.dt, "workers": self.workers,
        }
        tmp = p.with_name(p.name + ".tmp")
        np.savez_compressed(
            tmp, w_hat=self.w_hat, f_hat=self.f_hat, t=self.t, step_count=self.step_count,
            I_eps=self.I_eps, I_diss=self.I_diss, window_t0=self.window_t0,
            window_omega0=self.window_omega0, max_r_tail=self.max_r_tail,
            rng_name=state[0], rng_keys=state[1], rng_pos=state[2],
            rng_has_gauss=state[3], rng_cached_gauss=state[4], config_json=json.dumps(cfg),
        )
        actual_tmp = Path(str(tmp) + ".npz") if not tmp.name.endswith(".npz") else tmp
        actual_tmp.replace(p)

    @classmethod
    def from_checkpoint(cls, path: str | Path, workers: int | None = None) -> "ForcedRFFTTurbulence2D":
        data = np.load(path, allow_pickle=False)
        cfg = json.loads(str(data["config_json"]))
        obj = cls(workers=cfg.get("workers", -1) if workers is None else workers, **{k: cfg[k] for k in ("N", "nu", "alpha", "sigma_f", "tau_f", "seed", "dt")})
        obj.w_hat[:] = data["w_hat"]
        obj.f_hat[:] = data["f_hat"]
        obj.t = float(data["t"])
        obj.step_count = int(data["step_count"])
        obj.I_eps = float(data["I_eps"])
        obj.I_diss = float(data["I_diss"])
        obj.window_t0 = float(data["window_t0"])
        obj.window_omega0 = float(data["window_omega0"])
        obj.max_r_tail = float(data["max_r_tail"])
        obj.rng.set_state((str(data["rng_name"]), data["rng_keys"], int(data["rng_pos"]), int(data["rng_has_gauss"]), float(data["rng_cached_gauss"])))
        return obj
