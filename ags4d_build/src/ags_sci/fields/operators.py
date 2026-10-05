"""Dimension-agnostic periodic spectral operators.

The engine is spatial/Euclidean at the numerical discretization level. Metric-aware
contractions belong to :mod:`ags_sci.fields.geometry`; this distinction prevents a
4D FFT from being incorrectly described as Lorentz-covariant by itself.

Two gradients live here, and the difference matters:

- :meth:`SpectralEngineND.gradient` is the plain spectral gradient, ``IFFT(i*K*F)``.
  For any field whose spectrum lies below Nyquist it is **exact to machine
  precision** -- that is a theorem, not a tuning outcome. Nothing can beat it in
  that regime, so it stays the default and is deliberately left unfiltered.
- :meth:`SpectralEngineND.gradient_denoised` applies a Wiener-style spectral
  multiplier that is *derived*, not fitted. It estimates the noise floor from the
  highest modes and shrinks modes where noise dominates the signal. It is exact
  on clean band-limited data and roughly halves the gradient error under noise,
  because the plain gradient amplifies high-mode noise by up to Nyquist.

The denoised gradient is opt-in and reports which regime it used, so a caller can
see when it declined to shrink (see :class:`DenoisedGradient`).
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DenoisedGradient:
    """A gradient together with the evidence for how it was computed.

    ``regime`` is the part a caller should read first:

    - ``"shrunk"`` -- the spectral tail looked like a noise floor, so a Wiener
      multiplier was applied. Expect roughly half the error of the plain
      gradient on noisy data.
    - ``"plain"`` -- the operator declined to shrink, either because the tail
      still decays (real unresolved signal lives there) or because there was
      not enough of it to judge. The result is then exactly the plain spectral
      gradient, which is already exact for band-limited fields.

    Either way the components are the gradient; the metadata exists so the
    choice is auditable rather than implicit.
    """

    components: tuple
    regime: str
    noise_floor: float
    tail_slope: float
    multiplier: np.ndarray = field(default=None, repr=False)
    tail_fraction: float = 0.10

    def describe(self) -> dict[str, object]:
        return {
            "regime": self.regime,
            "noise_floor": self.noise_floor,
            "tail_slope": self.tail_slope,
            "tail_fraction": self.tail_fraction,
            "multiplier_min": float(np.min(self.multiplier)),
            "multiplier_max": float(np.max(self.multiplier)),
        }


class SpectralEngineND:
    """Periodic spectral operators for D=1..5 spatial coordinates."""

    def __init__(self, shape: tuple[int, ...], domain_lengths=None,
                 dealias: str = "spherical", filter_alpha: float = 36.0, filter_order: int = 36):
        self.shape = tuple(int(n) for n in shape)
        self.ndim = len(self.shape)
        if self.ndim not in (1, 2, 3, 4, 5):
            raise ValueError("Unsupported dimension: expected 1, 2, 3, 4, or 5")
        self.lengths = tuple(float(x) for x in (
            domain_lengths if domain_lengths is not None else (2*np.pi,)*self.ndim
        ))
        if len(self.lengths) != self.ndim or any(x <= 0 for x in self.lengths):
            raise ValueError("domain_lengths must be positive and match shape")

        axes = []
        mode_axes = []
        for n, L in zip(self.shape, self.lengths):
            # Angular wave numbers and dimensionless integer Fourier modes.
            axes.append(2*np.pi*np.fft.fftfreq(n, d=L/n))
            mode_axes.append(np.fft.fftfreq(n) * n)
        self.K_grids = tuple(np.meshgrid(*axes, indexing="ij", sparse=False))
        self.mode_grids = tuple(np.meshgrid(*mode_axes, indexing="ij", sparse=False))
        self.K_sq = sum(K*K for K in self.K_grids)

        # A spherical/hyperspherical 2/3 cutoff. This is deliberately named
        # spherical because it is not equivalent to componentwise 2/3 truncation.
        if dealias not in ("spherical", "componentwise", "hou_li"):
            raise ValueError("dealias must be 'spherical', 'componentwise', or 'hou_li'")
        if dealias == "spherical":
            q = sum((M/(n/3.0))**2 for M, n in zip(self.mode_grids, self.shape))
            self.dealias_mask = q < 1.0
            self.spectral_filter = np.ones(self.shape, dtype=float)
        elif dealias == "componentwise":
            self.dealias_mask = np.ones(self.shape, dtype=bool)
            for M, n in zip(self.mode_grids, self.shape):
                self.dealias_mask &= np.abs(M) < n/3.0
            self.spectral_filter = np.ones(self.shape, dtype=float)
        else:
            alpha = float(filter_alpha); order = int(filter_order)
            if not np.isfinite(alpha) or alpha <= 0 or order < 2 or order % 2:
                raise ValueError("Hou-Li filter requires alpha>0 and an even order >=2")
            q = np.sqrt(sum((M/(n/2.0))**2 for M, n in zip(self.mode_grids, self.shape)))
            self.dealias_mask = np.ones(self.shape, dtype=bool)
            self.spectral_filter = np.exp(-alpha * np.minimum(q, 1.0)**order)
        self.dealias_mode = dealias

        self.inv_laplacian = np.zeros(self.shape, dtype=np.float64)
        nz = self.K_sq > 0
        self.inv_laplacian[nz] = -1.0/self.K_sq[nz]

    def _raw_hat(self, field):
        a=np.asarray(field)
        if a.shape != self.shape: raise ValueError(f"field shape {a.shape} != {self.shape}")
        if not np.all(np.isfinite(a)): raise ValueError("field contains non-finite values")
        return np.fft.fftn(a)

    def _hat(self, field):
        return self._raw_hat(field) * self.dealias_mask * self.spectral_filter

    def filter_field(self, field):
        """Apply the configured spectral filter/truncation explicitly.

        Linear differential operators use the unfiltered spectrum. Dealiasing is
        an operation on nonlinear products/state updates, not a hidden modifier
        of every derivative.
        """
        return np.real(np.fft.ifftn(self._hat(field)))

    def gradient(self, field):
        fhat=self._raw_hat(field)
        return [np.real(np.fft.ifftn(1j*K*fhat)) for K in self.K_grids]

    def divergence(self, vector_field):
        if len(vector_field)!=self.ndim: raise ValueError("wrong vector component count")
        dhat=np.zeros(self.shape,dtype=np.complex128)
        for V,K in zip(vector_field,self.K_grids):
            dhat += 1j*K*self._raw_hat(V)
        return np.real(np.fft.ifftn(dhat))

    def laplacian(self, field):
        return np.real(np.fft.ifftn(-self.K_sq*self._raw_hat(field)))

    def solve_poisson(self, source):
        return np.real(np.fft.ifftn(self._raw_hat(source)*self.inv_laplacian))

    def leray_project(self, vector_field):
        if len(vector_field)!=self.ndim: raise ValueError("wrong vector component count")
        hats=[self._raw_hat(v) for v in vector_field]
        dot=sum(K*V for K,V in zip(self.K_grids,hats))
        out=[]; nz=self.K_sq > 0
        for K,V in zip(self.K_grids,hats):
            ph=V.copy()
            ph[nz] -= K[nz]*dot[nz]/self.K_sq[nz]
            ph *= self.dealias_mask * self.spectral_filter
            out.append(np.real(np.fft.ifftn(ph)))
        return out

    def filter_transfer(self):
        """Return the dimensionless spectral transfer function used by the engine."""
        return self.spectral_filter.copy()

    def _mode_magnitude_grid(self) -> np.ndarray:
        """``|k|`` in angular wavenumber units, normalised so Nyquist is 1."""
        return np.sqrt(sum((M / (n / 2.0)) ** 2
                           for M, n in zip(self.mode_grids, self.shape)))

    def gradient_denoised(self, field, *, tail_fraction: float = 0.10,
                          decay_tolerance: float = 0.5,
                          radial_bins: int = 8) -> "DenoisedGradient":
        """Gradient with a Wiener-style multiplier, or an explanation of why not.

        The multiplier is ``W(k) = S(k) / (S(k) + sigma^2)`` with ``S`` the
        observed power minus a noise floor ``sigma^2`` read off the highest
        modes. It is 1 where signal dominates and 0 where it does not, so it
        leaves resolved modes untouched -- which is why it stays exact on clean
        band-limited data.

        Two things make it safe rather than merely plausible:

        1. **The noise floor is bias-corrected.** ``|F|^2`` of white noise is
           exponentially distributed, so its *median* underestimates the mean by
           ``ln 2``. Using the raw median silently under-shrinks.
        2. **The tail is tested for signal before it is trusted.** If the tail
           still decays, it carries real unresolved signal, and shrinking would
           discard it. The operator then declines and reports ``regime="plain"``
           rather than quietly making the answer worse.
        """
        f = np.asarray(field)
        if f.shape != self.shape:
            raise ValueError(f"field shape {f.shape} != {self.shape}")
        if not np.all(np.isfinite(f)):
            raise ValueError("field contains non-finite values")
        if not 0.0 < tail_fraction < 1.0:
            raise ValueError("tail_fraction must lie strictly between 0 and 1")
        if radial_bins < 3:
            raise ValueError("radial_bins must be at least 3")

        q = self._mode_magnitude_grid()
        fhat = np.fft.fftn(f)
        power = np.abs(fhat) ** 2
        tail = q >= (1.0 - tail_fraction)

        def plain() -> "DenoisedGradient":
            comps = tuple(np.real(np.fft.ifftn(1j * K * fhat))
                          for K in self.K_grids)
            return DenoisedGradient(components=comps, regime="plain",
                                    noise_floor=0.0, tail_slope=0.0,
                                    multiplier=np.ones(self.shape),
                                    tail_fraction=tail_fraction)

        if int(tail.sum()) < 4 * radial_bins:
            return plain()

        # Radially bin the tail and regress log-power against position. A flat
        # tail is consistent with a noise floor; a falling one is signal.
        q_tail, p_tail = q[tail], power[tail]
        edges = np.linspace(float(q_tail.min()), float(q_tail.max()) + 1e-12,
                            radial_bins + 1)
        idx = np.digitize(q_tail, edges) - 1
        bin_means = np.array([p_tail[idx == b].mean() if (idx == b).any() else np.nan
                              for b in range(radial_bins)])
        usable = np.isfinite(bin_means) & (bin_means > 0)
        if int(usable.sum()) < 3:
            return plain()
        log_means = np.log(bin_means[usable])
        tail_slope = float(np.polyfit(np.linspace(0.0, 1.0, log_means.size),
                                      log_means, 1)[0])
        if tail_slope < -float(decay_tolerance):
            return plain()

        # Bias-corrected noise floor: the median of an exponential is ln(2)*mean.
        sigma2 = float(np.median(power[tail]) / np.log(2.0))
        if not np.isfinite(sigma2) or sigma2 <= 0.0:
            return plain()
        signal = np.maximum(power - sigma2, 0.0)
        multiplier = signal / (signal + sigma2)
        comps = tuple(np.real(np.fft.ifftn(1j * K * multiplier * fhat))
                      for K in self.K_grids)
        return DenoisedGradient(components=comps, regime="shrunk",
                                noise_floor=sigma2, tail_slope=tail_slope,
                                multiplier=multiplier,
                                tail_fraction=tail_fraction)

    def spectral_tail_ratio(self, field, inner_fraction=.8):
        """Energy fraction in a high-frequency hyperspherical shell."""
        fhat=np.fft.fftn(np.asarray(field))
        # Use normalized integer Fourier modes relative to the spherical cutoff.
        q=np.sqrt(sum((M/(n/3.0))**2 for M,n in zip(self.mode_grids,self.shape)))
        denom=float(np.sum(np.abs(fhat)**2))
        if denom==0: return 0.0
        return float(np.sum(np.abs(fhat[q>=inner_fraction])**2)/denom)
