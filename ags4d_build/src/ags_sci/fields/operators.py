"""Dimension-agnostic periodic spectral operators.

The engine is spatial/Euclidean at the numerical discretization level. Metric-aware
contractions belong to :mod:`ags_sci.fields.geometry`; this distinction prevents a
4D FFT from being incorrectly described as Lorentz-covariant by itself.
"""
from __future__ import annotations
import numpy as np

class SpectralEngineND:
    """Periodic spectral operators for D=1..4 spatial coordinates."""

    def __init__(self, shape: tuple[int, ...], domain_lengths=None,
                 dealias: str = "spherical", filter_alpha: float = 36.0, filter_order: int = 36):
        self.shape = tuple(int(n) for n in shape)
        self.ndim = len(self.shape)
        if self.ndim not in (1, 2, 3, 4):
            raise ValueError("Unsupported dimension: expected 1, 2, 3, or 4")
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

    def spectral_tail_ratio(self, field, inner_fraction=.8):
        """Energy fraction in a high-frequency hyperspherical shell."""
        fhat=np.fft.fftn(np.asarray(field))
        k2=self.K_sq
        # Use normalized integer Fourier modes relative to the spherical cutoff.
        q=np.sqrt(sum((M/(n/3.0))**2 for M,n in zip(self.mode_grids,self.shape)))
        denom=float(np.sum(np.abs(fhat)**2))
        if denom==0: return 0.0
        return float(np.sum(np.abs(fhat[q>=inner_fraction])**2)/denom)
