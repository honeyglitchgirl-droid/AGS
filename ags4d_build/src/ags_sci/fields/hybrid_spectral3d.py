"""3D Fourier--Chebyshev slab spectral foundation.

Periodic Fourier discretization is used in x/y and Chebyshev--Lobatto
collocation in z. This is a static-domain foundation for non-periodic vertical
boundaries; it is not a complete free-surface Navier--Stokes solver.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .spherical_shell3d import chebyshev_lobatto


@dataclass(frozen=True)
class HybridGrid3D:
    shape: tuple[int, int, int]
    lengths_xy: tuple[float, float]
    z_bounds: tuple[float, float]


class FourierChebyshev3D:
    def __init__(self, shape=(16, 16, 17), lengths_xy=(2*np.pi, 2*np.pi), z_bounds=(-1.0, 1.0)):
        nx, ny, nz = map(int, shape)
        if min(nx, ny) < 4 or nz < 3:
            raise ValueError("shape requires nx,ny>=4 and nz>=3")
        lx, ly = map(float, lengths_xy); z0, z1 = map(float, z_bounds)
        if lx <= 0 or ly <= 0 or z1 <= z0:
            raise ValueError("invalid domain")
        _, D = chebyshev_lobatto(nz)
        # Convert descending Chebyshev nodes to increasing z ordering.
        xi = np.cos(np.pi*np.arange(nz)/(nz-1))[::-1]
        D = D[::-1, ::-1]
        z = z0 + 0.5*(xi+1)*(z1-z0)
        self.grid = HybridGrid3D((nx,ny,nz),(lx,ly),(z0,z1))
        self.shape = (nx,ny,nz); self.z = z; self.Dz = 2.0/(z1-z0)*D
        kx = 2*np.pi*np.fft.fftfreq(nx, d=lx/nx)
        ky = 2*np.pi*np.fft.fftfreq(ny, d=ly/ny)
        self.kx, self.ky = np.meshgrid(kx, ky, indexing='ij')

    def _check(self, f):
        a=np.asarray(f,dtype=float)
        if a.shape != self.shape: raise ValueError(f"field shape {a.shape} != {self.shape}")
        if not np.all(np.isfinite(a)): raise ValueError("field contains non-finite values")
        return a

    def dx(self, f):
        a=self._check(f); return np.real(np.fft.ifft2(1j*self.kx[...,None]*np.fft.fft2(a,axes=(0,1)),axes=(0,1)))
    def dy(self, f):
        a=self._check(f); return np.real(np.fft.ifft2(1j*self.ky[...,None]*np.fft.fft2(a,axes=(0,1)),axes=(0,1)))
    def dz(self, f):
        a=self._check(f); return np.einsum('ij,xyj->xyi', self.Dz, a)
    def laplacian(self, f):
        a=self._check(f); F=np.fft.fft2(a,axes=(0,1)); horiz=-(self.kx[...,None]**2+self.ky[...,None]**2)*F
        horiz=np.real(np.fft.ifft2(horiz,axes=(0,1)))
        return horiz + np.einsum('ij,xyj->xyi', self.Dz@self.Dz, a)
    def divergence(self, u):
        if len(u)!=3: raise ValueError("3D vector requires three components")
        return self.dx(u[0])+self.dy(u[1])+self.dz(u[2])
