import numpy as np
from ags_sci.fields import SphericalShellToroidalDiffusion, chebyshev_lobatto, stretched_radius


def test_chebyshev_nodes_and_differentiation_polynomial():
    x, D = chebyshev_lobatto(17)
    y = x**4 - 2*x**2 + 3*x
    exact = 4*x**3 - 4*x + 3
    assert np.max(np.abs(D @ y - exact)) < 1e-10


def test_stretching_is_monotone_and_bounded():
    x, _ = chebyshev_lobatto(33)
    r = stretched_radius(x, 0.35, 1.0, beta=0.4)
    assert np.isclose(r[0], 1.0)
    assert np.isclose(r[-1], 0.35)
    assert np.all(np.diff(r) < 0)


def test_toroidal_diffusion_linear_operator_is_dissipative():
    op = SphericalShellToroidalDiffusion(33, l=1, viscosity=1e-3)
    rng = np.random.default_rng(4)
    y = rng.normal(size=31)
    power = float(y @ op.rhs(y))
    assert power < 0
    assert op.resolution.stable_linear_discretization


def test_integrating_factor_matches_matrix_exponential_eigen_action():
    op = SphericalShellToroidalDiffusion(25, l=2, viscosity=2e-3)
    y = np.sin(np.linspace(0, np.pi, 23))
    out = op.integrating_factor(y, 0.1)
    # For dissipative operator the norm cannot increase.
    assert np.linalg.norm(out) <= np.linalg.norm(y) + 1e-12
    assert np.all(np.isfinite(out))


def test_resolution_stability_improves_with_radial_resolution_for_linear_operator():
    low = SphericalShellToroidalDiffusion(9, l=8, viscosity=1e-3)
    high = SphericalShellToroidalDiffusion(33, l=8, viscosity=1e-3)
    assert low.resolution.max_positive_real_part <= 1e-8
    assert high.resolution.max_positive_real_part <= 1e-8
