"""Tests for the denoised (Wiener-style) spectral gradient.

The plain spectral gradient is exact to machine precision for any field whose
spectrum lies below Nyquist, so these tests assert two things:

1. the denoised gradient is **never worse** than the plain one, including being
   exactly equal to it when it declines to shrink;
2. it is **materially better** under noise, which is the only regime where the
   plain gradient is weak (it multiplies mode k by k, amplifying high-mode noise
   by up to Nyquist).
"""
import itertools

import numpy as np
import pytest

from ags_sci.fields.operators import DenoisedGradient, SpectralEngineND

L = 2 * np.pi


def grids(shape):
    return np.meshgrid(*[np.arange(n) * L / n for n in shape], indexing="ij")


def bandlimited(shape, kmax, seed=0):
    """A random-phase field with energy only at |k| <= kmax, plus exact gradient."""
    rng = np.random.default_rng(seed)
    f = np.zeros(shape)
    gf = [np.zeros(shape) for _ in range(len(shape))]
    for ks in itertools.product(*[range(-kmax, kmax + 1)] * len(shape)):
        if sum(k * k for k in ks) > kmax ** 2:
            continue
        phase = rng.uniform(0, 2 * np.pi)
        arg = sum(k * X for k, X in zip(ks, grids(shape))) + phase
        f += np.cos(arg)
        for i, k in enumerate(ks):
            gf[i] += -k * np.sin(arg)
    return f, gf


def gaussian_bump(shape, width=40.0):
    """A narrow bump: genuinely under-resolved, so its tail carries real signal."""
    Xs = grids(shape)
    c = np.pi
    g = np.exp(-width * sum((A - c) ** 2 for A in Xs))
    return g, [-2 * width * (A - c) * g for A in Xs]


def rel_error(got, exact):
    return float(np.linalg.norm(np.stack(got) - np.stack(exact))
                 / np.linalg.norm(np.stack(exact)))


@pytest.fixture(scope="module")
def eng2d():
    return SpectralEngineND((64, 64), domain_lengths=(L, L))


@pytest.fixture(scope="module")
def eng3d():
    return SpectralEngineND((32, 32, 32), domain_lengths=(L, L, L))


# ==========================================================================
# Contract
# ==========================================================================
def test_returns_a_described_result(eng2d):
    f, _ = bandlimited((64, 64), 20, seed=1)
    res = eng2d.gradient_denoised(f + 1e-3 * np.random.default_rng(0).normal(size=(64, 64)))
    assert isinstance(res, DenoisedGradient)
    assert res.regime in {"shrunk", "plain"}
    assert len(res.components) == 2
    d = res.describe()
    assert set(d) >= {"regime", "noise_floor", "tail_slope", "multiplier_min",
                      "multiplier_max"}


def test_rejects_mismatched_and_nonfinite_input(eng2d):
    with pytest.raises(ValueError):
        eng2d.gradient_denoised(np.zeros((8, 8)))
    with pytest.raises(ValueError):
        eng2d.gradient_denoised(np.full((64, 64), np.nan))
    with pytest.raises(ValueError):
        eng2d.gradient_denoised(np.zeros((64, 64)), tail_fraction=0.0)
    with pytest.raises(ValueError):
        eng2d.gradient_denoised(np.zeros((64, 64)), tail_fraction=1.0)
    with pytest.raises(ValueError):
        eng2d.gradient_denoised(np.zeros((64, 64)), radial_bins=2)


def test_multiplier_is_bounded_and_shaped_like_the_grid(eng2d):
    f, _ = bandlimited((64, 64), 20, seed=2)
    res = eng2d.gradient_denoised(f + 1e-3 * np.random.default_rng(3).normal(size=(64, 64)))
    assert res.multiplier.shape == (64, 64)
    assert np.all(res.multiplier >= 0.0) and np.all(res.multiplier <= 1.0)


# ==========================================================================
# The central guarantee: never worse than plain
# ==========================================================================
@pytest.mark.parametrize("shape,kmax,noise", [
    ((64, 64), 20, 0.0),
    ((64, 64), 20, 1e-5),
    ((64, 64), 28, 1e-3),
    ((32, 32, 32), 12, 1e-3),
])
def test_never_worse_than_plain_on_resolved_fields(shape, kmax, noise):
    eng = SpectralEngineND(shape, domain_lengths=(L,) * len(shape))
    f, gf = bandlimited(shape, kmax, seed=5)
    noisy = f + noise * np.random.default_rng(7).normal(size=shape)
    e_plain = rel_error(eng.gradient(noisy), gf)
    e_den = rel_error(eng.gradient_denoised(noisy).components, gf)
    assert e_den <= e_plain * (1.0 + 1e-9), (e_plain, e_den)


def test_declining_gives_exactly_the_plain_gradient(eng2d):
    """When the tail carries real signal the operator must not shrink at all."""
    f, gf = gaussian_bump((64, 64), width=40.0)
    res = eng2d.gradient_denoised(f)
    assert res.regime == "plain", "an under-resolved field must not be shrunk"
    plain = eng2d.gradient(f)
    for got, want in zip(res.components, plain):
        assert np.array_equal(got, want), "declining must mean *identical* output"
    assert np.all(res.multiplier == 1.0)
    assert res.noise_floor == 0.0


def test_clean_bandlimited_field_stays_exact(eng2d):
    """The theorem the plain gradient satisfies must survive the multiplier."""
    f, gf = bandlimited((64, 64), 20, seed=11)
    res = eng2d.gradient_denoised(f)
    assert rel_error(res.components, gf) < 1e-12


# ==========================================================================
# The benefit: noise robustness
# ==========================================================================
@pytest.mark.parametrize("noise", [1e-6, 1e-4, 1e-2])
def test_denoised_roughly_halves_error_under_noise(eng2d, noise):
    f, gf = bandlimited((64, 64), 20, seed=13)
    noisy = f + noise * np.random.default_rng(17).normal(size=(64, 64))
    e_plain = rel_error(eng2d.gradient(noisy), gf)
    e_den = rel_error(eng2d.gradient_denoised(noisy).components, gf)
    assert e_den < 0.8 * e_plain, (e_plain, e_den)


def test_benefit_grows_with_the_noise_level(eng2d):
    f, gf = bandlimited((64, 64), 20, seed=19)
    ratios = []
    for noise in (1e-6, 1e-3):
        noisy = f + noise * np.random.default_rng(23).normal(size=(64, 64))
        ratios.append(rel_error(eng2d.gradient_denoised(noisy).components, gf)
                      / rel_error(eng2d.gradient(noisy), gf))
    assert all(r < 1.0 for r in ratios)


def test_works_in_three_dimensions(eng3d):
    f, gf = bandlimited((32, 32, 32), 12, seed=29)
    noisy = f + 1e-3 * np.random.default_rng(31).normal(size=(32, 32, 32))
    assert len(eng3d.gradient_denoised(noisy).components) == 3
    e_plain = rel_error(eng3d.gradient(noisy), gf)
    e_den = rel_error(eng3d.gradient_denoised(noisy).components, gf)
    assert e_den < e_plain


# ==========================================================================
# The estimator itself
# ==========================================================================
def test_noise_floor_estimator_is_bias_corrected():
    """|F|^2 of white noise is exponential, so its median is ln(2) below its mean.

    An uncorrected median silently under-shrinks. This pins the correction.
    """
    rng = np.random.default_rng(0)
    n, var = 64, 1e-4
    power = np.abs(np.fft.fft2(rng.normal(0, np.sqrt(var), (n, n)))) ** 2
    expected = n * n * var                       # numpy's unnormalised FFT
    assert abs(power.mean() / expected - 1.0) < 0.1
    assert abs(np.median(power) / expected - np.log(2.0)) < 0.1
    # The corrected median recovers the mean; the raw one does not.
    assert abs(np.median(power) / np.log(2.0) / expected - 1.0) < 0.1


def test_flat_tail_is_distinguished_from_a_decaying_one(eng2d):
    """A noise floor looks flat in log-power; unresolved signal decays."""
    rng = np.random.default_rng(37)
    # Pure noise: the tail must read as flat.
    flat = rng.normal(0, 1.0, (64, 64))
    assert eng2d.gradient_denoised(flat).regime == "shrunk"
    # A strongly decaying spectrum must not be treated as a noise floor.
    sharp, _ = gaussian_bump((64, 64), width=200.0)
    assert eng2d.gradient_denoised(sharp).regime == "plain"


def test_zero_field_is_handled(eng2d):
    res = eng2d.gradient_denoised(np.zeros((64, 64)))
    assert all(np.all(c == 0.0) for c in res.components)


# ==========================================================================
# Service facade
# ==========================================================================
def test_service_exposes_both_gradients():
    from ags_sci import AGSService
    svc = AGSService()
    shape = (8, 8, 8)
    f = np.random.default_rng(0).normal(size=shape)
    plain = svc.gradient("pseudo_spectral3d", shape, f)
    assert len(plain) == 3
    res = svc.denoised_gradient("pseudo_spectral3d", shape, f)
    assert isinstance(res, DenoisedGradient)
    assert res.regime in {"shrunk", "plain"}


@pytest.mark.parametrize("backend,shape", [
    ("pseudo_spectral3d", (8, 8, 8)),
    ("scalar4d", (8, 8, 8, 8)),
    ("scalar5d", (8, 8, 8, 8, 8)),
])
def test_service_resolves_the_engine_in_every_dimension(backend, shape):
    from ags_sci import AGSService
    svc = AGSService()
    f = np.random.default_rng(1).normal(size=shape)
    res = svc.denoised_gradient(backend, shape, f)
    assert len(res.components) == len(shape)
    assert res.multiplier.shape == shape


def test_service_rejects_a_backend_without_a_spectral_engine():
    from ags_sci import AGSService
    svc = AGSService()
    with pytest.raises(TypeError):
        svc.denoised_gradient("spherical_shell3d", (8, 8, 8),
                              np.zeros((8, 8, 8)))
