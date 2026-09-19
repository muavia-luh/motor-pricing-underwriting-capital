"""Tests for the IRLS GLM engine. Run with `pytest` or `python3 tests/test_glm.py`.
Each test checks a property that must hold if the estimation is correct, rather
than a hard-coded number."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from motor_pricing.glm import fit_glm, PoissonFamily, GammaFamily, NegBinFamily


def test_poisson_recovers_known_coefficients():
    # Simulate from a known Poisson-log model; the GLM must recover the betas.
    rng = np.random.default_rng(0)
    n = 40_000
    x1 = rng.normal(size=n); x2 = rng.binomial(1, 0.4, size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    beta_true = np.array([-1.0, 0.5, -0.3])
    mu = np.exp(X @ beta_true)
    y = rng.poisson(mu).astype(float)
    res, _ = fit_glm(X, y, PoissonFamily(), names=["int", "x1", "x2"])
    assert np.allclose(res.coef, beta_true, atol=0.05), res.coef
    # Estimates should sit within a few standard errors of the truth.
    assert np.all(np.abs(res.coef - beta_true) < 4 * res.se)


def test_poisson_offset_is_respected():
    # With an offset, the intercept must absorb the offset shift correctly.
    rng = np.random.default_rng(1)
    n = 30_000
    expo = rng.uniform(0.1, 1.0, size=n)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta_true = np.array([-2.0, 0.4])
    mu = expo * np.exp(X @ beta_true)
    y = rng.poisson(mu).astype(float)
    res, _ = fit_glm(X, y, PoissonFamily(), offset=np.log(expo), names=["int", "x"])
    assert np.allclose(res.coef, beta_true, atol=0.06), res.coef


def test_gamma_recovers_mean_and_positive_dispersion():
    rng = np.random.default_rng(2)
    n = 30_000
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta_true = np.array([7.0, 0.3])           # mean severity ~ exp(7) ~ 1100
    mu = np.exp(X @ beta_true)
    shape = 2.0
    y = rng.gamma(shape, scale=mu / shape)
    res, _ = fit_glm(X, y, GammaFamily(), names=["int", "x"])
    assert np.allclose(res.coef, beta_true, atol=0.05), res.coef
    assert res.dispersion > 0
    # Estimated dispersion phi ~ 1/shape = 0.5.
    assert abs(res.dispersion - 0.5) < 0.1, res.dispersion


def test_negbin_collapses_to_poisson_for_large_theta():
    rng = np.random.default_rng(3)
    n = 20_000
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    mu = np.exp(X @ np.array([-0.5, 0.2]))
    y = rng.poisson(mu).astype(float)
    pois, _ = fit_glm(X, y, PoissonFamily())
    nb, _ = fit_glm(X, y, NegBinFamily(theta=1e6))
    assert np.allclose(pois.coef, nb.coef, atol=1e-3)


def test_deviance_and_se_are_valid():
    rng = np.random.default_rng(4)
    n = 5_000
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = rng.poisson(np.exp(X @ np.array([0.0, 0.1]))).astype(float)
    res, _ = fit_glm(X, y, PoissonFamily())
    assert res.deviance >= 0
    assert res.null_deviance >= res.deviance      # adding a covariate cannot worsen fit
    assert np.all(res.se > 0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\n{len(fns)} tests passed")
