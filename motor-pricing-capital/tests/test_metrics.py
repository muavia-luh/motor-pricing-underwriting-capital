"""Tests for the validation metrics. Run with `pytest` or
`python3 tests/test_metrics.py`."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from motor_pricing import metrics


def test_perfect_prediction_has_low_deviance():
    rng = np.random.default_rng(0)
    expo = rng.uniform(0.1, 1, 5000)
    rate = rng.uniform(0.02, 0.3, 5000)
    claims = rng.poisson(rate * expo).astype(float)
    d_true = metrics.poisson_deviance_rate(claims, expo, rate)
    d_flat = metrics.poisson_deviance_rate(claims, expo,
                                           np.full(5000, claims.sum() / expo.sum()))
    assert d_true < d_flat            # the true rate must beat a flat rate


def test_gini_between_zero_and_one_and_ranks_signal():
    rng = np.random.default_rng(1)
    expo = np.ones(5000)
    rate = rng.uniform(0.01, 0.5, 5000)
    claims = rng.poisson(rate).astype(float)
    g_signal = metrics.weighted_gini(claims, expo, rate)
    g_noise = metrics.weighted_gini(claims, expo, rng.permutation(rate))
    assert 0 <= g_signal <= 1
    assert g_signal > g_noise          # a predictive score separates risk better than noise


def test_gamma_deviance_zero_at_truth():
    y = np.array([100.0, 200.0, 300.0])
    assert metrics.gamma_deviance(y, y) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\n{len(fns)} tests passed")
