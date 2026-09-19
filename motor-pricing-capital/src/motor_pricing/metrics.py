"""Actuarial validation metrics: exposure-weighted Poisson deviance, an
exposure-weighted concentration Gini, and decile calibration tables. These are
the measures used to compare the transparent GLM with the ML challenger on the
same held-out data."""
from __future__ import annotations
import numpy as np
import pandas as pd


def poisson_deviance_rate(claims, exposure, predicted_rate):
    """Exposure-weighted mean Poisson deviance between observed and predicted
    *rates* (claims per exposure-year). Lower is better."""
    claims = np.asarray(claims, float); exposure = np.asarray(exposure, float)
    obs = claims / exposure
    mu = np.maximum(np.asarray(predicted_rate, float), 1e-12)
    # Evaluate obs*log(obs/mu) only where obs > 0 (no divide-by-zero computed).
    term = np.zeros_like(mu)
    pos = obs > 0
    term[pos] = obs[pos] * np.log(obs[pos] / mu[pos])
    dev = 2.0 * (term - (obs - mu))
    return float(np.sum(exposure * dev) / np.sum(exposure))


def gamma_deviance(y, mu):
    y = np.asarray(y, float); mu = np.maximum(np.asarray(mu, float), 1e-12)
    return float(np.mean(2.0 * (-np.log(y / mu) + (y - mu) / mu)))


def weighted_gini(claims, exposure, predicted_rate):
    """Exposure-weighted concentration Gini: order policies from lowest to highest
    predicted risk, trace the Lorenz curve of cumulative claims vs cumulative
    exposure, Gini = 1 - 2*area. Higher = better risk separation."""
    order = np.argsort(np.asarray(predicted_rate), kind="mergesort")
    c = np.asarray(claims, float)[order]; e = np.asarray(exposure, float)[order]
    cum_e = np.cumsum(e) / e.sum()
    cum_c = np.cumsum(c) / c.sum()
    area = np.trapezoid(cum_c, cum_e)
    return float(1.0 - 2.0 * area)


def calibration_by_risk(claims, exposure, predicted_rate, groups=10):
    t = pd.DataFrame({"claims": np.asarray(claims, float),
                      "exposure": np.asarray(exposure, float),
                      "rate": np.asarray(predicted_rate, float)})
    t["pred_claims"] = t["rate"] * t["exposure"]
    t["decile"] = pd.qcut(t["rate"], groups, labels=False, duplicates="drop") + 1
    g = t.groupby("decile", observed=True).agg(
        policies=("claims", "size"), exposure=("exposure", "sum"),
        observed_claims=("claims", "sum"), predicted_claims=("pred_claims", "sum"))
    g["observed_frequency"] = g["observed_claims"] / g["exposure"]
    g["predicted_frequency"] = g["predicted_claims"] / g["exposure"]
    g["actual_to_expected"] = g["observed_claims"] / g["predicted_claims"]
    return g.reset_index()
