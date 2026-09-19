"""Gradient-boosting challenger models (scikit-learn HistGradientBoosting), fitted
on the same train/test split as the GLM. The question these answer is the one a
model-validation team asks: does a flexible ML model beat the transparent GLM by
enough to justify the loss of interpretability?

Frequency: Poisson-loss boosting on the claim rate, exposure-weighted.
Severity : Gamma-loss boosting on individual claim amounts (body).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import ML_FEATURES, ML_CATEGORICAL
from . import metrics


def _align_categorical(train, test):
    """Convert both frames to categorical dtype using the *training* categories, so
    the test frame's category codes line up with what the model learned (an unseen
    test level becomes NaN, which HistGradientBoosting handles as missing)."""
    tr = train.copy(); te = test.copy()
    for c in ML_CATEGORICAL:
        cats = pd.Categorical(tr[c]).categories
        tr[c] = pd.Categorical(tr[c], categories=cats)
        te[c] = pd.Categorical(te[c], categories=cats)
    return tr, te


def fit_frequency_ml(train, test, seed=42):
    tr, te = _align_categorical(train, test)
    model = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.08, max_iter=300, max_leaf_nodes=31,
        min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
        categorical_features="from_dtype", random_state=seed)
    model.fit(tr[ML_FEATURES], tr["ClaimNb"] / tr["Exposure"],
              sample_weight=tr["Exposure"])
    rate_te = np.maximum(model.predict(te[ML_FEATURES]), 1e-8)

    claims = test["ClaimNb"].values.astype(float)
    expo = test["Exposure"].values.astype(float)
    return {
        "test_poisson_deviance_ml": round(metrics.poisson_deviance_rate(claims, expo, rate_te), 5),
        "test_weighted_gini_ml": round(metrics.weighted_gini(claims, expo, rate_te), 4),
        "boosting_iterations": int(model.n_iter_),
    }, rate_te


def fit_severity_ml(claims_frame, split_fn, u_tail, seed=42):
    body = claims_frame[claims_frame["ClaimAmount"] <= u_tail].copy()
    tr, te = split_fn(body)
    tr, te = _align_categorical(tr, te)
    model = HistGradientBoostingRegressor(
        loss="gamma", learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
        categorical_features="from_dtype", random_state=seed)
    model.fit(tr[ML_FEATURES], tr["ClaimAmount"])
    pred = np.maximum(model.predict(te[ML_FEATURES]), 1e-6)
    y = te["ClaimAmount"].values.astype(float)
    return {
        "test_gamma_deviance_ml": round(metrics.gamma_deviance(y, pred), 4),
        "boosting_iterations": int(model.n_iter_),
    }
