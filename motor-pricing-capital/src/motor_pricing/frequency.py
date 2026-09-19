"""Claim-frequency model: in-house Poisson GLM (log-exposure offset) with a
Negative-Binomial cross-check, evaluated out of sample by Poisson deviance, an
exposure-weighted Gini and decile calibration."""
from __future__ import annotations
import numpy as np

from .glm import fit_glm, fit_nb_theta, PoissonFamily
from .features import build_design
from . import metrics


def fit_frequency(train, test):
    Xtr, names, encoder = build_design(train)
    Xte, _, _ = build_design(test, encoder=encoder)
    ytr = train["ClaimNb"].values.astype(float)
    off_tr = np.log(train["Exposure"].values)
    off_te = np.log(test["Exposure"].values)

    pois, mu_tr = fit_glm(Xtr, ytr, PoissonFamily(), offset=off_tr, names=names)
    rate_te = np.exp(Xte @ pois.coef)                       # predicted claims / year

    # Over-dispersion check + NB cross-check (theta profiled on a small grid).
    pearson = np.sum((ytr - mu_tr) ** 2 / mu_tr)
    overdisp = pearson / (len(ytr) - len(names))
    nb, _, theta = fit_nb_theta(Xtr, ytr, offset=off_tr, names=names,
                                theta_grid=np.geomspace(0.5, 4.0, 12))

    claims_te = test["ClaimNb"].values.astype(float)
    expo_te = test["Exposure"].values.astype(float)
    dev = metrics.poisson_deviance_rate(claims_te, expo_te, rate_te)
    base_rate = np.full(len(test), ytr.sum() / train["Exposure"].sum())
    dev_base = metrics.poisson_deviance_rate(claims_te, expo_te, base_rate)
    gini = metrics.weighted_gini(claims_te, expo_te, rate_te)
    calib = metrics.calibration_by_risk(claims_te, expo_te, rate_te)

    result = {
        "n_train": int(len(train)), "n_test": int(len(test)), "n_params": len(names),
        "poisson_aic": round(pois.aic, 1), "negbin_aic": round(nb.aic, 1),
        "negbin_theta": round(theta, 3),
        "overdispersion_pearson_over_dof": round(overdisp, 3),
        "test_poisson_deviance_glm": round(dev, 5),
        "test_poisson_deviance_baseline": round(dev_base, 5),
        "deviance_improvement_pct": round(100 * (dev_base - dev) / dev_base, 2),
        "test_weighted_gini_glm": round(gini, 4),
        "lift_top_bottom": round(float(calib["observed_frequency"].iloc[-1] /
                                       calib["observed_frequency"].iloc[0]), 2),
    }
    coef_tbl = [{"term": names[i], "coef": round(float(pois.coef[i]), 4),
                 "relativity": round(float(np.exp(pois.coef[i])), 3),
                 "se": round(float(pois.se[i]), 4)} for i in range(len(names))]
    artefacts = {"coef": pois.coef, "names": names, "encoder": encoder,
                 "test_rate": rate_te, "calibration": calib, "summary": pois.summary()}
    return result, coef_tbl, artefacts
