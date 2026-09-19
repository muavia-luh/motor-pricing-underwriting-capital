"""Claim-severity model: a Gamma GLM for the attritional body and a
peaks-over-threshold generalised-Pareto fit for the large-loss tail.

Note on the pure premium: the body Gamma GLM is fitted and evaluated for
completeness, but it barely beats the intercept-only baseline, so the pure premium
(in reporting.py) multiplies the frequency by the POOLED mean severity rather than
the severity-GLM prediction -- a deliberate choice, not the GLM feeding through.
The tail fit feeds the capital model."""
from __future__ import annotations
import numpy as np
from scipy import stats

from .glm import fit_glm, GammaFamily
from .features import build_design
from . import metrics


def fit_severity(claims_frame, split_fn, tail_quantile=0.95):
    y_all = claims_frame["ClaimAmount"].values.astype(float)
    u = float(np.quantile(y_all, tail_quantile))

    # ---- Body: Gamma GLM on claims <= u ----
    # split_fn splits by policy (IDpol) so multiple claims on one policy never
    # straddle the train/test boundary.
    body = claims_frame[claims_frame["ClaimAmount"] <= u].copy()
    tr, te = split_fn(body)
    Xtr, names, encoder = build_design(tr)
    Xte, _, _ = build_design(te, encoder=encoder)
    gres, _ = fit_glm(Xtr, tr["ClaimAmount"].values.astype(float), GammaFamily(),
                      names=names)
    mu_te = np.exp(Xte @ gres.coef)
    yte = te["ClaimAmount"].values.astype(float)
    dev_model = metrics.gamma_deviance(yte, mu_te)
    dev_base = metrics.gamma_deviance(yte, np.full(len(yte), tr["ClaimAmount"].mean()))

    # ---- Tail: GPD via POT, with threshold sensitivity ----
    tail_sens = {}
    for q in [0.90, 0.95, 0.97]:
        ut = float(np.quantile(y_all, q)); exc = y_all[y_all > ut] - ut
        xi, _, beta = stats.genpareto.fit(exc, floc=0)
        tail_sens[str(q)] = {"threshold_eur": round(ut, 0), "n_exceedances": int(len(exc)),
                             "xi_shape": round(float(xi), 3), "beta_scale": round(float(beta), 1)}
    exc = y_all[y_all > u] - u
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    p_tail = float((y_all > u).mean())

    result = {
        "n_body_claims": int(len(body)),
        "large_loss_threshold_eur": round(u, 0),
        "gamma_dispersion_phi": round(gres.dispersion, 3),
        "test_gamma_deviance_glm": round(dev_model, 4),
        "test_gamma_deviance_baseline": round(dev_base, 4),
        "mean_severity_eur": round(float(y_all.mean()), 0),
        "tail_xi_shape": round(float(xi), 3),
        "tail_beta_scale": round(float(beta), 1),
        "tail_p_exceed": round(p_tail, 5),
        "tail_finite_mean": bool(xi < 1),
        "tail_finite_variance": bool(xi < 0.5),
        "tail_top1pct_loss_share": round(float(
            np.sort(y_all)[-int(0.01 * len(y_all)):].sum() / y_all.sum()), 4),
        "tail_sensitivity": tail_sens,
    }
    artefacts = {"small_claims": y_all[y_all <= u], "large_claims": y_all[y_all > u],
                 "u_tail": u, "xi": xi, "beta": beta, "p_tail": p_tail,
                 "mean_severity": float(y_all.mean()), "coef": gres.coef,
                 "names": names, "encoder": encoder, "summary": gres.summary()}
    return result, artefacts
