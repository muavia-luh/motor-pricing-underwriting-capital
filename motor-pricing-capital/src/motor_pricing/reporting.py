"""Run the full analysis and write JSON results + figures into reports/.
Frequency and severity are each modelled twice -- transparent GLM and ML
challenger -- on one shared split, then combined into a pure premium and a
1-in-200 capital figure with reinsurance stress tests."""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data as dat
from . import features as feat
from . import frequency as freqmod
from . import severity as sevmod
from . import ml_benchmark as ml
from . import capital as cap
from .split import train_test_split, grouped_train_test_split

BLUE, ORANGE, RED, GREEN = "#2b6cb0", "#dd6b20", "#c53030", "#2f855a"
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def run(root):
    raw = os.path.join(root, "data", "raw")
    fig = os.path.join(root, "reports", "figures")
    res = os.path.join(root, "reports", "results")
    os.makedirs(fig, exist_ok=True); os.makedirs(res, exist_ok=True)

    freq_raw = dat.load_frequency(os.path.join(raw, "freMTPL2freq.csv"))
    sev = dat.load_severity(os.path.join(raw, "freMTPL2sev.csv"))
    model_frame = feat.build_model_frame(freq_raw)
    claims = feat.build_severity_frame(freq_raw, sev)

    # ---------- Portfolio summary + reconciliation ----------
    reported = int(model_frame["ReportedClaimNb"].sum())
    capped = int(model_frame["ClaimNb"].sum())
    summary = {
        "n_policies": int(len(model_frame)),
        "exposure_years": round(float(model_frame["Exposure"].sum()), 1),
        "reported_claims": reported, "capped_claims": capped,
        "severity_rows": int(len(sev)),
        "reported_reconciles_severity": bool(reported == len(sev)),
        "claims_removed_by_capping": int(reported - capped),
        "observed_frequency": round(capped / model_frame["Exposure"].sum(), 5),
        "mean_severity_eur": round(float(sev["ClaimAmount"].mean()), 0),
        "max_severity_eur": round(float(sev["ClaimAmount"].max()), 0),
    }

    # ---------- Shared split ----------
    ftr, fte = train_test_split(model_frame, seed=42)

    # ---------- Frequency: GLM vs ML ----------
    fglm, fcoef, fart = freqmod.fit_frequency(ftr, fte)
    fml, ml_rate = ml.fit_frequency_ml(ftr, fte, seed=42)

    # ---------- Severity: GLM+tail vs ML (split by policy IDpol) ----------
    sev_split = lambda f: grouped_train_test_split(f, "IDpol", test_frac=0.2, seed=42)
    sglm, sart = sevmod.fit_severity(claims, sev_split, tail_quantile=0.95)
    sml = ml.fit_severity_ml(claims, sev_split, sart["u_tail"], seed=42)

    # ---------- Pure premium ----------
    # Pure premium = GLM-predicted frequency x POOLED mean severity. The severity
    # models (GLM and ML) showed essentially no lift over the pooled mean (Section
    # 4 / 5 of the report), so the pooled mean is used deliberately rather than a
    # severity-GLM prediction that would add noise without signal.
    X, names, _ = feat.build_design(model_frame, encoder=fart["encoder"])
    lam = np.exp(X @ fart["coef"])                     # expected claims / policy-year
    mu_N = float((lam * model_frame["Exposure"].values).sum())
    mean_sev = sart["mean_severity"]                   # overall observed mean severity
    pure_premium = lam * mean_sev
    pp = {
        "model_expected_count": round(mu_N, 0),
        "observed_claim_count": capped,
        "model_expected_annual_loss_eur": round(mu_N * mean_sev, 0),
        "observed_total_losses_eur": round(float(sev["ClaimAmount"].sum()), 0),
        "mean_pure_premium_eur": round(float(pure_premium.mean()), 2),
        "pure_premium_p95_eur": round(float(np.quantile(pure_premium, 0.95)), 2),
    }

    # ---------- Underwriting-risk capital: base + stress tests ----------
    N = 200_000
    small, large, u = sart["small_claims"], sart["large_claims"], sart["u_tail"]
    xi, beta, p_tail = sart["xi"], sart["beta"], sart["p_tail"]

    def scen(mode, cv, capv, seed, label):
        losses = cap.simulate_aggregate(mu_N, small, large, p_tail, mode, u_tail=u,
                                        xi=xi, beta=beta, cv_sys=cv, claim_cap=capv,
                                        n_sims=N, seed=seed)
        el = cap.model_expected_loss(mu_N, small, large, p_tail, mode, u_tail=u,
                                     xi=xi, beta=beta, claim_cap=capv)
        return cap.summarise(losses, label, expected_loss=el), losses

    rows = []
    base_r, base = scen("empirical", 0.05, None, 1, "Base: empirical tail, cv=5%, gross")
    rows.append(base_r)
    for cv in [0.0, 0.10]:
        r, _ = scen("empirical", cv, None, 2, f"Empirical tail, cv={int(cv*100)}%")
        rows.append(r)
    gpd_r, _ = scen("gpd", 0.05, None, 3, "GPD tail extrapolation, gross (stress)")
    rows.append(gpd_r)
    for capv in [2_000_000, 1_000_000]:
        r, _ = scen("empirical", 0.05, capv, 4, f"Net of per-claim cap EUR {capv:,}")
        rows.append(r)
    gpd_net_r, _ = scen("gpd", 0.05, 2_000_000, 5, "GPD stress, net of per-claim cap EUR 2,000,000")
    rows.append(gpd_net_r)

    cap_table = [{"scenario": r.scenario,
                  "expected_loss_eur_m": round(r.expected_loss/1e6, 2),
                  "var_995_eur_m": round(r.var_995/1e6, 2),
                  "tvar_995_eur_m_diagnostic": round(r.tvar_995/1e6, 2),
                  "uw_capital_eur_m": round(r.uw_capital/1e6, 2),
                  "uw_capital_pct_of_expected": round(r.uw_capital_pct_of_expected, 1)}
                 for r in rows]

    # ---------- GPD stress sensitivity (items 15-16): report a RANGE ----------
    gpd_sens = _gpd_sensitivity(mu_N, small, large, claims["ClaimAmount"].values)

    # ---------- Figures ----------
    _figures(fig, model_frame, sev, fart, fte, ml_rate, sart, y_all=claims["ClaimAmount"].values,
             base=base, capital_rows=rows, fglm=fglm, fml=fml, sglm=sglm, sml=sml)

    out = {"portfolio": summary,
           "frequency": {"glm": fglm, "ml_challenger": fml},
           "severity": {"glm_and_tail": sglm, "ml_challenger": sml},
           "pure_premium": pp,
           "reconciliation": {"model_expected_loss_eur_m": round(mu_N*mean_sev/1e6, 2),
                              "simulated_mean_loss_eur_m": round(float(base.mean())/1e6, 2),
                              "observed_total_losses_eur_m": round(float(sev["ClaimAmount"].sum())/1e6, 2)},
           "underwriting_capital": cap_table,
           "gpd_stress_sensitivity": gpd_sens,
           "n_sims": N}
    with open(os.path.join(res, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(fcoef).to_csv(os.path.join(res, "frequency_coefficients.csv"), index=False)
    fart["calibration"].to_csv(os.path.join(res, "frequency_calibration.csv"), index=False)
    print(json.dumps({k: out[k] for k in ["portfolio", "frequency", "severity",
                                           "pure_premium", "reconciliation"]}, indent=2))
    print("\nUnderwriting-risk capital (VaR99.5 - E[L]):")
    for r in cap_table:
        print(f"  {r['scenario']:<50} capital={r['uw_capital_eur_m']:>7} m "
              f"({r['uw_capital_pct_of_expected']}%)")
    print("\nGPD stress sensitivity (uw capital, EUR m):", gpd_sens["range_eur_m"])
    return out


def _gpd_sensitivity(mu_N, small, large, y_all, n_sims=100_000):
    """Report the GPD tail-extrapolation capital as a RANGE, varying the threshold,
    the simulation seed, GPD parameter uncertainty, and the number of simulations."""
    from scipy import stats
    import numpy as np
    results = {"by_threshold": {}, "by_seed": [], "by_param_bootstrap": {},
               "by_n_sims": {}}
    caps = []

    # (a) threshold 90/95/97%
    for q in [0.90, 0.95, 0.97]:
        ut = float(np.quantile(y_all, q)); exc = y_all[y_all > ut] - ut
        xi, _, beta = stats.genpareto.fit(exc, floc=0); p = float((y_all > ut).mean())
        sm = np.asarray(y_all[y_all <= ut], float); lg = np.asarray(y_all[y_all > ut], float)
        L = cap.simulate_aggregate(mu_N, sm, lg, p, "gpd", u_tail=ut, xi=xi, beta=beta,
                                   cv_sys=0.05, n_sims=n_sims, seed=1)
        el = cap.model_expected_loss(mu_N, sm, lg, p, "gpd", u_tail=ut, xi=xi, beta=beta)
        c = (float(np.quantile(L, 0.995)) - el) / 1e6
        results["by_threshold"][str(q)] = {"xi": round(float(xi), 3),
                                           "uw_capital_eur_m": round(c, 1)}
        caps.append(c)

    # fixed 95% threshold for the remaining axes
    ut = float(np.quantile(y_all, 0.95)); exc = y_all[y_all > ut] - ut
    xi, _, beta = stats.genpareto.fit(exc, floc=0); p = float((y_all > ut).mean())
    sm = np.asarray(y_all[y_all <= ut], float); lg = np.asarray(y_all[y_all > ut], float)

    # (b) simulation seed
    for s in range(1, 6):
        L = cap.simulate_aggregate(mu_N, sm, lg, p, "gpd", u_tail=ut, xi=xi, beta=beta,
                                   cv_sys=0.05, n_sims=n_sims, seed=s)
        el = cap.model_expected_loss(mu_N, sm, lg, p, "gpd", u_tail=ut, xi=xi, beta=beta)
        c = (float(np.quantile(L, 0.995)) - el) / 1e6
        results["by_seed"].append(round(c, 1)); caps.append(c)

    # (c) GPD parameter uncertainty: parametric bootstrap of the tail fit.
    # A resampled fit can land at xi >= 1, where the gross mean is infinite and
    # gross capital is undefined; those draws are counted and excluded from the
    # capital statistics (their VaR is still finite and is summarised separately).
    n_boot = 40
    rng = np.random.default_rng(0); boot = []; var_inf = []; n_infinite = 0
    for _ in range(n_boot):
        resample = stats.genpareto.rvs(xi, loc=0, scale=beta, size=len(exc), random_state=rng)
        xi_b, _, beta_b = stats.genpareto.fit(resample, floc=0)
        L = cap.simulate_aggregate(mu_N, sm, lg, p, "gpd", u_tail=ut, xi=xi_b, beta=beta_b,
                                   cv_sys=0.05, n_sims=50_000, seed=7)
        var = float(np.quantile(L, 0.995)) / 1e6
        el = cap.model_expected_loss(mu_N, sm, lg, p, "gpd", u_tail=ut, xi=xi_b, beta=beta_b)
        if not np.isfinite(el):
            n_infinite += 1; var_inf.append(var); continue
        boot.append(var - el / 1e6)
    results["by_param_bootstrap"] = {
        "n_fits": n_boot, "n_infinite_mean_xi_ge_1": n_infinite,
        "capital_p05_eur_m": round(float(np.quantile(boot, 0.05)), 1),
        "capital_p50_eur_m": round(float(np.quantile(boot, 0.50)), 1),
        "capital_p95_eur_m": round(float(np.quantile(boot, 0.95)), 1),
        "infinite_mean_var_range_eur_m": ([round(min(var_inf), 1), round(max(var_inf), 1)]
                                          if var_inf else None)}
    caps.extend([np.quantile(boot, 0.05), np.quantile(boot, 0.95)])

    # (d) number of simulations
    for n in [50_000, 100_000, 200_000]:
        L = cap.simulate_aggregate(mu_N, sm, lg, p, "gpd", u_tail=ut, xi=xi, beta=beta,
                                   cv_sys=0.05, n_sims=n, seed=1)
        el = cap.model_expected_loss(mu_N, sm, lg, p, "gpd", u_tail=ut, xi=xi, beta=beta)
        results["by_n_sims"][str(n)] = round((float(np.quantile(L, 0.995)) - el) / 1e6, 1)
        caps.append(results["by_n_sims"][str(n)])

    results["range_eur_m"] = [round(float(min(caps)), 1), round(float(max(caps)), 1)]
    return results


def _figures(fig, mf, sev, fart, fte, ml_rate, sart, y_all, base, capital_rows, fglm, fml, sglm, sml):
    # 1 severity hist
    f, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(np.log10(sev["ClaimAmount"]), bins=60, color=ORANGE)
    ax.set_xlabel("log10(claim amount, EUR)"); ax.set_ylabel("claims")
    ax.set_title("Claim severity is strongly right-skewed")
    f.tight_layout(); f.savefig(f"{fig}/01_severity_hist.png"); plt.close(f)

    # 2 frequency lift GLM vs ML (deciles by GLM prediction)
    claims = fte["ClaimNb"].values.astype(float); expo = fte["Exposure"].values.astype(float)
    order = np.argsort(fart["test_rate"] / expo)
    dec = np.array_split(order, 10)
    obs = [claims[d].sum() / expo[d].sum() for d in dec]
    prd = [fart["test_rate"][d].sum() / expo[d].sum() for d in dec]
    prd_ml = [ml_rate[d].sum() / expo[d].sum() for d in dec]
    f, ax = plt.subplots(figsize=(5.6, 3.6)); x = np.arange(1, 11)
    ax.plot(x, obs, "o-", color=BLUE, label="observed")
    ax.plot(x, prd, "s--", color=ORANGE, label="GLM")
    ax.plot(x, prd_ml, "^:", color=GREEN, label="ML (boosting)")
    ax.set_xlabel("decile of predicted frequency"); ax.set_ylabel("claims / exposure-year")
    ax.legend(); ax.set_title("Frequency lift, out of sample")
    f.tight_layout(); f.savefig(f"{fig}/02_frequency_lift.png"); plt.close(f)

    # 3 mean excess
    thr = np.quantile(y_all, np.linspace(0.80, 0.995, 40))
    me = [y_all[y_all > t].mean() - t for t in thr]
    f, ax = plt.subplots(figsize=(5.4, 3.4)); ax.plot(thr, me, color=BLUE)
    ax.set_xscale("log"); ax.set_xlabel("threshold u (EUR, log)"); ax.set_ylabel("mean excess")
    ax.set_title("Mean-excess plot: heavy Pareto tail")
    f.tight_layout(); f.savefig(f"{fig}/03_mean_excess.png"); plt.close(f)

    # 4 GLM vs ML metric comparison bars
    f, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
    ax[0].bar(["GLM", "ML"], [fglm["test_poisson_deviance_glm"], fml["test_poisson_deviance_ml"]],
              color=[ORANGE, GREEN]); ax[0].set_title("Frequency: test Poisson deviance (lower=better)")
    ax[1].bar(["GLM", "ML"], [fglm["test_weighted_gini_glm"], fml["test_weighted_gini_ml"]],
              color=[ORANGE, GREEN]); ax[1].set_title("Frequency: weighted Gini (higher=better)")
    for a in ax:
        for p in a.patches:
            a.text(p.get_x()+p.get_width()/2, p.get_height(), f"{p.get_height():.4f}",
                   ha="center", va="bottom", fontsize=8)
    f.tight_layout(); f.savefig(f"{fig}/04_glm_vs_ml.png"); plt.close(f)

    # 5 aggregate loss
    f, ax = plt.subplots(figsize=(6.0, 3.6)); hi = np.quantile(base, 0.995)
    ax.hist(base[base <= hi]/1e6, bins=80, color=BLUE, alpha=0.85)
    ax.axvline(base.mean()/1e6, color="black", lw=1.2, label="E[S]")
    ax.axvline(hi/1e6, color=ORANGE, lw=1.4, label="99.5% VaR")
    ax.set_xlabel("annual aggregate loss (EUR m)"); ax.set_ylabel("simulated years")
    ax.set_title("Annual loss distribution and 1-in-200 capital (gross)"); ax.legend()
    f.tight_layout(); f.savefig(f"{fig}/05_aggregate_loss.png"); plt.close(f)

    # 6 capital scenarios
    labels = ["Emp\ncv=0%", "Emp\ncv=5%", "Emp\ncv=10%", "GPD\nstress",
              "Cap\n2m", "Cap\n1m", "GPD net\ncap 2m"]
    capv = [capital_rows[i].uw_capital for i in [1, 0, 2, 3, 4, 5, 6]]
    colors = [BLUE]*3 + [RED] + [GREEN]*3
    f, ax = plt.subplots(figsize=(7.2, 3.8)); ax.bar(labels, np.array(capv)/1e6, color=colors)
    ax.set_yscale("log"); ax.set_ylabel("underwriting-risk capital (EUR m, log)")
    ax.set_title("Capital by tail assumption and per-claim cap")
    for i, v in enumerate(capv):
        ax.text(i, v/1e6, f"{v/1e6:.0f}", ha="center", va="bottom", fontsize=8)
    f.tight_layout(); f.savefig(f"{fig}/06_capital_scenarios.png"); plt.close(f)
