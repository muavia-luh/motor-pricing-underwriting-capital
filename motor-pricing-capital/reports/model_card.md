# Model card

## Purpose

An internal-model-style educational prototype for pricing and underwriting-risk capital on
French motor third-party-liability insurance. It estimates the expected annual claim cost
per policy and the 1-in-200 aggregate loss for the portfolio. It is **not** a production
tariff and **not** a regulatory capital model; it covers motor underwriting risk only.
Underwriting-risk capital is defined as VaR₉₉.₅(L) − E[L], with L the annual aggregate loss
and E[L] the deterministic model-implied expected loss.

## Data

CASdatasets `freMTPL2` — 677,991 policy records (~358,343 exposure-years), 26,444 individual
claims. Rating factors:
driver age, vehicle age and power, bonus-malus, area, population density, region,
vehicle brand and fuel. Claim counts capped at 4, exposure capped at 1. Reported claim
count reconciles with the number of severity records before any pure premium is formed.

## Models

| Component | Transparent model | Challenger |
|---|---|---|
| Frequency | Poisson GLM, log link, log-exposure offset (in-house IRLS) | HistGradientBoosting, Poisson loss |
| Frequency (dispersion) | Negative Binomial cross-check (θ ≈ 1.3) | — |
| Severity (body) | Gamma GLM, log link | HistGradientBoosting, Gamma loss |
| Severity (tail) | Generalised Pareto (peaks-over-threshold) | — |
| Capital | Mixed-Poisson collective risk model, VaR at 99.5% (TVaR diagnostic only) | — |

## Intended use and users

Illustrative portfolio work for actuarial pricing, actuarial analytics and model
validation / internal-model teams. Suitable for discussing method, validation and model
risk — not for setting real prices or capital.

## Metrics

Frequency: exposure-weighted Poisson deviance, exposure-weighted concentration Gini,
decile calibration and lift. Severity: Gamma deviance and actual-to-expected. Capital:
99.5% VaR, underwriting-risk capital = VaR − E[L] with a deterministic model-implied E[L],
a built-in reconciliation of E[L] against observed losses, and a GPD tail-extrapolation
stress reported as a range (threshold, seed, simulation count, parameter uncertainty). TVaR
is a supplementary diagnostic only and is unstable under the fitted heavy tail.

## Limitations and ethical considerations

- Rating factors carry little severity signal; the body severity model adds little over
  the mean and is retained mainly for the pure premium.
- The tail shape (ξ ≈ 0.9) is estimated from very few large claims and is unstable across
  thresholds; gross capital is therefore a range, not a point estimate.
- Motor underwriting risk only — not a regulatory capital figure; no market, counterparty,
  operational or catastrophe risk, no diversification across lines, no commercial loadings,
  no temporal validation. The reinsurance cap is a stylised per-claim retention; reinsurance
  premium, reinstatements, aggregate limits, expenses and counterparty-default risk are not
  modelled.
- The data is an anonymised French portfolio; region and density are geographic proxies.
  A production tariff would require a fairness and proxy-discrimination assessment before
  use, which is out of scope here.

## Reproducibility

Fixed seeds; one command (`python3 scripts/run_all.py`) regenerates every number and
figure. Data is fetched by `scripts/00_get_data.py`, not committed.
