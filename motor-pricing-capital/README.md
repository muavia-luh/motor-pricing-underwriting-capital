# Motor Insurance Pricing and Underwriting Capital: GLMs, Extreme-Value Tails and Reinsurance

An internal-model-style educational prototype on a French motor third-party-liability
portfolio. It fits a claim **frequency** and **severity** model, benchmarks the transparent
actuarial GLM against a **gradient-boosting challenger**, models the **large-loss tail** with
extreme-value theory, and runs a collective risk model to estimate **1-in-200
underwriting-risk capital** — with the reinsurance analysis that decides how big that number
really is.

Data: `freMTPL2` from CASdatasets — 677,991 policy records (~358,343 exposure-years), 26,444
claims.

> This is a demonstration prototype, **not** a regulatory capital calculation, and it covers
> **motor underwriting risk only**. "Capital" throughout means underwriting-risk capital as
> defined below, not a regulatory capital requirement of any kind.

## Why it's built this way

Two things a non-life insurer's actuarial and risk teams do, side by side, in one project:

- **Pricing / analytics** — a frequency-severity GLM, an ML challenger, and the out-of-sample
  validation you would use to decide whether the ML model is worth the loss of transparency.
- **Risk modelling** — a heavy-tail (extreme-value) model and a 1-in-200 underwriting-capital
  figure, stress-tested for the tail assumption and for reinsurance.

## Headline results

| | GLM | ML challenger |
|---|---:|---:|
| Frequency test Poisson deviance (lower better) | 0.4596 | **0.4521** |
| Frequency weighted Gini (higher better) | 0.2963 | **0.3283** |
| Severity test Gamma deviance (baseline ≈0.627) | ≈0.628 | ≈0.626 |

Boosting wins a little on frequency and roughly ties on severity — a realistic, honest
outcome.

**Underwriting-risk capital** = VaR₉₉.₅(L) − E[L], where L is annual aggregate loss and E[L]
is the deterministic model-implied expected loss:

| Scenario | Capital |
|---|---:|
| Principal estimate (observed tail, gross) | **€17.1m** (29% of expected loss) |
| Tail-extrapolation stress (GPD) — reported as a range | **≈ €120m – €515m** |
| Net of a stylised €2m per-claim cap | **€12.2m** |

The gross figure is a range in the hundreds of millions once you allow the tail to extrapolate
beyond history, and it cannot be pinned down from 26,444 claims. A €2m per-claim cap collapses
the whole range to ≈€12.6m. For this book, underwriting-risk capital is a tail-and-reinsurance
question far more than a pricing-model question. Expected loss reconciles with reality:
model-implied €59.7m vs observed €59.9m.

## Repository layout

```
src/motor_pricing/
  data.py         load + clean; keeps reported and cost-bearing claim definitions
  features.py     banded rating factors + one-hot design (learns all training categories)
  glm.py          IRLS GLM engine (Poisson / Negative Binomial / Gamma), from scratch
  metrics.py      Poisson/Gamma deviance, exposure-weighted Gini, decile calibration
  split.py        row-level and policy-grouped train/test splits
  frequency.py    frequency GLM + NB cross-check + validation
  severity.py     Gamma body GLM + peaks-over-threshold GPD tail
  ml_benchmark.py gradient-boosting challengers (scikit-learn), same split
  capital.py      collective risk model, VaR, deterministic E[L], reinsurance cap
  reporting.py    runs everything incl. GPD stress sensitivity; writes results + figures
  pipeline.py     entry point (python -m motor_pricing.pipeline)
scripts/          pinned data fetch, run-all wrapper, PDF builder
tests/            invariant and behavioural tests for the GLM, metrics and capital model
reports/          JSON results, figures, model card, and the written report
```

## Reproduce it (from a clean checkout)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[test]"                               # package + runtime deps + pytest
python scripts/00_get_data.py                          # fetch freMTPL2 (pinned CASdatasets commit)
python scripts/run_all.py                              # full pipeline (~5-7 min)
python -m pytest                                       # run the test suite
```

The data is **not** committed. `scripts/00_get_data.py` clones a pinned CASdatasets commit and
parses the `.rda` files. Random seeds are fixed; the numbers in
`reports/results/results.json` reproduce.

The written report is rebuilt with `pip install -e ".[report]" && playwright install chromium`
then `python scripts/make_pdf.py` (Playwright + Chromium render the PDF); this is optional and
not needed to run the analysis.

## Method in one paragraph each

**Frequency.** Poisson GLM, log link, log-exposure offset (in the fit and in the null model),
banded rating factors. Over-dispersed counts (Pearson/df ≈ 1.7; NB θ ≈ 1.3 lowers AIC), so a
Negative Binomial is fitted as a cross-check. Validated out of sample by Poisson deviance, an
exposure-weighted Gini and a decile lift chart. A histogram gradient-boosting model (Poisson
loss, same features and split) is the challenger.

**Severity.** A Gamma GLM for the attritional body and a generalised-Pareto (POT) fit for the
tail, split by policy ID. Rating factors barely move severity (both GLM and ML ≈ baseline); the
tail is heavy (ξ ≈ 0.90, finite mean but infinite variance; the largest 1% of claims are ~38%
of all cost).

**Capital.** A mixed-Poisson collective risk model, `N | Θ ~ Poisson(Θ·μ_N)`,
`Θ ~ Gamma(mean 1, cv=cv_sys)`. Capital = VaR₉₉.₅(L) − E[L] with a deterministic E[L]. The GPD
tail-extrapolation stress is reported as a range over threshold, seed, simulation count and
GPD parameter uncertainty. TVaR is a supplementary diagnostic only.

## Validation and honest limitations

- **Reconciliation:** model-implied E[L] €59.7m vs observed €59.9m (0.3%). Reported claim count
  (26,444) equals the number of severity records; capping at four claims removes 39.
- **Tests:** invariant and behavioural checks — GLM recovers known coefficients and respects the offset;
  NB → Poisson as θ → ∞; Gini ranks signal above noise; capital ≥ 0, TVaR ≥ VaR, capital rises
  with systematic risk and falls with the reinsurance cap; deterministic E[L] matches the
  simulation; results reproduce under a fixed seed.
- **Tail instability, stated plainly:** ξ ≈ 0.9 (infinite variance) moves with the threshold
  (0.85 → 0.98); the tail-extrapolation capital is a €120–515m range and the TVaR does not
  stabilise. In 2 of 40 bootstrap tail fits ξ ≥ 1, so the gross mean (and hence capital) is
  undefined; those are excluded from the range and their VaR (~€1.0bn) is reported separately.
- **Scope:** motor underwriting risk only — not a regulatory SCR; no market, counterparty or
  operational risk, no diversification across lines. The reinsurance cap is stylised;
  reinsurance premium, reinstatements, aggregate limits, expenses and counterparty-default risk
  are not modelled.

## Data licence

`freMTPL2` is third-party academic data from CASdatasets and is **not** committed;
`scripts/00_get_data.py` fetches a pinned version. See `data/raw/SOURCE.md`. Project code is
MIT-licensed (see `LICENSE`).
