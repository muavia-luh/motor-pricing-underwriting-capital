# Motor Insurance Pricing and Underwriting Capital: GLMs, Extreme-Value Tails and Reinsurance

**Technical memo — an internal-model-style educational prototype on the freMTPL2 portfolio**

---

## 1. What this memo does

It takes a real motor third-party-liability portfolio and does two jobs on it that sit
next to each other inside a non-life insurer: it builds a technical price per policy, and
it estimates how much capital the book needs to absorb a 1-in-200 year. Along the way it
asks whether a machine-learning model prices better than a classical GLM, and — the part
that turns out to matter most — how much the capital number depends on the severity tail
and on reinsurance. The data is `freMTPL2` from CASdatasets.

This is a prototype built to demonstrate method. It is **not** a regulatory capital
calculation and it covers **motor underwriting risk only**. Where "capital" appears below
it means underwriting-risk capital in the sense defined in Section 6, not a regulatory
requirement.

The conclusion in one line: the ML model prices slightly better, the GLM stays
competitive and interpretable, and the capital number is governed by the tail and the
reinsurance structure rather than by either pricing model.

## 2. The portfolio

The data holds 677,991 policy records representing approximately 358,343 exposure-years.
Claims are rare and usually cheap, but occasionally enormous: claim frequency is 7.37% per
exposure-year, the average claim costs €2,266 but the median is only €1,172, and the single
largest claim is €4.08m (Figure 1). That gap between the average and the maximum is the
whole story of the capital section — the largest 1% of claims account for roughly 38% of
all claim cost.

One data point worth stating precisely. The severity table contains 26,444 positive claim
records. The frequency table, after capping the claim count at four per policy, contains
26,405 claims — a difference of 39, the claims removed by that cap. The uncapped reported
count (26,444) equals the number of severity records exactly, so frequency and severity are
measured on the same claims before they are ever multiplied into a premium.

## 3. Frequency: GLM and an ML challenger

The transparent model is a Poisson GLM with a log link and a log-exposure offset, fitted by
iteratively reweighted least squares in `glm.py` — written from scratch, so the weight
update, the offset (including in the null model) and the standard errors are all visible.
Its relativities are sensible: relative to the worst bonus-malus band the maximum-discount
band carries about 0.12× the frequency, and older drivers claim far less often than
18–20-year-olds. The counts are over-dispersed (Pearson/df ≈ 1.7), which a Negative
Binomial confirms (θ ≈ 1.3, lower AIC) without much moving the central estimates.

The challenger is a histogram gradient-boosting model with a Poisson loss, given exactly
the same rating factors and the same train/test split so the comparison is fair. On the
held-out 20% of policies the boosting model prices slightly better: test Poisson deviance
0.4521 against the GLM's 0.4596, and a weighted Gini of 0.328 against 0.296. Both track
observed frequency closely across deciles and give a top decile with about 5.85× the claim
rate of the bottom (Figures 2 and 4).

The honest reading is that boosting adds real but modest lift, at the cost of a model a
regulator and an underwriter cannot read line by line. That trade-off — not the deviance
third decimal — is the actual decision, and it is the kind of judgement a model-validation
team is paid to make.

## 4. Severity: a weak body, a heavy tail

Severity splits into two very different problems. For the **body** (claims below the 95th
percentile, €4,765) I fit a Gamma GLM, and as a challenger the same gradient-boosting
machinery with a Gamma loss; the severity data is split by policy ID so a policy's claims
never straddle train and test. Both models land in the same unhelpful place: the test Gamma
deviance of the GLM (≈0.628) and of the ML model (≈0.626) sit right on top of the
intercept-only baseline (≈0.627) — the GLM is, if anything, a hair worse than predicting
the mean. Knowing the driver and the car tells you a lot about whether they
claim and little about how much it costs. That is a known property of motor severity, and
it is better reported than dressed up.

For the **tail** I use peaks-over-threshold: a generalised Pareto fit to the excesses above
a high threshold. The mean-excess plot slopes upward (Figure 3), the signature of a heavy
Pareto tail, and the fitted shape is ξ ≈ 0.904. That number has teeth. A shape this close to
one means the tail has a finite mean but an **infinite variance** — the theoretical variance
does not exist — so any quantity that leans on the variance, including the simulated sample
mean and the TVaR, is intrinsically unstable and will swing from run to run. Refitting at
higher thresholds pushes ξ from 0.85 (u = p90) to 0.98 (u = p97), right against the boundary
beyond which even the mean stops existing. With one €4.08m claim dominating, the tail
estimate is fragile, and that fragility is why the tail capital below is a range, not a
point.

## 5. Pure premium

The technical pure premium is each policy's expected frequency times the **pooled mean
severity** — a single portfolio-wide average of €2,266, not a severity-GLM prediction.
That is a deliberate choice, not an oversight: Section 4 showed the severity models
carry essentially no signal, so predicting each policy's claim cost from its rating
factors would add noise without lift. Using the pooled mean keeps the premium's risk
differentiation where the signal actually is — in the frequency model — and leaves
severity as a flat multiplier. (Whether to charge severity flat or by factor is itself
a real pricing decision; here the data says flat.)

The result is €177.69 on average, with a 95th-percentile policy at €399. Summed at the
observed exposure, expected annual losses are €59.7m, sitting right on the €59.9m actually
observed. The pieces multiply back to reality — the first validation of the whole chain.

## 6. Underwriting-risk capital

Capital here is defined as

> **underwriting-risk capital = VaR₉₉.₅(L) − E[L]**,

where L is the modelled annual aggregate claim loss and E[L] is the **deterministic
model-implied** expected loss (μ_N × mean severity), not the Monte-Carlo sample mean. The
sample mean is itself unstable under the heavy tail, so subtracting it would inject noise
into the capital figure; the closed-form E[L] does not.

The loss L is generated by a collective risk model. Claim count is mixed-Poisson —
`N | Θ ~ Poisson(Θ·μ_N)` with `Θ ~ Gamma(mean 1, cv = cv_sys)` — where Θ is a systematic
good-year / bad-year multiplier. This matters because process risk in the count is
negligible at this size (Poisson CV ≈ 0.4%), so the frequency risk that shows up at 1-in-200
is systematic, not random. Severities are resampled from the observed small and large
claims; because large claims are resampled at their observed relative frequency, the
observed maximum appears in a simulated year with the same frequency as in the data — a
random count, not a guaranteed once-a-year event. The year's total is simulated 200,000
times and the 99.5% VaR is read off.

**The principal estimate** uses the observed large claims, so the model never invents a loss
bigger than anything on record. That gives underwriting-risk capital of **€17.1m, about 29%
of expected loss** (Figures 5 and 6). Varying the systematic-frequency assumption moves it
between €15.1m (cv = 0%) and €22.3m (cv = 10%): it matters but does not dominate.

TVaR (the average loss beyond the 99.5% VaR) is reported in the results table only as a
supplementary tail diagnostic. Under ξ ≈ 0.9 it does not stabilise and should not be read as
a precise number.

## 7. The tail-extrapolation stress, as a range

Letting the fitted GPD extrapolate — allowing a future claim larger than any observed — is a
**stress scenario**, not the project's headline capital estimate. And it is deliberately
reported as a range, because a single number here would be false precision. Sweeping the
threshold, the simulation seed, the number of simulations, and GPD parameter uncertainty (a
parametric bootstrap of the tail fit), the stress capital lands anywhere from about **€120m
to €515m**, with a central value near €280–300m. The seed and the simulation count barely
move it (±€10m); the threshold and, above all, the parameter uncertainty drive the width.

The parameter bootstrap needs one caveat that is itself instructive. In 2 of the 40 bootstrap
fits the resampled shape came out at ξ ≥ 1, where the generalised Pareto has no finite mean at
all — so the gross expected loss, and therefore gross capital as VaR − E[L], is undefined, not
just large. Those two draws are excluded from the range above rather than being assigned a
spurious number; for the record their 99.5% VaR still comes out finite, around €1.0bn. The
honest content of the stress is "somewhere in the low hundreds of millions, sometimes formally
undefined, and not pin-downable from 26,444 claims" — which is itself the finding.

## 8. Why this is really a reinsurance question

That imprecision is exactly what reinsurance removes. Applying a stylised per-claim loss cap
(a retention) at €2m brings the principal capital to €12.2m and collapses the entire
€120–515m stress to about €12.6m as well (Figure 6). A €1m cap takes it to €10m. Net of the
cap the capital number is stable and modest no matter how heavy you believe the tail to be:
the whole of the tail uncertainty is transferred.

So the practical lever for this book is not another decimal of GLM or boosting accuracy — it
is the reinsurance structure. The pricing model sets the premium; the retention sets the
capital.

This is a stylised per-claim cap, not a full reinsurance treaty. Reinsurance premium,
reinstatements, aggregate limits, expenses and counterparty-default risk are not modelled;
the point here is only the effect of capping individual losses on the capital.

## 9. Limitations

- **Not a regulatory figure.** This is an internal-model-style educational prototype for
  motor underwriting risk. It is not a regulatory capital figure and carries no regulatory standing:
  no market, counterparty or operational risk, no diversification across lines, no
  catastrophe module.
- **Tail.** ξ ≈ 0.9 is estimated from very few large claims, is unstable across thresholds,
  and implies an infinite variance; the tail-extrapolation capital is a range, and the TVaR
  is a diagnostic only.
- **Severity signal.** Rating factors barely predict claim size; the body severity model
  could be replaced by the empirical distribution with little loss.
- **Assumptions.** The systematic-frequency CV is an assumption, stress-tested at 0/5/10%
  rather than estimated from a time series the data does not contain.

None of this is hidden in the code; each is a place a production model would go further.

---

*Reproducibility: all figures and numbers are generated by `scripts/run_all.py` on the
CASdatasets `freMTPL2` data (pinned commit) with fixed seeds. Methods follow McCullagh &
Nelder (GLMs), Embrechts, Klüppelberg & Mikosch (extreme-value / POT), and the
frequency-severity conventions in Wüthrich & Merz.*
