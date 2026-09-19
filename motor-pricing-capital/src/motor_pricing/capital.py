"""
Collective risk model for the portfolio's one-year aggregate loss L, and the
1-in-200 *underwriting-risk capital* for this motor book.

This is an internal-model-style educational prototype, not a regulatory capital
calculation. Underwriting-risk capital is defined here as

    UW capital = VaR_99.5(L) - E[L]

where E[L] is the deterministic model-implied expected loss (mu_N * mean severity),
not the Monte-Carlo sample mean -- the sample mean is itself unstable under the
heavy tail, so subtracting it would inject noise into the capital figure.

Structure (standard compound / mixed-Poisson model):

    Theta   ~ Gamma(mean 1, cv = cv_sys)        systematic ("good/bad year")
    N | Theta ~ Poisson(Theta * mu_N)           annual claim count
    each claim severity X is split at threshold u:
        small claim (<= u):  resampled from the observed small claims
        large claim ( > u):  either resampled from the observed large claims
                             ("empirical" base case) or modelled as u + GPD(xi,beta)
                             ("gpd" tail-extrapolation stress)
    L = sum of the N severities

The modelled year is calibrated to the observed exposure, so mu_N equals the
model's expected claim count over the data period and E[L] reconciles with the
actual total losses -- a built-in validation check. Because large claims are
resampled at their observed relative frequency, the observed maximum claim
appears in a simulated year with the same frequency as in the data; that is a
random count, not a fixed once-per-year occurrence.

Process risk in the count is negligible at this portfolio size (Poisson CV
~ 0.4%), so the frequency risk that matters at 1-in-200 is *systematic*, set by
cv_sys. Severity tail risk is carried by the large-claim model. Both are explicit
assumptions and are stress-tested rather than buried.

VaR is the 99.5% quantile of L. TVaR (average loss beyond the 99.5% VaR) is
reported only as a supplementary tail diagnostic: under the fitted heavy tail
(xi ~ 0.9, infinite variance) it does not stabilise and should not be read as a
precise number.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy import stats


@dataclass
class CapitalResult:
    scenario: str
    expected_loss: float          # deterministic model-implied E[L]
    sim_mean_loss: float          # Monte-Carlo sample mean (diagnostic only)
    sd_loss: float
    var_995: float
    tvar_995: float               # supplementary tail diagnostic; unstable
    uw_capital: float             # VaR_99.5(L) - E[L]
    uw_capital_pct_of_expected: float
    n_sims: int


def simulate_aggregate(mu_N, small_claims, large_claims, p_tail,
                       tail_mode="empirical", u_tail=None, xi=None, beta=None,
                       cv_sys=0.05, claim_cap=None, n_sims=200_000,
                       chunk=20_000, seed=0):
    """Monte-Carlo the annual aggregate loss S. The small-claim sum uses a normal
    approximation (justified: tens of thousands of light-tailed claims per year);
    large claims are simulated one by one so the heavy tail is captured exactly.

    tail_mode = "empirical" -> bootstrap observed large claims (base case)
    tail_mode = "gpd"       -> u_tail + GPD(xi, beta) draws (extrapolation stress)
    """
    rng = np.random.default_rng(seed)
    m_s = float(small_claims.mean())
    v_s = float(small_claims.var())
    large = np.asarray(large_claims, dtype=float)
    k = np.inf if cv_sys == 0 else 1.0 / cv_sys ** 2      # Gamma shape

    losses = np.empty(n_sims)
    done = 0
    while done < n_sims:
        c = min(chunk, n_sims - done)
        theta = np.ones(c) if cv_sys == 0 else rng.gamma(k, 1.0 / k, size=c)
        N = rng.poisson(theta * mu_N)
        N_tail = rng.binomial(N, p_tail)
        N_body = N - N_tail

        S_body = rng.normal(N_body * m_s, np.sqrt(np.maximum(N_body, 0) * v_s))
        S_body = np.maximum(S_body, 0.0)

        tot = int(N_tail.sum())
        S_tail = np.zeros(c)
        if tot > 0:
            if tail_mode == "empirical":
                claim = rng.choice(large, size=tot, replace=True)
            else:
                claim = u_tail + stats.genpareto.rvs(xi, loc=0, scale=beta,
                                                     size=tot, random_state=rng)
            if claim_cap is not None:
                claim = np.minimum(claim, claim_cap)
            ends = np.cumsum(N_tail); starts = ends - N_tail
            csum = np.concatenate([[0.0], np.cumsum(claim)])
            S_tail = csum[ends] - csum[starts]
        losses[done:done + c] = S_body + S_tail
        done += c
    return losses


def model_expected_loss(mu_N, small, large, p_tail, tail_mode="empirical",
                        u_tail=None, xi=None, beta=None, claim_cap=None):
    """Deterministic model-implied expected annual loss E[L] = mu_N * E[X], with
    the per-claim severity mean E[X] computed in closed form (no Monte Carlo):

    - empirical: mean of the observed claims, large claims capped where applicable;
    - gpd: (1-p_tail)*mean(small) + p_tail*(u + E[GPD excess, capped]).
      The uncapped GPD mean beta/(1-xi) exists only for xi < 1; for xi >= 1 the
      mean is infinite, so E[L] is returned as +inf and capital (VaR - E[L]) is
      undefined for that fit. A capped mean E[min(Y,t)] is always finite.
    """
    small = np.asarray(small, float); large = np.asarray(large, float)
    if tail_mode == "empirical":
        larg = large if claim_cap is None else np.minimum(large, claim_cap)
        exp_sev = np.concatenate([small, larg]).mean()
    else:
        mean_small = small.mean()
        if claim_cap is None:
            if xi >= 1.0:
                return float("inf")                       # mean does not exist
            e_excess = beta / (1.0 - xi)
        else:
            t = claim_cap - u_tail
            if abs(xi - 1.0) < 1e-9:
                e_excess = beta * np.log1p(t / beta)      # xi -> 1 limit
            else:
                e_excess = (beta / (1.0 - xi)) * (1.0 - (1.0 + xi * t / beta) ** (1.0 - 1.0 / xi))
        exp_sev = (1.0 - p_tail) * mean_small + p_tail * (u_tail + e_excess)
    return float(mu_N * exp_sev)


def summarise(losses, scenario, expected_loss, alpha=0.995):
    """Summarise a simulated loss vector. `expected_loss` is the deterministic
    model-implied E[L]; underwriting capital is VaR_99.5(L) - E[L], so a noisy
    Monte-Carlo sample mean never enters the capital figure."""
    var = float(np.quantile(losses, alpha))
    tvar = float(losses[losses >= var].mean())
    return CapitalResult(scenario=scenario, expected_loss=float(expected_loss),
                         sim_mean_loss=float(losses.mean()), sd_loss=float(losses.std()),
                         var_995=var, tvar_995=tvar,
                         uw_capital=var - expected_loss,
                         uw_capital_pct_of_expected=100 * (var - expected_loss) / expected_loss,
                         n_sims=len(losses))
