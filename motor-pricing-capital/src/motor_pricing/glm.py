"""
Generalised Linear Models by iteratively reweighted least squares (IRLS).

Implemented from scratch (numpy only) rather than pulled from statsmodels so that
every step of the estimation is visible and defensible: the weight update, the
offset handling, the dispersion estimate and the standard errors all live here.

Families provided:
    - Poisson (log link)   -- claim frequency, with an exposure offset
    - NegativeBinomial     -- claim frequency under over-dispersion (theta fixed,
                              profiled outside via fit_nb_theta)
    - Gamma (log link)     -- attritional claim severity

References for the formulae: McCullagh & Nelder, "Generalized Linear Models"
(2nd ed., 1989), chapters 2 and 8; Wuthrich & Merz, "Statistical Foundations of
Actuarial Learning" (2023), chapter 5.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy import special, stats


# ---------------------------------------------------------------------------
# Families: each knows its link (always log here), variance function and the
# unit deviance used both to fit and to score the model.
# ---------------------------------------------------------------------------
class PoissonFamily:
    name = "Poisson"

    def variance(self, mu):
        return mu

    def unit_deviance(self, y, mu):
        # 2 * (y log(y/mu) - (y - mu)); the y log y term -> 0 as y -> 0.
        # Evaluate the log only where y > 0 so no divide-by-zero is ever computed.
        term = np.zeros_like(mu, dtype=float)
        pos = y > 0
        term[pos] = y[pos] * np.log(y[pos] / mu[pos])
        return 2.0 * (term - (y - mu))

    def loglik(self, y, mu, weights):
        # weights = prior weights (exposure handled via offset, not here)
        return np.sum(weights * (y * np.log(mu) - mu - special.gammaln(y + 1)))


class GammaFamily:
    name = "Gamma"

    def variance(self, mu):
        return mu ** 2

    def unit_deviance(self, y, mu):
        return 2.0 * (-np.log(y / mu) + (y - mu) / mu)

    def loglik(self, y, mu, weights, dispersion):
        # dispersion = 1/shape for the Gamma; report using the estimated phi
        shape = 1.0 / dispersion
        rate = shape / mu
        return np.sum(weights * stats.gamma.logpdf(y, a=shape, scale=1.0 / rate))


class NegBinFamily:
    """Negative binomial with fixed dispersion theta (NB2 parameterisation):
    Var(y) = mu + mu^2 / theta.  As theta -> inf this collapses to Poisson."""
    name = "NegativeBinomial"

    def __init__(self, theta):
        self.theta = float(theta)

    def variance(self, mu):
        return mu + mu ** 2 / self.theta

    def unit_deviance(self, y, mu):
        th = self.theta
        term1 = np.zeros_like(mu, dtype=float)
        pos = y > 0
        term1[pos] = y[pos] * np.log(y[pos] / mu[pos])
        term2 = (y + th) * np.log((y + th) / (mu + th))
        return 2.0 * (term1 - term2)

    def loglik(self, y, mu, weights):
        th = self.theta
        ll = (special.gammaln(y + th) - special.gammaln(th) - special.gammaln(y + 1)
              + th * np.log(th / (th + mu)) + y * np.log(mu / (th + mu)))
        return np.sum(weights * ll)


@dataclass
class GLMResult:
    family: str
    coef: np.ndarray
    names: list
    se: np.ndarray
    deviance: float
    null_deviance: float
    dispersion: float
    loglik: float
    aic: float
    n_obs: int
    n_par: int
    n_iter: int

    def summary(self):
        z = self.coef / self.se
        p = 2 * (1 - stats.norm.cdf(np.abs(z)))
        lines = [f"GLM ({self.family}, log link)   n={self.n_obs}   params={self.n_par}",
                 f"deviance={self.deviance:.1f}   null deviance={self.null_deviance:.1f}"
                 f"   dispersion={self.dispersion:.4f}   AIC={self.aic:.1f}",
                 f"{'term':<28}{'coef':>10}{'se':>10}{'z':>9}{'p':>9}"]
        for i, nm in enumerate(self.names):
            lines.append(f"{nm:<28}{self.coef[i]:>10.4f}{self.se[i]:>10.4f}"
                         f"{z[i]:>9.2f}{p[i]:>9.3f}")
        return "\n".join(lines)


def fit_glm(X, y, family, offset=None, weights=None, names=None,
            max_iter=100, tol=1e-9):
    """Fit a log-link GLM by IRLS.

    X       : (n, p) design matrix, must already include an intercept column.
    y       : (n,) response.
    offset  : (n,) added to the linear predictor (log exposure for frequency).
    weights : (n,) prior weights (exposure for severity averages); default 1.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    names = names or [f"x{i}" for i in range(p)]

    # Start from a sensible mean; log link -> eta0 = log(mean response).
    mu = np.full(n, max(y.mean(), 1e-3))
    eta = np.log(mu)
    beta = np.zeros(p)

    for it in range(1, max_iter + 1):
        # log link: dmu/deta = mu.  IRLS working weights W = w * mu^2 / V(mu).
        V = family.variance(mu)
        Wt = w * (mu ** 2) / V
        z = (eta - offset) + (y - mu) / mu          # working response
        XtW = X.T * Wt
        A = XtW @ X
        b = XtW @ z
        beta_new = np.linalg.solve(A, b)
        eta = X @ beta_new + offset
        eta = np.clip(eta, -30, 30)
        mu = np.exp(eta)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Deviance and dispersion (Pearson) on the fitted model.
    dev = np.sum(w * family.unit_deviance(y, mu))
    pearson = np.sum(w * (y - mu) ** 2 / family.variance(mu))
    dof = n - p
    if isinstance(family, (PoissonFamily, NegBinFamily)):
        dispersion = 1.0                      # fixed by the family
    else:
        dispersion = pearson / dof            # Gamma: estimate phi

    # Covariance of beta = dispersion * (X' W X)^-1 with final weights.
    V = family.variance(mu)
    Wt = w * (mu ** 2) / V
    cov = dispersion * np.linalg.inv((X.T * Wt) @ X)
    se = np.sqrt(np.diag(cov))

    # Null model (intercept only) deviance for a McFadden-style reference.
    # The intercept-only MLE under a log link must include the offset, otherwise
    # the null model is misspecified for a frequency model with a log-exposure
    # offset. With offset o_i the intercept solves sum(w*y) = sum(w*exp(b0+o)),
    # i.e. b0 = log( sum(w*y) / sum(w*exp(o)) ), and mu0_i = exp(b0 + o_i).
    b0 = np.log(np.sum(w * y) / np.sum(w * np.exp(offset)))
    mu0 = np.exp(b0 + offset)
    null_dev = np.sum(w * family.unit_deviance(y, mu0))

    # Log-likelihood and AIC.
    if isinstance(family, PoissonFamily):
        ll = family.loglik(y, mu, w); k = p
    elif isinstance(family, NegBinFamily):
        ll = family.loglik(y, mu, w); k = p + 1     # theta counts as a parameter
    else:
        ll = family.loglik(y, mu, w, dispersion); k = p + 1
    aic = -2 * ll + 2 * k

    return GLMResult(family.name, beta, names, se, dev, null_dev, dispersion,
                     ll, aic, n, p, it), mu


def fit_nb_theta(X, y, offset=None, weights=None, names=None,
                 theta_grid=None):
    """Profile the NB dispersion theta: fit an NB GLM on a grid of theta and
    keep the one with the highest log-likelihood. Returns (result, mu, theta)."""
    if theta_grid is None:
        theta_grid = np.geomspace(0.1, 20.0, 40)
    best = None
    for th in theta_grid:
        res, mu = fit_glm(X, y, NegBinFamily(th), offset=offset,
                          weights=weights, names=names)
        if best is None or res.loglik > best[0].loglik:
            best = (res, mu, th)
    return best
