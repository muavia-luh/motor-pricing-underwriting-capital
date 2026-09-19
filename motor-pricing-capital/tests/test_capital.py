"""Tests for the collective risk / capital model. Run with `pytest` or
`python3 tests/test_capital.py`. These check economic properties the model must
satisfy, not fixed outputs."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from motor_pricing.capital import simulate_aggregate, summarise, model_expected_loss

# A small synthetic severity model reused across tests.
RNG = np.random.default_rng(0)
SMALL = RNG.gamma(2.0, 500.0, size=5000)                 # body claims
LARGE = 5000 + RNG.gamma(1.2, 8000.0, size=500)          # tail claims
P_TAIL = 500 / 5500
MU_N = 2000.0


def _sim(**kw):
    base = dict(mu_N=MU_N, small_claims=SMALL, large_claims=LARGE, p_tail=P_TAIL,
                tail_mode="empirical", cv_sys=0.05, n_sims=60_000, seed=1)
    base.update(kw)
    return simulate_aggregate(**base)


def _el(**kw):
    base = dict(mu_N=MU_N, small=SMALL, large=LARGE, p_tail=P_TAIL, tail_mode="empirical")
    base.update(kw)
    return model_expected_loss(**base)


def test_model_expected_loss_matches_simulation():
    # Deterministic E[L] should match the Monte-Carlo mean within MC error.
    losses = _sim()
    assert abs(losses.mean() - _el()) / _el() < 0.02


def test_capital_is_non_negative():
    r = summarise(_sim(), "t", expected_loss=_el())
    assert r.uw_capital >= 0 and r.var_995 >= r.expected_loss


def test_tvar_at_least_var():
    r = summarise(_sim(), "t", expected_loss=_el())
    assert r.tvar_995 >= r.var_995


def test_systematic_risk_increases_capital():
    low = summarise(_sim(cv_sys=0.0), "lo", expected_loss=_el())
    high = summarise(_sim(cv_sys=0.15), "hi", expected_loss=_el())
    assert high.var_995 > low.var_995


def test_reinsurance_cap_reduces_capital():
    gross = summarise(_sim(claim_cap=None), "gross", expected_loss=_el(claim_cap=None))
    net = summarise(_sim(claim_cap=1_000_000), "net", expected_loss=_el(claim_cap=1_000_000))
    assert net.uw_capital <= gross.uw_capital
    assert net.expected_loss <= gross.expected_loss


def test_gpd_tail_is_heavier_than_empirical():
    emp = summarise(_sim(tail_mode="empirical"), "emp", expected_loss=_el())
    gpd_kw = dict(tail_mode="gpd", u_tail=5000.0, xi=0.7, beta=8000.0)
    gpd = summarise(_sim(**gpd_kw), "gpd",
                    expected_loss=_el(tail_mode="gpd", u_tail=5000.0, xi=0.7, beta=8000.0))
    assert gpd.var_995 > emp.var_995


def test_reproducible_with_seed():
    a = _sim(seed=99); b = _sim(seed=99)
    assert np.array_equal(a, b)


def test_gpd_gross_mean_infinite_for_xi_ge_1():
    # Uncapped GPD with xi >= 1 has no finite mean -> E[L] must be +inf, not a
    # finite (and possibly negative) number.
    el = model_expected_loss(MU_N, SMALL, LARGE, P_TAIL, "gpd",
                             u_tail=5000.0, xi=1.05, beta=8000.0, claim_cap=None)
    assert not np.isfinite(el)
    # A per-claim cap makes the mean finite again.
    el_capped = model_expected_loss(MU_N, SMALL, LARGE, P_TAIL, "gpd",
                                    u_tail=5000.0, xi=1.05, beta=8000.0,
                                    claim_cap=1_000_000)
    assert np.isfinite(el_capped) and el_capped > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\n{len(fns)} tests passed")
