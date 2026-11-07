"""Each estimator must recover the effect planted by its simulator (run: pytest -q)."""
import numpy as np
import pandas as pd

from causalsimkit import did, events, experiments, propensity, rdd, simulate, synth
from causalsimkit.ols import ols


def test_ols_matches_lstsq():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    y = 1 + X @ np.array([0.5, -1.0, 2.0]) + rng.normal(size=500)
    r = ols(y, X)
    ref = np.linalg.lstsq(np.column_stack([np.ones(500), X]), y, rcond=None)[0]
    assert np.allclose(r.params.to_numpy(), ref)


def test_twfe_did_recovers_effect_under_parallel_trends():
    df = simulate.did_panel(effect=2.0, seed=1)
    r = did.twfe_did(df, "y", "unit", "period", "D")
    assert abs(r.params["D"] - 2.0) < 3 * r.bse["D"]


def test_event_study_flags_pretrend():
    df = simulate.did_panel(pretrend=0.3, seed=1)
    df["rel_time"] = np.where(df.treated == 1, df.period - 12, np.nan)
    tab, res = did.event_study(df, "y", "unit", "period", "rel_time")
    assert res.wald_test(did.lead_names(tab))["p"] < 0.01


def test_group_time_att_recovers_dynamic_effects():
    df = simulate.staggered_panel(seed=2)
    est = did.aggregate_event_time(did.group_time_att(df, "y", "unit", "period", "cohort"))
    truth = df[df.D == 1].groupby("rel_time")["true_te"].mean()
    common = truth.index.intersection(est.index)[:6]
    assert np.max(np.abs(est.loc[common] - truth.loc[common])) < 0.8


def test_synth_beats_did_when_trends_differ():
    wide, _ = simulate.synth_panel(effect=-3.0, seed=3)
    sc = synth.synthetic_control(wide, "treated_market", 30)
    dd = synth.did_counterfactual(wide, "treated_market", 30)
    assert abs(sc.att + 3.0) < 0.6 and abs(dd.att + 3.0) > abs(sc.att + 3.0)


def test_event_study_car():
    R, m, ed = simulate.stock_returns(seed=4)
    ar = events.market_model_abnormal_returns(R, m, ed)
    c = events.car(ar, 0, 5)
    assert abs(c.mean() - 0.03) < 3 * c.sem()


def test_rdd_recovers_jump_and_density_ok():
    df = simulate.rdd_data(effect=0.5, seed=5)
    r = rdd.local_poly_rd(df.score, df.y, bandwidth=0.3)
    assert abs(r.estimate - 0.5) < 3 * r.se
    assert rdd.density_test(df.score)["p"] > 0.01


def test_aipw_recovers_ate():
    df = simulate.observational(seed=6)
    X = df[["x1", "x2", "x3", "x4"]].assign(x3sq=df.x3**2)
    e = propensity.cross_fit_aipw(X, df.y, df.treated)
    assert abs(e.estimate - 1.0) < 3 * e.se


def test_tbr_recovers_lift():
    base = simulate.geo_markets(seed=7)
    vol = base[base.day < 90].groupby("geo").sales.sum()
    pairs = experiments.matched_pairs(vol, seed=7)
    tg = pairs.loc[pairs.treated == 1, "geo"]
    df = simulate.apply_geo_lift(base, tg, 90, 0.05)
    _, s = experiments.tbr(df, tg, 90)
    assert s["lower"] < s["cum_incremental"] < s["upper"] and 0.02 < s["relative_lift"] < 0.08


def test_switchback_burn_in_reduces_bias():
    a = experiments.switchback_assignment(24 * 56, 4, seed=8)
    df = pd.concat([simulate.switchback_marketplace(a.assignment.to_numpy(), seed=8), a[["block", "since_switch"]]], axis=1)
    naive = experiments.analyze_switchback(df, burn_in=0)
    trimmed = experiments.analyze_switchback(df, burn_in=2)
    assert abs(trimmed["estimate"] - 8) < abs(naive["estimate"] - 8)
