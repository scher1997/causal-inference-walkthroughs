"""Synthetic data-generating processes with KNOWN ground truth.

Each notebook estimates an effect we planted ourselves, so you can see exactly
when a method recovers the truth and when an assumption failure biases it.
Swap these for your real data once the workflow makes sense.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _time_effects(rng: np.random.Generator, n_periods: int) -> np.ndarray:
    trend = np.cumsum(rng.normal(0.2, 0.5, n_periods))
    season = 1.0 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
    return trend + season


# --------------------------------------------------------------------------- DiD
def did_panel(
    n_units: int = 40,
    n_periods: int = 20,
    treat_period: int = 12,
    effect: float = 2.0,
    pretrend: float = 0.0,
    frac_treated: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Block-adoption panel. `pretrend` != 0 adds a treated-only linear trend,
    i.e. a parallel-trends violation that continues after treatment."""
    rng = np.random.default_rng(seed)
    treated = (np.arange(n_units) < int(frac_treated * n_units)).astype(int)
    alpha = rng.normal(10, 2, n_units) + 1.5 * treated  # level gaps are fine for DiD
    gamma = _time_effects(rng, n_periods)
    u, t = np.meshgrid(np.arange(n_units), np.arange(n_periods), indexing="ij")
    u, t = u.ravel(), t.ravel()
    tr, post = treated[u], (t >= treat_period).astype(int)
    y = (alpha[u] + gamma[t] + pretrend * tr * (t - treat_period)
         + effect * tr * post + rng.normal(0, 1, u.size))
    return pd.DataFrame({"unit": u, "period": t, "treated": tr, "post": post,
                         "D": tr * post, "y": y})


def staggered_panel(
    n_units: int = 120,
    n_periods: int = 24,
    cohorts: tuple[int, ...] = (6, 11, 16),
    base_effect: dict | None = None,
    growth: float = 0.3,
    seed: int = 0,
) -> pd.DataFrame:
    """Staggered rollout (e.g. a feature launched market-by-market).

    Effects are heterogeneous by cohort and grow with time since adoption,
    exactly the setting where static TWFE is biased. cohort == 0 means never treated.
    """
    rng = np.random.default_rng(seed)
    base_effect = base_effect or {6: 3.0, 11: 2.0, 16: 1.0}
    groups = np.array(list(cohorts) + [0])
    cohort = groups[np.arange(n_units) % len(groups)]
    alpha = rng.normal(10, 2, n_units)
    gamma = _time_effects(rng, n_periods)
    u, t = np.meshgrid(np.arange(n_units), np.arange(n_periods), indexing="ij")
    u, t = u.ravel(), t.ravel()
    g = cohort[u]
    rel = np.where(g > 0, t - g, np.nan)
    on = (g > 0) & (t >= g)
    te = np.where(on, np.vectorize(lambda c: base_effect.get(c, 0.0))(g) + growth * np.nan_to_num(rel), 0.0)
    y = alpha[u] + gamma[t] + te + rng.normal(0, 1, u.size)
    return pd.DataFrame({"unit": u, "period": t, "cohort": g, "rel_time": rel,
                         "D": on.astype(int), "true_te": te, "y": y})


# ------------------------------------------------------------- synthetic control
def synth_panel(
    n_controls: int = 30,
    n_periods: int = 40,
    treat_period: int = 30,
    effect: float = -3.0,
    noise: float = 0.4,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Interactive fixed-effects (factor) model: y_jt = a_j + l1_j*trend_t + l2_j*season_t + e.

    The treated market loads heavily on the trend factor, so it does NOT share the
    average control trend (DiD fails) but IS a convex combination of a few controls
    (synthetic control works). Returns a wide frame (period x unit) and true weights.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_periods)
    f1, f2 = 10 * t / n_periods, 3 * np.sin(2 * np.pi * t / 12)
    a = rng.normal(50, 3, n_controls)
    l1, l2 = rng.uniform(0, 2, n_controls), rng.uniform(0, 2, n_controls)
    Yc = a + np.outer(f1, l1) + np.outer(f2, l2) + rng.normal(0, noise, (n_periods, n_controls))
    donors = np.argsort(l1)[-3:]
    w_true = np.array([0.2, 0.3, 0.5])
    a_t, l1_t, l2_t = a[donors] @ w_true, l1[donors] @ w_true, l2[donors] @ w_true
    yt = a_t + l1_t * f1 + l2_t * f2 + rng.normal(0, noise, n_periods)
    yt = yt + np.where(t >= treat_period, effect, 0.0)
    cols = [f"market_{j:02d}" for j in range(n_controls)]
    wide = pd.DataFrame(Yc, index=pd.Index(t, name="period"), columns=cols)
    wide.insert(0, "treated_market", yt)
    weights = pd.Series(0.0, index=cols)
    weights.iloc[donors] = w_true
    return wide, weights


# ----------------------------------------------------------- finance event study
def stock_returns(
    n_firms: int = 60,
    n_days: int = 500,
    event_jump: float = 0.02,
    drift: float = 0.002,
    drift_days: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Daily returns from a market model plus an announcement effect.

    True abnormal return: +event_jump on day 0 and +drift on days 1..drift_days,
    so the true CAR[0, +drift_days] = event_jump + drift * drift_days.
    """
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0004, 0.01, n_days)
    alpha = rng.normal(0, 0.0003, n_firms)
    beta = rng.uniform(0.6, 1.5, n_firms)
    sd = rng.uniform(0.01, 0.02, n_firms)
    R = alpha + np.outer(mkt, beta) + rng.normal(0, 1, (n_days, n_firms)) * sd
    event_day = rng.integers(300, n_days - 30, n_firms)
    for j, d0 in enumerate(event_day):
        R[d0, j] += event_jump
        R[d0 + 1: d0 + 1 + drift_days, j] += drift
    firms = [f"FIRM{j:02d}" for j in range(n_firms)]
    idx = pd.Index(np.arange(n_days), name="day")
    return (pd.DataFrame(R, index=idx, columns=firms),
            pd.Series(mkt, index=idx, name="market"),
            pd.Series(event_day, index=firms, name="event_day"))


# -------------------------------------------------------------------------- RDD
def rdd_data(n: int = 5000, cutoff: float = 0.0, effect: float = 0.5,
             manipulation: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Sharp RD: units with score >= cutoff are treated.

    `manipulation` in [0, 1) moves that share of units just below the cutoff to just
    above it (bunching), which the density test should flag.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    if manipulation > 0:
        near = (x < cutoff) & (x > cutoff - 0.1)
        flip = near & (rng.random(n) < manipulation)
        x[flip] = cutoff + rng.uniform(0, 0.05, flip.sum())
    z = 0.5 * x + rng.normal(0, 1, n)  # pre-treatment covariate: smooth through cutoff
    d = (x >= cutoff).astype(int)
    xc = x - cutoff
    # curvature differs by side (treatment also bends the slope), so wide bandwidths bias local-linear fits
    y = (1 + 1.2 * xc - 0.8 * xc**2 + 0.6 * xc**3 + 1.5 * d * xc**2
         + 0.3 * z + effect * d + rng.normal(0, 0.5, n))
    return pd.DataFrame({"score": x, "z": z, "treated": d, "y": y})


# ----------------------------------------------------------------- observational
def observational(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Confounded treatment with heterogeneous effect tau(x) = 1 + 0.5*x1.

    True ATE = 1.0 in the population; true ATT = mean of `true_cate` among treated.
    """
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.normal(size=(3, n))
    x4 = rng.binomial(1, 0.4, n)
    logit = -0.3 + 0.9 * x1 - 0.6 * x2 + 0.4 * x3 + 0.5 * x4
    t = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    cate = 1 + 0.5 * x1
    y = 2 + 1.5 * x1 + 1.0 * x2 + 0.5 * x3**2 + 0.8 * x4 + cate * t + rng.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "treated": t,
                         "y": y, "true_cate": cate})


# ---------------------------------------------------------- geo / switchback
def geo_markets(n_geos: int = 40, n_days: int = 120, seed: int = 0) -> pd.DataFrame:
    """Daily sales per geo under control (potential outcome Y(0)).
    Apply a treatment with `apply_geo_lift`."""
    rng = np.random.default_rng(seed)
    size = rng.lognormal(7, 0.8, n_geos)
    d = np.arange(n_days)
    common = (1 + 0.12 * np.sin(2 * np.pi * d / 7)) * (1 + 0.002 * d) * np.exp(
        np.cumsum(rng.normal(0, 0.01, n_days)))
    geo_trend = rng.normal(0, 0.0004, n_geos)
    G, D = np.meshgrid(np.arange(n_geos), d, indexing="ij")
    sales = size[G] * common[D] * (1 + geo_trend[G] * D) * np.exp(rng.normal(0, 0.04, G.shape))
    return pd.DataFrame({"geo": [f"geo_{g:02d}" for g in G.ravel()], "day": D.ravel(),
                         "sales": sales.ravel()})


def apply_geo_lift(df: pd.DataFrame, treated_geos, test_start: int, lift: float) -> pd.DataFrame:
    out = df.copy()
    mask = out["geo"].isin(set(treated_geos)) & (out["day"] >= test_start)
    out.loc[mask, "sales"] *= 1 + lift
    return out


def switchback_marketplace(
    assignment: np.ndarray,
    effect: float = 8.0,
    persistence: float = 0.6,
    seed: int = 0,
) -> pd.DataFrame:
    """Hourly marketplace metric under a switchback assignment.

    The treatment works through slow-adjusting supply: exposure e_t =
    persistence * e_{t-1} + (1 - persistence) * a_t. Always-on treatment has
    effect `effect`; frequent switching never lets exposure reach 1 (carryover).
    """
    rng = np.random.default_rng(seed)
    n = len(assignment)
    hour = np.arange(n) % 24
    day = np.arange(n) // 24
    base = 100 + 40 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 5, day.max() + 1)[day]
    e = np.zeros(n)
    for i in range(n):
        prev = e[i - 1] if i else 0.0
        e[i] = persistence * prev + (1 - persistence) * assignment[i]
    y = base + effect * e + rng.normal(0, 6, n)
    return pd.DataFrame({"t": np.arange(n), "day": day, "hour": hour,
                         "treated": assignment.astype(int), "exposure": e, "y": y})
