"""Difference-in-differences and panel event-study estimators."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .ols import OLSResult, fe_dummies, ols


def did_2x2(df: pd.DataFrame, y: str, treated: str, post: str, cluster: str | None = None) -> OLSResult:
    """Canonical regression: y ~ treated + post + treated:post. Coef of interest: 'treated:post'."""
    X = pd.DataFrame({treated: df[treated], post: df[post],
                      f"{treated}:{post}": df[treated] * df[post]})
    return ols(df[y], X, cluster=df[cluster] if cluster else None)


def twfe_did(df: pd.DataFrame, y: str, unit: str, time: str, treat: str,
             covariates: Sequence[str] = ()) -> OLSResult:
    """Two-way fixed-effects DiD, clustered by unit. Coef of interest: `treat`.

    Fine for block adoption. With staggered adoption + heterogeneous effects it is a
    weighted average that can even put negative weight on some effects: use
    `group_time_att` instead.
    """
    X = pd.concat([df[[treat, *covariates]].astype(float), fe_dummies(df, [unit, time])], axis=1)
    return ols(df[y], X, cluster=df[unit])


def _rt_name(k: int) -> str:
    return f"rt{k:+d}"


def event_study(df: pd.DataFrame, y: str, unit: str, time: str, rel_time: str,
                window: tuple[int, int] = (-5, 5), ref: int = -1) -> tuple[pd.DataFrame, OLSResult]:
    """Dynamic TWFE with lead/lag dummies (endpoints binned), reference period `ref`.

    `rel_time` = time - adoption time, NaN for never-treated units.
    Returns (coefficient table incl. the zero reference row, fitted OLSResult).
    """
    lo, hi = window
    binned = df[rel_time].clip(lo, hi)
    D = pd.DataFrame({_rt_name(k): (binned == k).astype(float)
                      for k in range(lo, hi + 1) if k != ref}, index=df.index)
    X = pd.concat([D, fe_dummies(df, [unit, time])], axis=1)
    res = ols(df[y], X, cluster=df[unit])
    tab = res.summary(list(D.columns)).assign(rel_time=[int(c[2:]) for c in D.columns])
    ref_row = pd.DataFrame({"coef": [0.0], "se": [0.0], "ci_lower": [0.0], "ci_upper": [0.0],
                            "rel_time": [ref]}, index=[_rt_name(ref)])
    tab = pd.concat([tab, ref_row]).sort_values("rel_time")
    return tab, res


def lead_names(table: pd.DataFrame, ref: int = -1) -> list[str]:
    return [i for i, k in zip(table.index, table["rel_time"]) if k < 0 and k != ref]


def group_time_att(df: pd.DataFrame, y: str, unit: str, time: str, cohort: str,
                   never_value=0) -> pd.DataFrame:
    """Callaway & Sant'Anna (2021) ATT(g, t), unconditional, never-treated comparison.

    ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - E[Y_t - Y_{g-1} | never treated].
    Pre-period cells (t < g) are placebo estimates and should be ~0.
    """
    wide = df.pivot(index=unit, columns=time, values=y)
    coh = df.groupby(unit)[cohort].first()
    never = coh.index[coh == never_value]
    if len(never) == 0:
        raise ValueError("no never-treated units; use not-yet-treated controls instead")
    rows = []
    for g in sorted(c for c in coh.unique() if c != never_value):
        units_g, base = coh.index[coh == g], g - 1
        if base not in wide.columns:
            continue
        dc = (wide.loc[never].sub(wide.loc[never, base], axis=0)).mean()
        dt = (wide.loc[units_g].sub(wide.loc[units_g, base], axis=0)).mean()
        for t in wide.columns:
            if t == base:
                continue
            rows.append({"cohort": g, "time": t, "event_time": t - g,
                         "att": dt[t] - dc[t], "n_units": len(units_g)})
    return pd.DataFrame(rows)


def aggregate_event_time(gt: pd.DataFrame) -> pd.Series:
    """Cohort-size-weighted average of ATT(g,t) by event time."""
    return gt.groupby("event_time").apply(
        lambda d: np.average(d["att"], weights=d["n_units"]), include_groups=False)


def aggregate_overall(gt: pd.DataFrame) -> float:
    post = gt[gt["event_time"] >= 0]
    return float(np.average(post["att"], weights=post["n_units"]))
