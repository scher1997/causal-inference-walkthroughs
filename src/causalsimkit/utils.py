"""Generic helpers: cluster bootstrap."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def cluster_bootstrap(
    df: pd.DataFrame,
    cluster: str,
    statistic: Callable[[pd.DataFrame], float | pd.Series],
    n_boot: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Resample whole clusters with replacement and re-run `statistic`.

    Duplicated clusters get fresh ids so panel estimators treat them as distinct units.
    Returns one row per bootstrap draw.
    """
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster, sort=False)}
    keys = np.array(list(groups))
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        boot = pd.concat(
            [groups[k].assign(**{cluster: i}) for i, k in enumerate(pick)], ignore_index=True
        )
        stat = statistic(boot)
        draws.append(stat if isinstance(stat, pd.Series) else pd.Series({"stat": stat}))
    return pd.DataFrame(draws).reset_index(drop=True)
