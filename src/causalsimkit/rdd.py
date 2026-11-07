"""Sharp regression discontinuity: local linear estimation + diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .ols import ols

KERNELS = {
    "triangular": lambda u: np.clip(1 - np.abs(u), 0, None),
    "uniform": lambda u: (np.abs(u) <= 1).astype(float),
    "epanechnikov": lambda u: np.clip(0.75 * (1 - u**2), 0, None),
}


@dataclass
class RDResult:
    estimate: float
    se: float
    bandwidth: float
    n_left: int
    n_right: int

    @property
    def ci(self) -> tuple[float, float]:
        return (self.estimate - 1.96 * self.se, self.estimate + 1.96 * self.se)

    @property
    def p(self) -> float:
        return float(2 * stats.norm.sf(abs(self.estimate / self.se)))


def local_poly_rd(x, y, cutoff: float = 0.0, bandwidth: float = 0.3,
                  kernel: str = "triangular", order: int = 1) -> RDResult:
    """Kernel-weighted local polynomial on each side; the jump at the cutoff is the
    coefficient on the treatment indicator. HC1 standard errors (conventional, not
    bias-corrected: see rdrobust for robust bias-corrected CIs)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - cutoff
    w = KERNELS[kernel](xc / bandwidth)
    keep = w > 0
    d = (xc >= 0).astype(float)
    X = {"D": d}
    for p in range(1, order + 1):
        X[f"x{p}"] = xc**p
        X[f"Dx{p}"] = d * xc**p
    X = pd.DataFrame(X)[keep]
    res = ols(y[keep], X, weights=w[keep])
    return RDResult(float(res.params["D"]), float(res.bse["D"]), bandwidth,
                    int((keep & (d == 0)).sum()), int((keep & (d == 1)).sum()))


def binned_means(x, y, cutoff: float = 0.0, n_bins: int = 20) -> pd.DataFrame:
    """Equal-width bins on each side of the cutoff (bins never straddle it)."""
    df = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)})
    out = []
    for side, part in (("left", df[df.x < cutoff]), ("right", df[df.x >= cutoff])):
        edges = np.linspace(part.x.min(), part.x.max(), n_bins + 1)
        b = pd.cut(part.x, edges, include_lowest=True)
        agg = part.groupby(b, observed=True).agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size"))
        out.append(agg.assign(side=side))
    return pd.concat(out, ignore_index=True)


def density_test(x, cutoff: float = 0.0, bandwidth: float = 0.3, bin_width: float = 0.02) -> dict:
    """Simplified McCrary-style test: fit linear regressions of histogram counts on each
    side and compare the implied densities at the cutoff (log ratio, delta-method SE).
    A significant jump suggests units are manipulating the running variable."""
    x = np.asarray(x, float) - cutoff
    est = {}
    for side, sign in (("left", -1), ("right", 1)):
        edges = np.arange(0, bandwidth + 1e-12, bin_width) * sign
        edges = np.sort(edges)
        counts, e = np.histogram(x, bins=edges)
        mids = (e[:-1] + e[1:]) / 2
        r = ols(counts.astype(float), pd.DataFrame({"mid": mids}), cov_type="HC1")
        est[side] = (r.params["const"], r.bse["const"])
    (fl, sl), (fr, sr) = est["left"], est["right"]
    log_ratio = np.log(fr / fl)
    se = np.sqrt((sr / fr) ** 2 + (sl / fl) ** 2)
    z = log_ratio / se
    return {"density_left": fl, "density_right": fr, "log_ratio": float(log_ratio),
            "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}
