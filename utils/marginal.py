"""The stationary marginal: 1-Wasserstein and the extreme quantiles.

In one dimension W_1 between empirical laws is the mean absolute gap between
their quantile functions, which is what `w1_marginal` computes.  It is reported
in units of the target's standard deviation so that runs at different S, whose
marginals have different spreads, are comparable.

For the tails, report the difference in sd units and not the RATIO: the ratio
of two quantiles blows up wherever the target quantile crosses zero, and for a
centred process that is exactly the median.  A table printing both makes the
point obvious; `quantile_table` returns both and marks the ratio as unreliable
where the denominator is small.
"""
import numpy as np

__all__ = ["w1_marginal", "quantile_table", "marginal_report",
           "DEFAULT_LEVELS"]

DEFAULT_LEVELS = (0.001, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 0.999)


def w1_marginal(x, y):
    """1-Wasserstein between two empirical marginals on the line."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    qs = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(x, qs) - np.quantile(y, qs))))


def quantile_table(tru, gen, levels=DEFAULT_LEVELS, sd=None):
    """Per-level target and generated quantiles, the gap in sd units, and the
    ratio with a reliability flag."""
    tru = np.asarray(tru, dtype=float)
    gen = np.asarray(gen, dtype=float)
    sd = float(tru.std()) if sd is None else float(sd)
    qt = np.quantile(tru, levels)
    qg = np.quantile(gen, levels)
    rows = []
    for lv, a, b in zip(levels, qt, qg):
        reliable = abs(a) > 0.1 * sd
        rows.append({"level": float(lv), "target": float(a), "generated": float(b),
                     "err_over_sd": float((b - a) / sd),
                     "ratio": float(b / a) if a != 0 else float("nan"),
                     "ratio_reliable": bool(reliable)})
    return rows


def marginal_report(tru, gen, baseline=None, levels=DEFAULT_LEVELS):
    sd = float(np.asarray(tru).std())
    out = {
        "sd_tru": sd,
        "sd_gen": float(np.asarray(gen).std()),
        "sd_ratio": float(np.asarray(gen).std() / sd),
        "mean_tru": float(np.mean(tru)),
        "mean_gen": float(np.mean(gen)),
        "w1": w1_marginal(tru, gen),
        "w1_over_sd": w1_marginal(tru, gen) / sd,
        "quantiles": quantile_table(tru, gen, levels, sd=sd),
    }
    tail = [r["err_over_sd"] for r in out["quantiles"]
            if r["level"] in (0.001, 0.01, 0.99, 0.999)]
    out["max_abs_tail_err_over_sd"] = float(np.max(np.abs(tail)))
    if baseline is not None:
        out["w1_base"] = w1_marginal(tru, baseline)
        out["w1_over_sd_base"] = out["w1_base"] / sd
        out["quantiles_base"] = quantile_table(tru, baseline, levels, sd=sd)
    return out
