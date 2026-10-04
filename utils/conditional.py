"""The conditional law given one lag: the diagnostic a path-law metric misses.

Everything else in `utils` compares unconditional or second-order features.
These two compare the conditional law of X_t given X_{t-1} directly, binned:

  `conditional_sd_profile`   the conditional standard deviation per bin, which
        sees a state-dependent SCALE (heteroskedastic, ARCH, oscillatory) and
        is pure sampling noise on a homoskedastic target.  Read the range ratio
        only on targets whose scale actually depends on the state.

  `conditional_w1_profile`   a per-bin 1-Wasserstein between the two
        conditional laws, aggregated by occupancy.  A crude one-step nested
        distance, and in the study the most stable statistic of all: 2 to 4 per
        cent relative standard deviation across seeds, against 11 to 32 per
        cent for theta and everything downstream of it.  It is also the only
        diagnostic in the suite that sees a target whose conditional SKEWNESS
        varies while its mean and variance do not, where it separated the
        generator from a matched Gaussian AR by a factor of 4.7.

If one number has to go in a table, it is the aggregated conditional W_1.
"""
import numpy as np

from .marginal import w1_marginal

__all__ = ["default_bins", "conditional_sd_profile", "conditional_w1_profile",
           "conditional_report"]


def default_bins(path, n_bins=18, lo_q=0.005, hi_q=0.995):
    """Equal-width bins spanning the bulk of the target's support."""
    path = np.asarray(path, dtype=float)
    return np.linspace(np.quantile(path, lo_q), np.quantile(path, hi_q), n_bins + 1)


def conditional_sd_profile(path, bins, min_count=30, stat="std"):
    """Per-bin conditional standard deviation (or mean) of X_t given X_{t-1}."""
    path = np.asarray(path, dtype=float)
    z, x = path[:-1], path[1:]
    idx = np.digitize(z, bins) - 1
    nb = len(bins) - 1
    out = np.full(nb, np.nan)
    cnt = np.zeros(nb, dtype=int)
    for b in range(nb):
        mask = idx == b
        cnt[b] = int(mask.sum())
        if cnt[b] >= min_count:
            out[b] = x[mask].std() if stat == "std" else x[mask].mean()
    return out, cnt


def conditional_w1_profile(path_a, path_b, bins, min_count=40):
    """Per-bin W_1 between the conditional laws, and the occupancy-weighted
    aggregate.  Returns (per_bin, weights, aggregate)."""
    a = np.asarray(path_a, dtype=float)
    b = np.asarray(path_b, dtype=float)
    za, xa = a[:-1], a[1:]
    zb, xb = b[:-1], b[1:]
    ia = np.digitize(za, bins) - 1
    ib = np.digitize(zb, bins) - 1
    nb = len(bins) - 1
    w = np.full(nb, np.nan)
    wt = np.zeros(nb)
    for j in range(nb):
        A, B = xa[ia == j], xb[ib == j]
        if len(A) >= min_count and len(B) >= min_count:
            w[j] = w1_marginal(A, B)
            wt[j] = len(A)
    ok = ~np.isnan(w)
    agg = float(np.average(w[ok], weights=wt[ok])) if ok.any() else float("nan")
    return w, wt, agg


def conditional_report(tru, gen, baseline=None, n_bins=18,
                       scale_is_state_dependent=None):
    """Both profiles for the generator, and the baseline for comparison."""
    bins = default_bins(tru, n_bins=n_bins)
    mid = 0.5 * (bins[:-1] + bins[1:])
    sd_t, cnt = conditional_sd_profile(tru, bins)
    sd_g, _ = conditional_sd_profile(gen, bins)
    w_g, _, agg_g = conditional_w1_profile(tru, gen, bins)

    def rng_(v):
        v = v[~np.isnan(v)]
        return float(v.max() - v.min()) if len(v) else float("nan")

    out = {
        "bins": bins.tolist(), "mid": mid.tolist(), "count": cnt.tolist(),
        "cond_sd_tru": sd_t.tolist(), "cond_sd_gen": sd_g.tolist(),
        "cond_sd_range_tru": rng_(sd_t), "cond_sd_range_gen": rng_(sd_g),
        "cond_w1_profile": w_g.tolist(), "cond_w1": agg_g,
        "scale_is_state_dependent": scale_is_state_dependent,
    }
    rt = out["cond_sd_range_tru"]
    out["cond_sd_range_ratio"] = (float(out["cond_sd_range_gen"] / rt)
                                  if rt and rt == rt and rt > 1e-9 else None)
    if baseline is not None:
        sd_b, _ = conditional_sd_profile(baseline, bins)
        w_b, _, agg_b = conditional_w1_profile(tru, baseline, bins)
        out["cond_sd_base"] = sd_b.tolist()
        out["cond_sd_range_base"] = rng_(sd_b)
        out["cond_w1_base"] = agg_b
        out["cond_w1_profile_base"] = w_b.tolist()
        out["cond_sd_range_ratio_base"] = (float(rng_(sd_b) / rt)
                                           if rt and rt == rt and rt > 1e-9 else None)
    if scale_is_state_dependent is False:
        out["note"] = ("the target is homoskedastic, so cond_sd_range is "
                       "sampling noise and its ratio carries no information")
    return out
