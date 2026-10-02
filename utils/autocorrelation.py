"""Autocorrelation of the process and of its squares.

Two series, and the second is the one that matters.

The ACF of X is reproduced by any correctly specified linear model, so a
matched Gaussian AR of the same order gets it right by construction and
agreement there is weak evidence.  The ACF of (X - mean)^2 is volatility
clustering, and a model whose conditional scale does not depend on the state
cannot produce it: it returns a value near zero, often slightly negative, which
is the small-sample behaviour of an estimator of an autocorrelation the model
has set to zero.  That negative value is not a puzzle and not a finite-sample
artefact to be explained away; it is the signature of the missing mechanism.

Report the pair (target, generated) rather than their ratio wherever the
target's own value is near zero, since the ratio of two small numbers is noise.
`acf_sq_ratio` therefore returns None below a floor.
"""
import numpy as np

__all__ = ["acf", "acf_squares", "acf_report", "acf_sq_ratio"]


def acf(x, nlags=20):
    """Sample autocorrelation, lags 0..nlags."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.zeros(nlags + 1)
    return np.array([1.0] + [float(np.dot(x[:-l], x[l:]) / denom)
                             for l in range(1, nlags + 1)])


def acf_squares(x, nlags=20):
    """Autocorrelation of the centred squares: the clustering diagnostic."""
    x = np.asarray(x, dtype=float)
    return acf((x - x.mean()) ** 2, nlags)


def acf_sq_ratio(target_acf, gen_acf, lag=1, floor=0.02):
    """Generated over target at one lag, or None when the target is too small
    for the ratio to carry information."""
    t = float(np.asarray(target_acf)[lag])
    if abs(t) < floor:
        return None
    return float(np.asarray(gen_acf)[lag] / t)


def acf_report(tru, gen, baseline=None, nlags=20):
    """ACF and ACF-of-squares for target, generator and optional baseline."""
    out = {
        "nlags": int(nlags),
        "acf_tru": acf(tru, nlags).tolist(),
        "acf_gen": acf(gen, nlags).tolist(),
        "acfsq_tru": acf_squares(tru, nlags).tolist(),
        "acfsq_gen": acf_squares(gen, nlags).tolist(),
    }
    if baseline is not None:
        out["acf_base"] = acf(baseline, nlags).tolist()
        out["acfsq_base"] = acf_squares(baseline, nlags).tolist()
    at, ag = np.array(out["acfsq_tru"]), np.array(out["acfsq_gen"])
    out["acfsq_ratio_lag1"] = acf_sq_ratio(at, ag, 1)
    out["acf_abs_err_lag1"] = float(abs(out["acf_tru"][1] - out["acf_gen"][1]))
    out["acf_max_abs_err"] = float(np.max(np.abs(np.array(out["acf_tru"][1:])
                                                 - np.array(out["acf_gen"][1:]))))
    if baseline is not None:
        ab = np.array(out["acfsq_base"])
        out["acfsq_ratio_lag1_base"] = acf_sq_ratio(at, ab, 1)
    return out
