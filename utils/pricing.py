"""European and Bermudan put values: the functional that needs the adapted
structure.

A European claim is a function of the marginal at one date, so two laws close
in W_1 price it to within Lip(g) W_1 and no conditional information enters.  A
Bermudan claim is an optimal stopping value, so its price depends on what is
PREDICTABLE from the observed path, which is exactly what a path-law distance
does not control.  The counterexample behind the adapted distance is priced:
two laws within epsilon in every W_p, p = infinity included, differ by one half
in the value of a two-period Bermudan.

So the early-exercise premium, Bermudan minus European, is the single number in
this package that a generator can only get right by getting the conditional
structure right.  It is reported for the target, the generator and the matched
Gaussian AR baseline.

Both are computed by backward induction on a binned empirical transition
kernel, so the two sides are priced by the same estimator and the comparison is
like for like.  Zero rates, undiscounted.
"""
import numpy as np

__all__ = ["empirical_kernel", "european_put", "bermudan_put", "pricing_report"]


def empirical_kernel(path, bins):
    """Row-stochastic transition matrix on bins of X_{t-1} -> bins of X_t,
    together with the occupancy of each bin."""
    path = np.asarray(path, dtype=float)
    nb = len(bins) - 1
    i = np.clip(np.digitize(path[:-1], bins) - 1, 0, nb - 1)
    j = np.clip(np.digitize(path[1:], bins) - 1, 0, nb - 1)
    P = np.zeros((nb, nb))
    np.add.at(P, (i, j), 1.0)
    occ = P.sum(1)
    P[occ > 0] /= occ[occ > 0, None]
    return P, occ


def _grid(path, n_bins, lo_q=0.002, hi_q=0.998):
    path = np.asarray(path, dtype=float)
    bins = np.linspace(np.quantile(path, lo_q), np.quantile(path, hi_q), n_bins + 1)
    return bins, 0.5 * (bins[:-1] + bins[1:])


def european_put(path, K, horizon=5, n_bins=60):
    """E[(K - X_{t+h})^+] under the binned kernel."""
    bins, mid = _grid(path, n_bins)
    P, occ = empirical_kernel(path, bins)
    V = np.maximum(K - mid, 0.0)
    for _ in range(horizon - 1):
        V = P @ V
    return float((occ / occ.sum()) @ V)


def bermudan_put(path, K, n_exercise=5, n_bins=60):
    """sup over stopping times adapted to the observed path, of E[(K - X_tau)^+],
    with n_exercise dates, by backward induction on the same kernel."""
    bins, mid = _grid(path, n_bins)
    P, occ = empirical_kernel(path, bins)
    imm = np.maximum(K - mid, 0.0)
    V = imm.copy()
    for _ in range(n_exercise - 1):
        V = np.maximum(imm, P @ V)
    return float((occ / occ.sum()) @ V)


def pricing_report(tru, gen, baseline=None, K=0.10, n_exercise=5, n_bins=60):
    """European, Bermudan and the early-exercise premium for each path."""
    def one(p):
        eu = european_put(p, K, horizon=n_exercise, n_bins=n_bins)
        be = bermudan_put(p, K, n_exercise=n_exercise, n_bins=n_bins)
        return {"european": eu, "bermudan": be, "premium": be - eu}

    out = {"K": float(K), "n_exercise": int(n_exercise), "n_bins": int(n_bins),
           "target": one(tru), "generated": one(gen)}
    if baseline is not None:
        out["baseline"] = one(baseline)
    t, g = out["target"]["premium"], out["generated"]["premium"]
    out["premium_rel_err"] = float((g - t) / t) if abs(t) > 1e-12 else None
    return out
