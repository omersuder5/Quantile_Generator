"""Is the repair lemma doing anything on this fit?

The generator proposition needs qhat(u;.) to be a bona fide conditional
quantile function: nondecreasing in u, and mapping a forward-invariant set into
itself.  Neither is imposed during training, because the repair lemma supplies
the increasing rearrangement R, a Lipschitz regularisation T, and the
projection Pi free of charge, each nonexpansive in the uniform norm so that
theta cannot grow.  The question these diagnostics answer is whether any of
them is actually needed.

Across the 21-run study, eleven of twenty-one fits needed none of the three.
The four that did are informative, and they are not the ones one would guess:

  Pi   the oscillatory target: sup|qhat| over the box reached 1.019 against
       M = 1, so the fitted map was not X-valued.
  R    on the DATA SUPPORT only once, at the chaotic end of the noise-budget
       sweep (crossing fraction 1.2e-4).  Three further fits were non-monotone
       somewhere in the box but nowhere the recursion goes.
  T    at m = 32, where sum_i Lip_i reached 1.076 on the box against a target
       S_m of 0.657, so the fit overshot a contractive target.

**The two domains are different and a table mixing them looks contradictory.**
`crossing_report` works on the data support; `du_min` and `sup_q_box` work on
the whole box.  A fit can be monotone where the process lives and not monotone
in a corner, and one was.

One thing T cannot repair: when the TARGET has S >= 1, a contractive
regularisation would destroy the fit rather than fix it.  There the hypothesis
is failing, not the estimator.
"""
import numpy as np
import torch

__all__ = ["crossing_report", "du_min", "sup_q_box", "invariant_radius",
           "repair_report"]


def crossing_report(net, Z, n_u=200, max_windows=2000):
    """Fraction of adjacent level pairs on which qhat DECREASES, on the data.

    Returns (fraction, worst_increment).  A strictly positive worst increment
    with a zero fraction is a monotone fit; a negative worst increment means
    genuine crossings and the rearrangement R is not the identity.
    """
    ug = np.linspace(1e-4, 1 - 1e-4, n_u)
    Zs = Z[::max(1, len(Z) // max_windows)]
    Qm = np.stack([net.q_np(np.full(len(Zs), u), Zs) for u in ug])
    d = np.diff(Qm, axis=0)
    return float((d < 0).mean()), float(d.min())


def du_min(net, M=1.0, n=3000, n_u=120, seed=0):
    """min over the BOX of d qhat / d u.  Negative means non-monotone there."""
    rng = np.random.default_rng(seed)
    z = torch.tensor(rng.uniform(-M, M, (n, net.m)), dtype=torch.float32)
    lo = np.inf
    for uu in np.linspace(1e-4, 1 - 1e-4, n_u):
        u = torch.full((n,), float(uu), requires_grad=True)
        g, = torch.autograd.grad(net(u, z).sum(), u)
        lo = min(lo, float(g.min()))
    return lo


def sup_q_box(net, M=1.0, n=5000, seed=0):
    """sup |qhat| over the box, on a level grid weighted toward the extremes.

    Above M the fitted map is not X-valued and Pi is not the identity.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(-M, M, (n, net.m))
    grid = np.concatenate([np.geomspace(1e-3, 0.5, 10),
                           1 - np.geomspace(1e-3, 0.5, 10)])
    return max(float(np.max(np.abs(net.q_np(np.full(n, uu), z)))) for uu in grid)


def invariant_radius(net, M=1.0, grid=None, n=3000, seed=0):
    """Smallest R on a grid with qhat(u; [-R,R]^m) contained in [-R,R].

    This is the compact forward-invariant K the hypotheses should be stated
    over: weaker than X^m, still checkable, and the reason theta over the box
    overstates the error that the recursion actually commits.  Returns
    (R, sup|qhat| there), or (nan, nan) if no R on the grid works.
    """
    grid = np.linspace(0.6 * M, 2.5 * M, 39) if grid is None else grid
    rng = np.random.default_rng(seed)
    lv = np.concatenate([np.geomspace(1e-3, 0.5, 8), 1 - np.geomspace(1e-3, 0.5, 8)])
    for R in grid:
        z = rng.uniform(-R, R, (n, net.m))
        mx = max(float(np.max(np.abs(net.q_np(np.full(n, uu), z)))) for uu in lv)
        if mx <= R:
            return float(R), float(mx)
    return float("nan"), float("nan")


def repair_report(net, Z, M=1.0, lip_box=None, S_m=None, seed=0):
    """All four diagnostics plus a verdict naming which operators are needed."""
    frac, worst = crossing_report(net, Z)
    dmin = du_min(net, M=M, seed=seed)
    supq = sup_q_box(net, M=M, seed=seed)
    R, mR = invariant_radius(net, M=M, seed=seed)
    needed = []
    if frac > 0:
        needed.append("R (on the data support)")
    elif dmin < 0:
        needed.append("R (in the box only)")
    if supq > M + 1e-9:
        needed.append("Pi")
    if lip_box is not None and lip_box >= 1.0:
        if S_m is not None and S_m >= 1.0:
            needed.append("none: the TARGET has S_m >= 1, so a contractive "
                          "repair would destroy the fit")
        else:
            needed.append("T")
    return {"cross_frac": frac, "cross_worst": worst, "du_min": dmin,
            "sup_q_box": supq, "box_invariant": bool(supq <= M + 1e-9),
            "inv_R": R, "inv_sup": mR,
            "repairs_needed": needed or ["none"]}
