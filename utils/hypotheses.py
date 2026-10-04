"""The standing hypotheses, measured.

Two quantities carry the theorem:

  S_m = sum_{i<=m} ell_i   the target's contraction constant at arity m.  A
        property of the TARGET, not of the fit; it is what the amplification
        1/(1 - S_m) is built from.

  theta = sup over (u, z) of |qhat(u;z) - q_m(u;z)|   the uniform quantile
        error of the fit against the m-anchored truncation of the target.

Three honest caveats, all of which the study quantified and none of which the
code can remove:

1.  theta as stated is a supremum over U x X^m.  What is computed here is a
    maximum over a grid, 64 levels by 3000 windows by default.  A defect on a
    set of small measure is invisible to it.

2.  theta over the BOX is between 1.7 and 7.8 times theta over the data
    support, because the fit is never asked about corners the recursion does
    not visit.  The proposition only ever evaluates q and qhat at points the
    two processes occupy, so the right hypothesis is a supremum over a compact
    forward-invariant K containing the support, not over X^m.  Both are
    reported; `invariant_radius` in `utils.repair` supplies K.

3.  theta is a supremum over LEVELS too, and the extreme levels dominate it:
    restricting to u in [0.05, 0.95] shrinks it by up to a factor of three.
    That gap is the uniformity-in-u cost and no change of domain removes it.

`lipschitz_empirical` reports sum_i sup_z |d qhat / d z_i|, which upper-bounds
the operator norm sup_z sum_i |d qhat / d z_i| that the uniqueness argument
actually needs.  The over-count grows with m: at m = 32 it was 1.08 on the box
against 0.81 on the data support.
"""
import numpy as np
import torch

from backend import to_np

__all__ = ["verify_assumptions", "lipschitz_empirical", "theta_hat",
           "theta_profile", "theta_trimmed", "hypothesis_report"]


def verify_assumptions(target, m=None, n_pairs=1500, n_u=101, delta=1e-3,
                       seed=0):
    """Check the analytic moduli against finite differences, and the range.

    Returns the analytic and numeric ell side by side.  A gap of order 1e-3 on
    ARCH and the logistic target is expected and not a bug: their moduli are
    attained at a corner of the box, which random sampling approaches but does
    not hit.
    """
    rng = np.random.default_rng(seed)
    M = target.M
    k = int(m) if m is not None else min(
        target.k_true if target.k_true is not None else 8, 8)
    ugrid = np.linspace(0.0, 1.0, n_u)
    ell_hat = np.zeros(k)
    for i in range(k):
        z = rng.uniform(-M, M, size=(n_pairs, k))
        zp = z.copy()
        step = rng.choice([-1.0, 1.0], size=n_pairs) * delta
        zp[:, i] = np.clip(z[:, i] + step, -M, M)
        d = np.abs(zp[:, i] - z[:, i])
        keep = d > 1e-12
        z, zp, d = z[keep], zp[keep], d[keep]
        worst = 0.0
        for u in ugrid:
            uu = np.full(len(z), u)
            worst = max(worst, float(np.max(np.abs(target.q(uu, zp)
                                                   - target.q(uu, z)) / d)))
        ell_hat[i] = worst
    z = rng.uniform(-M, M, size=(20000, k))
    u = rng.random(20000)
    vals = target.q(u, z)
    return {
        "m": k,
        "ell_analytic": target.moduli(k).tolist(),
        "ell_numeric": ell_hat.tolist(),
        "S_m_analytic": target.S_m(k),
        "S_m_numeric": float(ell_hat.sum()),
        "S_full": target.S,
        "L_tail": target.L_tail(k),
        "range_min": float(vals.min()),
        "range_max": float(vals.max()),
        "in_range": bool(vals.min() >= -M - 1e-9 and vals.max() <= M + 1e-9),
    }


def lipschitz_empirical(net, z, n_u=32):
    """Per-lag sup_z |d qhat / d z_i| over the supplied windows.

    Sum it for the quantity to compare against S_m.  The level grid spans
    [1e-3, 1-1e-3], so the extreme-level maxima are seen.  Runs on whatever
    device the network is on and returns numpy float64.
    """
    dev, dt = net.device, net.dtype
    z = torch.as_tensor(np.asarray(z), dtype=dt, device=dev).requires_grad_(True)
    per = torch.zeros(net.m, dtype=dt, device=dev)
    for uu in np.linspace(1e-3, 1 - 1e-3, n_u):
        u = torch.full((len(z),), float(uu), dtype=dt, device=dev)
        gr, = torch.autograd.grad(net(u, z).sum(), z)
        per = torch.maximum(per, gr.abs().max(0).values)
    return to_np(per)


def theta_hat(net, target, region="data", Z_data=None, M=1.0, m=None,
              n_z=3000, n_u=64, seed=0):
    """max over a level grid and a window sample of |qhat - q_m|.

    region="data"        windows drawn from the training path
    region="box"         windows uniform on [-M, M]^m
    region=(R,)          windows uniform on [-R, R]^m, for a forward-invariant R
    """
    rng = np.random.default_rng(seed)
    m = int(m if m is not None else net.m)
    if region == "data":
        if Z_data is None:
            raise ValueError("region='data' needs Z_data")
        z = Z_data[rng.integers(0, len(Z_data), size=n_z)]
    elif region == "box":
        z = rng.uniform(-M, M, size=(n_z, m))
    elif isinstance(region, (tuple, list)) and len(region) == 1:
        R = float(region[0])
        z = rng.uniform(-R, R, size=(n_z, m))
    else:
        raise ValueError(f"unknown region {region!r}")
    worst = 0.0
    for uu in np.linspace(1e-3, 1 - 1e-3, n_u):
        uv = np.full(len(z), uu)
        worst = max(worst, float(np.max(np.abs(net.q_np(uv, z) - target.q(uv, z)))))
    return worst


def theta_profile(net, target, Z_data=None, region="data", M=1.0, m=None,
                  n_z=2000, u_grid=None, seed=0):
    """|qhat - q_m| at each level: where in u the approximation is hard."""
    rng = np.random.default_rng(seed)
    m = int(m if m is not None else net.m)
    if region == "data":
        z = Z_data[rng.integers(0, len(Z_data), size=n_z)]
    else:
        z = rng.uniform(-M, M, size=(n_z, m))
    if u_grid is None:
        u_grid = np.concatenate([np.linspace(1e-3, 0.05, 12),
                                 np.linspace(0.05, 0.95, 37),
                                 np.linspace(0.95, 1 - 1e-3, 12)])
    err = np.array([float(np.max(np.abs(net.q_np(np.full(len(z), uu), z)
                                        - target.q(np.full(len(z), uu), z))))
                    for uu in u_grid])
    return u_grid, err


def theta_trimmed(u_grid, err, eps):
    """sup |qhat - q| over levels in [eps, 1-eps]."""
    mask = (u_grid >= eps) & (u_grid <= 1 - eps)
    return float(err[mask].max()) if mask.any() else float("nan")


def hypothesis_report(net, target, Z_data, m, M=1.0, inv_R=None, seed=0):
    """Everything in this module, as one dict."""
    rng = np.random.default_rng(seed)
    z_box = rng.uniform(-M, M, size=(3000, m))
    z_dat = Z_data[rng.integers(0, len(Z_data), size=3000)]
    lip_box = lipschitz_empirical(net, z_box)
    lip_dat = lipschitz_empirical(net, z_dat)
    th_d = theta_hat(net, target, "data", Z_data=Z_data, m=m, seed=seed)
    th_b = theta_hat(net, target, "box", M=M, m=m, seed=seed)
    ug, eg = theta_profile(net, target, Z_data=Z_data, m=m, seed=seed)
    out = {
        "S_m": target.S_m(m), "L_tail": target.L_tail(m), "S_full": target.S,
        "ell": target.moduli(m).tolist(),
        "lip_box": float(lip_box.sum()), "lip_data": float(lip_dat.sum()),
        "lip_per_lag_box": lip_box.tolist(),
        "theta_data": th_d, "theta_box": th_b,
        "theta_trim": {str(e): theta_trimmed(ug, eg, e) for e in (0.0, 0.01, 0.05)},
        "theta_profile_u": ug.tolist(), "theta_profile_err": eg.tolist(),
        "bound": target.bound(th_d, m),
        "bound_box": target.bound(th_b, m),
    }
    if inv_R is not None and np.isfinite(inv_R):
        th_K = theta_hat(net, target, (inv_R,), M=M, m=m, seed=seed)
        out["theta_invariant"] = th_K
        out["bound_invariant"] = target.bound(th_K, m)
    return out
