"""The top Lyapunov exponent of the learned recursion.

    lambda = E log | d qhat / d z |   along a stationary path.

At m = 1 this is exactly the top Lyapunov exponent of the random dynamical
system.  At m > 1 the companion Jacobian has first row (d_1 qhat, ..., d_m qhat)
and shifts below, so its row-sum norm is sum_i |d_i qhat| once that exceeds 1;
by Furstenberg and Kesten the average log of a submultiplicative norm is an
UPPER bound for the top exponent, and that is what is reported.  The function
says which case it is in.

Why this matters more than S.  S is a SUPREMUM of |d qhat / d z| over the state
space; lambda is an AVERAGE of its log over where the process actually sits.
The gap between them is the gap between uniform contraction and contraction on
average, and the noise-budget sweep makes it concrete: on the noisy logistic
target with ell_1 = 4(1-rho), the pathwise and adapted statements survived at
S = 2.0 and S = 2.8, with the spread across initial states still 9e-11 and
1.5e-2 respectively, and only died once lambda crossed zero near rho = 0.28.
S < 1 is sufficient; lambda < 0 is the operative boundary.

For the noisy logistic target the exponent also decomposes analytically,

    lambda = log 4(1-rho) + E log |X_t| / M,

the second term strictly negative because |X| <= M, which is exactly why
S = 4(1-rho) can exceed 1 while lambda stays negative.
"""
import numpy as np
import torch

from generators.simulate_paths import make_windows

__all__ = ["lyapunov", "lyapunov_decomposition"]


def lyapunov(net, path, n_u=16, n_max=20000):
    """E log sum_i |d qhat / d z_i| along a stationary path of the LEARNED map.

    Returns (value, exact) with exact=True only when m = 1, where the quantity
    is the top exponent itself rather than an upper bound for it.

    The path must be a stationary path of the learned recursion, not of the
    target: the exponent is a property of the fitted map along its own
    invariant law.
    """
    m = int(net.m)
    p = np.asarray(path, dtype=float)[:n_max + m]
    if len(p) <= m:
        raise ValueError("path too short for the lag dimension")
    Zw, _ = make_windows(p, m)
    z = torch.tensor(Zw, dtype=torch.float32, requires_grad=True)
    tot = 0.0
    for uu in np.linspace(0.02, 0.98, n_u):
        g, = torch.autograd.grad(net(torch.full((len(z),), float(uu)), z).sum(),
                                 z, retain_graph=True)
        s = np.abs(g.detach().numpy()).sum(1)
        tot += float(np.mean(np.log(s + 1e-300)))
    return tot / n_u, (m == 1)


def lyapunov_decomposition(target, path):
    """The TARGET's exponent, decomposed: log 4(1-rho) + E log|X|/M.

    Noisy logistic only; returns None for any other target.  Note what this is
    and is not: it evaluates the target's analytic slope along the supplied
    path, so it is the target's exponent, whereas `lyapunov` measures the
    LEARNED map's by autograd.  The two agree once the fit has converged and a
    gap between them means it has not, which makes the pair a useful
    convergence check as well as the cleanest statement of why S and lambda
    separate: the first term is the log of a supremum, the second is strictly
    negative because |X| <= M.
    """
    if not hasattr(target, "rho"):
        return None
    x = np.asarray(path, dtype=float)
    sup_term = float(np.log(4.0 * (1.0 - target.rho)))
    avg_term = float(np.mean(np.log(np.abs(x) / target.M + 1e-300)))
    return {"log_sup_slope": sup_term,
            "E_log_abs_X_over_M": avg_term,
            "lambda_analytic": sup_term + avg_term,
            "geometric_mean_abs_X": float(np.exp(avg_term) * target.M)}
