"""The pathwise bound, under the synchronous coupling.

Driving target and learned recursion with the SAME innovation stream gives a
coupling that is bicausal, because each process is adapted to the innovations
and closed in its own observable.  The realised

    sup_t | X_t - Xhat_t |

is therefore an UPPER bound for the adapted Wasserstein distance between the
two laws, for every p and every weight sequence, and it is the quantity the
comparison lemma bounds above by

    ( theta + 2 M L_(m+1) ) / ( 1 - S_m ).

Two things to keep in mind when reading the slack factor this module reports.

**It is an upper bound on an upper bound.**  Nothing here produces a lower
bound for the adapted distance, so a large slack says the bound is loose, not
that the generator is better than the bound allows.

**The slack decomposes, and only part of it is removable.**  Across the arity
sweep the realised error satisfied, by least squares on five points,

    sup_t |X_t - Xhat_t| (1 - S_m)  =  0.169 L_(m+1) - 0.008,

where the theorem charges 2M = 2 for that slope: 8.4 per cent of the charge is
realised.  Two causes.  (a) 2M is the diameter of the state space, and the
truncation error at time t is really sum_{i>m} ell_i |X_{t-i} - anchor|, of the
order of the stationary spread rather than the diameter; replacing 2M by the
diameter of a forward-invariant set recovers a factor of about 1.7 of the 12.
(b) The rest is that a supremum over t of a weighted sum of deviations is not
the sum of the suprema, which no change of domain removes: removing it means
replacing the worst-case telescoping by something distributional, which turns
the conclusion from surely into in expectation.
"""
import numpy as np

from generators.simulate_paths import shared_pair, initial_spread

__all__ = ["pathwise_report", "synchronisation_report"]


def pathwise_report(target, recursion, theta, m, n=20000, burn=2000, seed=7,
                    n_head=500):
    """Realised pathwise error against its bound, on a shared stream.

    `x_head` and `xhat_head` are the first `n_head` steps of the two coupled
    paths themselves, not only their difference.  Overlaying them is the figure
    that shows what the coupling IS, and a summary that reports only the error
    hides whether the two paths track each other or merely happen to have
    similar amplitude.
    """
    xt, xh, err = shared_pair(target, recursion, n=n, burn=burn, seed=seed)
    bound = target.bound(theta, m)
    finite = bound == bound                                   # not nan
    return {
        "sup_err": float(err.max()),
        "mean_err": float(err.mean()),
        "q999_err": float(np.quantile(err, 0.999)),
        "bound": float(bound) if finite else None,
        "slack": float(bound / max(err.max(), 1e-15)) if finite else None,
        "bound_holds": bool(err.max() <= bound) if finite else None,
        "theta": float(theta),
        "truncation_term": float(target.truncation_term(m)),
        "amplification": float(1.0 / (1.0 - target.S_m(m)))
                         if target.S_m(m) < 1 else None,
        "err_head": err[:1500].tolist(),
        "x_head": xt[:n_head].tolist(),
        "xhat_head": xh[:n_head].tolist(),
    }


def synchronisation_report(recursion, M=1.0, n_steps=80, n_starts=24,
                           seed_u=11, seed_z=3):
    """Does the learned recursion forget its initial condition?

    Collapse to machine zero is uniqueness of the closed recursion seen
    directly.  A spread of the order of the state space means the solution is
    not unique and the adapted conclusion is void, which happens exactly when
    the top Lyapunov exponent is positive, not when S_m >= 1: in the
    noise-budget sweep the spread was still 9e-11 at S = 2.0.
    """
    trj, spread = initial_spread(recursion, n_steps=n_steps, n_starts=n_starts,
                                 M=M, seed_u=seed_u, seed_z=seed_z)
    half = n_steps // 2
    return {"spread_half": float(spread[half]),
            "spread_end": float(spread[-1]),
            "spread": spread.tolist(),
            # the trajectories themselves, so the collapse can be SEEN as the
            # chains merging and not only inferred from a decaying scalar
            "trajectories": trj.tolist(),
            "n_starts": int(n_starts), "n_steps": int(n_steps)}
