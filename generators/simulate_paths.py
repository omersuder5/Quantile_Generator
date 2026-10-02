"""One entry point for simulating any generator in this package.

Three protocols are used throughout and they must not be confused, because
they answer different questions:

  INDEPENDENT   target and generator driven by unrelated innovation streams.
                Everything about the LAW is measured this way: the marginal,
                the autocorrelation, the conditional distance, the block-MMD
                test.  Using a shared stream here would make the two paths
                close by construction and the comparison meaningless.

  SHARED        both driven by the SAME stream u_1, u_2, ...  This is the
                synchronous coupling.  sup_t |X_t - Xhat_t| is the quantity the
                comparison lemma bounds by (theta + 2 M L_(m+1))/(1 - S_m), and
                it is an upper bound for the adapted Wasserstein distance,
                never a lower one.

  SPREAD        one stream, many initial states.  Measures whether the learned
                recursion forgets its initial condition, i.e. whether the
                solution of the closed recursion is unique.  It collapses to
                machine zero when the fitted map contracts and stays of the
                order of the state space when it does not.

`shared_pair` and `initial_spread` exist so that no caller has to rebuild the
stream bookkeeping and accidentally use the wrong one.
"""
import numpy as np

__all__ = ["simulate", "shared_pair", "initial_spread", "make_windows",
           "draw_stream"]


def draw_stream(n, burn=0, seed=0):
    """A stream of burn+n i.i.d. uniforms, for the shared-innovation protocols."""
    return np.random.default_rng(seed).random(burn + n)


def simulate(gen, n, burn=2000, seed=0, u_stream=None, z0=None):
    """Simulate any generator: a synthetic target or a learned recursion.

    Synthetic targets always run at their OWN memory order, never at the
    fitting arity m; `ARMATarget` and `LongMemoryTarget` override `simulate`
    with their exact recursions.  Learned recursions run at m.
    """
    sim = getattr(gen, "simulate", None)
    if callable(sim):
        return sim(n, burn=burn, seed=seed, u_stream=u_stream, z0=z0)
    gen_fn = getattr(gen, "generate", None)
    if callable(gen_fn):
        return gen_fn(n, burn=burn, seed=seed, u_stream=u_stream, z0=z0)
    raise TypeError(f"{type(gen).__name__} has neither simulate nor generate")


def shared_pair(target, recursion, n=20000, burn=2000, seed=7):
    """Target and learned recursion on ONE innovation stream.

    Returns (x_target, x_learned, abs_error).  This is the section-8 pathwise
    diagnostic: compare `abs_error.max()` with `target.bound(theta, m)`.
    """
    u = draw_stream(n, burn=burn, seed=seed)
    xt = simulate(target, n, burn=burn, seed=0, u_stream=u)
    xh = simulate(recursion, n, burn=burn, seed=0, u_stream=u)
    return xt, xh, np.abs(xt - xh)


def initial_spread(recursion, n_steps=80, n_starts=24, M=1.0, seed_u=11,
                   seed_z=3):
    """Spread across `n_starts` initial states under one shared stream.

    Returns (trajectories, spread) with spread[t] = max_j x_j(t) - min_j x_j(t).
    Decay to machine zero is uniqueness of the closed recursion, seen directly.
    """
    m = int(getattr(recursion, "m", getattr(recursion, "k")))
    u = np.random.default_rng(seed_u).random(n_steps)
    z0s = np.random.default_rng(seed_z).uniform(-M, M, size=(n_starts, m))
    trj = np.stack([simulate(recursion, n_steps, burn=0, seed=0, u_stream=u, z0=z)
                    for z in z0s])
    return trj, trj.max(0) - trj.min(0)


def make_windows(path, m):
    """Supervised pairs (z_t, x_t) with z_t = (X_{t-1}, ..., X_{t-m}).

    m is the fitting arity, which is what the user presumes the memory to be.
    It is NOT raised to the target's true order: fitting below the truth is a
    legitimate and informative experiment, priced by L_(m+1).
    """
    path = np.asarray(path, dtype=float)
    T = len(path)
    if T <= m:
        raise ValueError(f"path of length {T} is too short for m = {m}")
    idx = np.arange(m, T)
    Z = np.stack([path[idx - j] for j in range(1, m + 1)], axis=1)
    X = path[idx]
    return Z, X
