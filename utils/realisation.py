"""Diagnostics for `conj:approx`: is an open-loop ESN a realisation of the
generator, and in what sense?

Three different questions, answered by three different diagnostics, and they
must not be conflated.

1. THE HYPOTHESIS: is the ESN close to the generator's FILTER?
   `sync_errors`.  Drive the generator and the ESN with the same innovations and
   measure ||xhat_t - y_t|| in L^1, L^2 and L^inf.  This is the eps of the
   conjecture, ||X_0 - Y_0||_{L^p} <= eps, and it is what least squares
   minimises.

2. THE KERNEL: does the ESN's output behave like the generator's KERNEL,
   given the ESN's own history?
   `pit_residuals`.  For each t invert the generator in the level along the
   ESN's own path,

       uhat_t = Fhat( y_t | y_{t-1}, ..., y_{t-m} ),

   i.e. solve qhat(uhat; y_{t-1..t-m}) = y_t.  If the ESN's conditional law
   given its own past were the kernel, uhat would be exactly iid U(0,1).  The
   quantity behind it is the average kernel discrepancy

       kappa_1 = E W_1( P( . | Ybar_t), L(Y_{t+1} | Ybar_t) ),

   the intermediate quantity of Lemma A: step 1 bounds it ABOVE by
   (1 + S') eps_1 using the shared innovations, and step 2 (Knothe-Rosenblatt,
   bicausal) turns it into AW <= kappa / (1 - S').  The PIT bounds it BELOW:
   Fhat( . | z) is f(z)-Lipschitz, so W_1(L(uhat), U(0,1)) <= f* kappa_1
   (`density_sup`, `kernel_sandwich`).  Independence is tested by the lag-1
   autocorrelations of uhat and of (uhat - 1/2)^2.  Run on the generator itself
   it is the control (exact up to quantile crossings); run against the target's
   q it measures the generator-to-target gap on the same scale.

   The synchronous coupling alone cannot certify an adapted statement: it is
   causal from xhat to y but not back, since nothing makes the ESN invertible.
   That is why the argument goes through kappa.

3. THE LAW: on independent streams, is L(y) close to L(xhat)?
   Use the package's marginal / ACF / conditional reports with the GENERATOR
   as the reference.  These are lower bounds for any adapted distance, so they
   can refute the conclusion but never certify it.

`esp_spread` checks the echo state property of the reservoir itself, which
Gonon and Ortega's theorem asserts and the construction needs: the readout is
only a function of the innovations if the reservoir forgets its initial state.
"""
import numpy as np

__all__ = ["sync_errors", "esp_spread", "pit_residuals", "pit_report",
           "density_sup", "kernel_sandwich", "conjectured_bound", "truncated_path"]


def sync_errors(x, y):
    """L^1, L^2, L^inf distances between two paths driven by the same
    innovations, raw and relative to sd(x)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    sd = float(x.std())
    out = {"L1": float(np.mean(np.abs(d))), "L2": float(np.sqrt(np.mean(d ** 2))),
           "Linf": float(np.max(np.abs(d))), "sd": sd}
    out.update({k + "_rel": out[k] / sd for k in ("L1", "L2", "Linf")})
    return out


def esp_spread(esn, u, n_starts=8, seed=0):
    """Spread of the ESN OUTPUT across n random initial reservoir states under
    one stream, as a function of t.  -> 0 is the echo state property."""
    rng = np.random.default_rng(seed)
    H0 = rng.uniform(-1, 1, size=(n_starts, esn.N))
    Hs = esn.states_many(u, H0)                                 # (n, T, N)
    Y = np.stack([esn.predict_from_states(H) for H in Hs])      # (n, T)
    return Y.max(0) - Y.min(0)


def pit_residuals(qfun, path, m, n_bisect=50):
    """uhat_t = F(path_t | path_{t-1..t-m}) by bisection on the level.

    `qfun(u, Z)` is vectorised: u of length n, Z of shape (n, m).  Returns
    (uhat, clip_frac): clip_frac is the share of steps where path_t lies
    outside [q(0;z), q(1;z)], i.e. where the observed value is not in the
    support the kernel allows at all.
    """
    p = np.asarray(path, float)
    T = len(p) - m
    Z = np.stack([p[m - 1 - i: m - 1 - i + T] for i in range(m)], axis=1)
    x = p[m:]
    lo_u, hi_u = np.full(T, 1e-9), np.full(T, 1 - 1e-9)
    q_lo, q_hi = qfun(lo_u, Z), qfun(hi_u, Z)
    clip = (x < np.minimum(q_lo, q_hi)) | (x > np.maximum(q_lo, q_hi))
    incr = q_hi >= q_lo                       # orientation, per window
    lo, hi = lo_u.copy(), hi_u.copy()
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        qm = qfun(mid, Z)
        below = np.where(incr, qm < x, qm > x)
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi), float(clip.mean())


def _acf1(v):
    v = np.asarray(v, float) - np.mean(v)
    return float(np.dot(v[:-1], v[1:]) / np.dot(v, v))


def pit_report(uhat):
    """Uniformity and independence of PIT residuals.

    W1 to U(0,1) is computed from the empirical quantile function.  It is not
    zero even for an exact iid U(0,1) sample: its expectation is
    int_0^1 E|F_n(x) - x| dx ~ sqrt(2/pi) * (pi/8) / sqrt(n) = sqrt(pi/(32 n)),
    about 0.0022 for n = 20000.  That is `W1_iid_scale`, the level to read W1
    against and the amount to subtract before using it as a lower bound.  The
    lag-1 autocorrelations are read against their iid sd, 1/sqrt(n).
    """
    u = np.sort(np.asarray(uhat, float))
    n = len(u)
    grid = (np.arange(n) + 0.5) / n
    w1 = float(np.mean(np.abs(u - grid)))
    ks = float(max(np.max(np.arange(1, n + 1) / n - u), np.max(u - np.arange(n) / n)))
    return {"n": n, "W1_to_uniform": w1, "KS": ks,
            "W1_iid_scale": float(np.sqrt(np.pi / (32 * n))),
            "acf1": _acf1(uhat), "acf1_sq": _acf1((np.asarray(uhat) - 0.5) ** 2),
            "acf_iid_scale": 1.0 / np.sqrt(n)}


def density_sup(qfun, path, m, n_grid=199, edge=0.005):
    """Largest conditional density of the kernel along the windows of `path`.

    For each window z_t = (path_{t-1}, ..., path_{t-m}), the density of
    q(U; z_t)_# Unif is 1 / d_u q(u; z_t); its supremum over u is estimated by
    finite differences on a level grid in [edge, 1 - edge].  Returns
    (f_star, f_per_window), f_star the maximum over windows.

    Why it is needed: F(. | z) is f(z)-Lipschitz, so a W1 discrepancy d(z)
    between the ESN's conditional law and the kernel moves the PIT by at most
    f(z) d(z).  Hence  W1(L(uhat), U(0,1)) <= f_star * kappa_1,  which turns the
    PIT diagnostic into a LOWER bound on the average kernel discrepancy kappa_1.
    """
    p = np.asarray(path, float)
    T = len(p) - m
    Z = np.stack([p[m - 1 - i: m - 1 - i + T] for i in range(m)], axis=1)
    g = np.linspace(edge, 1.0 - edge, n_grid)
    du = g[1] - g[0]
    prev = np.asarray(qfun(np.full(T, g[0]), Z), float)
    f = np.zeros(T)
    for j in range(1, n_grid):
        cur = np.asarray(qfun(np.full(T, g[j]), Z), float)
        f = np.maximum(f, du / np.maximum(np.abs(cur - prev), 1e-300))
        prev = cur
    return float(f.max()), f


def kernel_sandwich(w1_pit, w1_iid, f_star, eps1, S):
    """Both sides of the first step of the conjecture's argument, from measured
    quantities.  kappa_1 = E W1( P(. | Ybar), L(Y_next | Ybar) ), P the
    generator's kernel, Ybar the ESN's own past:

        (W1(uhat) - iid level)_+ / f_star  <=  kappa_1  <=  (1 + S) eps_1 .

    The left side comes from the PIT residuals (C.2), the right side from the
    synchronous L^1 error (C.1) and the generator's contraction constant S.
    left > right means a measured constant is wrong: in practice S, if the
    Lipschitz estimate missed the supremum.  nan on the right when S >= 1.
    """
    lo = max(w1_pit - w1_iid, 0.0) / f_star
    hi = (1.0 + S) * eps1 if S < 1 else float("nan")
    return lo, hi


def conjectured_bound(eps, S):
    """(1+S) eps / (1-S): the adapted bound conj:approx predicts from an L^p
    filter error eps.  nan when S >= 1."""
    return float((1 + S) * eps / (1 - S)) if S < 1 else float("nan")


def truncated_path(target, m, u_stream, burn=4000):
    """The m-truncated recursion  x_t = q_m(u_t; x_{t-1..t-m})  driven by a given
    stream: the TRUNCATION realisation that conj:approx(ii) compares against.
    Its dimension is m (a shift register of length m, linear readout)."""
    u = np.asarray(u_stream, float)
    z = np.zeros(m)
    out = np.empty(len(u))
    for t in range(len(u)):
        out[t] = float(target.q(np.array([u[t]]), z[None, :])[0])
        z[1:] = z[:-1]
        z[0] = out[t]
    return out[burn:]
