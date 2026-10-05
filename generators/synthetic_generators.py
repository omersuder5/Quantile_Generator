"""Synthetic targets, each parameterised by its contraction constant S.

Every target here is a strictly stationary process on X = [-M, M] whose one-step
conditional quantile given the past is known in closed form,

    X_t = q(u_t ; X_{t-1}, X_{t-2}, ...),      u_t iid U(0,1),

with per-lag W_infinity moduli

    |q(u;z) - q(u;z')| <= sum_i ell_i |z_i - z'_i|,     S = sum_i ell_i.

Two things are different from the earlier code and both are deliberate.

**S is an input, not an output.**  Each target takes a fixed *shape* (its
specifics: the relative lag profile, the oscillation count, the ARCH floor
omega, the decay exponent) plus the scalar S, and solves internally for the one
amplitude that makes sum_i ell_i equal to S exactly.  Three targets invert
analytically (`linear` and `smoothnl` scale their lag vector; `logistic` has
ell_1 = 4(1-rho) independently of r, so rho = 1 - S/4); the rest bisect on a
single monotone knob.  This makes S an orthogonal axis of the experiment, so
"same S, different mechanism" is a fair comparison, which it was not when each
target's S fell out of hard-coded coefficients.

**There is no padding to a common arity.**  A target has whatever memory it
has; the learner fits at arity m, which is what the user *presumes* the memory
to be, and m may be smaller, equal to or larger than the truth.  Accordingly
`q(u, z)` accepts z with any number of columns and treats absent lags as the
anchor 0, which makes it exactly q_m, the m-anchored truncation of
Definition (quantile generator).  The cost of fitting at m is then

    S_m = sum_{i<=m} ell_i ,     L_(m+1) = sum_{i>m} ell_i ,

uniformly across finite and infinite memory, and the pathwise bound is
(theta + 2 M L_(m+1)) / (1 - S_m).  Fitting a 2-lag linear target at m = 1 is
misspecified in exactly the way, and priced by exactly the term, that fitting a
long-memory target at m = 4 is.

Simulation always uses the target's *full* memory.  m never enters it.

**Nine of the eleven are inside the theory; two are not.**  `GARCHTarget` and
`EGARCHTarget` are latent-volatility models, which `prop:gauss` excludes
outright, and they are here so the exclusion can be MEASURED rather than
asserted.  Both are made runnable by bounding the innovation so |X| <= M
exactly, and both unroll in the observable -- GARCH's latent variance is a
geometrically weighted sum of past SQUARES -- so q_m, the moduli, S_m and
L_(m+1) are all well defined for them too.  See their own docstrings for what
is faithful and what is a concession.
"""
import numpy as np

from .noise import noise_fn

__all__ = [
    "Target", "LinearTarget", "HeteroTarget", "SkewTarget", "SmoothNonlinearAR",
    "ARCHTarget", "OscDriftTarget", "NoisyLogisticTarget", "ARMATarget",
    "LongMemoryTarget", "GARCHTarget", "EGARCHTarget",
    "TARGETS", "build_target",
]


# ---------------------------------------------------------------------------
# Amplitude solving
# ---------------------------------------------------------------------------

class Infeasible(Exception):
    """Raised by a shape evaluation when the amplitude violates the range
    constraint (no room left for the innovation)."""


def solve_amplitude(S_of_lam, S_target, lam_init=1.0, tol=1e-12, label=""):
    """Smallest lam > 0 with S_of_lam(lam) = S_target.

    `S_of_lam` must be continuous and nondecreasing on the feasible interval,
    with S_of_lam(0+) = 0, and should raise `Infeasible` beyond it.  Returns
    (lam, S_achieved).  Raises ValueError naming the attainable range when the
    request is out of reach, which is the useful failure: it tells you to change
    the shape, not the solver.
    """
    if not (S_target > 0):
        raise ValueError(f"{label}: S must be > 0, got {S_target}")

    lo, hi = 0.0, float(lam_init)
    S_hi = None
    for _ in range(200):
        try:
            S_hi = S_of_lam(hi)
        except Infeasible:
            break
        if S_hi >= S_target:
            break
        lo, hi = hi, hi * 1.6
    else:
        raise ValueError(f"{label}: amplitude search did not bracket S = {S_target}")

    for _ in range(400):
        mid = 0.5 * (lo + hi)
        try:
            s = S_of_lam(mid)
        except Infeasible:
            hi = mid
            continue
        if s < S_target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break

    lam = 0.5 * (lo + hi)
    try:
        S_got = S_of_lam(lam)
    except Infeasible:
        lam = lo
        S_got = S_of_lam(lam) if lam > 0 else 0.0
    if abs(S_got - S_target) > 1e-6 * max(1.0, S_target):
        raise ValueError(
            f"{label}: S = {S_target:.4f} is not attainable with this shape; "
            f"the largest reachable value is about {S_got:.4f}. "
            "Change the shape in TARGET_SPECIFICS (the lag profile, kappa, "
            "omega or the ARCH floor), not S.")
    return lam, float(S_got)


def _normalise(profile):
    w = np.abs(np.asarray(profile, dtype=float))
    if w.sum() <= 0:
        raise ValueError("lag profile must have positive mass")
    return w / w.sum()


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Target:
    """Common interface.

    Subclasses set, in `_finalise`:  `_ell_full` (the full modulus sequence),
    `S`, `c` (innovation amplitude), `g`, `k_true` (int, or None when the memory
    is only numerically finite), `name`.
    """

    M = 1.0
    noise = "truncnorm"
    _ell_full = None
    k_true = None

    # -- modulus bookkeeping ------------------------------------------------
    def moduli(self, m=None):
        """The first m per-lag moduli, zero-padded if m exceeds the memory."""
        e = np.asarray(self._ell_full, dtype=float)
        if m is None:
            return e.copy()
        if m <= len(e):
            return e[:m].copy()
        return np.concatenate([e, np.zeros(m - len(e))])

    def S_m(self, m):
        """Retained mass at fitting arity m."""
        return float(np.asarray(self._ell_full)[:m].sum())

    def L_tail(self, m):
        """L_(m+1) = sum_{i>m} ell_i, the mass the m-anchored truncation drops.
        The truncation term in the bound is 2 M L_(m+1)."""
        return float(np.asarray(self._ell_full)[m:].sum())

    def truncation_term(self, m):
        return 2.0 * self.M * self.L_tail(m)

    def bound(self, theta, m):
        """(theta + 2 M L_(m+1)) / (1 - S_m), or nan when S_m >= 1."""
        Sm = self.S_m(m)
        if Sm >= 1.0:
            return float("nan")
        return float((theta + self.truncation_term(m)) / (1.0 - Sm))

    # -- the quantile -------------------------------------------------------
    def q(self, u, z):
        """q(u; z) with z of any arity: absent lags are the anchor 0, so this is
        q_m when z has m columns."""
        raise NotImplementedError

    def step(self, gu, z):
        """One step given the already-transformed innovation and the state."""
        raise NotImplementedError

    # -- simulation ---------------------------------------------------------
    def simulate(self, n, burn=2000, seed=0, u_stream=None, z0=None):
        """Free-running simulation at the target's OWN memory order."""
        k = self._sim_order()
        rng = np.random.default_rng(seed)
        total = burn + n
        u = (rng.random(total) if u_stream is None
             else np.asarray(u_stream, dtype=float)[:total])
        if len(u) < total:
            raise ValueError(f"u_stream too short: need {total}, got {len(u)}")
        gu = self.g(u)
        z = np.zeros(k) if z0 is None else np.asarray(z0, dtype=float).copy()
        if len(z) < k:
            z = np.concatenate([z, np.zeros(k - len(z))])
        out = np.empty(total)
        for t in range(total):
            out[t] = self.step(gu[t], z)
            z[1:] = z[:-1]
            z[0] = out[t]
        return out[burn:]

    def _sim_order(self):
        return int(self.k_true if self.k_true is not None else len(self._ell_full))

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _as_uz(u, z):
        return (np.atleast_1d(np.asarray(u, dtype=float)),
                np.atleast_2d(np.asarray(z, dtype=float)))

    def _coeffs_for(self, ncol, vec):
        """vec truncated or zero-padded to ncol, for linear-in-z targets."""
        v = np.asarray(vec, dtype=float)
        if ncol <= len(v):
            return v[:ncol]
        return np.concatenate([v, np.zeros(ncol - len(v))])

    def _pad_state(self, z, k):
        """z padded with the anchor 0 (or truncated) to exactly k columns."""
        ncol = z.shape[1]
        if ncol == k:
            return z
        if ncol > k:
            return z[:, :k]
        return np.hstack([z, np.zeros((len(z), k - ncol))])

    def describe(self):
        return {"name": self.name, "S": self.S, "M": self.M, "c": self.c,
                "k_true": self.k_true, "noise": self.noise,
                "ell_head": self.moduli(min(8, len(self._ell_full))).tolist()}


# ---------------------------------------------------------------------------
# Affine in the state, additive innovation
# ---------------------------------------------------------------------------

class LinearTarget(Target):
    """q(u;z) = sum_i a_i z_i + g(u),  ell_i = |a_i| exactly.

    The control: affine mean, constant conditional scale.  S is matched by
    scaling the lag profile, which is exact.  This is also the family in which
    the Perron bound is tight.
    """
    def __init__(self, S, profile=(0.45, 0.20), M=1.0, noise="truncnorm"):
        w = _normalise(profile)
        self.M = float(M)
        self.noise = noise
        self.a = float(S) * w
        self.k_true = len(w)
        self._ell_full = np.abs(self.a)
        self.S = float(self._ell_full.sum())
        if self.S >= 1.0:
            raise ValueError("LinearTarget needs S < 1 for the range to close")
        self.c = self.M * (1.0 - self.S)
        self.g = noise_fn(noise, self.c)
        self.name = f"Linear(k={self.k_true}, S={self.S:.3f})"

    def q(self, u, z):
        u, z = self._as_uz(u, z)
        return z @ self._coeffs_for(z.shape[1], self.a) + self.g(u)

    def step(self, gu, z):
        z = np.asarray(z, dtype=float)
        return float(z @ self._coeffs_for(len(z), self.a) + gu)


class SmoothNonlinearAR(Target):
    """X_t = H tanh( (sum_i b_i X_{t-i}) / H ) + g(u_t),  H = h M.

    Nonlinear conditional mean, homoskedastic.  dq/dz_i = b_i sech^2(.), so
    ell_i = |b_i| attained at z = 0 and S = sum |b_i| is matched exactly.
    Small h saturates the tanh and makes the nonlinearity strong at fixed S.
    """
    def __init__(self, S, profile=(0.47, 0.53), h=0.55, M=1.0, noise="truncnorm"):
        w = _normalise(profile)
        self.M, self.h = float(M), float(h)
        if not 0.0 < self.h < 1.0:
            raise ValueError("h must lie in (0,1)")
        self.noise = noise
        self.beta = float(S) * w
        self.k_true = len(w)
        self.H = self.h * self.M
        self.c = (1.0 - self.h) * self.M
        self._ell_full = np.abs(self.beta)
        self.S = float(self._ell_full.sum())
        if self.S >= 1.0:
            raise ValueError("SmoothNonlinearAR needs S < 1")
        self.g = noise_fn(noise, self.c)
        self.name = f"SmoothNL(k={self.k_true}, S={self.S:.3f}, h={self.h})"

    def mean_of(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return self.H * np.tanh((z @ self._coeffs_for(z.shape[1], self.beta)) / self.H)

    def q(self, u, z):
        u, z = self._as_uz(u, z)
        return self.mean_of(z) + self.g(u)

    def step(self, gu, z):
        z = np.asarray(z, dtype=float)
        b = self._coeffs_for(len(z), self.beta)
        return float(self.H * np.tanh(float(z @ b) / self.H) + gu)


# ---------------------------------------------------------------------------
# State-dependent conditional shape
# ---------------------------------------------------------------------------

class HeteroTarget(Target):
    """q(u;z) = sum_i a_i z_i + s(z_1) g(u),  s(z) = s0 + s1 tanh(kappa z).

    Conditional scale is a ridge function of z_1, so one hidden unit can in
    principle express it.  ell_1 = a_1 + s1 kappa c, ell_i = a_i.

    The amplitude lam scales the lag profile AND kappa together, so that both
    routes by which the state enters the conditional law shrink to nothing as
    lam -> 0 and S(lam) runs from 0 to 1.
    """
    def __init__(self, S, profile=(0.70, 0.30), s0=0.45, s1=0.45, kappa=1.2,
                 M=1.0, noise="truncnorm"):
        w = _normalise(profile)
        self.M, self.noise = float(M), noise
        self.s0, self.s1, self.kappa0 = float(s0), float(s1), float(kappa)

        def S_of(lam):
            if lam >= 1.0:
                raise Infeasible
            a = lam * w
            c = self.M * (1.0 - lam) / (self.s0 + self.s1)
            if c <= 0:
                raise Infeasible
            ell = a.copy()
            ell[0] += self.s1 * (lam * self.kappa0) * c
            return float(ell.sum())

        lam, self.S = solve_amplitude(S_of, float(S), label="HeteroTarget")
        self.alpha = lam * w
        self.kappa = lam * self.kappa0
        self.c = self.M * (1.0 - lam) / (self.s0 + self.s1)
        self.k_true = len(w)
        ell = self.alpha.copy()
        ell[0] += self.s1 * self.kappa * self.c
        self._ell_full = ell
        self.g = noise_fn(noise, self.c)
        self.name = f"Hetero(k={self.k_true}, S={self.S:.3f})"

    def scale(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return self.s0 + self.s1 * np.tanh(self.kappa * z[:, 0])

    def q(self, u, z):
        u, z = self._as_uz(u, z)
        return z @ self._coeffs_for(z.shape[1], self.alpha) + self.scale(z) * self.g(u)

    def step(self, gu, z):
        z = np.asarray(z, dtype=float)
        a = self._coeffs_for(len(z), self.alpha)
        return float(z @ a + (self.s0 + self.s1 * np.tanh(self.kappa * z[0])) * gu)


class SkewTarget(Target):
    """State-dependent conditional SKEWNESS, conditional mean held fixed.

        v = 2u - 1,   psi(v; l) = v + l|v| - l/2      (mean zero, increasing for |l|<1)
        q(u;z) = sum_i a_i z_i + c psi(2u-1; l(z)),   l(z) = l0 tanh(kappa z_1)

    The conditional variance moves only mildly and the third moment flips sign
    with z_1.  No second-order diagnostic sees this: not the ACF, not the ACF of
    squares, not the conditional standard deviation.  A matched Gaussian AR and
    a matched GARCH are both blind to it by construction, which makes this the
    sharpest test of a conditional criterion in the suite.

        d psi / d l = |v| - 1/2 in [-1/2, 1/2]
        ell_1 = a_1 + c l0 kappa / 2,  ell_i = a_i

    Consumes the raw uniform, because the transform depends on the state.
    """
    def __init__(self, S, profile=(0.70, 0.30), lam0=0.7, kappa=1.2, M=1.0,
                 noise="raw"):
        w = _normalise(profile)
        self.M = float(M)
        self.lam0, self.kappa0 = float(lam0), float(kappa)
        if not 0.0 <= self.lam0 < 1.0:
            raise ValueError("lam0 must lie in [0,1) for monotonicity in u")

        def S_of(lam):
            if lam >= 1.0:
                raise Infeasible
            a = lam * w
            c = self.M * (1.0 - lam) / (1.0 + self.lam0 / 2.0)
            if c <= 0:
                raise Infeasible
            ell = a.copy()
            ell[0] += c * self.lam0 * (lam * self.kappa0) / 2.0
            return float(ell.sum())

        lam, self.S = solve_amplitude(S_of, float(S), label="SkewTarget")
        self.alpha = lam * w
        self.kappa = lam * self.kappa0
        self.c = self.M * (1.0 - lam) / (1.0 + self.lam0 / 2.0)
        self.k_true = len(w)
        ell = self.alpha.copy()
        ell[0] += self.c * self.lam0 * self.kappa / 2.0
        self._ell_full = ell
        self.noise = "raw"
        self.g = noise_fn("raw", self.c)      # identity: q applies the transform
        self.name = f"Skew(k={self.k_true}, S={self.S:.3f}, lam0={self.lam0})"

    def lam_of(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return self.lam0 * np.tanh(self.kappa * z[:, 0])

    def q(self, u, z):
        u, z = self._as_uz(u, z)
        v = 2.0 * u - 1.0
        l = self.lam_of(z)
        return (z @ self._coeffs_for(z.shape[1], self.alpha)
                + self.c * (v + l * np.abs(v) - l / 2.0))

    def step(self, u, z):
        z = np.asarray(z, dtype=float)
        v = 2.0 * float(u) - 1.0
        l = self.lam0 * np.tanh(self.kappa * float(z[0]))
        a = self._coeffs_for(len(z), self.alpha)
        return float(z @ a + self.c * (v + l * abs(v) - l / 2.0))


class ARCHTarget(Target):
    """Bounded ARCH:  X_t = sqrt(omega + sum_i a_i X_{t-i}^2) g(u_t).

    Zero conditional mean, conditional scale even in each coordinate.  This is
    the volatility-clustering target, and the one the earlier MMD-trained
    architecture could not reproduce.  Genuinely Markov in the observable,
    unlike GARCH, whose latent volatility has no finite invertible
    representation in X and which is excluded outright by the Gaussian-tail
    proposition.

        dq/dz_i = g(u) a_i z_i / sqrt(omega + sum_j a_j z_j^2)
        ell_i   = c a_i M / sqrt(omega + a_i M^2)
        range   |q| <= c sqrt(omega + M^2 sum a) = M  fixes c.

    omega is the shape (the volatility floor); the amplitude scales the a_i.
    Note the identity sigma_max / sigma_min <= (1-S)^{-1/2}, with equality at
    one lag: the attainable scale contrast and the amplification constant are
    the same parameter.
    """
    def __init__(self, S, profile=(0.667, 0.333), omega=0.2, M=1.0,
                 noise="truncnorm"):
        w = _normalise(profile)
        self.M, self.omega, self.noise = float(M), float(omega), noise
        if self.omega <= 0:
            raise ValueError("omega must be > 0 so the conditional scale is bounded below")

        def S_of(lam):
            a = lam * w
            c = self.M / np.sqrt(self.omega + self.M ** 2 * a.sum())
            ell = c * a * self.M / np.sqrt(self.omega + a * self.M ** 2)
            return float(ell.sum())

        lam, self.S = solve_amplitude(S_of, float(S), label="ARCHTarget")
        self.a = lam * w
        self.k_true = len(w)
        self.c = self.M / np.sqrt(self.omega + self.M ** 2 * self.a.sum())
        self._ell_full = self.c * self.a * self.M / np.sqrt(self.omega + self.a * self.M ** 2)
        self.g = noise_fn(noise, self.c)
        self.name = f"ARCH(k={self.k_true}, S={self.S:.3f}, omega={self.omega})"

    def sigma_of(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return np.sqrt(self.omega + z ** 2 @ self._coeffs_for(z.shape[1], self.a))

    def q(self, u, z):
        u, z = self._as_uz(u, z)
        return self.sigma_of(z) * self.g(u)

    def step(self, gu, z):
        z = np.asarray(z, dtype=float)
        a = self._coeffs_for(len(z), self.a)
        return float(np.sqrt(self.omega + float(z ** 2 @ a)) * gu)

    def scale_contrast_bound(self):
        """sigma_max / sigma_min <= (1-S)^{-1/2}; equality at one lag."""
        return float((1.0 - self.S) ** -0.5)


class OscDriftTarget(Target):
    """q(u;z) = (a1/W) sin(W z_1) + a2 z_2 + s(z_1) g(u),  s = s0 + s1 tanh(kappa z).

    The oscillation count W is the shape and is NOT touched by the amplitude:
    a drift that is Lipschitz-a1 and oscillates W times over the state space has
    amplitude a1/W.  So (S fixed, W varying) separates the amplification
    constant 1/(1-S), which the theory supplies, from the width the network
    needs, which it does not.

        ell_1 = a1 + s1 kappa c,  ell_2 = |a2|
        |q| <= a1/W + |a2| M + (s0+s1) c = M  fixes c.
    """
    def __init__(self, S, profile=(0.70, 0.30), omega_osc=6.0, s0=0.40, s1=0.10,
                 kappa=1.2, M=1.0, noise="truncnorm"):
        w = _normalise(profile)
        if len(w) < 2:
            raise ValueError("OscDriftTarget needs at least two lags")
        self.M, self.noise = float(M), noise
        self.omega_osc = float(omega_osc)
        self.s0, self.s1, self.kappa0 = float(s0), float(s1), float(kappa)

        def S_of(lam):
            a = lam * w
            c = (self.M - a[0] / self.omega_osc - np.abs(a[1:]).sum() * self.M) \
                / (self.s0 + self.s1)
            if c <= 0:
                raise Infeasible
            ell = np.abs(a).copy()
            ell[0] = a[0] + self.s1 * (lam * self.kappa0) * c
            return float(ell.sum())

        lam, self.S = solve_amplitude(S_of, float(S), label="OscDriftTarget")
        self.a = lam * w
        self.kappa = lam * self.kappa0
        self.c = (self.M - self.a[0] / self.omega_osc
                  - np.abs(self.a[1:]).sum() * self.M) / (self.s0 + self.s1)
        self.k_true = len(w)
        ell = np.abs(self.a).copy()
        ell[0] = self.a[0] + self.s1 * self.kappa * self.c
        self._ell_full = ell
        self.g = noise_fn(noise, self.c)
        self.name = f"OscDrift(k={self.k_true}, S={self.S:.3f}, W={self.omega_osc:g})"

    def scale(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return self.s0 + self.s1 * np.tanh(self.kappa * z[:, 0])

    def drift(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        zf = self._pad_state(z, self.k_true)
        return ((self.a[0] / self.omega_osc) * np.sin(self.omega_osc * zf[:, 0])
                + zf[:, 1:] @ self.a[1:])

    def q(self, u, z):
        u, z = self._as_uz(u, z)
        return self.drift(z) + self.scale(z) * self.g(u)

    def step(self, gu, z):
        z = np.asarray(z, dtype=float)
        zf = np.concatenate([z, np.zeros(max(0, self.k_true - len(z)))])[:self.k_true]
        d = (self.a[0] / self.omega_osc) * np.sin(self.omega_osc * zf[0]) + zf[1:] @ self.a[1:]
        return float(d + (self.s0 + self.s1 * np.tanh(self.kappa * zf[0])) * gu)


class NoisyLogisticTarget(Target):
    """A chaotic drift with bounded stochastic forcing.

    The logistic map at parameter r, conjugated to [-M, M], centred and
    rescaled so that its range covers a fraction 1-rho of the interval:

        m0(x) = r(x+M)(M-x)/(2M) - M,   m0'(x) = -r x / M,   sup|m0'| = r
        m(x)  = gamma (m0(x) - mid),    gamma = 4(1-rho)/r
        X_t   = m(X_{t-1}) + rho M g(u_t)

    Since m0 ranges over an interval of width M r / 2,

        ell_1 = sup |m'| = gamma r = 4 (1 - rho)   INDEPENDENTLY OF r,

    so S determines rho = 1 - S/4 analytically and r stays free as the shape.
    S < 1 iff rho > 3/4.  **S >= 1 is permitted here** (the standing hypothesis
    then fails and the Lipschitz penalty must be switched off), because the
    noise-budget sweep across S = 1 is the empirical content of the
    average-contraction statement: the operative boundary for the pathwise and
    adapted conclusions is a negative Lyapunov exponent, not S < 1.
    """
    def __init__(self, S, r=3.7, M=1.0, noise="truncnorm"):
        self.M, self.r, self.noise = float(M), float(r), noise
        self.S = float(S)
        self.rho = 1.0 - self.S / 4.0
        if not 0.0 < self.rho <= 1.0:
            raise ValueError(f"S = {S} gives rho = {self.rho:.3f}; need 0 < S < 4")
        self.gamma = 4.0 * (1.0 - self.rho) / self.r
        self.mid = self.M * (self.r / 4.0 - 1.0)
        self.c = self.rho * self.M
        self.k_true = 1
        self._ell_full = np.array([4.0 * (1.0 - self.rho)])
        self.g = noise_fn(noise, self.c)
        self.name = f"NoisyLogistic(r={self.r:g}, rho={self.rho:.3f}, S={self.S:.3f})"

    def drift(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))[:, 0]
        m0 = self.r * (z + self.M) * (self.M - z) / (2 * self.M) - self.M
        return self.gamma * (m0 - self.mid)

    def q(self, u, z):
        u, z = self._as_uz(u, z)
        return self.drift(z) + self.g(u)

    def step(self, gu, z):
        return float(self.drift(np.asarray(z, dtype=float)[None, :])[0] + gu)

    def lyapunov_analytic(self, path):
        """E log|m'(X)| = log 4(1-rho) + E log|X|/M, exactly, along a path."""
        x = np.asarray(path, dtype=float)
        return float(np.mean(np.log(np.abs(-4 * (1 - self.rho) * x / self.M) + 1e-300)))


# ---------------------------------------------------------------------------
# Infinite memory
# ---------------------------------------------------------------------------

class ARMATarget(Target):
    """Bounded ARMA(p,q), seen through its AR(infinity) inversion.

        X_t = sum_j phi_j X_{t-j} + e_t + sum_j theta_j e_{t-j},   e_t = g(u_t)

    Inverted, q(u; X_<t) = sum_i pi_i X_{t-i} + g(u), so the target has no
    finite memory order and the moduli are |pi_i|, decaying geometrically.
    The amplitude scales phi and theta jointly, which keeps the process a
    genuine ARMA (so the exact (p+q)-state simulation still applies) while
    moving S = sum |pi_i| continuously from 0.

    pi comes from theta(B) A(B) = phi(B) with A(B) = 1 - sum_i pi_i B^i:
        a_0 = 1, a_n = -phi_n 1{n<=p} - sum_{j<=min(n,q)} theta_j a_{n-j},
        pi_n = -a_n.
    """
    def __init__(self, S, phi=(0.20,), theta=(0.25,), M=1.0, noise="truncnorm",
                 n_pi=400):
        phi0 = np.asarray(phi, dtype=float)
        theta0 = np.asarray(theta, dtype=float)
        self.M, self.noise, self.n_pi = float(M), noise, int(n_pi)

        def pi_of(lam):
            ph, th = lam * phi0, lam * theta0
            if len(th) and np.abs(th).sum() >= 1.0:
                raise Infeasible                      # MA part must be invertible
            a = np.zeros(self.n_pi + 1)
            a[0] = 1.0
            for n in range(1, self.n_pi + 1):
                s = -(ph[n - 1] if n <= len(ph) else 0.0)
                for j in range(1, min(n, len(th)) + 1):
                    s -= th[j - 1] * a[n - j]
                a[n] = s
            return ph, th, -a[1:]

        def S_of(lam):
            _, _, pi = pi_of(lam)
            s = float(np.abs(pi).sum())
            if s >= 1.0:
                raise Infeasible
            return s

        lam, self.S = solve_amplitude(S_of, float(S), label="ARMATarget")
        self.phi, self.theta, self.pi = pi_of(lam)
        self.p, self.q_ord = len(self.phi), len(self.theta)
        self._ell_full = np.abs(self.pi)
        self.c = self.M * (1.0 - self.S)
        self.g = noise_fn(noise, self.c)
        self.k_true = None
        self.name = (f"ARMA(p={self.p},q={self.q_ord}, S={self.S:.3f}) "
                     f"[AR(inf), geometric moduli]")

    def q(self, u, z):
        u, z = self._as_uz(u, z)
        return z @ self._coeffs_for(z.shape[1], self.pi) + self.g(u)

    def step(self, gu, z):
        z = np.asarray(z, dtype=float)
        return float(z @ self._coeffs_for(len(z), self.pi) + gu)

    def simulate(self, n, burn=2000, seed=0, u_stream=None, z0=None):
        """Exact, in ARMA form: a (p+q) state, no truncation of the moduli."""
        rng = np.random.default_rng(seed)
        total = burn + n
        u = (rng.random(total) if u_stream is None
             else np.asarray(u_stream, dtype=float)[:total])
        e = self.g(u)
        x = np.zeros(total)
        for t in range(total):
            v = e[t]
            for j in range(1, self.p + 1):
                if t - j >= 0:
                    v += self.phi[j - 1] * x[t - j]
            for j in range(1, self.q_ord + 1):
                if t - j >= 0:
                    v += self.theta[j - 1] * e[t - j]
            x[t] = v
        return x[burn:]


class LongMemoryTarget(Target):
    """X_t = sum_{i>=1} phi_i X_{t-i} + g(u_t), phi_i >= 0 summable, sum = S.

    The coefficient sequence is given directly rather than induced by a rational
    filter, which is the only way to get a POLYNOMIAL modulus tail:

      decay="geometric"    phi_i ~ r^(i-1)          L_(m+1) ~ r^m        easy
      decay="polynomial"   phi_i ~ i^-(1+alpha)     L_(m+1) ~ m^-alpha   hard
      decay="fractional"   the ARFIMA shape, ~ i^-(1+d)

    Normalised to the prescribed S by construction, so no bisection is needed.
    Truncated at n_lag, which is in the thousands while you fit at arity m in
    the tens: the "thousand-Markov target fitted at m = 2" case, with L_tail(m)
    pricing it.
    """
    def __init__(self, S, decay="polynomial", n_lag=4000, r=0.8, alpha=0.5,
                 d=0.3, M=1.0, noise="truncnorm"):
        i = np.arange(1, int(n_lag) + 1, dtype=float)
        if decay == "geometric":
            w = float(r) ** (i - 1)
        elif decay == "polynomial":
            w = i ** (-(1.0 + float(alpha)))
        elif decay == "fractional":
            w = np.empty(int(n_lag))
            w[0] = float(d)
            for j in range(2, int(n_lag) + 1):
                w[j - 1] = w[j - 2] * (j - 1 - float(d)) / j
        else:
            raise ValueError(f"unknown decay {decay!r}")
        self.phi = float(S) * w / w.sum()
        self.n_lag = int(n_lag)
        self.decay = decay
        self.M, self.noise = float(M), noise
        self._ell_full = self.phi.copy()
        self.S = float(self.phi.sum())
        if self.S >= 1.0:
            raise ValueError("LongMemoryTarget needs S < 1")
        self.c = self.M * (1.0 - self.S)
        self.g = noise_fn(noise, self.c)
        self.k_true = None
        self.name = f"LongMemory({decay}, S={self.S:.3f}, n_lag={self.n_lag})"

    def q(self, u, z):
        u, z = self._as_uz(u, z)
        return z @ self._coeffs_for(z.shape[1], self.phi) + self.g(u)

    def step(self, gu, z):
        z = np.asarray(z, dtype=float)
        return float(z @ self._coeffs_for(len(z), self.phi) + gu)

    def m_required(self, eps):
        """Smallest m with 2 M L_(m+1) / (1 - S) < eps."""
        tail = np.concatenate([self.phi[::-1].cumsum()[::-1][1:], [0.0]])
        ok = np.nonzero(2 * self.M * tail / (1 - self.S) < eps)[0]
        return int(ok[0]) + 1 if len(ok) else None

    def simulate(self, n, burn=4000, seed=0, u_stream=None, z0=None):
        rng = np.random.default_rng(seed)
        total = burn + n
        u = (rng.random(total) if u_stream is None
             else np.asarray(u_stream, dtype=float)[:total])
        e = self.g(u)
        x = np.zeros(total)
        L = self.n_lag
        for t in range(total):
            lo = max(0, t - L)
            hist = x[lo:t][::-1]
            x[t] = hist @ self.phi[:len(hist)] + e[t]
        return x[burn:]


# ---------------------------------------------------------------------------
# Latent-volatility targets.  These are the ones the theory EXCLUDES, kept so
# the exclusion can be measured rather than asserted.
# ---------------------------------------------------------------------------

class GARCHTarget(Target):
    """Bounded GARCH(1,1), optionally with a GJR leverage term.

        sigma_t^2 = omega + (alpha + gamma 1{X_{t-1}<0}) X_{t-1}^2
                          + beta sigma_{t-1}^2 ,
        X_t       = sigma_t g(u_t) .

    WHY THIS IS NOT ONE OF THE NINE.  The nine targets are m-Markov in the
    observable, exactly or after a truncation whose cost `L_tail` prices.
    GARCH is different in kind: its state is the LATENT sigma_t^2, and the
    model is `prop:gauss`'s explicit exclusion.  It is here because the
    exclusion is worth measuring.

    The latent state does unroll in the observable, which is what makes the
    experiment possible at all.  Substituting the recursion into itself,

        sigma_t^2 = omega/(1-beta)
                    + sum_{j>=0} beta^j a(x_{t-1-j}) x_{t-1-j}^2 ,
        a(x)      = alpha + gamma 1{x < 0} ,

    so GARCH is a chain with complete connections in the observable, with
    GEOMETRICALLY decaying memory in the SQUARES.  That is the same shape as
    `LongMemoryTarget` with `decay="geometric"`, one level up: there the memory
    is in X, here in X^2.  Truncating at m lags therefore gives an honest q_m
    and an honest L_(m+1), and the whole apparatus applies.

    Three things still differ from the nine, and they are the experiment:

    1.  BOUNDEDNESS IS IMPOSED, NOT INHERITED.  A real GARCH is unbounded; the
        theory needs X = [-M, M].  Here the innovation is bounded, |g| <= c,
        and c is set so that |X| <= M exactly:

            sigma_max^2 = [omega + (alpha+gamma) M^2] / (1 - beta),
            c           = M / sigma_max .

        So this is a bounded process with GARCH's dependence structure, not a
        GARCH.  The Gaussian-tail obstruction is removed by hand; what remains
        to be tested is whether the DEPENDENCE is learnable.
    2.  THE MEMORY IS IN THE SQUARES, so the conditional MEAN is zero at every
        state and all the structure is in the conditional SCALE.  The pinball
        criterion sees it, an MMD between window laws does not.
    3.  S IS LARGE.  The per-lag moduli decay like beta^j, so S runs over many
        lags and `1/(1-S)` is unavailable for any realistic beta.  See
        `S_ceiling`.

    Moduli.  dq/dz_j = g(u) beta^j a(z_j) z_j / sigma(z), worst case at
    z_j = -M (so the leverage term fires) with the other lags at zero:

        ell_j = c A_j M / sqrt(omega/(1-beta) + A_j M^2),
        A_j   = beta^j (alpha + gamma) .

    Exactly `ARCHTarget`'s formula with a_i replaced by A_j, which it should
    be: ARCH(q) is GARCH with beta = 0 and q lags.

    The amplitude knob scales (alpha, gamma) together at fixed ratio, with beta
    and omega the shape.  As it grows, S saturates at

        S_ceiling = sqrt(1 - beta) / (1 - sqrt(beta)) ,

    which is 4.96 at beta = 0.85 and 2.41 at beta = 0.5: the ceiling is a
    property of the variance persistence alone.
    """
    def __init__(self, S, beta=0.85, omega=0.05, leverage=0.0, n_lag=400,
                 M=1.0, noise="truncnorm"):
        self.M, self.noise = float(M), noise
        self.beta, self.omega = float(beta), float(omega)
        self.leverage = float(leverage)
        self.n_lag = int(n_lag)
        if not (0.0 <= self.beta < 1.0):
            raise ValueError("GARCHTarget needs 0 <= beta < 1")
        if self.omega <= 0:
            raise ValueError("omega must be > 0 so the conditional scale is bounded below")
        if not (0.0 <= self.leverage <= 1.0):
            raise ValueError("leverage is gamma/alpha and must be in [0, 1]")

        bj = self.beta ** np.arange(self.n_lag, dtype=float)
        self._c0 = self.omega / (1.0 - self.beta)          # the variance floor
        w = 1.0 + self.leverage                            # (alpha + gamma)/alpha

        def S_of(lam):
            A = lam * w * bj                               # beta^j (alpha+gamma)
            smax2 = (self.omega + lam * w * self.M ** 2) / (1.0 - self.beta)
            c = self.M / np.sqrt(smax2)
            ell = c * A * self.M / np.sqrt(self._c0 + A * self.M ** 2)
            return float(ell.sum())

        lam, self.S = solve_amplitude(S_of, float(S), label="GARCHTarget")
        self.alpha = lam
        self.gamma = lam * self.leverage
        self.A = (self.alpha + self.gamma) * bj
        self.sigma_max = float(np.sqrt((self.omega + (self.alpha + self.gamma)
                                        * self.M ** 2) / (1.0 - self.beta)))
        self.sigma_min = float(np.sqrt(self._c0))
        self.c = self.M / self.sigma_max
        self._ell_full = self.c * self.A * self.M / np.sqrt(self._c0 + self.A * self.M ** 2)
        self.g = noise_fn(noise, self.c)
        self.k_true = None                                 # geometric, not finite
        lev = f", gamma/alpha={self.leverage:g}" if self.leverage else ""
        self.name = (f"GARCH(1,1), S={self.S:.3f}, beta={self.beta:g}, "
                     f"omega={self.omega:g}{lev})")

    # -- the m-truncated quantile, which is what the learner is scored against
    def _sigma_trunc(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        A = self._coeffs_for(z.shape[1], self.A)
        lev = 1.0 + self.leverage * (z < 0)                # a(x)/alpha, per entry
        return np.sqrt(self._c0 + (z ** 2 * lev) @ (A / (1.0 + self.leverage)))

    def q(self, u, z):
        u, z = self._as_uz(u, z)
        return self._sigma_trunc(z) * self.g(u)

    def step(self, gu, z):
        return float(self._sigma_trunc(np.asarray(z, dtype=float)[None, :])[0] * gu)

    def scale_contrast_bound(self):
        """sigma_max / sigma_min, the attainable conditional-scale contrast."""
        return float(self.sigma_max / self.sigma_min)

    def S_ceiling(self):
        """sup over the amplitude of S, a function of beta alone."""
        return float(np.sqrt(1.0 - self.beta) / (1.0 - np.sqrt(self.beta)))

    def simulate(self, n, burn=4000, seed=0, u_stream=None, z0=None):
        """The EXACT GARCH recursion, not the n_lag truncation.

        Same split as `ARMATarget` and `LongMemoryTarget`: paths come from the
        exact process, while q, the moduli and L_tail come from the
        m-truncation, so the truncation cost is measured rather than hidden.
        """
        rng = np.random.default_rng(seed)
        total = burn + n
        u = (rng.random(total) if u_stream is None
             else np.asarray(u_stream, dtype=float)[:total])
        if len(u) < total:
            raise ValueError(f"u_stream too short: need {total}, got {len(u)}")
        gu = self.g(u)
        s2_max = self.sigma_max ** 2
        # start at the unconditional variance, which exists because
        # alpha + gamma/2 + beta < 1 is implied by the boundedness scaling
        s2 = min(self.omega / max(1.0 - self.alpha - 0.5 * self.gamma - self.beta,
                                  1e-6), s2_max)
        x_prev = 0.0 if z0 is None else float(np.atleast_1d(z0)[0])
        out = np.empty(total)
        for t in range(total):
            a = self.alpha + (self.gamma if x_prev < 0 else 0.0)
            s2 = min(self.omega + a * x_prev ** 2 + self.beta * s2, s2_max)
            out[t] = np.sqrt(s2) * gu[t]
            x_prev = out[t]
        return out[burn:]


class EGARCHTarget(Target):
    """Exponential GARCH with leverage, in log-variance form.

        log sigma_t^2 = omega + beta log sigma_{t-1}^2
                        + alpha (|X_{t-1}|/M - kappa) + gamma X_{t-1}/M ,
        X_t           = sigma_t g(u_t) .

    ONE DELIBERATE DEPARTURE FROM TEXTBOOK EGARCH, and it has to be stated.
    Nelson's EGARCH drives the log-variance with the STANDARDISED residual
    z_{t-1} = X_{t-1}/sigma_{t-1}, not with X_{t-1}.  Standardising makes the
    recursion depend on the latent sigma path, so unrolling it gives no closed
    form in the observable, and without a closed form there is no q_m to
    compute moduli against, no theta, and nothing to compare.  Driving it with
    the observable return keeps every structural feature the experiment is
    about -- an exponential link, so the scale is positive by construction
    rather than by a floor, and a signed term, so the response to a negative
    return differs from the response to a positive one -- while unrolling
    exactly:

        log sigma_t^2 = h0 + sum_{j>=0} beta^j [alpha |x_{t-1-j}| + gamma x_{t-1-j}] / M ,
        h0            = (omega - alpha kappa) / (1 - beta) .

    So this is log-GARCH with leverage.  Call it EGARCH's structure, not
    EGARCH.

    Two identities make it the right companion to `GARCHTarget`:

        ell_j = beta^j (alpha + |gamma|) / 2 ,
        S     = (alpha + |gamma|) / (2 (1 - beta)) ,

    so the amplitude is EXACTLY invertible, alpha + |gamma| = 2 S (1 - beta),
    with no bisection; and

        sigma_max / sigma_min = exp(S) ,

    against ARCH's and GARCH's (1 - S)^{-1/2}.  That is the structural point of
    including it.  The square-root family's attainable scale contrast BLOWS UP
    at S = 1, so S < 1 is simultaneously the theory's hypothesis and the limit
    of what the family can express; the exponential family's contrast is finite
    for every S, so it separates the two and lets a target with a large, honest
    conditional-scale range be posed at any S.

    omega and kappa shift the log-variance level only; the innovation amplitude
    c = M / sigma_max absorbs them, so neither changes S, the moduli, or the
    scale contrast.  They are kept for interpretability.
    """
    def __init__(self, S, beta=0.85, omega=0.0, kappa=0.5, leverage=0.35,
                 n_lag=400, M=1.0, noise="truncnorm"):
        self.M, self.noise = float(M), noise
        self.beta, self.omega, self.kappa = float(beta), float(omega), float(kappa)
        self.leverage = float(leverage)
        self.n_lag = int(n_lag)
        if not (0.0 <= self.beta < 1.0):
            raise ValueError("EGARCHTarget needs 0 <= beta < 1")
        if not (0.0 <= self.leverage <= 1.0):
            raise ValueError("leverage is |gamma|/alpha and must be in [0, 1]")

        # S = (alpha + |gamma|)/(2(1-beta)) inverts in closed form.
        tot = 2.0 * float(S) * (1.0 - self.beta)           # alpha + |gamma|
        self.alpha = tot / (1.0 + self.leverage)
        self.gamma = -self.alpha * self.leverage           # negative: the
        self.S = float(S)                                  # leverage effect
        bj = self.beta ** np.arange(self.n_lag, dtype=float)
        self._ell_full = bj * tot / 2.0
        self._h0 = (self.omega - self.alpha * self.kappa) / (1.0 - self.beta)
        # sup and inf of the log-variance over the box, used for the scaling
        per = (self.alpha + abs(self.gamma)) / (1.0 - self.beta)
        neg = max(abs(self.gamma) - self.alpha, 0.0) / (1.0 - self.beta)
        self._h_sup = self._h0 + per
        self._h_inf = self._h0 - neg
        self.sigma_max = float(np.exp(0.5 * self._h_sup))
        self.sigma_min = float(np.exp(0.5 * self._h_inf))
        self.c = self.M / self.sigma_max
        self.g = noise_fn(noise, self.c)
        self.k_true = None
        self.name = (f"EGARCH-structure, S={self.S:.3f}, beta={self.beta:g}, "
                     f"|gamma|/alpha={self.leverage:g})")

    def _sigma_trunc(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        bj = self._coeffs_for(z.shape[1], self.beta ** np.arange(self.n_lag))
        h = self._h0 + (self.alpha * np.abs(z) + self.gamma * z) @ bj / self.M
        return np.exp(0.5 * h)

    def q(self, u, z):
        u, z = self._as_uz(u, z)
        return self._sigma_trunc(z) * self.g(u)

    def step(self, gu, z):
        return float(self._sigma_trunc(np.asarray(z, dtype=float)[None, :])[0] * gu)

    def scale_contrast_bound(self):
        """sigma_max / sigma_min = exp(S) when alpha >= |gamma|."""
        return float(self.sigma_max / self.sigma_min)

    def simulate(self, n, burn=4000, seed=0, u_stream=None, z0=None):
        """The exact log-variance recursion, not the n_lag truncation."""
        rng = np.random.default_rng(seed)
        total = burn + n
        u = (rng.random(total) if u_stream is None
             else np.asarray(u_stream, dtype=float)[:total])
        if len(u) < total:
            raise ValueError(f"u_stream too short: need {total}, got {len(u)}")
        gu = self.g(u)
        h = self._h0
        x_prev = 0.0 if z0 is None else float(np.atleast_1d(z0)[0])
        out = np.empty(total)
        for t in range(total):
            h = (self.omega + self.beta * h
                 + self.alpha * (abs(x_prev) / self.M - self.kappa)
                 + self.gamma * x_prev / self.M)
            h = min(h, self._h_sup)
            out[t] = np.exp(0.5 * h) * gu[t]
            x_prev = out[t]
        return out[burn:]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TARGETS = {
    "linear":      LinearTarget,
    "hetero":      HeteroTarget,
    "skew":        SkewTarget,
    "smoothnl":    SmoothNonlinearAR,
    "arch":        ARCHTarget,
    "oscillatory": OscDriftTarget,
    "logistic":    NoisyLogisticTarget,
    "arma":        ARMATarget,
    "longmem":     LongMemoryTarget,
    "garch":       GARCHTarget,
    "egarch":      EGARCHTarget,
}


def build_target(name, S, specifics=None, M=1.0, noise="truncnorm"):
    """Instantiate a target by key, at the requested S, with a fixed shape.

    `specifics` is the target's entry in TARGET_SPECIFICS: shape only, never S.
    `noise` is ignored by `skew`, which consumes the raw uniform.
    """
    if name not in TARGETS:
        raise ValueError(f"unknown target {name!r}; choose from {sorted(TARGETS)}")
    kw = dict(specifics or {})
    kw.setdefault("M", M)
    if name != "skew":
        kw.setdefault("noise", noise)
    else:
        kw.pop("noise", None)
    return TARGETS[name](S=float(S), **kw)
