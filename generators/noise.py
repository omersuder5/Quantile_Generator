"""Innovation transforms.

An innovation transform is a continuous nondecreasing  g : [0,1] -> [-c, c].
Composed with a uniform draw it produces the target's innovation, and because
it is nondecreasing it is the quantile function of the law it induces, which is
what makes the conditional probability integral transform of the target exact
rather than approximate.

Every target in `synthetic_generators` draws its randomness through one of
these, with the single exception of `SkewTarget`, whose transform depends on
the state and therefore consumes the raw uniform itself (registered here as
"raw" so the simulation loop does not have to special-case it).
"""
import numpy as np
from scipy.stats import norm

__all__ = ["g_uniform", "g_truncnorm", "g_raw", "NOISE", "noise_fn"]


def g_uniform(u, c):
    """Uniform on [-c, c].  Flat density, so the conditional quantile is affine
    in u and the level parameterisation is as easy as it can be."""
    return c * (2.0 * np.asarray(u, dtype=float) - 1.0)


def g_truncnorm(u, c, T=2.0):
    """Normal truncated at +/- T standard deviations, rescaled to [-c, c].

    This is the default.  It has a genuine tail shape, so dq/du at u = 0.001 is
    several times its value at the median, which is exactly the level-resolution
    problem the tail investigation was about; `tail_frac` in the pinball level
    sampler exists to put training mass there.
    """
    lo, hi = norm.cdf(-T), norm.cdf(T)
    return c * norm.ppf(lo + np.asarray(u, dtype=float) * (hi - lo)) / T


def g_raw(u, c):
    """Identity on the uniform.  For targets whose innovation transform depends
    on the state and is therefore applied inside q itself."""
    return np.asarray(u, dtype=float)


NOISE = {"uniform": g_uniform, "truncnorm": g_truncnorm, "raw": g_raw}


def noise_fn(name, c):
    """Bind an amplitude, returning g : [0,1] -> [-c, c]."""
    if name not in NOISE:
        raise ValueError(f"unknown noise {name!r}; choose from {sorted(NOISE)}")
    f = NOISE[name]
    return lambda u: f(u, c)
