"""The null model.

A Gaussian AR(m) fitted by least squares to the SAME training path and
simulated on its own innovation stream.  It matches the mean, the variance and
the linear autocorrelation by construction, and it has a constant conditional
spread, also by construction.  So it is the model that passes every standard
second-order diagnostic and gets the conditional structure wrong, and anything
the quantile generator gets right that this does not is a genuinely
non-Gaussian, non-linear feature rather than a well-fitted mean.

It is also the model class the earlier MMD-trained architecture was effectively
confined to, which is why its ACF of squares reproduces the near-zero,
occasionally negative values recorded there.

Expect it to WIN on the marginal W_1 for an affine, near-Gaussian target: on a
correctly specified problem a correctly specified model should win, and a
nonparametric method that beat it there would be suspicious.
"""
import numpy as np

__all__ = ["MatchedGaussianAR"]


class MatchedGaussianAR:
    """AR(p) with Gaussian innovations, fitted by OLS."""

    def __init__(self, p=1):
        self.p = int(p)
        self.b0 = None
        self.phi = None
        self.sigma = None

    def fit(self, path):
        p = self.p
        path = np.asarray(path, dtype=float)
        T = len(path)
        if T <= p + 1:
            raise ValueError("path too short to fit")
        Z = np.stack([path[p - j - 1: T - j - 1] for j in range(p)], axis=1)
        y = path[p:]
        Zc = np.hstack([np.ones((len(y), 1)), Z])
        beta, *_ = np.linalg.lstsq(Zc, y, rcond=None)
        resid = y - Zc @ beta
        self.b0, self.phi, self.sigma = float(beta[0]), beta[1:], float(resid.std())
        return self

    def simulate(self, n, burn=2000, seed=0, u_stream=None, z0=None):
        """Signature matches the other generators so `simulate_paths.simulate`
        accepts it.  `u_stream` is ignored: the baseline draws Gaussian
        innovations of its own, which is the point of it being a baseline.
        """
        if self.phi is None:
            raise RuntimeError("fit() first")
        rng = np.random.default_rng(seed)
        p = self.p
        out = np.zeros(burn + n)
        for t in range(p, burn + n):
            out[t] = (self.b0 + self.phi @ out[t - p:t][::-1]
                      + self.sigma * rng.standard_normal())
        return out[burn:]

    def describe(self):
        return {"p": self.p, "b0": self.b0,
                "phi": None if self.phi is None else self.phi.tolist(),
                "sigma": self.sigma}
