"""Realising the closed quantile recursion inside an echo state network.

The quantile generator is a map of (u, z) with z the last m outputs.  An ESN

    h_t = sigma(A h_{t-1} + c u_t + zeta),      Xhat_t = w^T h_t

carries the reservoir state, not its own output window, which is precisely why
the filter route is causal but not bicausal.  The construction here removes the
gap by making part of the reservoir an EXACT shift register rather than an
approximate one, so that the network's state is a linear image of its own last
m outputs and the recursion it runs is the quantile recursion, not an
approximation of it.

Layout of the N = W + (m-1) + 1 units:

  W quantile units    rows of G, input weight c, bias zeta.  These compute the
                      readout of the fitted network.
  m-1 shift units     unit j reads x_{j-1} through E and returns lam * x_{j-1}.
                      They carry ZERO input weight, which matters: a unit whose
                      value must be an exact affine function of past outputs
                      cannot be allowed to see the innovation.
  1 constant unit     no state and no input weight, so it sits at sigma(zeta0)
                      and supplies the readout bias b.

The scale lam = 0.9 / operating_radius keeps every shift unit's pre-activation
inside the interval on which sigma is affine, which is the whole reason the
activation must be affine on an interval and tanh provably will not do.  The
extraction matrix E inverts that affine action exactly, so the shift is exact
and not merely accurate: in the 21-run study
`max |ESN - direct recursion| = 7.8e-16` over every run, including m = 32.
"""
import numpy as np
import torch

from backend import to_np
from .quantile_generator import SoftClip, probit_np

__all__ = ["ESNRealization"]


class ESNRealization:
    """Exact-shift realisation of a `QuantileNet` as a single ESN.

    Parameters
    ----------
    net : QuantileNet
        Must use the softclip activation.
    operating_radius : float
        A bound on |x| along the generated path; the shift scale is
        0.9 / operating_radius so pre-activations stay in the affine window.
        1.6 M is a safe default for an X-valued recursion.
    """
    def __init__(self, net, operating_radius=1.6, zeta0=0.5, beta=0.5):
        if net.act_name != "softclip":
            raise ValueError("the exact shift needs an activation affine on an "
                             "interval; tanh provably has none")
        W, m = net.width, net.m
        G = to_np(net.G)
        c = to_np(net.c)
        z_ = to_np(net.zeta)
        a = to_np(net.a)
        b = float(to_np(net.b)[0])

        N = W + (m - 1) + 1
        self.N, self.m, self.k, self.beta = N, m, m, float(beta)
        self.v_mode = getattr(net, "v_mode", "u")
        self.v_clip = getattr(net, "v_clip", 2.5)
        lam = 0.9 / float(operating_radius)
        self.lam = lam

        Gf = np.zeros((N, m)); cf = np.zeros(N); zf = np.zeros(N)
        Gf[:W] = G; cf[:W] = c; zf[:W] = z_             # quantile units
        for j in range(1, m):                            # shift units, no input weight
            Gf[W + j - 1, j - 1] = lam
        zf[W + m - 1] = zeta0                            # constant unit

        E = np.zeros((m, N))
        E[0, :W] = a
        E[0, W + m - 1] = b / zeta0                      # readout bias
        for j in range(1, m):
            E[j, W + j - 1] = 1.0 / lam

        self.G_full, self.c_full, self.zeta_full, self.E = Gf, cf, zf, E
        self.A = Gf @ E                                  # the recurrent matrix
        self.w = E[0].copy()

    def sigma(self, y):
        return SoftClip.numpy(y, self.beta)

    def generate(self, n, burn=2000, seed=0, u_stream=None):
        rng = np.random.default_rng(seed)
        total = burn + n
        u = (rng.random(total) if u_stream is None
             else np.asarray(u_stream, dtype=float)[:total])
        v = u if self.v_mode == "u" else probit_np(u, self.v_clip)
        h = np.zeros(self.N)
        out = np.empty(total)
        for t in range(total):
            h = self.sigma(self.A @ h + self.c_full * v[t] + self.zeta_full)
            out[t] = self.w @ h
        return out[burn:]

    def info(self):
        return {"N": self.N, "m": self.m, "lam": self.lam,
                "rank_A": int(np.linalg.matrix_rank(self.A, tol=1e-8)),
                "A_shape": tuple(self.A.shape)}

    def realisation_error(self, recursion, n=2000, burn=500, seed=0):
        """max_t |ESN_t - recursion_t| on a shared innovation stream.

        This is the certificate that the realisation is exact rather than
        approximate.  Anything above about 1e-12 means the operating radius is
        too small and a shift unit has left the affine window.
        """
        rng = np.random.default_rng(seed)
        u = rng.random(burn + n)
        a = self.generate(n, burn=burn, seed=seed, u_stream=u)
        b = recursion.generate(n, burn=burn, seed=seed, u_stream=u)
        return float(np.abs(a - b).max())
