"""The learned quantile generator.

The object the theory is about is a continuous map

    qhat : [0,1] x X^m -> X

closed into the recursion  xhat_t = qhat(u_t ; xhat_{t-1}, ..., xhat_{t-m})
driven by i.i.d. uniforms.  No window of the target, no seed and no burn-in are
required; the burn-in used below is purely numerical, to let the unique
solution be reached from an arbitrary start.

Three things are deliberately NOT imposed during fitting, because the repair
lemma makes each of them free after the fact:

  R   the increasing rearrangement in u, so qhat(.;z) is a genuine quantile
      function.  Nonexpansive in the uniform norm, so theta cannot grow.
  T   a Lipschitz regularisation in z, which supplies the contraction the
      uniqueness argument needs.
  Pi  the projection onto X, so the recursion is X-valued.

`repair_report` measures whether any of them is doing anything on a given fit;
across the 21-run study eleven of twenty-one fits needed none of the three.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

__all__ = ["SoftClip", "ACTS", "QuantileNet", "ClosedQuantileRecursion",
           "probit_np", "rearrange_levels", "project_to_X", "RepairedRecursion"]


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

class SoftClip(nn.Module):
    """Identity on [-1,1], C^1, bounded by 1+beta.

    Affine on an interval, which is exactly what the exact-shift realisation
    needs and what tanh provably does not have.  Using it is the one place the
    ESN corollary constrains the architecture.
    """
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = float(beta)

    def forward(self, y):
        b = self.beta
        a = y.abs()
        sat = 1.0 + b * (1.0 - torch.exp(-(a - 1.0).clamp(min=0.0) / b))
        return torch.where(a <= 1.0, y, torch.sign(y) * sat)

    @staticmethod
    def numpy(y, beta=0.5):
        a = np.abs(y)
        sat = 1.0 + beta * (1.0 - np.exp(-np.maximum(a - 1.0, 0.0) / beta))
        return np.where(a <= 1.0, y, np.sign(y) * sat)


#            module      sup |sigma'|
ACTS = {"tanh": (nn.Tanh(), 1.0), "softclip": (SoftClip(), 1.0)}


def probit_np(u, v_clip=2.5):
    return np.clip(norm.ppf(np.clip(u, 1e-7, 1 - 1e-7)), -v_clip, v_clip)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class QuantileNet(nn.Module):
    """qhat(u, z) = a^T sigma(G z + c v(u) + zeta) + b.

    One hidden layer, the quantile level entering as an extra input coordinate.
    That shape is not incidental: it is the form the exact-shift block realises
    inside an echo state network whose state is its own last m outputs, so the
    network fitted here IS the ESN readout, up to the shift register that
    `esn_realization` adds.

    v_mode:
      "u"       v = u, the level itself.  The default and what the study used.
      "probit"  v = Phi^{-1}(u), clipped.  A composition with a fixed increasing
                bijection, so the per-lag moduli, uniqueness, bicausality, the
                adapted bound and the exact-shift realisation all carry over
                verbatim.  It flattens dq/dv in the tails by about a factor of
                17 for a truncated-normal innovation, which helps level
                resolution and hurts nothing in the theory; it is off by default
                because it is a change to the estimator's inputs rather than to
                the estimator, and the tail problem turned out to be fixable
                without it.
    """
    def __init__(self, m, width=256, act="softclip", seed=0, v_mode="u",
                 v_clip=2.5):
        super().__init__()
        if act not in ACTS:
            raise ValueError(f"unknown activation {act!r}; choose from {sorted(ACTS)}")
        torch.manual_seed(seed)
        self.m = int(m)
        self.k = int(m)                      # alias: torch code elsewhere reads .k
        self.width, self.act_name = int(width), act
        self.v_mode, self.v_clip = v_mode, float(v_clip)
        self.sigma, self.dsup = ACTS[act]
        self.G = nn.Parameter(torch.randn(width, self.m) / np.sqrt(self.m))
        self.c = nn.Parameter(torch.randn(width))
        self.zeta = nn.Parameter(0.1 * torch.randn(width))
        self.a = nn.Parameter(torch.randn(width) / np.sqrt(width))
        self.b = nn.Parameter(torch.zeros(1))

    def phi(self, u):
        if self.v_mode == "u":
            return u
        return torch.clamp(torch.special.ndtri(u.clamp(1e-7, 1 - 1e-7)),
                           -self.v_clip, self.v_clip)

    def forward(self, u, z):
        v = self.phi(u)
        h = self.sigma(z @ self.G.T + self.c[None, :] * v[:, None] + self.zeta[None, :])
        return h @ self.a + self.b

    def lipschitz_bound(self):
        """Analytic upper bound on sum_i Lip_i(qhat): sup|sigma'| sum_j |a_j| ||G_j||_1.
        Usually very conservative; the gradient estimate is what to report."""
        return self.dsup * (self.a.abs() * self.G.abs().sum(dim=1)).sum()

    def rescale_G(self, factor):
        with torch.no_grad():
            self.G.mul_(factor)

    @torch.no_grad()
    def q_np(self, u, z):
        u = torch.as_tensor(np.atleast_1d(u), dtype=torch.float32)
        z = torch.as_tensor(np.atleast_2d(z), dtype=torch.float32)
        return self.forward(u, z).numpy()


# ---------------------------------------------------------------------------
# The closed recursion
# ---------------------------------------------------------------------------

class ClosedQuantileRecursion:
    """xhat_t = qhat(u_t; xhat_{t-1}, ..., xhat_{t-m}), in numpy.

    Detached from the torch graph on construction, so generation is cheap and
    the object is picklable.
    """
    def __init__(self, net):
        self.net = net
        self.m = int(net.m)
        self.k = self.m
        self.G = net.G.detach().numpy().astype(np.float64)
        self.c = net.c.detach().numpy().astype(np.float64)
        self.zeta = net.zeta.detach().numpy().astype(np.float64)
        self.a = net.a.detach().numpy().astype(np.float64)
        self.b = float(net.b.detach().numpy()[0])
        self.beta = getattr(net.sigma, "beta", None)
        self.act = net.act_name
        self.v_mode = getattr(net, "v_mode", "u")
        self.v_clip = getattr(net, "v_clip", 2.5)

    def _sigma(self, y):
        if self.act == "tanh":
            return np.tanh(y)
        return SoftClip.numpy(y, self.beta)

    def q(self, u, z):
        """One step of the learned quantile map."""
        v = u if self.v_mode == "u" else probit_np(u, self.v_clip)
        return float(self._sigma(self.G @ z + self.c * v + self.zeta) @ self.a + self.b)

    def q_vec(self, u, Z):
        """Vectorised over a batch of (u, z) pairs."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        v = u if self.v_mode == "u" else probit_np(u, self.v_clip)
        H = self._sigma(Z @ self.G.T + self.c[None, :] * v[:, None] + self.zeta[None, :])
        return H @ self.a + self.b

    def generate(self, n, burn=2000, seed=0, u_stream=None, z0=None):
        rng = np.random.default_rng(seed)
        total = burn + n
        u = (rng.random(total) if u_stream is None
             else np.asarray(u_stream, dtype=float)[:total])
        if len(u) < total:
            raise ValueError(f"u_stream too short: need {total}, got {len(u)}")
        z = np.zeros(self.m) if z0 is None else np.asarray(z0, dtype=float).copy()
        if len(z) < self.m:
            z = np.concatenate([z, np.zeros(self.m - len(z))])
        out = np.empty(total)
        for t in range(total):
            out[t] = self.q(u[t], z)
            z[1:] = z[:-1]
            z[0] = out[t]
        return out[burn:]


# ---------------------------------------------------------------------------
# The repair operators of the repair lemma
# ---------------------------------------------------------------------------

def rearrange_levels(vals):
    """Increasing rearrangement along the level axis.

    `vals` has shape (n_u, n_z): the map evaluated on a grid of levels for each
    state.  Sorting each column is the increasing rearrangement R, which is
    nonexpansive in the uniform norm, so it can only reduce |qhat - q|
    (Chernozhukov, Fernandez-Val and Galichon).  Returns the sorted array.
    """
    return np.sort(np.asarray(vals, dtype=float), axis=0)


def project_to_X(x, M=1.0):
    """Pi_X: the projection onto the state space.  1-Lipschitz, fixes q."""
    return np.clip(x, -M, M)


class RepairedRecursion(ClosedQuantileRecursion):
    """The closed recursion with the repair operators switched on.

    `project=True` applies Pi_X at every step, which makes the recursion
    X-valued even when the fitted map is not (the oscillatory target in the
    study overshot M by 1.9 per cent).

    `monotone_grid=n` applies R at every step on an n-point level grid: the map
    is evaluated on the grid at the current state, sorted, and the value at the
    requested level is read off by interpolation.  This costs a factor of n per
    step, so it is off by default and is meant for the runs where
    `repair_report` says the crossing fraction is nonzero.
    """
    def __init__(self, net, M=1.0, project=True, monotone_grid=0):
        super().__init__(net)
        self.M = float(M)
        self.project = bool(project)
        self.monotone_grid = int(monotone_grid)
        if self.monotone_grid:
            self._ug = np.linspace(1e-4, 1 - 1e-4, self.monotone_grid)

    def q(self, u, z):
        if self.monotone_grid:
            vals = np.sort(self.q_vec(self._ug, np.repeat(z[None, :],
                                                          self.monotone_grid, 0)))
            x = float(np.interp(u, self._ug, vals))
        else:
            x = super().q(u, z)
        return float(np.clip(x, -self.M, self.M)) if self.project else x
