"""The randomised pinball criterion, and the level sampler it draws from.

For a level u in (0,1) the pinball loss

    rho_u(x, p) = max( u (x - p), (u - 1)(x - p) )

is minimised over p by the u-quantile of the law of x, so it is a strictly
consistent scoring function for that quantile.  Averaging it over u ~ U(0,1)
gives, up to the factor 1/2, the continuous ranked probability score of the
conditional law: for any weight w(u) > 0 the weighted average is still proper
pointwise in u, which is why reweighting the levels is legitimate and is not a
change of estimand.  That is the whole content of `tail_frac` below.

This is the point on which the construction differs from a path-law criterion
such as an MMD between windows.  The pinball criterion scores the CONDITIONAL
law, pointwise in the level, with one observation per conditioning window; an
MMD between window laws has no conditional content at all.  The empirical
consequence is in the ARCH runs: a generator trained this way reproduces the
autocorrelation of squares, and the earlier MMD-trained architecture, with
five different fixes attempted, did not.
"""
import numpy as np
import torch

__all__ = ["pinball", "pinball_weighted", "sample_levels", "crps_from_pinball"]


def pinball(pred, x, u):
    """Mean pinball loss.  `pred`, `x`, `u` are 1-D tensors of equal length."""
    e = x - pred
    return torch.maximum(u * e, (u - 1.0) * e).mean()


def pinball_weighted(pred, x, u, w):
    """Pinball with a positive level weight.  Still proper pointwise in u, so
    the minimiser is unchanged; only the weighting of the levels moves."""
    e = x - pred
    return (w * torch.maximum(u * e, (u - 1.0) * e)).mean()


def sample_levels(n, gen, tail_frac=0.0, edge=0.02, u_min=1e-4):
    """Draw n quantile levels.

    With `tail_frac` = 0 this is U(0,1).  Otherwise a fraction `tail_frac` of
    the draws is replaced by a log-uniform draw in (u_min, edge), reflected to
    the upper tail half the time.  The point is resolution, not estimand: for a
    truncated-normal innovation dq/du at u = 0.001 is about seven times its
    value at the median, so uniform levels put almost no training signal where
    the conditional quantile is steepest, and the generated tails come out
    short.

    The signature is keyword-safe on purpose.  An earlier version was called
    positionally with an extra argument, which silently bound a boolean into
    `u_min`: True disabled the tail mixture entirely and False gave log(0) and
    NaNs in a fifth of the batch.  Call this with keywords.
    """
    u = torch.rand(n, generator=gen)
    if tail_frac <= 0.0:
        return u
    if not (0.0 < u_min < edge < 0.5):
        raise ValueError(f"need 0 < u_min < edge < 0.5, got {u_min}, {edge}")
    m = torch.rand(n, generator=gen) < tail_frac
    s = torch.rand(n, generator=gen)
    lo, hi_ = float(np.log(u_min)), float(np.log(edge))
    t = torch.exp(lo + (hi_ - lo) * s)
    upper = torch.rand(n, generator=gen) < 0.5
    return torch.where(m, torch.where(upper, 1.0 - t, t), u)


def crps_from_pinball(mean_pinball):
    """CRPS = 2 * E_u[pinball].  Recorded so the training loss can be quoted on
    the scale the forecasting literature uses."""
    return 2.0 * float(mean_pinball)
