"""Gaussian RBF kernel with the median heuristic, in numpy and in torch.

    k(x, y) = exp( - ||x - y||^2 / (2 sigma^2) )

The RBF kernel on R^d is characteristic, and integrally strictly positive
definite, so the MMD it induces metrises weak convergence on laws over R^d.
Applied to windows of length L of a path, viewed as vectors in R^L, it
therefore metrises weak convergence of the L-dimensional finite-dimensional
distributions.  Two caveats worth keeping in mind when reading a block-MMD
number: it sees the joint law of L consecutive states but is blind to the time
ordering within the window, and no finite L sees the whole path law.

The signature kernel is the natural object on streams and respects the order,
but its sample-size requirements made it impractical here; a plain RBF on
blocks is what this package uses.

**Median heuristic.**  sigma^2 = median of the squared pairwise distances over
the pooled sample, which puts the kernel on the scale of the data.  Computed
once on a subsample and then held FIXED for every evaluation that is to be
compared, including every permutation of a permutation test: recomputing it per
permutation would make the null and the observed statistic use different
kernels and the test invalid.
"""
import numpy as np
import torch

__all__ = ["median_sigma", "rbf_gram", "median_sigma_torch", "rbf_gram_torch"]


def _sqdist(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)


def median_sigma(*samples, max_n=500, seed=0):
    """sigma from the median of squared pairwise distances on the pooled sample.

    Returns sigma, not sigma^2.  Degenerate input (all points equal) falls back
    to 1.0 rather than producing a zero bandwidth.
    """
    rng = np.random.default_rng(seed)
    Z = np.vstack([np.atleast_2d(np.asarray(s, dtype=float)) for s in samples])
    if len(Z) > max_n:
        Z = Z[rng.choice(len(Z), max_n, replace=False)]
    d2 = _sqdist(Z, Z)
    pos = d2[d2 > 0]
    if pos.size == 0:
        return 1.0
    med = float(np.median(pos))
    return float(np.sqrt(med / 2.0)) if med > 0 else 1.0


def rbf_gram(A, B, sigma):
    """exp(-||a-b||^2 / (2 sigma^2)) for every pair."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    return np.exp(-_sqdist(A, B) / (2.0 * float(sigma) ** 2))


# ----------------------------------------------------------------- torch ---

def median_sigma_torch(*samples, max_n=500, generator=None):
    Z = torch.cat([s.reshape(len(s), -1) for s in samples], dim=0)
    if len(Z) > max_n:
        idx = torch.randperm(len(Z), generator=generator)[:max_n]
        Z = Z[idx]
    d2 = torch.cdist(Z, Z) ** 2
    pos = d2[d2 > 0]
    if pos.numel() == 0:
        return torch.tensor(1.0)
    med = pos.median()
    return torch.sqrt(med / 2.0).detach()


def rbf_gram_torch(A, B, sigma):
    """Differentiable in A and B; `sigma` should be detached (see the module
    docstring: the bandwidth is data-derived but held fixed)."""
    A = A.reshape(len(A), -1)
    B = B.reshape(len(B), -1)
    return torch.exp(-(torch.cdist(A, B) ** 2) / (2.0 * sigma ** 2))
