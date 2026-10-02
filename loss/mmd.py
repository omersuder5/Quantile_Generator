"""Maximum mean discrepancy on path blocks, as an objective and as a test.

Two uses, deliberately separated:

**As a training objective** (`mmd2_torch`).  Differentiable, biased-but-smooth
V-statistic by default or the unbiased U-statistic, between a batch of real
path blocks and a batch of blocks unrolled from the generator.  This is the
criterion the earlier architecture was trained on, kept here so that the
comparison "same model class, same diagnostics, pinball against MMD" can be run
rather than asserted.  Note what it does and does not see: an MMD between
window laws is a PATH-LAW criterion with no conditional content, and the
counterexample behind the adapted distance applies to it unchanged.

**As an evaluation statistic** (`mmd2_unbiased`, `permutation_test`).  The
unbiased estimator plus a permutation test on overlapping blocks of two
independently simulated paths.  Failure to reject is not evidence of equality;
with 900 blocks of length 4 and 150 permutations the power is real but limited,
so report the p-value next to the effect sizes, never instead of them.

The bandwidth is fixed once from the pooled sample and reused for the observed
statistic and every permutation.
"""
import numpy as np
import torch

from .rbf_kernel import median_sigma, rbf_gram, median_sigma_torch, rbf_gram_torch

__all__ = ["blocks", "mmd2_unbiased", "permutation_test", "mmd2_torch"]


def blocks(path, L=4, stride=1):
    """Overlapping windows of length L, as rows of an (n, L) array."""
    path = np.asarray(path, dtype=float)
    if len(path) < L:
        raise ValueError(f"path shorter than the block length ({len(path)} < {L})")
    idx = np.arange(0, len(path) - L + 1, stride)
    return np.stack([path[i:i + L] for i in idx])


def _subsample(A, max_n, rng):
    return A if len(A) <= max_n else A[rng.choice(len(A), max_n, replace=False)]


def mmd2_unbiased(A, B, sigma=None, max_n=1200, seed=0):
    """Unbiased MMD^2 estimator with an RBF kernel.

    Returns (mmd2, sigma).  Unbiased means it can be negative when the two
    samples come from the same law; a small negative value is the expected
    behaviour under the null, not an error.
    """
    rng = np.random.default_rng(seed)
    A = _subsample(np.atleast_2d(A), max_n, rng)
    B = _subsample(np.atleast_2d(B), max_n, rng)
    if sigma is None:
        sigma = median_sigma(A, B, seed=seed)
    m, n = len(A), len(B)
    if m < 2 or n < 2:
        raise ValueError("need at least two blocks on each side")
    Kaa, Kbb, Kab = rbf_gram(A, A, sigma), rbf_gram(B, B, sigma), rbf_gram(A, B, sigma)
    np.fill_diagonal(Kaa, 0.0)
    np.fill_diagonal(Kbb, 0.0)
    val = (Kaa.sum() / (m * (m - 1)) + Kbb.sum() / (n * (n - 1))
           - 2.0 * Kab.mean())
    return float(val), float(sigma)


def permutation_test(A, B, n_perm=200, sigma=None, max_n=1200, seed=0):
    """Two-sample permutation test on the unbiased MMD^2.

    The bandwidth is fixed before permuting, so the null and the observed
    statistic use the same kernel.
    """
    rng = np.random.default_rng(seed)
    A = _subsample(np.atleast_2d(A), max_n, rng)
    B = _subsample(np.atleast_2d(B), max_n, rng)
    if sigma is None:
        sigma = median_sigma(A, B, seed=seed)
    obs, _ = mmd2_unbiased(A, B, sigma=sigma, max_n=max_n, seed=seed)
    Z = np.vstack([A, B])
    m = len(A)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(Z))
        null[i], _ = mmd2_unbiased(Z[p[:m]], Z[p[m:]], sigma=sigma,
                                   max_n=max_n, seed=i)
    return {"mmd2": obs,
            "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm)),
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "sigma": float(sigma),
            "n_a": int(len(A)), "n_b": int(len(B)), "n_perm": int(n_perm)}


# ------------------------------------------------------------- objective ---

def mmd2_torch(A, B, sigma=None, unbiased=False, generator=None):
    """Differentiable MMD^2 between two batches of blocks.

    `A` is typically the data and `B` the unrolled generator output, so only
    `B` carries gradient.  `sigma` is detached: the bandwidth is a property of
    the data scale, not a parameter to be optimised, and letting it move makes
    the objective degenerate (shrink sigma and every MMD goes to zero).

    `unbiased=True` uses the U-statistic, which is what the evaluation side
    reports but is noisier as a training signal; the default V-statistic is
    biased upward by O(1/n) and smoother.
    """
    A = A.reshape(len(A), -1)
    B = B.reshape(len(B), -1)
    if sigma is None:
        sigma = median_sigma_torch(A.detach(), B.detach(), generator=generator)
    sigma = sigma.detach() if torch.is_tensor(sigma) else torch.tensor(float(sigma))
    Kaa = rbf_gram_torch(A, A, sigma)
    Kbb = rbf_gram_torch(B, B, sigma)
    Kab = rbf_gram_torch(A, B, sigma)
    if not unbiased:
        return Kaa.mean() + Kbb.mean() - 2.0 * Kab.mean()
    m, n = len(A), len(B)
    saa = Kaa.sum() - Kaa.diagonal().sum()
    sbb = Kbb.sum() - Kbb.diagonal().sum()
    return saa / (m * (m - 1)) + sbb / (n * (n - 1)) - 2.0 * Kab.mean()
