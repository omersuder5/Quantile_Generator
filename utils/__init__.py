"""One module per analysis.

hypotheses       S_m, per-lag Lipschitz, theta on three domains
repair           is the repair lemma doing anything on this fit
pathwise         the bound under the synchronous coupling, and its slack
lyapunov         the top exponent, and why it, not S, is the boundary
autocorrelation  the ACF and the ACF of squares
marginal         W_1 and the extreme quantiles
conditional      the conditional sd profile and the conditional W_1
pricing          European, Bermudan, and the early-exercise premium
baselines        the matched Gaussian AR null
plotting         figures, with a fixed three-series colour assignment
reporting        results.txt, summary.csv, COMPARISON.md
"""
from .hypotheses import (verify_assumptions, lipschitz_empirical, theta_hat,
                         theta_profile, theta_trimmed, hypothesis_report)
from .repair import (crossing_report, du_min, sup_q_box, invariant_radius,
                     repair_report)
from .pathwise import pathwise_report, synchronisation_report
from .lyapunov import lyapunov, lyapunov_decomposition
from .autocorrelation import acf, acf_squares, acf_report
from .marginal import w1_marginal, quantile_table, marginal_report
from .conditional import (default_bins, conditional_sd_profile,
                          conditional_w1_profile, conditional_report)
from .pricing import european_put, bermudan_put, pricing_report
from .baselines import MatchedGaussianAR
