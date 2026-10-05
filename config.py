"""The experiment configuration.

Pull this file, edit it, and run `python quantile_learning.py`.  Nothing in the
notebook needs changing.

HOW THE GRID IS BUILT
---------------------
`TARGETS`, `M`, `S_TARGET` and every value inside `TRAIN`, `NET`, `DATA` and
`EVAL` are LISTS, and the sweep is their full Cartesian product.  A list of
length one is a fixed setting, so you turn an axis off by shortening it.
`TARGET_SPECIFICS` is NOT swept: it fixes each target's shape.

That split is the whole design.  The amplitude of every target is solved for so
that sum_i ell_i equals the requested `S_TARGET` exactly, so S is an axis in its
own right and "same S, different mechanism" is a fair comparison.  The
specifics say what KIND of dependence the target has (a ridge scale, an even
scale, a varying skewness, an oscillation count, a decay exponent); S says how
strong it is.

`M` is the fitting arity: the memory you PRESUME the target has.  It is never
raised to the target's true order.  Fitting below the truth is a legitimate
experiment and is priced by L_(m+1) = sum_{i>m} ell_i, uniformly for finite and
infinite memory: a 2-lag linear target fitted at m = 1 is misspecified in
exactly the way, and by exactly the term, that a long-memory target fitted at
m = 4 is.

NOTEBOOK vs SCRIPT
------------------
`quantile_learning.ipynb` takes the FIRST element of every list and runs once.
`quantile_learning.py` runs the whole grid into
`reports/run_<timestamp>/<target>_<varying axes>/`.
"""

# ---------------------------------------------------------------------------
# 1. What to fit, at what arity, at what contraction constant.  All vectors.
# ---------------------------------------------------------------------------

TARGETS = [
    "linear",        # affine mean, constant scale: the control
    "hetero",        # conditional scale a ridge function of z_1
    "skew",          # conditional SKEWNESS varies; mean and variance do not
    "smoothnl",      # nonlinear conditional mean, homoskedastic
    "arch",          # conditional scale even in each coordinate: clustering
    "oscillatory",   # high-frequency drift at fixed S
    "logistic",      # chaotic drift; S >= 1 is allowed here only
    "arma",          # infinite memory, geometric moduli
    "longmem",       # infinite memory, polynomial moduli
]

# The learner's lag dimension, and what you presume the target's memory to be.
M = [2, 4, 8]

# The target's contraction constant, matched exactly.  Keep below 1 except for
# the noise-budget experiment on `logistic` (see NOTE at the foot of this file).
S_TARGET = [0.75]


# ---------------------------------------------------------------------------
# 2. Target shapes.  FIXED, never swept.
#    Each entry says what kind of dependence the target has; S says how strong.
# ---------------------------------------------------------------------------

TARGET_SPECIFICS = {
    # profile = the relative weight of each lag; rescaled so sum ell_i = S.
    "linear":      dict(profile=[0.45, 0.20]),

    # s(z) = s0 + s1 tanh(kappa z_1); the amplitude scales the profile and
    # kappa together, so both routes from state to conditional law shrink
    # with S.
    "hetero":      dict(profile=[0.70, 0.30], s0=0.45, s1=0.45, kappa=1.2),

    # lam0 sets how far the conditional skewness can swing; |lam0| < 1 keeps
    # the map increasing in u.
    "skew":        dict(profile=[0.70, 0.30], lam0=0.7, kappa=1.2),

    # h = the share of the range the tanh mean may use; small h saturates it
    # and makes the nonlinearity strong at fixed S.
    "smoothnl":    dict(profile=[0.47, 0.53], h=0.55),

    # omega is the volatility floor.  Note sigma_max/sigma_min <= (1-S)^-1/2:
    # the attainable scale contrast and the amplification are one parameter.
    "arch":        dict(profile=[0.667, 0.333], omega=0.2),

    # omega_osc = how many oscillations the drift makes across the state
    # space.  It does NOT change S, which is the point: it separates the
    # approximation difficulty from the amplification constant.
    "oscillatory": dict(profile=[0.70, 0.30], omega_osc=6.0, s0=0.40,
                        s1=0.10, kappa=1.2),

    # r is the logistic parameter; 3.7 is chaotic.  ell_1 = 4(1-rho)
    # independently of r, so S fixes rho = 1 - S/4 and r stays free.
    "logistic":    dict(r=3.7),

    # the ARMA shape; the amplitude scales phi and theta together, keeping it
    # a genuine ARMA so the exact (p+q)-state simulation still applies.
    "arma":        dict(phi=[0.20], theta=[0.25], n_pi=400),

    # decay="polynomial" gives L_(m+1) ~ m^-alpha, the hard case;
    # "geometric" (with r) and "fractional" (with d) are the alternatives.
    "longmem":     dict(decay="polynomial", n_lag=4000, alpha=0.5),
}


# ---------------------------------------------------------------------------
# 3. The data.  All vectors.
# ---------------------------------------------------------------------------

DATA = dict(
    T_train=[60_000],     # length of the training path
    burn=[2_000],         # burn-in discarded before the path is kept
    seed_path=[1],        # seed of the training path
    M_state=[1.0],        # the state space is [-M_state, M_state]
    noise=["truncnorm"],  # "truncnorm" or "uniform"; `skew` ignores it
)


# ---------------------------------------------------------------------------
# 4. The network.  All vectors.
# ---------------------------------------------------------------------------

NET = dict(
    width=[256],
    act=["softclip"],     # softclip is affine on an interval, which the exact
                          # -shift ESN realisation needs; tanh provably is not
    seed_net=[0],
    v_mode=["u"],         # "u" or "probit"; see QuantileNet for why probit is
                          # legitimate and why it is off
    v_clip=[2.5],
)


# ---------------------------------------------------------------------------
# 5. Training.  All vectors.
# ---------------------------------------------------------------------------

TRAIN = dict(
    objective=["pinball"],   # "pinball" (proper, conditional) or "mmd"
                             # (path-law, for the comparison)
    epochs=[300],            # a COUNT: the rate anneals to zero over exactly
                             # this many, and there is no early stopping
    batch=[512],
    lr=[3e-3],

    # --- the learning rate ----------------------------------------------
    # A cosine anneal from `lr` to zero over `epochs`, returning the LAST
    # iterate.  No knob: no schedule choice, no early stopping, no best-loss
    # restore.  That is measured, not lazy.  With the loss held constant to 0.6
    # per cent, variants moved W1/sd over a factor of twenty (0.0156 to 0.3098,
    # linear at S=0.75), because anything selecting an iterate BY LOSS leaves a
    # bias in the conditional median that 1/(1-S_m) then multiplies.  Do not
    # re-add them; the table is in claude/quantile-generator-package.md.
    # If a run feels slow, lower `epochs` or `T_train`.
    # --------------------------------------------------------------------
    lip_mode=["penalty"],    # "penalty" | "project" | "none".  Forced to
                             # "none" automatically when the target has S >= 1
    pen_region=["hood"],     # "hood" (a neighbourhood of the data) | "box" | "data"
    pen_sigma=[0.15],        # the width of that neighbourhood
    L_max=[0.95],            # the cap the penalty enforces on sum_i Lip_i
    lam=[1.0],               # the penalty weight
    tail_frac=[0.2],         # share of training levels drawn log-uniform in
                             # the tails; 0 is plain U(0,1)
    edge=[0.02],             # the tail region is (u_min, edge) and its mirror
    u_min=[1e-4],
    # MMD objective only:
    mmd_block=[4],           # path-block length
    mmd_batch=[256],         # chains unrolled per step
    mmd_burn=[32],           # unroll burn-in before a block is collected
)


# ---------------------------------------------------------------------------
# 6. Evaluation.  All vectors.
# ---------------------------------------------------------------------------

EVAL = dict(
    T_eval=[40_000],       # length of each evaluation path
    seed_eval=[99],        # target path; generated uses +1, baseline +2
    T_shared=[20_000],     # length of the shared-innovation comparison
    seed_shared=[7],
    n_starts=[24],         # initial states for the synchronisation spread
    n_spread_steps=[80],
    block_len=[4],         # block length for the MMD permutation test
    block_stride=[1],      # 1 = every overlapping window, then a random
                           # subsample; the earlier study used 3
    n_perm=[150],
    mmd_max_n=[900],
    n_bins=[18],           # bins for the conditional profiles
    strike=[0.10],         # the put strike
    n_exercise=[5],        # Bermudan exercise dates
)


# ---------------------------------------------------------------------------
# 7. Which diagnostics to run.  Flags, not vectors.
# ---------------------------------------------------------------------------

DIAGNOSTICS = dict(
    verify_assumptions=True,   # finite-difference check of the moduli
    hypotheses=True,           # S_m, Lipschitz, theta
    repair=True,               # monotonicity, invariance, invariant radius
    pathwise=True,             # the bound under the shared stream
    synchronisation=True,      # spread across initial states
    lyapunov=True,
    autocorrelation=True,
    marginal=True,
    conditional=True,
    mmd_test=True,             # block-MMD permutation test
    pricing=True,              # European, Bermudan, premium
    esn=True,                  # the exact-shift realisation check
    baseline=True,             # the matched Gaussian AR null
    figures=True,
    save_net=True,             # net.pt per run, so a run can be re-analysed
)


# ---------------------------------------------------------------------------
# 8. Output
# ---------------------------------------------------------------------------

REPORTS_DIR = "reports"
RESUME = True          # skip a configuration whose results.json already exists
TORCH_THREADS = 2      # CPU threads; ignored on a GPU

# ---------------------------------------------------------------------------
# 9. Where the tensors live.  Scalars, not swept.
# ---------------------------------------------------------------------------
# DEVICE  "cpu" | "auto" | "cuda" | "cuda:0" | "mps"
#         The default is "cpu", deliberately, and on a Mac it should stay that
#         way.  The Lipschitz penalty differentiates a gradient (`grad_lip`
#         calls autograd.grad with create_graph=True, then backpropagates
#         through it), and that double backward through the MPS kernels can
#         abort the process rather than raise: in a notebook the kernel dies
#         and Jupyter reports "notebook controller is DISPOSED" with no Python
#         traceback at all.  "auto" picks MPS on a Mac, so "auto" plus
#         lip_mode "penalty" is the combination to avoid there.  Nothing is
#         lost by the CPU here: generation and every diagnostic are numpy on
#         the CPU regardless, and at width 256 the fit gains little.  Set
#         "cuda" on the cluster, where the double backward is well trodden.
#
#         "auto" prefers CUDA, then MPS, then CPU, and skips MPS when float64
#         is asked for because MPS has no float64 at all.  Anything
#         unavailable falls back to the CPU with a warning rather than
#         killing the sweep on configuration one.
#
# DTYPE   "float32" | "float64"
#
# A GPU shortens the FIT; generation and the diagnostics are numpy on the CPU.
# That is a choice about where the sequential work belongs, and the reason is
# narrower than "generation cannot be parallelised", which is false.  ALONG a
# chain nothing can be batched, x_t being a function of x_{t-1}; ACROSS chains
# the recursion is embarrassingly parallel, and the two places where the
# quantity wanted really is a set of chains are batched: the MMD objective
# (`unroll_blocks`, in torch, so it does move to a GPU) and the synchronisation
# spread (`q_batch`, in numpy, measured 3x to 5x faster than the loop).  The
# free-running evaluation path stays ONE long trajectory because the ACF and
# the Lyapunov average are estimates along a single stationary path; cutting it
# into B pieces would change the estimand, not the arithmetic.
#
# For scale: one scalar step costs 14.3 us at width 256, identically at m = 2
# and m = 16, so it is call overhead rather than the matrix product, and it is
# the same order as a single GPU kernel launch.  Batching 24 chains brings that
# to 2.5-4.5 us per chain-step; the gain saturates near 4x.
#
# On an NVIDIA card float64 throughput is half of float32 on a datacentre part
# (A100, H100, V100) and a thirty-second or sixty-fourth of it on a consumer
# one (RTX, GeForce), so float64 there is a real decision, not a free upgrade.
# On a Mac there is no CUDA at all and MPS has no float64, so float64 means
# the CPU.
#
# float32 is the default and is what every quoted number was produced with.
# Changing the dtype does NOT change which fit you get: every random draw is
# made on the CPU through the seeded generator and only then cast, so the
# stream is identical across devices and precisions and a float64 run is the
# same fit computed more precisely, agreeing to about seven decimals on the
# loss.  Measured: float32 [0.04288297, 0.03723715, 0.03768581] against
# float64 [0.04288296, 0.03723714, 0.03768581].
DEVICE = "cpu"
DTYPE = "float32"


# ---------------------------------------------------------------------------
# NOTE on S >= 1
#
# Only `logistic` accepts it, because ell_1 = 4(1-rho) is exact there and the
# target remains well defined.  It is how the average-contraction experiment is
# expressed: S = 4(1-rho) crosses 1 at rho = 3/4, while the top Lyapunov
# exponent crosses 0 near rho = 0.28, and between those two points the standing
# hypothesis has failed but the pathwise and adapted conclusions still hold.
# The Lipschitz penalty is forced off automatically when S >= 1, since a
# contractive fit cannot approximate an expansive target.
#
#   TARGETS   = ["logistic"]
#   M         = [1]
#   S_TARGET  = [0.80, 2.00, 2.80, 3.40]      # rho = 0.80, 0.50, 0.30, 0.15
#
# Two more sweeps worth having, for reference:
#
#   the arity sweep          TARGETS = ["longmem"]; M = [2, 4, 8, 16, 32]
#   seed replicates          DATA["seed_path"] = [1, 2, 3]
#                            NET["seed_net"]   = [0, 1, 2]
#     (note this crosses to nine runs; for three paired replicates run three
#      separate sweeps, or accept the nine and read the diagonal)
# ---------------------------------------------------------------------------
