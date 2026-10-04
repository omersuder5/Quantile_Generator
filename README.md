# Quantile_Generator

Fitting a conditional quantile, closing it into a recursion driven by i.i.d.
uniforms, and checking that the process this produces has the target's law —
in adapted Wasserstein distance, not merely in path law.

```
X_t      = q (u_t ; X_{t-1}, X_{t-2}, ... )          the target
Xhat_t   = qhat(u_t ; Xhat_{t-1}, ..., Xhat_{t-m})   the generator, closed in
                                                     its own observable
```

Under a one-sided Dobrushin condition on the target's conditional quantile,
`sum_i ell_i = S < 1`, the generated process is stationary and ergodic, is a
function of the innovations alone with no window, seed or burn-in, and

```
AW_p^w(mu, muhat)  <=  ( theta + 2 M L_(m+1) ) / ( 1 - S_m )
```

for every p and every weight sequence, with `theta = sup |qhat - q_m|`,
`S_m = sum_{i<=m} ell_i`, `L_(m+1) = sum_{i>m} ell_i`, and the constant
carrying the **target's** contraction.  Nothing is assumed of `qhat` beyond
continuity: monotonicity in `u` and contractivity are free, by the repair
lemma.

---

## Two conventions that shape everything here

**`S` is an input, not an output.**  Each target takes a fixed *shape* — its
relative lag profile, its oscillation count, its ARCH floor, its decay
exponent — plus the scalar `S`, and solves internally for the one amplitude
that makes `sum_i ell_i` equal `S` exactly.  Three invert analytically
(`linear` and `smoothnl` scale their lag vector; `logistic` has
`ell_1 = 4(1-rho)` independently of `r`, so `rho = 1 - S/4`); the rest bisect
on a single monotone knob.  So `S` is an axis of the experiment in its own
right and "same S, different mechanism" is a fair comparison.

**`m` is what you presume, not the truth.**  It is the learner's lag dimension
and it is never silently raised to the target's own memory order.  Fitting
below the truth is a legitimate experiment and is priced by `L_(m+1)`,
uniformly for finite and infinite memory: a two-lag linear target fitted at
`m = 1` is misspecified in exactly the way, and by exactly the term, that a
long-memory target fitted at `m = 4` is.  The training path is always simulated
at the target's **own** order; only the fit happens at `m`.

---

## Layout

```
config.py                 the only file you edit.  Vectors are swept.
fit_generator.py          the fitting: pinball (default) or MMD
quantile_learning.py      runs the whole grid into reports/
quantile_learning.ipynb   one run, with the reasoning next to the code
appendix.ipynb            A.1, A.2 and B; not part of the experiment

generators/
    noise.py                  innovation transforms g : [0,1] -> [-c, c]
    synthetic_generators.py   the nine targets, each S-parameterised
    quantile_generator.py     SoftClip, QuantileNet, the closed recursion,
                              and the repair operators
    esn_realization.py        the exact-shift ESN block
    simulate_paths.py         one simulate() for every generator, plus the
                              shared-stream and spread protocols

loss/
    pinball.py                the randomised pinball criterion and its
                              level sampler
    rbf_kernel.py             RBF kernel, median-heuristic bandwidth
    mmd.py                    MMD on path blocks: differentiable objective,
                              unbiased estimator, permutation test

utils/                        one module per analysis
    hypotheses.py             S_m, per-lag Lipschitz, theta on three domains
    repair.py                 is the repair lemma doing anything
    pathwise.py               the bound under the synchronous coupling
    lyapunov.py               the top exponent, and why it is the real boundary
    autocorrelation.py        ACF and ACF of squares
    marginal.py               W_1 and the extreme quantiles
    conditional.py            conditional sd profile, conditional W_1
    pricing.py                European, Bermudan, early-exercise premium
    baselines.py              the matched Gaussian AR null
    plotting.py               figures, fixed three-series colour assignment
    reporting.py              results.txt, summary.csv, COMPARISON.md

reports/                      created at run time
```

Requires `numpy`, `scipy`, `torch`, `matplotlib`.  Nothing else.
`backend.py` is the one place the device and precision are decided.

---

## GPU and precision

`config.py` sets two scalars:

```python
DEVICE = "cpu"       # "cpu" | "auto" | "cuda" | "cuda:0" | "mps"
DTYPE  = "float32"   # "float32" | "float64"
```

`"auto"` prefers CUDA, then MPS, then CPU, and skips MPS when float64 is
requested.  Anything unavailable falls back to the CPU with a warning rather
than killing a sweep on its first configuration.

**The default is `"cpu"`, and on a Mac it should stay that way.**  The Lipschitz
penalty differentiates a gradient — `grad_lip` calls `autograd.grad` with
`create_graph=True` and the result is then backpropagated — and that double
backward through the MPS kernels can **abort the process rather than raise**.
In a notebook that kills the kernel, and Jupyter reports only
`notebook controller is DISPOSED`, with no Python traceback to go on.  `"auto"`
picks MPS on a Mac, so `"auto"` together with `lip_mode = "penalty"` is the
combination to avoid there; `fit` now detects it, warns, and moves that fit to
the CPU rather than let it take the session down.  Nothing is lost: generation
and every diagnostic are numpy on the CPU regardless, and at `width = 256` the
fit gains little.  Set `"cuda"` on the cluster, where the double backward is
well trodden.

**A GPU shortens the fit.**  Generation and the diagnostics are numpy on the
CPU, and the reason is narrower than "generation cannot be parallelised", which
is false.  There are two axes and they behave oppositely:

* **along a chain, strictly sequential.**  `x_t` is a function of
  `x_{t-1},...,x_{t-m}`, so the steps of one path must be taken in order.  This
  is the definition of the recursion and no implementation changes it.
* **across chains, embarrassingly parallel.**  `B` chains are `B` rows of one
  state matrix advanced by one matrix product per time step.  That is
  `ClosedQuantileRecursion.q_batch`.

So the question is never whether generation parallelises but whether the
quantity wanted is many chains or one long path, and both occur:

| quantity | form | where |
|---|---|---|
| MMD objective | `n_blocks` chains, differentiable | `unroll_blocks`, torch, moves to a GPU |
| synchronisation spread | 24 chains, one shared stream | `initial_spread` via `q_batch`, numpy |
| evaluation path, ACF, Lyapunov, shared coupling | **one** stationary path | single chain, by necessity |

The last row is a statistical constraint, not a technical one.  `B` burnt-in
chains of length `T/B` give the marginal more cheaply, but they cannot give the
lag-`h` dependence for `h` near `T/B`, and the dependence is the point of the
ACF panel.

Measured, width 256, numpy float64:

| | cost |
|---|---|
| one chain, scalar step | 14.3 µs/step, **identical at `m = 2` and `m = 16`** |
| 24 chains, batched | 59–109 µs/step for all 24, i.e. 2.5–4.5 µs per chain-step |
| 1000 chains, batched | 3.3 µs per chain-step |
| `initial_spread(80, 24)` | 19.7 ms loop → 4.3 ms batched, **4.5×** |

That the scalar step costs the same at `m = 16` as at `m = 2` says the cost is
interpreter overhead, not the `256 × m` product, which is also why the gain
from batching saturates near 4×: once `B × 256` is large the elementwise
activation becomes real arithmetic.  And 14 µs is the same order as one GPU
kernel launch, which is the actual reason a sequential chain gains nothing from
a GPU however it is written.

Whether the fit itself gains depends on the width.  At `width = 256`,
`batch = 512` the matrices are small enough that launch overhead can swallow
the benefit, and the Lipschitz penalty needs a second backward pass through
`autograd.grad` with its own fixed cost.  Measure before believing; `width`
of 1024 or more is where a GPU starts to pay.

**Apple Silicon.**  `torch.cuda.is_available()` is always False on a Mac; the
GPU is MPS, and **MPS implements no float64 at all**.  So on a Mac
`DTYPE = "float64"` means the CPU, whatever `DEVICE` says.

**float64 on NVIDIA** is half the float32 throughput on a datacentre part
(A100, H100, V100) and a thirty-second or a sixty-fourth of it on a consumer
one (RTX, GeForce).  It is worth switching on when a diagnostic sits near the
precision floor, which here means only the ESN realisation check — and that
already lands at about `4e-16` in float32, because the recursion side is numpy
float64 regardless.

**Changing device or precision does not change which fit you get.**  Every
random draw is made on the CPU through the seeded generator and only then
cast, so the stream of levels, permutations and penalty windows is identical
everywhere.  A float64 run agrees with its float32 twin to about seven decimal
places on the loss; it is the same fit computed more precisely, not a
different experiment.

---

## Running

```bash
python quantile_learning.py --dry-run    # print the grid, run nothing
python quantile_learning.py --quick      # tiny budget, ~3 s a run, to check
                                         # the pipeline end to end
python quantile_learning.py              # the real thing
python quantile_learning.py --only arch_m02 hetero_m02
python quantile_learning.py --resume-into reports/run_2026-10-02_031430
```

Output:

```
reports/run_<timestamp>/
    <target>_<the axes that vary>/
        results.json   every number, machine readable
        results.txt    the same, laid out
        figures.png    nine panels
        net.pt         the fitted weights, so a run can be re-analysed
                       without retraining
    summary.csv        one flat row per configuration
    COMPARISON.md      the tables
    overview.png       every run side by side
    sweep_<target>_<axis>.png
    config_used.json   exactly what was run
    run.log
```

A configuration whose `results.json` already exists is skipped, so an
interrupted sweep restarts with `--resume-into`.

## The learning rate

A cosine anneal from `lr` to zero over `epochs`, and the **last** iterate is what
comes back.  There is no schedule knob, no early stopping and no best-loss
restore, so `epochs` is a count rather than a ceiling.

That is a measured choice, not a simplification: with the loss held constant to
0.6 per cent, variants of the schedule and of which iterate is returned moved
`W₁`/sd over a factor of twenty, from 0.0156 to 0.3098 on the linear target at
S = 0.75.  Anything that selects an iterate by loss — early stopping, a
best-loss restore, a plateau trigger — selects by a quantity that does not rank
iterates here, because the bias it leaves in the conditional median is then
multiplied by `1/(1−S_m)`.  **Do not re-add any of them.**  The full table and
the mechanism are in the project note `claude/quantile-generator-package.md`.

If a run feels slow, lower `epochs` or `T_train`.  Those change the budget and
say so in the report; an early stop changed the answer and did not.

---

At the shipped budget (`T_train = 60 000`, 300 epochs) a configuration takes
roughly fifteen minutes on two CPU cores.  The clustering diagnostic is genuinely
budget-sensitive: on ARCH the lag-1 autocorrelation of squares reaches about
0.88 of the target's at 300 epochs and markedly less below that, so `--quick`
numbers are for checking that the pipeline runs, never for quoting.

---

## The config, in one paragraph

`TARGETS`, `M`, `S_TARGET` and every value inside `DATA`, `NET`, `TRAIN` and
`EVAL` are lists, and the sweep is their full Cartesian product; a list of
length one is a fixed setting.  `TARGET_SPECIFICS` is **not** swept — it fixes
each target's shape, on the view that a method that learns one parameterisation
of a mechanism should learn them all.  Folder names carry only the axes that
actually vary, so a single-axis sweep gives `longmem_m04` rather than a name
listing every setting.

Three sweeps worth having are written out at the foot of `config.py`: the
arity sweep, seed replicates, and the noise-budget sweep across `S = 1`.

---

## The nine targets

Each isolates one feature, and the column that can see it differs from target
to target.

| key | mechanism | what sees it |
|---|---|---|
| `linear` | affine mean, constant scale | the control |
| `hetero` | conditional scale a ridge function of `z_1` | conditional sd, ACF of squares |
| `skew` | conditional **skewness** varies; mean and variance do not | **only** the conditional `W_1` |
| `smoothnl` | nonlinear conditional mean, homoskedastic | the quantile-map overlay |
| `arch` | conditional scale even in each coordinate | ACF of squares: volatility clustering |
| `oscillatory` | high-frequency drift at **fixed** `S` | separates approximation difficulty from `1/(1-S)` |
| `logistic` | chaotic drift; `S >= 1` allowed here only | the Lyapunov exponent |
| `arma` | infinite memory, geometric moduli | `L_(m+1)` |
| `longmem` | infinite memory, polynomial moduli | `L_(m+1)`, decaying slowly |

`skew` is the sharpest test in the suite: no second-order diagnostic can see
it, not the ACF, not the ACF of squares, not the conditional standard
deviation, and a matched Gaussian AR and a matched GARCH are both blind to it
by construction.

---

## Reading the output

Three innovation protocols are used and they answer different questions.

* **shared** — one stream drives both. `sup_t |X - Xhat|` is what the
  comparison lemma bounds, and it is an **upper** bound for the adapted
  Wasserstein distance, never a lower one. A large slack means the bound is
  loose, not that the generator beats it.
* **independent** — unrelated streams. Everything about the law uses this.
* **matched Gaussian AR** — the null. It reproduces the linear autocorrelation
  and has constant conditional spread, both by construction, so it passes every
  standard second-order diagnostic and gets the conditional structure wrong.
  Expect it to **win** on the marginal for an affine, near-Gaussian target;
  that is the correct outcome.

Four cautions that the code cannot remove and the reports therefore state:

1. `theta` is a maximum over a grid (64 levels by 3000 windows), not a
   supremum.
2. `theta` over the box runs several times larger than over the data support.
   The proposition only ever evaluates `q` and `qhat` where the two processes
   are, so the right domain is a compact forward-invariant `K`; `inv_R`
   reports the smallest such radius and `theta_invariant` the value there.
3. `theta` is also a supremum over levels and the extreme levels dominate it;
   the trimmed values show by how much. No change of domain removes that.
4. Nothing here produces a lower bound for the adapted distance.
