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

At the shipped budget (`T_train = 60 000`, 300 epochs) a configuration takes
roughly fifteen minutes on two cores.  The clustering diagnostic is genuinely
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

---

## What changed from the earlier `ESN_MMD_Generator` code

Removed: `stratify` (unused, and a positional-argument bug in its old call
site silently disabled the tail mixture or produced NaNs); `PadTarget` and all
`max(m, target order)` logic; `sigkernel` and the old `loss/loss.py`;
duplicate pricing functions; `_empirical_kernel` as a public name.

Kept but off by default: `v_mode="probit"`, which is a composition with a fixed
increasing bijection and therefore leaves the moduli, uniqueness, bicausality,
the adapted bound and the exact-shift realisation untouched — it flattens
`dq/dv` in the tails by about a factor of seventeen, and it is off because the
tail problem turned out to be fixable without changing the estimator's inputs.
`pen_region` keeps all three of `box`, `hood` and `data`, defaulting to `hood`.

Added: the `S`-parameterisation of every target; `theta` on the
forward-invariant set; the `repairs_needed` verdict; a differentiable MMD, so
"pinball against MMD on the same model class and the same diagnostics" can be
run rather than argued about; `net.pt` per run.
