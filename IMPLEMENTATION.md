# `Quantile_Generator`: the implementation, the targets, and what the runs show

What this document is for: to say exactly what the code computes, to put every
piece of it next to the theoretical object it stands for, and to report an
`m`-sweep honestly, including the parts that did not work.

It is written to be read alongside the paper, not instead of it. Where a name
like `prop:p1p3` appears it is the draft's label.

---

## 1. The object

The theory is about a continuous map

```
    qhat : [0,1] x X^m -> X,        X = [-M, M],
```

closed into the recursion

```
    Xhat_t = qhat(u_t ; Xhat_{t-1}, ..., Xhat_{t-m}),      u_t ~ iid U(0,1).
```

Three features of that statement drive the whole design and are worth naming
before anything else.

**No seed.** The recursion is driven by innovations alone. There is no window
of the target to start from, no latent state drawn from a law one is trying to
learn. The burn-in in the code is numerical, to reach the unique solution from
an arbitrary start, and nothing in the theory needs it. This is the point on
which the earlier Theorem A failed, and the reason this draft exists.

**Bicausality, not just causality.** `Xhat` must be a measurable function of
`u^{<=t}`, which is what makes the synchronous coupling bicausal and the
adapted conclusion available. That requires the closed recursion to have a
*unique* bounded solution, which is what the contraction hypothesis buys.

**The estimand is the conditional law**, pointwise in the level, not the law of
a window. That is why the objective is a randomised pinball loss and not an
MMD, and the difference is measurable: see §5.

---

## 2. Code ↔ theory

| code | theory | file |
|---|---|---|
| `QuantileNet` | `qhat(u,z) = a' sigma(G z + c v(u) + zeta) + b`, the hypothesis class | `generators/quantile_generator.py` |
| `ClosedQuantileRecursion` | the closed recursion driven by iid uniforms | same |
| `RepairedRecursion` | `Pi_X` and `R` applied at every step | same |
| `rearrange_levels` | `R`, the increasing rearrangement in `u` | same |
| `project_to_X` | `Pi_X`, the projection onto `X` | same |
| `ESNRealization` | `prop:exactshift`: the exact-shift ESN block | `generators/esn_realization.py` |
| `Target.moduli(m)` | `ell_i = sup_{u,z} \|dq/dz_i\|`, the per-lag `W_inf` moduli | `generators/synthetic_generators.py` |
| `Target.S_m(m)` | `S_m = sum_{i<=m} ell_i`, the one-sided Dobrushin sum | same |
| `Target.L_tail(m)` | `L_{m+1} = sum_{i>m} ell_i`, the mass the truncation drops | same |
| `Target.bound(theta,m)` | `(theta + 2 M L_{m+1}) / (1 - S_m)` | same |
| `pinball` | `rho_u(x,p)`, strictly consistent for the `u`-quantile | `loss/pinball.py` |
| `grad_lip` | a differentiable estimate of `sum_i Lip_i(qhat)` | `fit_generator.py` |
| `theta_hat` | `theta = sup_{u,z} \|qhat - q_m\|`, on three domains | `utils/hypotheses.py` |
| `lipschitz_empirical` | `sum_i sup_z \|d qhat / d z_i\|` | same |
| `synchronisation_report` | uniqueness, seen as the spread across starts collapsing | `utils/pathwise.py` |
| `pathwise_report` | `sup_t \|X_t - Xhat_t\|` under the synchronous coupling | same |
| `lyapunov` | `lambda = E log sum_i \|d qhat / d z_i\|` along the fitted map's own path | `utils/lyapunov.py` |
| `conditional_report` | the conditional `W_1`, which is the only diagnostic with conditional content | `utils/conditional.py` |

Two deliberate asymmetries in that table.

`lipschitz_empirical` returns `sum_i sup_z |d qhat/dz_i|`, which **upper
bounds** the operator norm `sup_z sum_i |d qhat/dz_i|` that uniqueness
actually needs. The over-count grows with `m`: in the earlier 21-run study it
was 1.08 on the box against 0.81 on the data support at `m = 32`. The reported
number is therefore conservative in the right direction.

`theta_hat` is reported on **three** domains and they are not the same number:
the data support (what the bound uses, and the honest quantity, because the
recursion never visits the corners of the box), the box `[-M,M]^m` (what the
proposition as usually stated asks for), and the forward-invariant set
`K = [-R,R]^m` with `qhat(u;K^m) subseteq K` (the refinement: weaker than the
box, still checkable, and all the proof actually uses). The study measured the
box at 1.7x to 7.8x the data-support value.

---

## 3. The targets

Every target is a strictly stationary process on `X = [-M,M]` whose one-step
conditional quantile `q(u;z)` is available in closed form, and **`S` is an
input, not an output**: each takes a fixed *shape* and a requested `S`, and
`solve_amplitude` bisects a single monotone amplitude knob until
`sum_i ell_i = S` exactly (verified to 1e-13; three targets invert in closed
form and skip the bisection). This is what makes the nine comparable: in the
old code each target's `S` fell out of hard-coded coefficients, so no two
targets were at the same point of the theory.

There is **no padding to a common arity**. A target has whatever memory it has;
`m` is the learner's lag dimension *and* what the user assumes the target has.
Fitting a 2-lag target at `m = 1` is a real misspecification and `L_{m+1}`
prices it; fitting a long-memory target at `m = 4` likewise. Simulation always
uses the target's full memory, so `m` never enters the data-generating process.

### 3.1 The nine inside the theory

| key | `q(u;z)` | moduli | what only it can show |
|---|---|---|---|
| `linear` | `a'z + g(u)` | `ell_i = \|a_i\|` exactly | the control; the family where the Perron bound is tight |
| `hetero` | `alpha'z + s(z_1) g(u)`, `s = s0 + s1 tanh(kappa z_1)` | mean plus scale terms | a conditional scale that is a ridge function of one lag |
| `skew` | an asymmetric map whose conditional **skewness** varies | — | **only** the conditional `W_1` sees it: mean and variance are constant, so the ACF, the ACF of squares and the conditional sd are all blind, and a matched Gaussian AR *and* a matched GARCH are blind by construction |
| `smoothnl` | nonlinear conditional mean, homoskedastic | — | the quantile-map overlay |
| `arch` | `sqrt(omega + sum a_i z_i^2) g(u)` | `ell_i = c a_i M / sqrt(omega + a_i M^2)` | volatility clustering in a genuinely Markov observable |
| `oscillatory` | high-frequency drift at **fixed** `S` | — | separates approximation difficulty from the amplification `1/(1-S)` |
| `logistic` | the logistic map at `r`, rescaled, plus noise | `ell_1 = 4(1-rho)` **independently of `r`** | the only target where `S >= 1` is allowed; the Lyapunov boundary |
| `arma` | the `pi`-weights of an ARMA, truncated | geometric | `L_{m+1}` |
| `longmem` | `phi_i ~ i^{-(1+alpha)}` given directly | polynomial | `L_{m+1}` decaying *slowly*: the hard case |

Two exact identities fall out of the `S`-parameterisation and are worth
keeping, because each ties a hypothesis to an observable:

```
    logistic:   ell_1 = 4(1 - rho)  independently of r,   so  rho = 1 - S/4
    arch:       sigma_max / sigma_min <= (1 - S)^{-1/2},  equality at one lag
```

The second says something uncomfortable: in the square-root family the
attainable conditional-scale contrast and the amplification constant are *the
same parameter*. `S < 1` is simultaneously the theory's hypothesis and the
limit of what the family can express, so an experiment inside that family
cannot tell the two apart. That is the gap `egarch` was added to fill.

### 3.2 The two outside it

`garch` and `egarch` are **not** claimed to satisfy the hypotheses.
`prop:gauss` excludes latent-volatility models outright. They are here so the
exclusion can be measured instead of asserted, and so that the question
"*which* assumption actually bites" has an answer.

**What makes the experiment possible.** Substituting the variance recursion
into itself,

```
    sigma_t^2 = omega/(1-beta) + sum_{j>=0} beta^j a(x_{t-1-j}) x_{t-1-j}^2,
    a(x)      = alpha + gamma 1{x < 0},
```

so the latent state *does* unroll in the observable. GARCH is a chain with
complete connections with geometrically decaying memory **in the squares** —
structurally `longmem` with `decay="geometric"`, one level up. So `q_m`, the
moduli, `S_m` and `L_{m+1}` are all well defined, and nothing in the apparatus
needs changing.

**The one concession.** A real GARCH is unbounded; the theory needs
`X = [-M,M]`. The innovation is bounded and its amplitude set to
`c = M / sigma_max` with
`sigma_max^2 = [omega + (alpha+gamma) M^2] / (1 - beta)`, which makes
`|X| <= M` exactly. These are therefore **bounded processes with GARCH's
dependence structure**, not GARCH processes. The Gaussian-tail obstruction is
removed by hand so that what remains to be tested is whether the *dependence*
is learnable. Reporting it any other way would be dishonest about which
assumption the experiment is probing.

**`egarch` departs from Nelson in one stated way.** Textbook EGARCH drives the
log-variance with the standardised residual `z_{t-1} = X_{t-1}/sigma_{t-1}`.
Standardising makes the recursion depend on the latent `sigma` path, so it has
no closed form in the observable — and without a closed form there is no `q_m`,
no `theta`, and nothing to compare. Driving it with the observable return
keeps both features the experiment is about (an exponential link, so the scale
is positive by construction rather than by a floor; a signed term, so a
negative return moves the variance differently from a positive one) and unrolls
exactly:

```
    log sigma_t^2 = h0 + sum_{j>=0} beta^j [alpha |x_{t-1-j}| + gamma x_{t-1-j}] / M.
```

It is log-GARCH with leverage. It is EGARCH's *structure*, not EGARCH, and the
docstring says so.

Its two identities are the reason it earns a place:

```
    ell_j = beta^j (alpha + |gamma|) / 2,
    S     = (alpha + |gamma|) / (2 (1 - beta))        -> inverts in closed form
    sigma_max / sigma_min = exp(S)                    -> finite at every S
```

Against `arch`'s and `garch`'s `(1-S)^{-1/2}`, which is infinite at `S = 1`.
The exponential link therefore **separates the hypothesis from the
expressiveness of the family**: it can pose a target with a large, honest
conditional-scale range at any `S`, which the square-root family cannot.

**`beta` is the experiment.** At fixed `S` it decides where the modulus mass
sits, and the trade is sharp. Measured at `S = 0.75`, `T = 60k`:

| `beta` | GARCH `ACFsq(1)` | `L(2)` | `L(8)` | EGARCH `ACFsq(1)` | `L(2)` | `L(8)` |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | +0.0865 | 0.087 | 0.000 | +0.1317 | 0.067 | 0.000 |
| 0.50 | +0.0585 | 0.212 | 0.003 | +0.0947 | 0.187 | 0.003 |
| **0.70** (default) | **+0.0335** | **0.384** | **0.047** | **+0.0565** | **0.367** | **0.043** |
| 0.85 | +0.0162 | 0.548 | 0.211 | +0.0280 | 0.542 | 0.204 |
| `arch`(2) reference | +0.0851 | 0.000 | 0.000 | | | |

Low `beta` is ARCH with extra steps. High `beta` is the real GARCH case, and
the row says the obstruction quantitatively: **at matched `S` the clustering
per unit `S` collapses while the truncation term grows.** `S` is a poor
currency for latent-volatility models, because matching `S` to a Markov target
forces the GARCH to be nearly homoskedastic. `beta = 0.70` is the default
because both effects are visible at once.

Leverage, `gamma/alpha`, measured as `corr(X_t, X_{t+1}^2)` at `beta = 0.70`:
GARCH `+0.003 -> -0.005`, EGARCH `+0.003 -> -0.032`. The exponential link
carries leverage about six times better, because its signed term enters the
log-variance linearly rather than through a square.

---

## 4. The algorithm, step by step

### 4.1 Data

One path of length `T_train` from the target's own full memory, then
`(z_t, x_t)` windows with `z_t = (X_{t-1},...,X_{t-m})`. **One observation per
conditioning window** — the conditional law at `z` is never observed, only one
draw from it. That is the whole difficulty, and the criterion below works
anyway because it is pointwise proper.

### 4.2 The objective: randomised pinball

```
    rho_u(x,p) = max( u(x-p), (u-1)(x-p) ),     minimised at the u-quantile
```

so averaging over `u ~ U(0,1)` gives, up to `1/2`, the CRPS of the conditional
law. For **any** positive weight `w(u)` the weighted average is still proper
pointwise in `u`, which is what licenses `tail_frac`: a fraction of levels is
drawn log-uniformly in `(u_min, edge)` and reflected to the upper tail half the
time. That is a change of *resolution*, not of estimand. It is needed because
`dq/du` at `u = 0.001` is about seven times its value at the median for a
truncated-normal innovation, so uniform levels put almost no signal where the
conditional quantile is steepest and the generated tails come out short.

**This is the point on which the construction differs from a path-law
criterion.** The pinball criterion scores the conditional law pointwise in the
level; an MMD between window laws has no conditional content at all. The
empirical consequence is in the ARCH runs and is reported in §5.

### 4.3 The Lipschitz penalty

`lam * relu( grad_lip(qhat) - L_max )^2`, with `grad_lip` differentiating a
gradient (`autograd.grad` with `create_graph=True`). It targets
`sum_i sup_batch |d qhat/dz_i|`, i.e. the uniqueness condition, rather than the
closed-form bound `sum_j |a_j| ||G_j||_1`, which is usually vacuous.

`pen_region="hood"` applies it near the data (a Gaussian neighbourhood of
sampled windows, clipped to `X`) rather than on uniform draws from the box. The
hinge is a maximum over the batch, so on box draws it fires at points the
recursion never visits and, at a finite budget, drives the fitted map flatter
than the truth. That was the main cause of the earlier tail problem: at
`L_max = 0.85`, `lam = 5` the fitted lag-sensitivity sum came out at 0.62
against a target `S` of 0.705, and both the generated tails and the volatility
clustering were short.

The penalty is forced off when the target has `S >= 1`: a contractive map
cannot approximate an expansive target, and those runs exist precisely to show
that `S < 1` is sufficient and not necessary.

### 4.4 The learning rate

A cosine anneal from `lr` to zero over `epochs`, returning the **last**
iterate. No early stopping, no best-loss restore. That is measured, not lazy:
with the loss held constant to 0.6 per cent, variants moved `W1/sd` over a
factor of twenty (0.0156 to 0.3098 on `linear` at `S = 0.75`). **Anything that
selects an iterate by loss** — early stopping, a best-loss restore, a plateau
trigger — selects by a quantity that does not rank iterates here, because the
bias it leaves in the conditional median is then multiplied by `1/(1-S_m)`.

### 4.5 Closing the recursion, and the repair lemma

Three operators are deliberately **not** imposed during fitting, because the
repair lemma makes each free after the fact, and each is nonexpansive in the
uniform norm so `theta` cannot grow:

| operator | what it fixes | measured by |
|---|---|---|
| `R` | the increasing rearrangement in `u`, so `qhat(.;z)` is a genuine quantile function | `cross_frac`, `du_min` |
| `T` | a Lipschitz regularisation in `z`, supplying the contraction uniqueness needs | `lip_box` |
| `Pi_X` | the projection onto `X`, so the recursion is `X`-valued | `sup_q_box` |

`repair_report` measures whether any of them is doing anything on a given fit,
and names which are needed. Across the earlier 21-run study, eleven of
twenty-one fits needed none of the three, and three different runs needed three
different pieces — so the lemma is not decoration.

### 4.6 The ESN realisation

`prop:exactshift`. The reservoir has `N = W + (m-1) + 1` units: `W` quantile
units carrying the fitted readout, `m-1` shift units, and one constant unit.
The shift units carry **zero input weight** — a unit whose value must be an
exact affine function of past outputs cannot be allowed to see the innovation —
and the scale `lam = 0.9 / operating_radius` keeps their pre-activations inside
the interval on which `sigma` is affine. That is the whole reason the
activation must be affine on an interval, and why `tanh` provably will not do.
The extraction matrix inverts the affine action exactly, so the shift is exact,
not accurate: `max |ESN - direct recursion| = 7.8e-16` across the 21-run study,
including at `m = 32`.

### 4.7 Three innovation protocols, never mixed

| protocol | streams | what it answers |
|---|---|---|
| **independent** | unrelated on the two sides | everything about the **law**: marginal, ACF, conditional `W_1`, block MMD |
| **shared** | one stream drives both | the synchronous coupling. `sup_t\|X-Xhat\|` is bounded by `(theta + 2ML_{m+1})/(1-S_m)` and is an **upper** bound for `AW`, never a lower one |
| **spread** | one stream, 24 initial states | uniqueness of the closed recursion, seen directly: collapse to machine zero is `Xhat` being a function of the innovations alone |

Using a shared stream for a law-level comparison would make the two paths close
by construction and the comparison meaningless. The three are kept in separate
functions so no caller has to rebuild the bookkeeping and get it wrong.

---

## 5. The `m`-sweep

One sweep, 20 runs, 146 minutes. Four targets at `m = 1, 2, 4, 8, 16`, every
other knob held at the shipped default (`S = 0.75`, `T_train = 60 000`,
`width = 256`, 300 epochs, cosine to zero, `lip_mode="penalty"`,
`pen_region="hood"`, `L_max = 0.95`, `tail_frac = 0.2`, one seed throughout).
Nothing was tuned per target. Raw output in `reports/msweep`.

The four targets are chosen to separate one thing: **where the dependence
lives.**

| target | memory | in what |
|---|---|---|
| `arch` | exactly 2 lags | the conditional **scale** |
| `garch` | geometric, `beta = 0.70` | the conditional **scale** (the squares) |
| `egarch` | geometric, `beta = 0.70`, exponential link + leverage | the conditional **scale** |
| `longmem` | polynomial, `alpha = 0.5`, hundreds of lags | the conditional **mean** |

### 5.1 The table

`theta` is on the data support. `W1/sd` and `cond W1` are against an
independent stream, with the matched Gaussian AR(`m`) in brackets. `ACFsq` is
the lag-1 autocorrelation of squares, generated against target. `sdrng` is the
generated conditional-sd range over the target's. `q999` is the 99.9% marginal
quantile error in sd units.

| run | `S_m` | `L_(m+1)` | `theta` | `sum Lip` | bound | `sup err` | slack | `W1/sd` | (AR) | `cond W1` | (AR) | `ACFsq` gen / tru | `sdrng` | `q999` | `lambda` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `garch_m01` | 0.211 | 0.539 | 0.071 | 0.260 | 1.455 | 0.080 | 18.2 | 0.0263 | 0.0422 | 0.0143 | 0.0161 | +0.0240 / 0.026 | 0.820 | −0.158 | −4.57 |
| `garch_m02` | 0.366 | 0.384 | 0.119 | 0.299 | 1.399 | 0.089 | 15.7 | 0.0251 | 0.0422 | 0.0143 | 0.0163 | +0.0203 / 0.026 | 0.819 | −0.161 | −3.70 |
| `garch_m04` | 0.557 | 0.193 | 0.143 | 0.642 | 1.194 | 0.106 | 11.2 | 0.0241 | 0.0422 | 0.0147 | 0.0164 | +0.0117 / 0.026 | 0.741 | −0.172 | −2.88 |
| `garch_m08` | 0.703 | 0.047 | 0.123 | 0.432 | 0.732 | 0.107 | 6.8 | 0.0291 | 0.0422 | 0.0152 | 0.0162 | +0.0064 / 0.026 | 0.578 | −0.188 | −2.88 |
| `garch_m16` | 0.747 | 0.003 | 0.173 | 0.822 | 0.706 | 0.111 | 6.4 | 0.0318 | 0.0421 | 0.0161 | 0.0158 | +0.0106 / 0.026 | 0.488 | −0.178 | −2.07 |
| `egarch_m01` | 0.225 | 0.525 | 0.047 | 0.314 | 1.416 | 0.078 | 18.1 | 0.0249 | 0.0413 | 0.0108 | 0.0121 | +0.0201 / 0.032 | 0.721 | −0.167 | −3.99 |
| `egarch_m02` | 0.383 | 0.367 | 0.070 | 0.396 | 1.304 | 0.066 | 19.9 | 0.0245 | 0.0413 | **0.0104** | 0.0122 | +0.0224 / 0.032 | 0.870 | −0.168 | −3.37 |
| `egarch_m04` | 0.570 | 0.180 | 0.084 | 0.647 | 1.034 | 0.075 | 13.9 | 0.0238 | 0.0413 | 0.0109 | 0.0121 | +0.0138 / 0.032 | 0.692 | −0.189 | −2.71 |
| `egarch_m08` | 0.707 | 0.043 | 0.091 | 0.710 | 0.604 | 0.080 | 7.5 | 0.0286 | 0.0413 | 0.0111 | 0.0121 | +0.0098 / 0.032 | 0.732 | −0.192 | −2.52 |
| `egarch_m16` | 0.748 | 0.002 | 0.111 | 0.850 | 0.461 | 0.087 | 5.3 | 0.0281 | 0.0412 | 0.0113 | 0.0119 | +0.0071 / 0.032 | 0.741 | −0.182 | −1.93 |
| `arch_m01` | 0.474 | 0.276 | 0.090 | 0.304 | 1.221 | 0.106 | 11.5 | 0.0275 | 0.0401 | 0.0128 | 0.0170 | **+0.0822** / 0.083 | 1.148 | −0.195 | −3.84 |
| `arch_m02` | 0.750 | 0.000 | 0.165 | 0.534 | 0.660 | 0.099 | 6.7 | 0.0279 | 0.0401 | 0.0133 | 0.0169 | +0.0673 / 0.083 | 0.911 | −0.205 | −2.82 |
| `arch_m04` | 0.750 | 0.000 | 0.184 | 0.906 | 0.735 | 0.124 | 5.9 | 0.0270 | 0.0401 | 0.0137 | 0.0171 | +0.0684 / 0.083 | 0.913 | −0.220 | −2.37 |
| `arch_m08` | 0.750 | 0.000 | 0.185 | 0.697 | 0.741 | 0.147 | 5.1 | 0.0324 | 0.0402 | 0.0149 | 0.0169 | +0.0401 / 0.083 | 0.673 | −0.288 | −2.55 |
| `arch_m16` | 0.750 | 0.000 | 0.237 | 0.870 | 0.948 | 0.159 | 6.0 | 0.0325 | 0.0400 | 0.0152 | 0.0165 | +0.0396 / 0.083 | 0.632 | −0.286 | −2.05 |
| `longmem_m01` | 0.291 | 0.459 | 0.051 | 0.711 | 1.367 | 0.124 | 11.0 | 0.0243 | 0.0260 | 0.0054 | 0.0058 | +0.1427 / 0.152 | — | −0.036 | −0.97 |
| `longmem_m02` | 0.393 | 0.357 | 0.039 | 0.915 | 1.240 | 0.094 | 13.2 | 0.0236 | 0.0271 | 0.0046 | 0.0057 | +0.1505 / 0.152 | — | −0.000 | −0.76 |
| `longmem_m04` | 0.486 | 0.264 | 0.032 | 0.901 | 1.090 | 0.063 | 17.2 | **0.0199** | 0.0274 | 0.0047 | 0.0056 | +0.1552 / 0.152 | — | −0.005 | −0.61 |
| `longmem_m08` | 0.560 | 0.190 | **0.025** | 0.953 | 0.921 | 0.045 | 20.3 | 0.0255 | 0.0272 | 0.0050 | 0.0060 | +0.1512 / 0.152 | — | +0.004 | −0.52 |
| `longmem_m16` | 0.616 | 0.134 | 0.029 | 0.991 | 0.773 | **0.040** | 19.2 | 0.0258 | 0.0273 | 0.0055 | 0.0061 | +0.1573 / 0.152 | — | +0.008 | −0.44 |

(`longmem` is homoskedastic, so its `sdrng` is sampling noise and is omitted
rather than reported as information. `garch` and `egarch` were missing from
`STATE_DEPENDENT_SCALE` on this run and have been added, which changes a label
and no number.)

### 5.2 What is achieved

**1. The bound holds on all 20 runs, slack 5.1x to 20.3x.** No violations,
including on the two targets the theory excludes, and including every
misspecified arity. The range matches the 21-run study's 3.1x–20.3x.

**2. `prop:exactshift` is numerically exact, again.**
`max |ESN − direct recursion| = 6.7e−16` over all 20 runs, up to `m = 16` with a
16-lag shift register.

**3. Uniqueness is comfortable everywhere.** The 24-start spread collapses to
exactly zero on every run, and `lambda` runs from −0.44 to −4.57, far from the
boundary. No run needed `T`, and the one repair that fired at all was `R` in the
box only, on `longmem_m02` — not on the data support.

**4. The conditional law beats the matched Gaussian AR on 19 of 20 runs.**
`cond W1` generated against AR: `garch` 0.0143–0.0161 vs 0.0158–0.0164,
`egarch` 0.0104–0.0113 vs 0.0119–0.0122, `arch` 0.0128–0.0152 vs 0.0165–0.0171,
`longmem` 0.0046–0.0055 vs 0.0056–0.0061. The margin is 8–25% on `arch`, 5–15%
on `egarch`, 11% on `garch` at `m = 1`. **The one exception is `garch_m16`,
where it reverses: 0.0161 against the AR's 0.0158.** That is the only run in
the sweep where a Gaussian AR is the better conditional model, and it is the
largest arity on the target with the weakest conditional signal, which is
exactly where §5.3 predicts the degradation to land.

**5. The sign reversal reproduces, and the generator does not have it.** The
matched AR's lag-1 ACF of squares is **−0.009** on every single volatility run,
against targets of +0.026 to +0.083. A model with state-independent conditional
scale produces a small *negative* sample autocorrelation of squares, which is
exactly what `progress_report.pdf` recorded for the earlier MMD-trained
architecture and called "not fully understood". It is the small-sample behaviour
of an estimator of an autocorrelation the model has set to zero. The pinball-
fitted generator is positive on every run.

**6. GARCH and EGARCH are learnable at the level of the law.** This is the
answer to "I know it is not supposed to work". `W1/sd` is 0.024–0.032 for both,
against the AR's 0.041–0.042, so the marginal is recovered to within 3% of a
standard deviation and about 1.5x better than the Gaussian baseline; the block
MMD cannot distinguish the laws on any run (`p` = 0.54–0.63). At `m = 1` and
`m = 2` the clustering is recovered at 70–94% of target. The reason this is
possible at all is §3.2's unrolling: the latent variance is a geometrically
weighted sum of past squares, so `q_m` exists, and the obstruction `prop:gauss`
names is the *tail*, which the bounded construction removes. **So the experiment
separates the two assumptions, and it is the tail assumption that was doing the
work, not the latent state.**

**7. EGARCH is easier to approximate than GARCH, at matched `S`.** `theta` is
lower at every arity (0.047–0.111 against 0.071–0.173, a factor of 1.5–1.6) and
`cond W1` is 25–30% lower. The exponential link also holds its conditional-scale
range far better as `m` grows: `sdrng` 0.72–0.87 with no trend, against GARCH's
monotone collapse 0.82 → 0.49. That is the `exp(S)` versus `(1−S)^{−1/2}`
identity showing up in a fit.

### 5.3 What is not achieved

**1. Raising `m` improves the bound and degrades the law — on three of four
targets.** This is the sharpest negative result and it reproduces the 21-run
study's finding on new targets.

| | bound | `sup err` | `cond W1` | `ACFsq` recovered |
|---|---|---|---|---|
| `garch`, `m=1 -> 16` | 1.455 → 0.706 (**2.1x better**) | 0.080 → 0.111 (1.4x worse) | 0.0143 → 0.0161 | 94% → 41% |
| `egarch`, `m=1 -> 16` | 1.416 → 0.461 (**3.1x better**) | 0.078 → 0.087 (1.1x worse) | 0.0108 → 0.0113 | 63% → 22% |
| `arch`, `m=2 -> 16` | 0.660 → 0.948 (1.4x worse) | 0.099 → 0.159 (1.6x worse) | 0.0133 → 0.0152 | 81% → 48% |

**2. `arch` at `m = 4, 8, 16` is the cleanest version of that, because nothing
in the bound changes.** ARCH(2) has `L_(m+1) = 0` and `S_m = 0.75` for every
`m >= 2`, so the truncation term is exactly zero and the amplification is
exactly 4 across those four runs. The only thing varying is `m`. And `theta`
goes 0.165 → 0.184 → 0.185 → 0.237, the realised error 0.099 → 0.159, the
clustering 81% → 48%, and the 99.9% tail error −0.205 → −0.286. **Pure cost of
over-specification, with the bound's own terms held fixed, and nothing in the
theory that prices it.** It is a curse-of-dimensionality signature: `theta` is a
supremum over an `m`-dimensional conditioning set estimated from a fixed `T`.

**3. Under-specification beat correct specification on the conditional
structure.** `arch_m01` fits one lag to a two-lag target — a real
misspecification, charged `L_(m+1) = 0.276` and a bound of 1.221 against
`arch_m02`'s 0.660 — and yet recovers **99%** of the clustering (0.0822 against
0.083) where the correctly specified `arch_m02` recovers 81%, with a lower
`theta` (0.090 against 0.165) and a better conditional sd range. The target's
first lag carries about twice the second's modulus, so dropping the second lag
concentrates a fixed estimation budget where the signal is. The bound cannot see
this, because it charges for the dropped mass and not for the estimation it buys.

**4. The dependence being in the conditional SCALE rather than the MEAN is what
decides the sign.** `longmem` is the control, and it is the mirror image: every
quantity improves monotonically with `m`. `theta` 0.051 → 0.025, realised error
0.124 → 0.040 (**3.1x better**), and the lag-5 autocorrelation goes
+0.007 → +0.050 → +0.120 → +0.159 → +0.163 against a target of +0.149 — the
longer-lag structure is recovered exactly as the arity allows it. So adding lags
is not intrinsically harmful; it is harmful **when what the added lags carry is
second-order**. A mean dependence is linear in `z` and each added lag delivers
first-order information; a scale dependence lives in `sum_j beta^j z_j^2`, and
with one draw per conditioning window the extra lags add parameters faster than
they add usable signal, so the fit regresses toward homoskedasticity. That is
exactly what losing the ACF of squares means.

**5. The bound's composition flips over the sweep, and the amplification is the
hidden cost of raising `m`.** Decomposing `(theta + 2 M L_(m+1)) / (1 - S_m)`:

| | `theta` share of the numerator | `1/(1-S_m)` |
|---|---:|---:|
| `garch_m01` | 6% | 1.27 |
| `garch_m04` | 27% | 2.26 |
| `garch_m16` | **97%** | **3.96** |

Raising `m` does not simply remove the truncation penalty. It **trades** it for
amplification: `2 M L_(m+1)` falls by a factor of 216 while `1/(1-S_m)` rises by
a factor of 3.1 and `theta` by 2.4, so the bound improves only 2.1x. At `m = 1`
the bound is 94% truncation penalty; at `m = 16` it is 97% `theta`. Any statement
about "take `m` large enough" is a statement about that trade, and the sweep says
the trade stops paying around `m = 4` for these targets.

**6. The tails are short on every volatility target, at every arity.** The
99.9% marginal quantile error is −0.158 to −0.188 sd for `garch`, −0.167 to
−0.192 for `egarch`, −0.195 to −0.288 for `arch`, and it *worsens* with `m`.
`longmem` has −0.036 to +0.008, i.e. essentially none. So the tail deficiency is
specific to scale dependence and survives `tail_frac = 0.2`. The earlier study
measured ARCH at −0.197 ± 0.017 over three seeds, so this is real and about ten
seed-sd, not noise.

**7. At matched `S` the GARCH targets are nearly homoskedastic, which blunts the
whole experiment.** `garch` at `S = 0.75, beta = 0.70` has a target lag-1 ACF of
squares of only 0.026, against `arch`'s 0.083 at the same `S`. §3.2 explains
why — spreading a fixed modulus mass over geometrically many lags costs
clustering — but the consequence for the experiment is that the GARCH runs are
testing the recovery of a weak signal. A fair test of GARCH's clustering needs
`S` well above 1, where the bound is void and only the law-level diagnostics
remain. That run has not been done.

**8. `sum_i Lip_i` does not track `S_m` reliably on the volatility targets.**
`garch_m08` fits a map with `sum Lip = 0.432` against the target's
`S_m = 0.703`, i.e. 39% *under*, which is the same fit whose clustering
collapsed to 25% of target; `garch_m16` is 0.822 against 0.747, 10% over.
`longmem` behaves quite differently: `sum Lip` sits *above* `S_m` at every arity
and rises monotonically toward the penalty cap — 0.711/0.291, 0.915/0.393,
0.901/0.486, 0.953/0.560, 0.991/0.616 — so the fitted map is consistently more
lag-sensitive than the target and the ratio falls from 2.4 to 1.6 simply because
`S_m` is catching up. So on the scale-dependent targets the fitted lag
sensitivity is a poor proxy for the target's and can land either side of it, and
when it lands below, the clustering goes with it; on the mean-dependent target it
is reliably above and pinned by `L_max`.

### 5.4 Two things to run next

- **`garch` and `egarch` at `S > 1`**, with `lip_mode="none"`, at `m = 2` and
  `m = 4`. The bound is void there, which is the point: it isolates whether the
  *law* survives when the amplification does not, as the `logistic` noise-budget
  sweep already showed for a chaotic mean. These are the runs where GARCH has
  the clustering a practitioner would recognise.
- **`arch` over-specification against `T_train`.** If §5.3's items 2 and 4 are a
  sample-size effect, then `arch_m16` at `T = 240 000` should recover what
  `arch_m02` recovers at `T = 60 000`. If it does not, the cost is in the
  architecture and not in the data, which is a different and more serious claim.

---

## 6. Limitations to state before anyone quotes a number

- Every number here is one seed unless it says otherwise. The 21-run study put
  the relative seed spread at 2–4% for the conditional `W_1` (by a distance the
  most stable scalar), 5–8% for `W1/sd`, 6–7% for the ACF-of-squares ratio,
  10–12% for `sum Lip`, and 11–32% for `theta` and everything downstream of it.
  So `theta` and the bound should be quoted with a spread, not as point values.
- `garch` and `egarch` are bounded by construction. They do not test the
  Gaussian-tail obstruction, which is removed by hand; they test whether the
  dependence structure is learnable once it has been.
- `egarch` is log-GARCH with leverage, not Nelson's EGARCH. §3.2 says why.
- `lipschitz_empirical` over-counts, increasingly with `m`.
- The sup over levels in `theta` is dominated by the extremes; trimming to
  `u in [0.05, 0.95]` shrank it by up to 2.9x in the earlier study. That gap is
  the uniformity-in-`u` cost and no change of domain removes it.
