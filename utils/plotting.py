"""Figures.

Three series appear everywhere and they are assigned FIXED colours, linestyles
and markers, never matplotlib's cycle:

    target            blue    solid    circle
    generated         orange  dashed   square
    matched AR        aqua    dotted   triangle

Fixed assignment matters because these figures are read side by side across
dozens of runs: if the colour followed the plotting order rather than the
entity, a run where the baseline is omitted would repaint the generator and
every comparison by eye would be wrong.  The three hues clear colour-vision
separation on all pairs (worst deutan Delta E 9.2, normal-vision 24.0), and
linestyle and marker carry the identity redundantly so the figures survive
greyscale printing.

Reference lines (y = 0, y = 1, the bound) are annotations, not series: they are
drawn in a recessive grey or, for the bound, in red with a direct label.  No
figure here uses two y-scales.
"""
import numpy as np
import matplotlib.pyplot as plt

# NOTE: the backend is deliberately NOT forced here.  `quantile_learning.py`
# selects Agg before importing this module, because it runs headless; a
# notebook keeps its inline backend, which forcing Agg here would silently
# break (every plt.show() would render nothing).

__all__ = ["STYLE", "C_TARGET", "C_GEN", "C_BASE", "panel_figure",
           "sweep_figure", "overview_figure"]

C_TARGET = "#2a78d6"
C_GEN = "#eb6834"
C_BASE = "#1baf7a"
C_RULE = "#52514e"
C_BOUND = "#e34948"

STYLE = {
    "target":    dict(color=C_TARGET, ls="-",  marker="o", ms=3.5, lw=1.6,
                      label="target"),
    "generated": dict(color=C_GEN,    ls="--", marker="s", ms=3.5, lw=1.6,
                      label="generated"),
    "baseline":  dict(color=C_BASE,   ls=":",  marker="^", ms=3.5, lw=1.6,
                      label="matched Gaussian AR"),
}


def _tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_RULE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=C_RULE, labelsize=8)
    ax.grid(True, color=C_RULE, alpha=0.12, lw=0.6)
    ax.set_axisbelow(True)


def _style(which, **over):
    d = dict(STYLE[which])
    d.update(over)
    return d


def panel_figure(res, tru, gen, base, target, net, path_out, title=""):
    """The nine-panel per-run figure.

    Row 1  the paths, the marginals, the marginal quantile error
    Row 2  the ACF, the ACF of squares, the conditional standard deviation
    Row 3  q(u; z_1, 0, ...) overlay, theta by level, the pathwise error
    """
    M = target.M
    m = net.m
    sd = float(np.std(tru))
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))

    # --- paths -------------------------------------------------------------
    a = ax[0, 0]
    a.plot(tru[:500], **_style("target", marker="", lw=0.8))
    a.plot(gen[:500], **_style("generated", marker="", lw=0.8))
    a.set_title("paths, independent innovation streams", fontsize=10)
    a.set_xlabel("t"); a.legend(fontsize=7, frameon=False); _tidy(a)

    # --- marginals ---------------------------------------------------------
    a = ax[0, 1]
    lo = min(tru.min(), gen.min()); hi = max(tru.max(), gen.max())
    bb = np.linspace(lo, hi, 100)
    a.hist(tru, bins=bb, density=True, alpha=0.55, color=C_TARGET, label="target")
    a.hist(gen, bins=bb, density=True, alpha=0.55, color=C_GEN, label="generated")
    w1 = res["marginal"]["w1_over_sd"]
    a.set_title(f"stationary marginal   $W_1$/sd = {w1:.4f}", fontsize=10)
    a.set_xlabel("$X$"); a.legend(fontsize=7, frameon=False); _tidy(a)

    # --- marginal quantile error (sd units, never a ratio) -----------------
    a = ax[0, 2]
    qs = [r["level"] for r in res["marginal"]["quantiles"]]
    eg = [r["err_over_sd"] for r in res["marginal"]["quantiles"]]
    a.plot(qs, eg, **_style("generated"))
    if "quantiles_base" in res["marginal"]:
        eb = [r["err_over_sd"] for r in res["marginal"]["quantiles_base"]]
        a.plot(qs, eb, **_style("baseline"))
    a.axhline(0, color=C_RULE, lw=0.8)
    a.set_xscale("logit")
    a.set_title(r"marginal quantile error $(\hat q_\alpha-q_\alpha)/\mathrm{sd}$",
                fontsize=10)
    a.set_xlabel(r"level $\alpha$"); a.legend(fontsize=7, frameon=False); _tidy(a)

    # --- ACF and ACF of squares -------------------------------------------
    ac = res["autocorrelation"]
    for a, key, ttl in ((ax[1, 0], "acf", "autocorrelation of $X_t$"),
                        (ax[1, 1], "acfsq", r"autocorrelation of $(X_t-\bar X)^2$")):
        a.plot(ac[f"{key}_tru"], **_style("target"))
        a.plot(ac[f"{key}_gen"], **_style("generated"))
        if f"{key}_base" in ac:
            a.plot(ac[f"{key}_base"], **_style("baseline"))
        a.axhline(0, color=C_RULE, lw=0.8)
        a.set_title(ttl, fontsize=10); a.set_xlabel("lag")
        a.legend(fontsize=7, frameon=False); _tidy(a)

    # --- conditional sd ----------------------------------------------------
    a = ax[1, 2]
    cd = res["conditional"]
    mid = np.array(cd["mid"])
    a.plot(mid, cd["cond_sd_tru"], **_style("target"))
    a.plot(mid, cd["cond_sd_gen"], **_style("generated"))
    if "cond_sd_base" in cd:
        a.plot(mid, cd["cond_sd_base"], **_style("baseline"))
    note = "" if cd.get("scale_is_state_dependent") else "\n(homoskedastic target: this is sampling noise)"
    a.set_title("conditional sd given $X_{t-1}$" + note, fontsize=10)
    a.set_xlabel("$X_{t-1}$"); a.legend(fontsize=7, frameon=False); _tidy(a)

    # --- the quantile map overlay -----------------------------------------
    a = ax[2, 0]
    zg = np.linspace(-M, M, 300)
    Zm = np.zeros((len(zg), m)); Zm[:, 0] = zg
    for uu, ls in ((0.05, ":"), (0.5, "-"), (0.95, "--")):
        a.plot(zg, target.q(np.full(len(zg), uu), Zm), color=C_TARGET, ls=ls, lw=1.4)
        a.plot(zg, net.q_np(np.full(len(zg), uu), Zm), color=C_GEN, ls=ls, lw=1.2)
    a.plot([], [], color=C_TARGET, lw=1.4, label="target $q_m$")
    a.plot([], [], color=C_GEN, lw=1.2, label=r"learned $\hat q$")
    a.set_title("$q(u;z_1,0,\\dots)$ at $u$ = 0.05 (:), 0.5 (-), 0.95 (--)",
                fontsize=10)
    a.set_xlabel("$z_1$"); a.legend(fontsize=7, frameon=False); _tidy(a)

    # --- theta by level ----------------------------------------------------
    a = ax[2, 1]
    hy = res["hypotheses"]
    a.plot(hy["theta_profile_u"], hy["theta_profile_err"], color=C_GEN, lw=1.6,
           label="on the data support")
    a.axhline(hy["theta_data"], color=C_RULE, ls="--", lw=0.9)
    a.annotate(f"$\\theta$ = {hy['theta_data']:.4f}",
               (0.02, hy["theta_data"]), fontsize=8, color=C_RULE,
               va="bottom")
    a.set_title(r"$\max_z|\hat q - q_m|$ by level: the uniformity-in-$u$ gap",
                fontsize=10)
    a.set_xlabel("$u$"); a.legend(fontsize=7, frameon=False); _tidy(a)

    # --- pathwise error ----------------------------------------------------
    a = ax[2, 2]
    pw = res["pathwise"]
    err = np.maximum(np.array(pw["err_head"]), 1e-14)
    a.semilogy(err, color=C_GEN, lw=0.7, label=r"$|X_t-\hat X_t|$, shared stream")
    if pw["bound"]:
        a.axhline(pw["bound"], color=C_BOUND, ls="--", lw=1.4)
        a.annotate(f"bound {pw['bound']:.3f}  (slack x{pw['slack']:.1f})",
                   (0.02, pw["bound"]), xycoords=("axes fraction", "data"),
                   fontsize=8, color=C_BOUND, va="bottom")
    else:
        a.set_title("", fontsize=10)
    a.set_title("pathwise error under the synchronous coupling", fontsize=10)
    a.set_xlabel("t"); a.legend(fontsize=7, frameon=False, loc="lower right")
    _tidy(a)

    fig.suptitle(title or res.get("label", ""), fontsize=12)
    plt.tight_layout()
    fig.savefig(path_out, dpi=110)
    plt.close(fig)
    return path_out


# ---------------------------------------------------------------- cross-run ---

def _axis_labels(ax, labels):
    x = np.arange(len(labels))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    return x


def overview_figure(rows, path_out, title="all runs"):
    """Four panels across every run of a sweep: the law, the conditional law,
    whether the fit inherits the contraction, and the pathwise error."""
    if not rows:
        return None
    labels = [r["label"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(max(12, 1.1 * len(rows)), 8))
    x = np.arange(len(rows))

    a = ax[0, 0]
    a.bar(x - 0.2, [r["marginal"]["w1_over_sd"] for r in rows], 0.4,
          color=C_GEN, label="generated")
    if "w1_over_sd_base" in rows[0]["marginal"]:
        a.bar(x + 0.2, [r["marginal"]["w1_over_sd_base"] for r in rows], 0.4,
              color=C_BASE, label="matched Gaussian AR")
    a.set_yscale("log"); a.set_title(r"marginal $W_1$ / sd", fontsize=10)
    a.legend(fontsize=8, frameon=False); _axis_labels(a, labels); _tidy(a)

    a = ax[0, 1]
    a.bar(x - 0.2, [r["conditional"]["cond_w1"] for r in rows], 0.4,
          color=C_GEN, label="generated")
    if "cond_w1_base" in rows[0]["conditional"]:
        a.bar(x + 0.2, [r["conditional"]["cond_w1_base"] for r in rows], 0.4,
              color=C_BASE, label="matched Gaussian AR")
    a.set_yscale("log")
    a.set_title(r"conditional $W_1$ given $X_{t-1}$", fontsize=10)
    a.legend(fontsize=8, frameon=False); _axis_labels(a, labels); _tidy(a)

    a = ax[1, 0]
    a.plot(x, [r["hypotheses"]["S_m"] for r in rows],
           **_style("target", label=r"$S_m$ (target)"))
    a.plot(x, [r["hypotheses"]["lip_box"] for r in rows],
           **_style("generated", label=r"$\sum_i$Lip$_i(\hat q)$, box"))
    a.plot(x, [r["hypotheses"]["lip_data"] for r in rows],
           **_style("baseline", label=r"$\sum_i$Lip$_i(\hat q)$, data"))
    a.axhline(1, color=C_BOUND, ls=":", lw=1)
    a.set_title("does the fit inherit the contraction?", fontsize=10)
    a.legend(fontsize=8, frameon=False); _axis_labels(a, labels); _tidy(a)

    a = ax[1, 1]
    a.semilogy(x, [max(r["pathwise"]["sup_err"], 1e-14) for r in rows],
               **_style("generated", label=r"realised $\sup_t|X_t-\hat X_t|$"))
    bd = [r["pathwise"]["bound"] if r["pathwise"]["bound"] else np.nan for r in rows]
    a.semilogy(x, bd, color=C_BOUND, ls="--", marker="v", ms=3.5, lw=1.6,
               label=r"$(\theta+2ML_{m+1})/(1-S_m)$")
    a.set_title("pathwise error against its bound", fontsize=10)
    a.legend(fontsize=8, frameon=False); _axis_labels(a, labels); _tidy(a)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(); fig.savefig(path_out, dpi=110); plt.close(fig)
    return path_out


def sweep_figure(rows, axis_key, path_out, title="", logx=True):
    """Four panels along ONE varying axis (m, S, rho, ...).

    No panel uses two y-scales: the exponent gets a panel of its own rather
    than being hung off a second axis.
    """
    if len(rows) < 2:
        return None
    xv = np.array([r["axes"][axis_key] for r in rows], dtype=float)
    o = np.argsort(xv)
    xv = xv[o]
    rows = [rows[i] for i in o]
    fig, ax = plt.subplots(1, 4, figsize=(19, 3.8))
    sx = (lambda a: a.set_xscale("log", base=2)) if logx else (lambda a: None)

    a = ax[0]
    a.plot(xv, [r["hypotheses"]["S_m"] for r in rows],
           **_style("target", label=r"$S_m$"))
    a.plot(xv, [r["hypotheses"]["L_tail"] for r in rows],
           **_style("generated", label=r"$L_{m+1}$"))
    a.plot(xv, [r["hypotheses"]["lip_box"] for r in rows],
           **_style("baseline", label=r"$\sum_i$Lip$_i(\hat q)$"))
    a.axhline(1, color=C_BOUND, ls=":", lw=1)
    a.set_title("the two competing terms", fontsize=10)

    a = ax[1]
    a.plot(xv, [r["hypotheses"]["theta_data"] for r in rows],
           **_style("generated", label=r"$\theta$ (data support)"))
    a.plot(xv, [r["hypotheses"]["theta_box"] for r in rows],
           **_style("baseline", label=r"$\theta$ (box)"))
    a.set_yscale("log")
    a.set_title("approximation error", fontsize=10)

    a = ax[2]
    a.plot(xv, [max(r["pathwise"]["sup_err"], 1e-14) for r in rows],
           **_style("generated", label=r"$\sup_t|X_t-\hat X_t|$"))
    bd = [r["pathwise"]["bound"] if r["pathwise"]["bound"] else np.nan for r in rows]
    a.plot(xv, bd, color=C_BOUND, ls="--", marker="v", ms=3.5, lw=1.6,
           label="bound")
    a.set_yscale("log")
    a.set_title("realised pathwise error vs its bound", fontsize=10)

    a = ax[3]
    a.plot(xv, [r["marginal"]["w1_over_sd"] for r in rows],
           **_style("generated", label=r"$W_1$/sd"))
    a.plot(xv, [r["conditional"]["cond_w1"] for r in rows],
           **_style("baseline", label=r"conditional $W_1$"))
    a.set_yscale("log")
    a.set_title("the law", fontsize=10)

    for a in ax:
        sx(a)
        a.set_xlabel(axis_key)
        a.set_xticks(xv)
        a.set_xticklabels([f"{v:g}" for v in xv])
        a.legend(fontsize=7, frameon=False)
        _tidy(a)
    fig.suptitle(title or f"sweep over {axis_key}", fontsize=12)
    plt.tight_layout(); fig.savefig(path_out, dpi=110); plt.close(fig)
    return path_out
