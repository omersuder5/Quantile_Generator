"""Turning a run, or a sweep, into files a person can read.

Per run:   results.json  every number, machine readable
           results.txt   the same, laid out

Per sweep: summary.csv      one row per configuration, flat, for pivoting
           COMPARISON.md    the tables, with the protocols spelled out
           overview.png, sweep_<axis>.png

`run_dirname` names a run folder by the axes that actually VARY in this sweep,
so a single-axis sweep gives `arch_m04` rather than a name carrying every
setting.  `varying_axes` computes that set.
"""
import csv
import json
import os

import numpy as np

__all__ = ["varying_axes", "run_dirname", "write_json", "render_txt",
           "write_summary_csv", "write_comparison_md", "PROTOCOL_NOTE"]


PROTOCOL_NOTE = """Three innovation protocols are used and they must not be confused.

* **shared** - target and learned recursion driven by the SAME stream. This is
  the synchronous coupling; `sup|X-Xhat|` is what the comparison lemma bounds,
  and it is an upper bound for the adapted Wasserstein distance, never a lower
  one.
* **independent** - fresh, unrelated streams on both sides. Everything under
  "the law" uses this, so the marginal and the autocorrelation are genuine
  law-level comparisons and not artefacts of a shared driver.
* **matched Gaussian AR** - an AR(m) fitted to the same training path and
  simulated on its own stream. The null: it reproduces the linear
  autocorrelation by construction, so anything the generator gets right that it
  does not is a non-Gaussian, non-linear feature.

`theta` is a maximum over a grid of 64 levels and 3000 windows, reported on the
data support and on the whole box. The bound uses the data-support value; the
proposition as usually stated asks for the box. The two differ by a factor of
roughly 2 to 8, and the gap is not an artefact: the recursion lives on a
forward-invariant set strictly inside the box, which is what `inv_R` reports.
"""


def _fmt(x, n=4):
    if x is None:
        return "--"
    if isinstance(x, float) and x != x:
        return "--"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{n}f}"


def varying_axes(configs):
    """Keys of `cfg['axes']` that take more than one value across the sweep."""
    if not configs:
        return []
    keys = sorted(configs[0]["axes"].keys())
    out = []
    for k in keys:
        vals = {repr(c["axes"].get(k)) for c in configs}
        if len(vals) > 1:
            out.append(k)
    return out


_NAME_ORDER = ["m", "S_target"]          # then everything else, alphabetically
_SHORT = {"m": "m", "S_target": "S", "seed_path": "sp", "seed_net": "sn",
          "epochs": "ep", "width": "w", "objective": "", "lip_mode": "lip",
          "pen_region": "pen", "tail_frac": "tf", "L_max": "Lmax"}


def run_dirname(axes, varying, index=None):
    """`arch_m04_S0p80` style, using only the axes that actually vary.

    The order is target, then m, then S, then the rest alphabetically, so a
    directory listing sorts the way the sweep reads.  m is zero-padded to two
    digits for the same reason.
    """
    rest = sorted(k for k in varying if k not in _NAME_ORDER and k != "target")
    bits = [str(axes.get("target", "run"))]
    for k in [k for k in _NAME_ORDER if k in varying] + rest:
        v = axes[k]
        if k == "m":
            s = f"{int(v):02d}"
        elif isinstance(v, float):
            s = f"{v:g}".replace(".", "p").replace("-", "neg")
        else:
            s = str(v).replace(".", "p")
        pre = _SHORT.get(k, k)
        bits.append(f"{pre}{s}")
    name = "_".join(bits)
    return name if index is None else f"{index:02d}_{name}"


def write_json(obj, path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=default)
    return path


def render_txt(r):
    """The per-run report, laid out."""
    L, A = [], None
    out = []
    A = out.append
    A("=" * 78)
    A(f"{r['label']}    {r['target_name']}")
    A("=" * 78)
    ax = r["axes"]
    A(f"target key          {ax['target']}")
    A(f"fitting arity m     {ax['m']}   "
      f"(the memory you presume; the target's own order is "
      f"{r['k_true'] if r['k_true'] is not None else 'infinite'})")
    A(f"requested S         {ax['S_target']}   achieved {r['hypotheses']['S_full']:.6f}")
    hy = r["hypotheses"]
    A(f"S_m = sum_{{i<=m}}     {hy['S_m']:.4f}" +
      (f"    amplification 1/(1-S_m) = {1/(1-hy['S_m']):.3f}"
       if hy["S_m"] < 1 else "    (>= 1: the standing hypothesis fails)"))
    A(f"L_(m+1) tail        {hy['L_tail']:.4f}    2 M L = {2*r['M']*hy['L_tail']:.4f}")
    A(f"per-lag moduli      {np.round(hy['ell'][:8], 4).tolist()}"
      + (" ..." if len(hy["ell"]) > 8 else ""))
    v = r["verify"]
    A(f"moduli num vs ana   max dev "
      f"{max(abs(a-b) for a, b in zip(v['ell_analytic'], v['ell_numeric'])):.2e}")
    A(f"range contained     {v['in_range']}  "
      f"[{v['range_min']:.4f}, {v['range_max']:.4f}]")
    A("")
    A("HYPOTHESES OF THE GENERATOR PROPOSITION")
    A(f"  sum_i Lip_i(qhat)   box {hy['lip_box']:.4f}   data {hy['lip_data']:.4f}"
      f"   (target S_m {hy['S_m']:.4f})")
    A(f"  per lag, box        {np.round(hy['lip_per_lag_box'][:8], 4).tolist()}")
    A(f"  theta               data {hy['theta_data']:.4f}   box {hy['theta_box']:.4f}"
      f"   (ratio {hy['theta_box']/max(hy['theta_data'],1e-12):.1f}x)")
    if "theta_invariant" in hy:
        A(f"  theta on K=[-R,R]   {hy['theta_invariant']:.4f}   "
          f"bound there {_fmt(hy['bound_invariant'], 3)}")
    A("  theta trimmed       " + "  ".join(f"eps={k}: {_fmt(v_,4)}"
                                           for k, v_ in hy["theta_trim"].items()))
    A(f"  bound (theta+2ML)/(1-S_m) = {_fmt(hy['bound'], 4)}")
    A("")
    rp = r["repair"]
    A("IS THE REPAIR LEMMA DOING ANYTHING?")
    A(f"  crossings (data)    fraction {rp['cross_frac']:.2e}   "
      f"worst increment {rp['cross_worst']:+.2e}")
    A(f"  min d qhat/du (box) {rp['du_min']:+.4f}   (monotone there iff >= 0)")
    A(f"  sup|qhat| on box    {rp['sup_q_box']:.4f}   X-valued: {rp['box_invariant']}")
    A(f"  invariant radius R  {_fmt(rp['inv_R'], 3)}   (sup|qhat| there "
      f"{_fmt(rp['inv_sup'], 3)})")
    A(f"  repairs needed      {', '.join(rp['repairs_needed'])}")
    A("")
    pw, sy = r["pathwise"], r["synchronisation"]
    A("PATHWISE, SHARED INNOVATION STREAM")
    A(f"  sup_t |X-Xhat|      {pw['sup_err']:.4f}    mean {pw['mean_err']:.4f}"
      f"    99.9pct {pw['q999_err']:.4f}")
    if pw["bound"]:
        A(f"  bound               {pw['bound']:.4f}   slack x{pw['slack']:.1f}"
          f"   holds: {pw['bound_holds']}")
    else:
        A("  bound               void (S_m >= 1)")
    A(f"  spread, 24 starts   t={sy['n_steps']//2} {sy['spread_half']:.3e}   "
      f"t={sy['n_steps']-1} {sy['spread_end']:.3e}")
    ly = r["lyapunov"]
    A(f"  E log sum|d/dz|     {ly['value']:+.4f}   "
      + ("(= the top Lyapunov exponent, m = 1)" if ly["exact"]
         else "(an upper bound for the top exponent, m > 1)"))
    if ly.get("decomposition"):
        d = ly["decomposition"]
        A(f"    = log 4(1-rho) {d['log_sup_slope']:+.4f}  +  E log|X|/M "
          f"{d['E_log_abs_X_over_M']:+.4f}  =  {d['lambda_analytic']:+.4f}")
    A("")
    mg, cd, ac = r["marginal"], r["conditional"], r["autocorrelation"]
    A("THE TWO LAWS, FRESH INDEPENDENT STREAMS")
    A(f"  sd                  target {mg['sd_tru']:.4f}   generated {mg['sd_gen']:.4f}"
      f"   ratio {mg['sd_ratio']:.4f}")
    A(f"  W1 marginal         generated {mg['w1']:.5f} ({mg['w1_over_sd']:.4f} sd)"
      + (f"   matched AR {mg['w1_base']:.5f} ({mg['w1_over_sd_base']:.4f} sd)"
         if "w1_base" in mg else ""))
    A(f"  ACF lags 1-5 target  {np.round(ac['acf_tru'][1:6], 4).tolist()}")
    A(f"               learned {np.round(ac['acf_gen'][1:6], 4).tolist()}")
    if "acf_base" in ac:
        A(f"               AR      {np.round(ac['acf_base'][1:6], 4).tolist()}")
    A(f"  ACF sq  target       {np.round(ac['acfsq_tru'][1:6], 4).tolist()}")
    A(f"          learned      {np.round(ac['acfsq_gen'][1:6], 4).tolist()}")
    if "acfsq_base" in ac:
        A(f"          AR           {np.round(ac['acfsq_base'][1:6], 4).tolist()}")
    if ac["acfsq_ratio_lag1"] is not None:
        A(f"  ACFsq lag-1 ratio   generated/target {ac['acfsq_ratio_lag1']:.3f}"
          + (f"   AR/target {ac['acfsq_ratio_lag1_base']:.3f}"
             if ac.get("acfsq_ratio_lag1_base") is not None else ""))
    else:
        A("  ACFsq lag-1 ratio   not reported: the target's own value is near zero")
    A(f"  cond sd range       target {cd['cond_sd_range_tru']:.4f}   "
      f"generated {cd['cond_sd_range_gen']:.4f}"
      + (f"   AR {cd['cond_sd_range_base']:.4f}" if "cond_sd_range_base" in cd else ""))
    if cd.get("note"):
        A(f"                      ({cd['note']})")
    A(f"  conditional W1      generated {cd['cond_w1']:.5f}"
      + (f"   matched AR {cd['cond_w1_base']:.5f}" if "cond_w1_base" in cd else ""))
    mm = r["mmd"]
    A(f"  block-MMD^2 (L={mm['block_len']})  generated {mm['generated']['mmd2']:+.6f} "
      f"p={mm['generated']['p_value']:.3f}"
      + (f"   AR {mm['baseline']['mmd2']:+.6f} p={mm['baseline']['p_value']:.3f}"
         if "baseline" in mm else ""))
    A("")
    A("  marginal quantiles   (err/sd is the honest column; the ratio blows up")
    A("                        wherever the target quantile crosses zero)")
    A(f"  {'level':>8}{'target':>10}{'learned':>10}{'err/sd':>10}{'ratio':>10}")
    for q in mg["quantiles"]:
        rr = f"{q['ratio']:.3f}" if q["ratio_reliable"] else "n/a"
        A(f"  {q['level']:8.3f}{q['target']:10.4f}{q['generated']:10.4f}"
          f"{q['err_over_sd']:10.4f}{rr:>10}")
    A("")
    pr = r["pricing"]
    A(f"  European put (K={pr['K']}, {pr['n_exercise']} dates)")
    A(f"    target {pr['target']['european']:.5f}   "
      f"generated {pr['generated']['european']:.5f}"
      + (f"   AR {pr['baseline']['european']:.5f}" if "baseline" in pr else ""))
    A("  Bermudan put")
    A(f"    target {pr['target']['bermudan']:.5f}   "
      f"generated {pr['generated']['bermudan']:.5f}"
      + (f"   AR {pr['baseline']['bermudan']:.5f}" if "baseline" in pr else ""))
    A("  early-exercise premium  (the functional that needs the adapted structure)")
    A(f"    target {pr['target']['premium']:.5f}   "
      f"generated {pr['generated']['premium']:.5f}"
      + (f"   AR {pr['baseline']['premium']:.5f}" if "baseline" in pr else ""))
    A("")
    A(f"  ESN realisation  max|ESN - direct recursion| = {r['esn']['error']}")
    A(f"  ESN size N = {r['esn']['N']}  (W + (m-1) shift units + 1 constant)")
    A(f"  final training loss {r['fit']['final_loss']:.6f}"
      f"   wall time {r['secs']:.0f}s")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ sweep ---

_SUMMARY_FIELDS = [
    ("label", lambda r: r["label"]),
    ("target", lambda r: r["axes"]["target"]),
    ("m", lambda r: r["axes"]["m"]),
    ("S_target", lambda r: r["axes"]["S_target"]),
    ("S_m", lambda r: r["hypotheses"]["S_m"]),
    ("L_tail", lambda r: r["hypotheses"]["L_tail"]),
    ("theta_data", lambda r: r["hypotheses"]["theta_data"]),
    ("theta_box", lambda r: r["hypotheses"]["theta_box"]),
    ("lip_box", lambda r: r["hypotheses"]["lip_box"]),
    ("lip_data", lambda r: r["hypotheses"]["lip_data"]),
    ("bound", lambda r: r["pathwise"]["bound"]),
    ("sup_err", lambda r: r["pathwise"]["sup_err"]),
    ("slack", lambda r: r["pathwise"]["slack"]),
    ("bound_holds", lambda r: r["pathwise"]["bound_holds"]),
    ("spread_end", lambda r: r["synchronisation"]["spread_end"]),
    ("lyapunov", lambda r: r["lyapunov"]["value"]),
    ("w1_over_sd", lambda r: r["marginal"]["w1_over_sd"]),
    ("w1_over_sd_base", lambda r: r["marginal"].get("w1_over_sd_base")),
    ("cond_w1", lambda r: r["conditional"]["cond_w1"]),
    ("cond_w1_base", lambda r: r["conditional"].get("cond_w1_base")),
    ("acfsq1_tru", lambda r: r["autocorrelation"]["acfsq_tru"][1]),
    ("acfsq1_gen", lambda r: r["autocorrelation"]["acfsq_gen"][1]),
    ("acfsq1_base", lambda r: r["autocorrelation"].get("acfsq_base", [None, None])[1]),
    ("cond_sd_range_ratio", lambda r: r["conditional"]["cond_sd_range_ratio"]),
    ("tail_err_999", lambda r: next(q["err_over_sd"] for q in r["marginal"]["quantiles"]
                                    if q["level"] == 0.999)),
    ("mmd_p", lambda r: r["mmd"]["generated"]["p_value"]),
    ("mmd_p_base", lambda r: r["mmd"].get("baseline", {}).get("p_value")),
    ("premium_tru", lambda r: r["pricing"]["target"]["premium"]),
    ("premium_gen", lambda r: r["pricing"]["generated"]["premium"]),
    ("esn_err", lambda r: r["esn"]["error"]),
    ("cross_frac", lambda r: r["repair"]["cross_frac"]),
    ("inv_R", lambda r: r["repair"]["inv_R"]),
    ("repairs", lambda r: "; ".join(r["repair"]["repairs_needed"])),
    ("secs", lambda r: r["secs"]),
]


def write_summary_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([n for n, _ in _SUMMARY_FIELDS])
        for r in rows:
            out = []
            for _, fn in _SUMMARY_FIELDS:
                try:
                    out.append(fn(r))
                except Exception:
                    out.append(None)
            w.writerow(out)
    return path


def _esc(s):
    return str(s).replace("|", "\\|")


def _table(head, rows):
    al = ["---:"] * len(head)
    al[0] = ":---"
    return ("| " + " | ".join(_esc(h) for h in head) + " |\n"
            + "| " + " | ".join(al) + " |\n"
            + "".join("| " + " | ".join(_esc(c) for c in r) + " |\n" for r in rows))


def write_comparison_md(rows, path, cfg_echo="", varying=()):
    L = ["# Comparison across the sweep", ""]
    L.append(f"{len(rows)} configuration(s). Axes varying in this sweep: "
             f"**{', '.join(varying) if varying else 'none (single run)'}**.")
    L.append("")
    if cfg_echo:
        L.append("Fixed configuration:\n\n```\n" + cfg_echo.rstrip() + "\n```\n")
    L.append(PROTOCOL_NOTE)

    L.append("\n## 1. Hypotheses\n")
    L.append("`S_m` and `L_(m+1)` are properties of the TARGET at the fitting "
             "arity m; `Lip` is the same quantity measured on the FIT. A fit is "
             "not told S.\n")
    L.append(_table(
        ["run", "m", "S req", "S_m", "L(m+1)", "ΣLip box", "ΣLip data",
         "θ data", "θ box", "bound"],
        [[r["label"], r["axes"]["m"], _fmt(r["axes"]["S_target"], 3),
          _fmt(r["hypotheses"]["S_m"]), _fmt(r["hypotheses"]["L_tail"]),
          _fmt(r["hypotheses"]["lip_box"]), _fmt(r["hypotheses"]["lip_data"]),
          _fmt(r["hypotheses"]["theta_data"]), _fmt(r["hypotheses"]["theta_box"]),
          _fmt(r["hypotheses"]["bound"], 3)] for r in rows]))

    L.append("\n## 2. Repair conditions\n")
    L.append("Crossings are measured on the DATA SUPPORT; the minimum of "
             "dq̂/du and sup|q̂| on the whole BOX. A fit can be monotone where "
             "the process lives and not monotone in a corner.\n")
    L.append(_table(
        ["run", "crossing frac", "min ∂q̂/∂u (box)", "sup|q̂| (box)",
         "X-valued", "inv. R", "repairs needed"],
        [[r["label"], f"{r['repair']['cross_frac']:.2e}",
          f"{r['repair']['du_min']:+.4f}", _fmt(r["repair"]["sup_q_box"], 3),
          "yes" if r["repair"]["box_invariant"] else "no",
          _fmt(r["repair"]["inv_R"], 2),
          ", ".join(r["repair"]["repairs_needed"])] for r in rows]))

    L.append("\n## 3. Pathwise, shared innovations\n")
    L.append(_table(
        ["run", "sup|X−X̂|", "mean", "bound", "slack", "holds",
         "spread end", "E log Σ|∂q̂/∂z|"],
        [[r["label"], _fmt(r["pathwise"]["sup_err"]),
          _fmt(r["pathwise"]["mean_err"]), _fmt(r["pathwise"]["bound"], 3),
          (f"×{r['pathwise']['slack']:.1f}" if r["pathwise"]["slack"] else "--"),
          str(r["pathwise"]["bound_holds"]),
          f"{r['synchronisation']['spread_end']:.1e}",
          f"{r['lyapunov']['value']:+.3f}"] for r in rows]))

    L.append("\n## 4. The two laws, independent streams\n")
    L.append(_table(
        ["run", "W₁/sd", "W₁/sd (AR)", "cond W₁", "cond W₁ (AR)", "sd ratio",
         "ACF₁ tru/gen", "ACFsq₁ tru/gen/AR", "q₀.₉₉₉ err/sd", "MMD p"],
        [[r["label"], _fmt(r["marginal"]["w1_over_sd"]),
          _fmt(r["marginal"].get("w1_over_sd_base")),
          _fmt(r["conditional"]["cond_w1"]),
          _fmt(r["conditional"].get("cond_w1_base")),
          _fmt(r["marginal"]["sd_ratio"], 3),
          f"{r['autocorrelation']['acf_tru'][1]:+.3f} / "
          f"{r['autocorrelation']['acf_gen'][1]:+.3f}",
          f"{r['autocorrelation']['acfsq_tru'][1]:+.3f} / "
          f"{r['autocorrelation']['acfsq_gen'][1]:+.3f} / "
          + (f"{r['autocorrelation']['acfsq_base'][1]:+.3f}"
             if "acfsq_base" in r["autocorrelation"] else "--"),
          _fmt(next(q["err_over_sd"] for q in r["marginal"]["quantiles"]
                    if q["level"] == 0.999), 3),
          _fmt(r["mmd"]["generated"]["p_value"], 3)] for r in rows]))

    L.append("\n## 5. Early-exercise premium\n")
    L.append("Bermudan minus European, the functional that needs the adapted "
             "structure rather than the path law.\n")
    L.append(_table(
        ["run", "premium target", "premium generated", "premium AR",
         "relative error"],
        [[r["label"], _fmt(r["pricing"]["target"]["premium"], 5),
          _fmt(r["pricing"]["generated"]["premium"], 5),
          _fmt(r["pricing"].get("baseline", {}).get("premium"), 5),
          _fmt(r["pricing"]["premium_rel_err"], 4)] for r in rows]))

    L.append("\n## Files\n")
    L.append("```\n"
             "<run folder>/results.json   every number\n"
             "<run folder>/results.txt    the same, laid out\n"
             "<run folder>/figures.png    nine panels\n"
             "<run folder>/net.pt         the fitted weights, to re-analyse\n"
             "                            without retraining\n"
             "summary.csv                 one flat row per configuration\n"
             "COMPARISON.md               this file\n"
             "overview.png                every run side by side\n"
             "sweep_<axis>.png            one panel set per varying axis\n"
             "config_used.json            the exact configuration of this run\n"
             "```\n")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path
