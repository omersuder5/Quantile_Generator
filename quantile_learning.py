"""Run the whole grid in `config.py`.

    python quantile_learning.py                 # everything in config.py
    python quantile_learning.py --dry-run       # print the grid and stop
    python quantile_learning.py --only arch_m02 # one run, by folder name
    python quantile_learning.py --quick         # tiny budget, to shake it out

Output lands in

    reports/run_<YYYY-mm-dd_HHMMSS>/
        <target>_<the axes that vary>/ results.json, results.txt,
                                       figures.png, net.pt
        summary.csv        one flat row per configuration
        COMPARISON.md      the tables
        overview.png       every run side by side
        sweep_<axis>.png   one panel set per varying axis
        config_used.json   exactly what was run
        run.log

Resumable: a configuration whose results.json already exists is skipped, so an
interrupted sweep can be restarted by pointing REPORTS_DIR at the same folder
(or passing --resume-into).
"""
import argparse
import itertools
import json
import os
import sys
import time
import traceback
from datetime import datetime

import matplotlib
matplotlib.use("Agg")          # headless: must precede any pyplot import

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as CFG
from backend import resolve, describe
from generators.synthetic_generators import build_target
from generators.quantile_generator import ClosedQuantileRecursion
from generators.esn_realization import ESNRealization
from generators.simulate_paths import simulate, make_windows
from fit_generator import fit
from loss.mmd import blocks, permutation_test
from utils.hypotheses import verify_assumptions, hypothesis_report
from utils.repair import repair_report
from utils.pathwise import pathwise_report, synchronisation_report
from utils.lyapunov import lyapunov, lyapunov_decomposition
from utils.autocorrelation import acf_report
from utils.marginal import marginal_report
from utils.conditional import conditional_report
from utils.pricing import pricing_report
from utils.baselines import MatchedGaussianAR
from utils.plotting import panel_figure, overview_figure, sweep_figure
from utils.reporting import (varying_axes, run_dirname, write_json, render_txt,
                             write_summary_csv, write_comparison_md)

# Targets whose conditional SCALE depends on the state.  Used only to label
# the conditional-sd diagnostic, whose range ratio is sampling noise elsewhere.
STATE_DEPENDENT_SCALE = {"hetero", "arch", "oscillatory"}


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

def expand_grid(cfg=CFG):
    """Cartesian product of TARGETS x M x S_TARGET x DATA x NET x TRAIN x EVAL."""
    sections = [("data", cfg.DATA), ("net", cfg.NET), ("train", cfg.TRAIN),
                ("eval", cfg.EVAL)]
    keys, pools = [], []
    for sec, d in sections:
        for k, v in d.items():
            if not isinstance(v, (list, tuple)):
                raise TypeError(f"{sec}['{k}'] must be a list, got {type(v).__name__}")
            keys.append((sec, k))
            pools.append(list(v))

    out = []
    for target, m, S in itertools.product(cfg.TARGETS, cfg.M, cfg.S_TARGET):
        for combo in itertools.product(*pools):
            flat = {"target": target, "m": int(m), "S_target": float(S)}
            resolved = {"data": {}, "net": {}, "train": {}, "eval": {}}
            for (sec, k), val in zip(keys, combo):
                resolved[sec][k] = val
                flat[k] = val
            out.append({"axes": flat, **resolved})
    return out


def _quick(grid):
    for g in grid:
        g["data"]["T_train"] = 6000
        g["train"]["epochs"] = 6
        g["eval"]["T_eval"] = 6000
        g["eval"]["T_shared"] = 3000
        g["eval"]["n_perm"] = 30
        g["eval"]["mmd_max_n"] = 300
        g["net"]["width"] = 64
        # keep the override OUT of `axes`, so it cannot become a sweep axis
        # and leak into the folder names
        if g["axes"]["target"] == "longmem":
            g["n_lag_override"] = 400
    return grid


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------

def run_one(spec, outdir, specifics=None, diagnostics=None, verbose=False,
            quick=False):
    """Fit and diagnose one configuration.  Returns the results dict."""
    os.makedirs(outdir, exist_ok=True)
    diag = dict(CFG.DIAGNOSTICS if diagnostics is None else diagnostics)
    spec_all = dict(CFG.TARGET_SPECIFICS if specifics is None else specifics)
    ax, D, N, T, E = (spec["axes"], spec["data"], spec["net"], spec["train"],
                      spec["eval"])
    t_start = time.time()

    name = ax["target"]
    m = int(ax["m"])
    M = float(D["M_state"])
    shape = dict(spec_all.get(name, {}))
    if spec.get("n_lag_override") is not None:
        shape["n_lag"] = spec["n_lag_override"]

    # --- the target, at exactly the requested S ----------------------------
    target = build_target(name, ax["S_target"], shape, M=M, noise=D["noise"])

    # S >= 1 is only coherent for the logistic target, and a contractive fit
    # cannot approximate an expansive one, so the penalty comes off.
    lip_mode = T["lip_mode"]
    forced = None
    if target.S >= 1.0 and lip_mode != "none":
        forced = (f"the target has S = {target.S:.3f} >= 1, so lip_mode was "
                  f"forced from {lip_mode!r} to 'none'")
        lip_mode = "none"

    # --- data, always at the target's own memory order ---------------------
    path = simulate(target, D["T_train"], burn=D["burn"], seed=D["seed_path"])
    Z, Xs = make_windows(path, m)

    # --- fit ---------------------------------------------------------------
    dev, dt = resolve(getattr(CFG, "DEVICE", "auto"),
                      getattr(CFG, "DTYPE", "float32"))
    net, fitinfo = fit(Z, Xs, m, objective=T["objective"], width=N["width"],
                       act=N["act"], seed_net=N["seed_net"], v_mode=N["v_mode"],
                       v_clip=N["v_clip"], epochs=T["epochs"], batch=T["batch"],
                       lr=T["lr"], lip_mode=lip_mode, pen_region=T["pen_region"],
                       pen_sigma=T["pen_sigma"], L_max=T["L_max"], lam=T["lam"],
                       tail_frac=T["tail_frac"], edge=T["edge"],
                       u_min=T["u_min"], M=M, mmd_block=T["mmd_block"],
                       mmd_batch=T["mmd_batch"], mmd_burn=T["mmd_burn"],
                       device=dev, dtype=dt, verbose=verbose)
    rec = ClosedQuantileRecursion(net)
    if diag.get("save_net"):
        torch.save({"state_dict": net.state_dict(), "m": m,
                    "width": N["width"], "act": N["act"],
                    "v_mode": N["v_mode"], "v_clip": N["v_clip"]},
                   os.path.join(outdir, "net.pt"))

    res = {"label": os.path.basename(outdir), "axes": ax,
           "target_name": target.name, "k_true": target.k_true, "M": M,
           "fit": fitinfo, "lip_mode_used": lip_mode, "lip_mode_forced": forced}

    # --- the standing hypotheses -------------------------------------------
    res["verify"] = (verify_assumptions(target, m=m) if diag.get("verify_assumptions")
                     else {})
    rp = (repair_report(net, Z, M=M) if diag.get("repair") else {})
    res["repair"] = rp
    res["hypotheses"] = (hypothesis_report(net, target, Z, m, M=M,
                                           inv_R=rp.get("inv_R"))
                         if diag.get("hypotheses") else {})
    theta = res["hypotheses"].get("theta_data", float("nan"))

    # --- pathwise, shared innovations --------------------------------------
    res["pathwise"] = (pathwise_report(target, rec, theta, m,
                                       n=E["T_shared"], burn=D["burn"],
                                       seed=E["seed_shared"])
                       if diag.get("pathwise") else {})
    res["synchronisation"] = (synchronisation_report(rec, M=M,
                                                     n_steps=E["n_spread_steps"],
                                                     n_starts=E["n_starts"])
                              if diag.get("synchronisation") else {})

    # --- the two laws, independent streams ---------------------------------
    tru = simulate(target, E["T_eval"], burn=D["burn"], seed=E["seed_eval"])
    gen = simulate(rec, E["T_eval"], burn=D["burn"], seed=E["seed_eval"] + 1)
    base = None
    if diag.get("baseline"):
        base = MatchedGaussianAR(p=m).fit(path).simulate(
            E["T_eval"], burn=D["burn"], seed=E["seed_eval"] + 2)

    if diag.get("lyapunov"):
        val, exact = lyapunov(net, gen)
        res["lyapunov"] = {"value": val, "exact": exact,
                           "decomposition": lyapunov_decomposition(target, gen)}
    else:
        res["lyapunov"] = {"value": float("nan"), "exact": False}

    res["autocorrelation"] = (acf_report(tru, gen, base) if diag.get("autocorrelation") else {})
    res["marginal"] = (marginal_report(tru, gen, base) if diag.get("marginal") else {})
    res["conditional"] = (conditional_report(
        tru, gen, base, n_bins=E["n_bins"],
        scale_is_state_dependent=(name in STATE_DEPENDENT_SCALE))
        if diag.get("conditional") else {})

    if diag.get("mmd_test"):
        L, st = E["block_len"], E.get("block_stride", 1)
        bt = blocks(tru, L, st)
        mm = {"block_len": L, "block_stride": st,
              "generated": permutation_test(bt, blocks(gen, L, st),
                                            n_perm=E["n_perm"],
                                            max_n=E["mmd_max_n"])}
        if base is not None:
            mm["baseline"] = permutation_test(bt, blocks(base, L, st),
                                              n_perm=E["n_perm"],
                                              max_n=E["mmd_max_n"])
        res["mmd"] = mm
    else:
        res["mmd"] = {"block_len": E["block_len"],
                      "generated": {"mmd2": float("nan"), "p_value": float("nan")}}

    res["pricing"] = (pricing_report(tru, gen, base, K=E["strike"],
                                     n_exercise=E["n_exercise"])
                      if diag.get("pricing") else {})

    # --- the ESN realisation ------------------------------------------------
    if diag.get("esn") and N["act"] == "softclip":
        try:
            esn = ESNRealization(net, operating_radius=1.6 * M)
            res["esn"] = {"error": esn.realisation_error(rec, n=2000, burn=500),
                          **esn.info()}
        except Exception as exc:                       # noqa: BLE001
            res["esn"] = {"error": None, "failed": f"{type(exc).__name__}: {exc}"}
    else:
        res["esn"] = {"error": None,
                      "skipped": "the exact shift needs the softclip activation"}

    res["secs"] = time.time() - t_start

    write_json(res, os.path.join(outdir, "results.json"))
    with open(os.path.join(outdir, "results.txt"), "w") as f:
        f.write(render_txt(res))
    if diag.get("figures"):
        panel_figure(res, tru, gen, base, target, net,
                     os.path.join(outdir, "figures.png"),
                     title=f"{res['label']}   {target.name}   m={m}")
    return res


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid and exit")
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget, to check the pipeline end to end")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these folder names")
    ap.add_argument("--resume-into", default=None,
                    help="an existing reports/run_* folder to continue")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(getattr(CFG, "TORCH_THREADS", 2))
    grid = expand_grid(CFG)
    if args.quick:
        grid = _quick(grid)
    varying = varying_axes(grid)
    for i, g in enumerate(grid, 1):
        g["dirname"] = run_dirname(g["axes"], varying)
    # disambiguate any collision (possible when a varying axis is not in the name)
    seen = {}
    for g in grid:
        n = g["dirname"]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            g["dirname"] = f"{n}_{seen[n]:02d}"

    if args.only:
        grid = [g for g in grid if g["dirname"] in set(args.only)]

    dev, dt = resolve(getattr(CFG, "DEVICE", "auto"),
                      getattr(CFG, "DTYPE", "float32"))
    print(describe(dev, dt))
    print("  (the fit runs there; generation is numpy on the CPU, sequential "
          "along a chain and\n   batched across chains, see backend.py)")
    print(f"{len(grid)} configuration(s); axes varying: "
          f"{', '.join(varying) if varying else 'none'}")
    for g in grid:
        print("   ", g["dirname"])
    if args.dry_run:
        return 0

    root = args.resume_into or os.path.join(
        CFG.REPORTS_DIR, "run_" + datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(root, exist_ok=True)
    write_json({"targets": CFG.TARGETS, "M": CFG.M, "S_TARGET": CFG.S_TARGET,
                "TARGET_SPECIFICS": CFG.TARGET_SPECIFICS, "DATA": CFG.DATA,
                "NET": CFG.NET, "TRAIN": CFG.TRAIN, "EVAL": CFG.EVAL,
                "DIAGNOSTICS": CFG.DIAGNOSTICS, "quick": bool(args.quick),
                "varying_axes": varying,
                "device": str(dev), "dtype": str(dt).replace("torch.", "")},
               os.path.join(root, "config_used.json"))
    log = open(os.path.join(root, "run.log"), "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"# sweep started {datetime.now().isoformat(timespec='seconds')}  "
        f"-> {root}")
    rows, t0 = [], time.time()
    for i, g in enumerate(grid, 1):
        outdir = os.path.join(root, g["dirname"])
        rj = os.path.join(outdir, "results.json")
        if getattr(CFG, "RESUME", True) and os.path.exists(rj):
            say(f"[{i}/{len(grid)}] skip {g['dirname']} (already done)")
            rows.append(json.load(open(rj)))
            continue
        say(f"[{i}/{len(grid)}] {g['dirname']} ...")
        try:
            r = run_one(g, outdir, verbose=args.verbose, quick=args.quick)
            rows.append(r)
            pw = r["pathwise"]
            say(f"    done in {r['secs']:.0f}s   "
                f"W1/sd {r['marginal']['w1_over_sd']:.4f}   "
                f"cond W1 {r['conditional']['cond_w1']:.4f}   "
                f"sup err {pw.get('sup_err', float('nan')):.4f}"
                + (f" vs bound {pw['bound']:.3f}" if pw.get("bound") else "")
                + (f"\n    {r['fit']['epochs_run']} epochs"
                   + f", final lr {r['fit']['final_lr']:.1e}")
                + (f"\n    NOTE: {r['lip_mode_forced']}" if r["lip_mode_forced"] else ""))
        except Exception:                                    # noqa: BLE001
            say(f"    FAILED\n{traceback.format_exc()}")

        # rewrite the cross-run artefacts after every run, so an interrupted
        # sweep still leaves a usable report
        if rows:
            write_summary_csv(rows, os.path.join(root, "summary.csv"))
            write_comparison_md(rows, os.path.join(root, "COMPARISON.md"),
                                varying=varying)

    if rows:
        write_json(rows, os.path.join(root, "all_results.json"))
        overview_figure(rows, os.path.join(root, "overview.png"),
                        title=f"{len(rows)} runs")
        for axis in varying:
            if axis == "target":
                continue
            vals = {r["axes"][axis] for r in rows}
            if len(vals) >= 2 and all(isinstance(v, (int, float)) for v in vals):
                for tgt in sorted({r["axes"]["target"] for r in rows}):
                    sub = [r for r in rows if r["axes"]["target"] == tgt]
                    if len(sub) >= 2:
                        sweep_figure(sub, axis,
                                     os.path.join(root, f"sweep_{tgt}_{axis}.png"),
                                     title=f"{tgt}: sweep over {axis}",
                                     logx=(axis == "m"))
    say(f"# ALL DONE  {len(rows)}/{len(grid)} in {(time.time()-t0)/60:.1f} min")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
