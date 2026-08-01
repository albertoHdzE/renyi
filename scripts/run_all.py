#!/usr/bin/env python
"""Run the complete replication and cache every result to ``results/``.

The notebook can either recompute results live or load these artefacts. Run::

    python scripts/run_all.py                # CollegeMsg, all experiments
    python scripts/run_all.py --quiet        # summaries only
    python scripts/run_all.py --seeds 42 43 44
    python scripts/run_all.py --weibo        # include the Weibo extension
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtwre import (Config, load_collegemsg, load_weibo_ced, prepare_split,
                   run_experiment, METRICS, BASELINES, ALPHA_GRID,
                   LAMBDA_GRID, NEG_RATIO_GRID, WINDOW_GRID, RESULTS)
from dtwre.experiments import (EmbeddingCache, baseline_comparison,
                               alpha_sweep, lambda_sweep, ratio_sweep,
                               window_sweep, training_curves, save_results)


def multi_seed_comparison(tg, cfg, seeds, quiet=False):
    """Table 1 across several seeds -> mean and standard deviation.

    The paper reports point estimates with no seed, repetition count or error
    bars. With only a few hundred test pairs, run-to-run spread is the first
    thing a reader needs in order to judge whether a gap is real.
    """
    per_seed = {}
    for s in seeds:
        c = replace(cfg, seed=s)
        cache = EmbeddingCache()
        if not quiet:
            print(f"  seed {s}")
        per_seed[s] = baseline_comparison(tg, c, cache=cache, verbose=not quiet)

    summary = {}
    for method in per_seed[seeds[0]]:
        summary[method] = {}
        for m in METRICS:
            vals = [per_seed[s][method][m] for s in seeds]
            summary[method][m] = float(np.mean(vals))
            summary[method][f"{m}_std"] = float(np.std(vals))
    return summary, {str(k): v for k, v in per_seed.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-run lines; print summaries only")
    ap.add_argument("--weibo", action="store_true")
    ap.add_argument("--max-cascades", type=int, default=100)
    args = ap.parse_args()
    q = args.quiet

    t_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    tg = load_collegemsg()
    cfg = Config()
    print("dataset:", tg.summary())

    timings = {}

    # ---- Table 1 / Figure 5 ----
    print("\n[1/7] Baseline comparison (Table 1, Figure 5)")
    t0 = time.time()
    summary, per_seed = multi_seed_comparison(tg, cfg, args.seeds, quiet=q)
    timings["baselines"] = time.time() - t0
    save_results({"summary": summary, "per_seed": per_seed},
                 "table1_collegemsg.json")
    for k, v in summary.items():
        print(f"  {k:14s} AUC={v['auc']:.4f}+-{v['auc_std']:.4f}  "
              f"F1={v['f1']:.4f}+-{v['f1_std']:.4f}")

    # A single shared cache for the sweeps (graph is fixed across them).
    cache = EmbeddingCache()

    print("\n[2/7] Alpha sweep (Figure 6)")
    t0 = time.time()
    alpha_res = alpha_sweep(tg, cfg, cache=cache, verbose=not q, seeds=args.seeds)
    timings["alpha"] = time.time() - t0
    save_results({"grid": ALPHA_GRID, "results": alpha_res}, "alpha_sweep.json")

    print("\n[3/7] Lambda sweep (Figure 7)")
    t0 = time.time()
    lam_res = lambda_sweep(tg, cfg, cache=cache, verbose=not q, seeds=args.seeds)
    timings["lambda"] = time.time() - t0
    save_results({"grid": LAMBDA_GRID, "results": lam_res}, "lambda_sweep.json")

    print("\n[4/7] Negative-ratio sweep (Figure 8)")
    t0 = time.time()
    ratio_res = ratio_sweep(tg, cfg, cache=cache, verbose=not q, seeds=args.seeds)
    timings["ratio"] = time.time() - t0
    save_results({"grid": NEG_RATIO_GRID, "results": ratio_res},
                 "ratio_sweep.json")

    print("\n[5/7] Window sweep (Table 2, Figure 9)")
    t0 = time.time()
    win_res = window_sweep(tg, cfg, cache=cache, verbose=not q, seeds=args.seeds)
    timings["window"] = time.time() - t0
    save_results({"grid": WINDOW_GRID, "results": win_res}, "window_sweep.json")

    print("\n[6/7] Training curves (Figure 10)")
    t0 = time.time()
    curves = training_curves(tg, cfg, cache=cache)
    timings["curves"] = time.time() - t0
    save_results({"history": curves["history"], "test": curves["test"]},
                 "training_curves.json")
    print(f"  final test AUC {curves['test']['auc']:.4f}")

    # ---- Weibo extension ----
    if args.weibo:
        print("\n[7/7] Weibo extension")
        t0 = time.time()
        wtg = load_weibo_ced(max_cascades=args.max_cascades)
        print("  ", wtg.summary())
        # Section 3.1 uses a 1 h window for Weibo, but the CED corpus we can
        # actually obtain aggregates cascades over ~1100 days: a 1 h window
        # would mean ~26,000 near-empty snapshots. A 30-day window gives a
        # snapshot count comparable to CollegeMsg. Duration-based splitting is
        # also degenerate here (4 test positives), so we split by edge count.
        wcfg = Config(window_seconds=30 * 86400, augment=True,
                      split_by="count")
        wres = training_curves(wtg, wcfg)
        timings["weibo"] = time.time() - t0
        save_results({"summary": wtg.summary(), "history": wres["history"],
                      "test": wres["test"]}, "weibo.json")
        print(f"   test AUC {wres['test']['auc']:.4f}")
    else:
        print("\n[7/7] Weibo extension skipped (--weibo to enable)")

    timings["total"] = time.time() - t_start
    save_results(timings, "timings.json")
    print(f"\nDone in {timings['total']:.1f}s -> {RESULTS}")


if __name__ == "__main__":
    main()
