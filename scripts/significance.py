#!/usr/bin/env python
"""Paired significance test for the paper's central claim.

The paper reports point estimates with no seeds and no error bars. On a test set
of a few hundred pairs, that is not enough to tell an improvement from noise: at
three seeds the DTWRE-vs-static-Rényi difference already changes sign.

This runs every method over many seeds on identical splits, then compares methods
*pairwise within each seed* (the paired design removes split-to-split variance)
and reports a Wilcoxon signed-rank test alongside the raw win counts.

    python scripts/significance.py --seeds 10 --quiet
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtwre import Config, load_collegemsg, METRICS, BASELINES, BASELINE_LABELS
from dtwre.experiments import EmbeddingCache, baseline_comparison, save_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--metric", default="auc", choices=METRICS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    tg = load_collegemsg()
    base = Config()
    seeds = list(range(42, 42 + args.seeds))

    per_seed = {}
    for s in seeds:
        cfg = replace(base, seed=s)
        per_seed[s] = baseline_comparison(tg, cfg, cache=EmbeddingCache(),
                                          verbose=False)
        if not args.quiet:
            print(f"  seed {s}: " + "  ".join(
                f"{m}={per_seed[s][m][args.metric]:.4f}" for m in BASELINES))

    mat = pd.DataFrame({m: [per_seed[s][m][args.metric] for s in seeds]
                        for m in BASELINES}, index=seeds)

    print(f"\n{args.metric.upper()} over {len(seeds)} seeds")
    summary = pd.DataFrame({"mean": mat.mean(), "std": mat.std(ddof=1),
                            "min": mat.min(), "max": mat.max()}).round(4)
    summary.index = [BASELINE_LABELS[i] for i in summary.index]
    print(summary.to_string())

    print(f"\nPaired comparisons against DTWRE (positive = DTWRE better)")
    rows = []
    for m in BASELINES:
        if m == "dtwre":
            continue
        d = mat["dtwre"].values - mat[m].values
        try:
            _, p = stats.wilcoxon(d)
        except ValueError:                       # all-zero differences
            p = float("nan")
        rows.append({
            "vs": BASELINE_LABELS[m],
            "mean diff": round(float(d.mean()), 4),
            "std diff": round(float(d.std(ddof=1)), 4),
            "DTWRE wins": f"{int((d > 0).sum())}/{len(d)}",
            "wilcoxon p": round(float(p), 4),
            "significant (p<0.05)": bool(p < 0.05),
        })
    print(pd.DataFrame(rows).set_index("vs").to_string())

    save_results({"metric": args.metric, "seeds": seeds,
                  "matrix": {k: list(map(float, v)) for k, v in mat.items()},
                  "paired": rows}, f"significance_{args.metric}.json")
    print(f"\nsaved -> results/significance_{args.metric}.json")


if __name__ == "__main__":
    main()
