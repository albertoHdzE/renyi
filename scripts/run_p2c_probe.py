#!/usr/bin/env python
"""WP-A -- the censoring probe (PLAN-02-ext-research.md, review finding D1).

Question, asked before anything else is built: **how much classifier-separable
signal does observation-window truncation alone produce through the exact P2
feature pipeline?**

Design (Definition D5 of the plan, pre-registered 2026-08-24): for each
generator g in {periodic(jitter 0.5), poisson, bursty(tail 1.2)} -- each from
its OWN renewal process; there is no separate Poisson process -- draw 120 + 120
accounts. Class A is observed over the full 900-day timeline; class B is the
same process truncated to its first W days, W in {30, 90, 400}. Identical
feature pipeline (renyiext.features.temporal_blocks_ts, the core that
temporal_blocks delegates to), identical classifier protocol as P2. Binary
AUC(B vs A) per metric:

    SPEC_T        the spectrum alone
    COUNT+SPEC_T  the P2-headline composite -- THIS cell owns the trigger
    SHAPE         level removed (per-order H_a - H_1 within each half)

Amendment trigger (pre-registered): any COUNT+SPEC_T cell at AUC >= 0.85 fires
the censoring amendment -- SPEC_T's edge may be substantially censoring, and
the "shape" reading of P2 is formally downgraded.

This is a control, not a family claim: it runs on pure synthetic data, needs no
corpus and no cache, and its job is to bound what the real corpus result could
mean. Pipeline-fidelity note: features are computed with the pipeline AS IT IS
when this runs ("overflow_cell" flag in the JSON records whether the WP-D fix
has landed); plan WP-A task 4 obliges a re-run after WP-D with verdict unchanged.

Usage:
    python scripts/run_p2c_probe.py [--quiet]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import FIGURES, RESULTS
from renyiext.features import temporal_blocks_ts, MS_PER_DAY
from renyiext.spectrum import spectrum_labels, SPECTRUM_ALPHAS

HORIZON_D = 900
WINDOWS_D = (30, 90, 400)
N_PER_CLASS = 120
SEEDS = list(range(42, 52))
TRIGGER = 0.85

GENERATORS = ("periodic", "poisson", "bursty")
# bursty's scale is calibrated so its MEDIAN inter-arrival equals the other
# two generators' 6 h (median of Pareto(tail)+1 is 2**(1/tail)-1); the tail
# exponent is untouched. Without this the renewal process emits ~7e5
# events/account over 900 d -- a compute artefact, not a behaviour. Printed in
# the JSON config echo (G3).
_BURSTY_SCALE_MS = 6 * 3_600_000.0 / (2.0 ** (1.0 / 1.2) - 1.0)
GEN_KW = {"periodic": {"jitter": 0.5}, "poisson": {},
          "bursty": {"tail": 1.2, "scale_ms": _BURSTY_SCALE_MS}}


def _draw_open_ended(gen: str, rng, horizon_ms: int) -> np.ndarray:
    """Event timestamps of ONE account from generator g's renewal process,
    run until the timeline exceeds ``horizon_ms``."""
    kw = GEN_KW[gen]
    dts, total = [], 0
    while total <= horizon_ms:
        chunk = GENERATOR_FNS[gen](4096, rng, **kw)
        cums = np.cumsum(chunk)
        keep = np.searchsorted(cums, horizon_ms - total, side="right")
        dts.append(chunk[:keep])
        total += int(cums[keep - 1]) if keep else 0
        if keep < len(chunk):
            break
    dt = np.concatenate(dts) if dts else np.empty(0, np.float64)
    ts = np.cumsum(dt).astype(np.int64)
    return ts[ts <= horizon_ms]


GENERATOR_FNS = None  # set in main() to avoid import-order surprises


def eval_auc(X, y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200,
                                             early_stopping=False)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def shape_arm(spec: np.ndarray) -> np.ndarray:
    """Level removed: ia orders minus H1_ia, cd orders minus H1_cd."""
    out = spec.copy()
    out[:, :6] -= spec[:, [2]]          # index 2 = alpha = 1
    out[:, 6:] -= spec[:, 6:][:, [2]]
    return out


def main():
    global GENERATOR_FNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    from renyiext.generators import (periodic_account, poisson_account,
                                     bursty_account)
    GENERATOR_FNS = {"periodic": periodic_account, "poisson": poisson_account,
                     "bursty": bursty_account}

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    horizon_ms = HORIZON_D * MS_PER_DAY
    cells = {}
    rasters = {}

    if not args.quiet:
        print(f"censoring probe: {len(GENERATORS)} generators x "
              f"{len(WINDOWS_D)} windows x {N_PER_CLASS}+{N_PER_CLASS} "
              f"accounts, seeds {SEEDS[0]}-{SEEDS[-1]}", flush=True)

    for gi, gen in enumerate(GENERATORS):
        for W_d in WINDOWS_D:
            # One rng per (generator, window), same seed every time: the
            # underlying processes are IDENTICAL across windows, so any
            # class difference is truncation and nothing else.
            rng = np.random.default_rng(1_000_003 * (gi + 1))
            series_a, series_b = [], []
            for _ in range(N_PER_CLASS):
                series_a.append(_draw_open_ended(gen, rng, horizon_ms))
                s = _draw_open_ended(gen, rng, horizon_ms)
                series_b.append(s[s <= W_d * MS_PER_DAY])

            ts_list = series_b + series_a          # y=1 truncated, y=0 full
            y = np.array([1] * N_PER_CLASS + [0] * N_PER_CLASS)
            bl = temporal_blocks_ts(ts_list, labels=y, n_bins=24, lo=1.0,
                                    hi=400 * MS_PER_DAY, min_events=5)
            blocks, yy = bl["blocks"], bl["y"]

            arms = {
                "SPEC_T": blocks["SPEC_T"],
                "COUNT+SPEC_T": np.hstack([blocks["COUNT"], blocks["SPEC_T"]]),
                "SHAPE": shape_arm(blocks["SPEC_T"]),
            }
            cell = {}
            for name, X in arms.items():
                aucs = [eval_auc(X, yy, s) for s in SEEDS]
                cell[name] = {"auc_mean": float(np.mean(aucs)),
                              "auc_sd": float(np.std(aucs)),
                              "auc_per_seed": aucs}
            cells[f"{gen}|W={W_d}d"] = cell

            n_events_a = [len(s) for s in series_a]
            n_events_b = [len(s) for s in series_b]
            cell["_n"] = {
                "kept_A": int(sum(len(s) >= 5 for s in series_a)),
                "kept_B": int(sum(len(s) >= 5 for s in series_b)),
                "median_events_A": float(np.median(n_events_a)),
                "median_events_B": float(np.median(n_events_b)),
            }
            rasters[f"{gen}|W={W_d}d"] = (series_a[:15], series_b[:15],
                                          W_d, gen)

            if not args.quiet:
                cs = cell["COUNT+SPEC_T"]
                print(f"  {gen:<9} W={W_d:>4}d  "
                      f"COUNT+SPEC_T {cs['auc_mean']:.4f}±{cs['auc_sd']:.4f}  "
                      f"(nB median {cell['_n']['median_events_B']:.0f})",
                      flush=True)

    # ---- trigger ----------------------------------------------------------
    trig_cells = {k: v["COUNT+SPEC_T"]["auc_mean"] for k, v in cells.items()}
    worst_key = max(trig_cells, key=trig_cells.get)
    fired = bool(trig_cells[worst_key] >= TRIGGER)

    # ---- figures ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    width = 0.25
    xs = np.arange(len(WINDOWS_D))
    for gi, gen in enumerate(GENERATORS):
        vals = [cells[f"{gen}|W={W_d}d"]["COUNT+SPEC_T"]["auc_mean"]
                for W_d in WINDOWS_D]
        ax.bar(xs + (gi - 1) * width, vals, width,
               color=["#2a78d6", "#eb6834", "#1baf7a"][gi], label=gen)
        for x, v in zip(xs + (gi - 1) * width, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(TRIGGER, color="crimson", ls="--", lw=2,
               label=f"amendment trigger {TRIGGER}")
    ax.set_xticks(xs); ax.set_xticklabels([f"W = {w} d" for w in WINDOWS_D])
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("AUC(B vs A)")
    ax.set_title("Censoring probe — same generator, same rate, different "
                 "window (COUNT+SPEC_T arm)")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "p2c_probe_grid.png", dpi=130)
    plt.close(fig)

    sa, sb, W_d, gen = max(
        rasters.values(),
        key=lambda r: cells[f"{r[3]}|W={r[2]}d"]["COUNT+SPEC_T"]["auc_mean"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
    day = MS_PER_DAY
    for row, (series, name, colour, W) in enumerate(
            [(sa, f"class A — observed {HORIZON_D} d", "#2a78d6", HORIZON_D),
             (sb, f"class B — same generator, first {W_d} d", "#eb6834", W_d)]):
        axc = axes[row]
        for k, ts in enumerate(series):
            axc.plot(np.array(ts) / day, np.full(len(ts), k), "|",
                     ms=3, color=colour, alpha=0.65)
        axc.axvline(W, color="k", ls=":", lw=1.5)
        axc.set_ylabel("account")
        axc.set_title(name, loc="left", fontsize=10)
    axes[1].set_xlabel("days since account birth")
    fig.suptitle(f"Worst cell: {gen}, W = {W_d} d — the objects (G1)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "p2c_probe_rasters.png", dpi=130)
    plt.close(fig)

    report = {
        "phase": "P2c-probe (WP-A)",
        "definition": "plan PLAN-02-ext-research.md sect. 8 D5 (v1.1)",
        "generators": {g: GEN_KW[g] for g in GENERATORS},
        "horizon_days": HORIZON_D, "windows_days": WINDOWS_D,
        "n_per_class": N_PER_CLASS, "seeds": SEEDS,
        "feature_config": {"n_bins": 24, "lo": 1.0, "hi_days": 400,
                           "min_events": 5},
        "pipeline_version": {"overflow_cell": False,
                             "note": "pre-WP-D log_bin_counts; re-run "
                                     "obliged after WP-D (plan WP-A task 4)"},
        "classifier": "HistGradientBoostingClassifier(max_iter=200, "
                      "early_stopping=False), StratifiedKFold(5)",
        "trigger": {"threshold": TRIGGER, "metric": "COUNT+SPEC_T",
                    "worst_cell": worst_key,
                    "worst_value": trig_cells[worst_key], "fired": fired},
        "cells": cells,
    }
    path = RESULTS / "p2c_probe.json"
    path.write_text(json.dumps(report, indent=1))

    print("\n" + "=" * 72)
    print("WP-A — CENSORING PROBE")
    print("=" * 72)
    print(f"{'cell':<22}{'SPEC_T':>12}{'COUNT+SPEC_T':>14}{'SHAPE':>10}"
          f"{'med ev B':>10}")
    for key, cell in cells.items():
        print(f"{key:<22}{cell['SPEC_T']['auc_mean']:>12.4f}"
              f"{cell['COUNT+SPEC_T']['auc_mean']:>14.4f}"
              f"{cell['SHAPE']['auc_mean']:>10.4f}"
              f"{cell['_n']['median_events_B']:>10.0f}")
    print(f"\nTRIGGER ({TRIGGER} on COUNT+SPEC_T): "
          f"{'FIRED' if fired else 'NOT FIRED'} "
          f"— worst cell {worst_key} at {trig_cells[worst_key]:.4f}")
    print(f"figures -> {FIGURES}/p2c_probe_grid.png, p2c_probe_rasters.png")
    print(f"json     -> {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
