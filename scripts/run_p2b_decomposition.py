#!/usr/bin/env python
"""WP-C -- backfill the four P2 qualification arms quoted by bitacora 04.

Review finding A1: the SHAPE decomposition and the second burstiness
comparison were quoted in bitacora/04 sect. 3-4, docs/04-DECISIONS.md,
HANDOFF.md and notebook 04 -- but no committed script or JSON produced them.
This script is the missing producer.

Arms (headline config n_bins=24, hi=400d, min_events=5, seeds 42-51, HGB,
StratifiedKFold(5) -- identical protocol to run_p2_temporal.py):

    COUNT, BURST, SPEC_T          fidelity cross-check against
                                  results/p2_temporal.json (elementwise,
                                  per seed -- any drift anywhere fails loud)
    SPEC_T_MINUS_H0               primary reading: BOTH H_0 columns removed
                                  (ia and cd) -- matches the stated rationale
                                  ("the single most volume-contaminated
                                  order"); per-half removals are sensitivity
                                  rows so the original computation's variant,
                                  whichever it was, is visible
    SHAPE                         level removed within each half:
                                  H_a - H_1(ia) for the ia six,
                                  H_a - H_1(cd) for the cd six
    COUNT+SHAPE                   1 + 12 dims
    COUNT+BURST, COUNT+BURST+SPEC_T

Paired Wilcoxon (the three comparisons bitacora 04 quotes):

    COUNT+SHAPE          vs COUNT                (quoted +0.0273 CLEARS)
    SPEC_T               vs SPEC_T_MINUS_H0      (quoted +0.0104 fails)
    COUNT+BURST+SPEC_T   vs COUNT+BURST          (quoted +0.0192 fails)

Expected values are from bitacora 04 sect. 4; |delta| > 0.005 on any of them
is a FINDING (recorded, never tuned to).

NOTE: eval_arm here is copied verbatim from run_p2_temporal.py rather than
imported; scripts/ is not a package. WP-E consolidates both onto
renyiext.evaluate with a regression gate, at which point this duplication ends.

Usage:
    python scripts/run_p2b_decomposition.py [--quiet]
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
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import DATA_PROCESSED, RESULTS
from renyiext.events import load_events_cached
from renyiext.features import temporal_blocks, MS_PER_DAY

CACHE = DATA_PROCESSED / "cresci_events_d9.npz"
P2_JSON = RESULTS / "p2_temporal.json"
SEEDS = list(range(42, 52))
HEADLINE = {"n_bins": 24, "hi": 400 * MS_PER_DAY, "min_events": 5}


def tpr_at_fpr(y, s, target=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target, fpr, tpr))


def eval_arm(X, y, seed):
    """Verbatim from run_p2_temporal.py (see module docstring)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200,
                                             early_stopping=False)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return {
        "auc": float(roc_auc_score(y, oof)),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, oof),
        "macro_f1": float(f1_score(y, (oof > 0.5).astype(int), average="macro")),
        "accuracy": float(accuracy_score(y, (oof > 0.5).astype(int))),
    }


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return {"mean_diff": float(d.mean()), "std_diff": float(d.std()),
            "wins": f"{int((d > 0).sum())}/{len(d)}", "p": p,
            "significant": bool(p < 0.05),
            "clears_floor": bool(d.mean() > 0.02 and p < 0.05)}


def shape_arm(spec: np.ndarray) -> np.ndarray:
    """Level removed within each half: index 2 of each half is alpha = 1."""
    out = spec.copy()
    out[:, :6] -= spec[:, :6][:, [2]]
    out[:, 6:] -= spec[:, 6:][:, [2]]
    return out


def overflow_mass_render(ev):
    """WP-D G1 render: who loses intervals above hi=400d? (review C1)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from renyiext.config import FIGURES

    hi = 400 * MS_PER_DAY
    bot = ev.labels == 1
    n_over = np.zeros(ev.n_users, dtype=np.int64)
    for i in range(ev.n_users):
        ts, _ = ev.events_of(i)
        if len(ts) > 1:
            dt = np.diff(ts)
            n_over[i] = int((dt > hi).sum())
    tot_bot = int(np.diff(ev.offsets)[bot].clip(min=1).sum())
    tot_hum = int(np.diff(ev.offsets)[~bot].clip(min=1).sum())
    out = {
        "hi_days": hi / MS_PER_DAY,
        "intervals_above_hi": {"bot": int(n_over[bot].sum()),
                               "human": int(n_over[~bot].sum())},
        "interval_totals": {"bot": tot_bot, "human": tot_hum},
        "share_of_intervals": {
            "bot": float(n_over[bot].sum() / max(tot_bot, 1)),
            "human": float(n_over[~bot].sum() / max(tot_hum, 1))},
        "accounts_affected_share": {
            "bot": float((n_over[bot] > 0).mean()),
            "human": float((n_over[~bot] > 0).mean())},
    }
    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels = ["bot", "human"]
    xs = np.arange(2)
    bars = [out["intervals_above_hi"]["bot"], out["intervals_above_hi"]["human"]]
    ax.bar(xs, bars, color=["#e76f51", "#2a9d8f"])
    for x, v in zip(xs, bars):
        ax.text(x, v + max(bars) * 0.02, str(v), ha="center")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("inter-arrival intervals above 400 d")
    ax.set_title(f"Out-of-range mass the pre-WP-D pipeline silently dropped "
                 f"(hi = {hi/MS_PER_DAY:.0f} d)", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "p2d_overflow_mass.png", dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    ev = load_events_cached(CACHE)
    bl = temporal_blocks(ev, **HEADLINE)
    blocks, y = bl["blocks"], bl["y"]

    minus_both = np.delete(blocks["SPEC_T"], [0, 6], axis=1)
    minus_ia = np.delete(blocks["SPEC_T"], [0], axis=1)
    minus_cd = np.delete(blocks["SPEC_T"], [6], axis=1)

    ARMS = {
        "COUNT":              blocks["COUNT"],
        "BURST":              blocks["BURST"],
        "SPEC_T":             blocks["SPEC_T"],
        "SPEC_T_MINUS_H0":    minus_both,
        "SPEC_T_MINUS_H0_IA": minus_ia,
        "SPEC_T_MINUS_H0_CD": minus_cd,
        "SHAPE":              shape_arm(blocks["SPEC_T"]),
        "COUNT+SHAPE":        np.hstack([blocks["COUNT"],
                                         shape_arm(blocks["SPEC_T"])]),
        "COUNT+BURST":        np.hstack([blocks["COUNT"], blocks["BURST"]]),
        "COUNT+BURST+SPEC_T": np.hstack([blocks["COUNT"], blocks["BURST"],
                                         blocks["SPEC_T"]]),
    }

    res = {}
    for name, X in ARMS.items():
        per_seed = [eval_arm(X, y, s) for s in SEEDS]
        res[name] = {"n_features": int(X.shape[1]),
                     "auc": [r["auc"] for r in per_seed],
                     "tpr01": [r["tpr_at_1pct_fpr"] for r in per_seed],
                     "macro_f1": [r["macro_f1"] for r in per_seed],
                     "accuracy": [r["accuracy"] for r in per_seed]}
        if not args.quiet:
            a = np.array(res[name]["auc"])
            print(f"  {name:<22} {X.shape[1]:>2}d  AUC {a.mean():.4f} "
                  f"+- {a.std():.4f}", flush=True)

    # ---- fidelity gate vs p2_temporal.json (elementwise, per seed) --------
    fidelity = {"file": str(P2_JSON), "checked": [], "max_abs_diff": None,
                "pass": None}
    if P2_JSON.exists():
        stored = json.loads(P2_JSON.read_text())["arms"]
        worst = 0.0
        for arm in ("COUNT", "BURST", "SPEC_T"):
            for metric in ("auc", "tpr01", "macro_f1", "accuracy"):
                got = np.array(res[arm][metric])
                want = np.array(stored[arm][metric])
                worst = max(worst, float(np.abs(got - want).max()))
            fidelity["checked"].append(arm)
        fidelity["max_abs_diff"] = worst
        fidelity["pass"] = bool(worst < 1e-9)
        if not fidelity["pass"]:
            raise AssertionError(
                f"fidelity gate FAILED: max |diff| {worst:.3e} vs "
                f"{P2_JSON} -- pipeline drift; do not quote these numbers")

    comparisons = {
        "COUNT+SHAPE_vs_COUNT": paired(res["COUNT+SHAPE"]["auc"],
                                       res["COUNT"]["auc"]),
        "SPEC_T_vs_MINUS_H0": paired(res["SPEC_T"]["auc"],
                                     res["SPEC_T_MINUS_H0"]["auc"]),
        "CB_SPEC_T_vs_CB": paired(res["COUNT+BURST+SPEC_T"]["auc"],
                                  res["COUNT+BURST"]["auc"]),
        # sensitivity variants for the ambiguous MINUS_H0 semantics
        "SPEC_T_vs_MINUS_H0_IA_only": paired(res["SPEC_T"]["auc"],
                                             res["SPEC_T_MINUS_H0_IA"]["auc"]),
        "SPEC_T_vs_MINUS_H0_CD_only": paired(res["SPEC_T"]["auc"],
                                             res["SPEC_T_MINUS_H0_CD"]["auc"]),
    }

    ovf = overflow_mass_render(ev)
    if not args.quiet:
        print(f"  [G1] overflow mass: bot {ovf['intervals_above_hi']['bot']} "
              f"({100*ovf['share_of_intervals']['bot']:.4f}% of bot "
              f"intervals), human {ovf['intervals_above_hi']['human']} "
              f"({100*ovf['share_of_intervals']['human']:.4f}%)")

    expected = {
        "SHAPE_auc": 0.9596, "COUNT+SHAPE_auc": 0.9673,
        "COUNT+SHAPE_minus_COUNT": 0.0273,
        "SPEC_T_minus_MINUS_H0": 0.0104,
        "CB_SPEC_T_minus_CB": 0.0192,
    }
    measured = {
        "SHAPE_auc": float(np.mean(res["SHAPE"]["auc"])),
        "COUNT+SHAPE_auc": float(np.mean(res["COUNT+SHAPE"]["auc"])),
        "COUNT+SHAPE_minus_COUNT":
            comparisons["COUNT+SHAPE_vs_COUNT"]["mean_diff"],
        "SPEC_T_minus_MINUS_H0": comparisons["SPEC_T_vs_MINUS_H0"]["mean_diff"],
        "CB_SPEC_T_minus_CB": comparisons["CB_SPEC_T_vs_CB"]["mean_diff"],
    }
    reconciliation = {k: {"quoted": v,
                          "measured": round(measured[k], 6),
                          "abs_diff": round(abs(measured[k] - v), 6),
                          "finding": bool(abs(measured[k] - v) > 0.005)}
                      for k, v in expected.items()}

    report = {
        "phase": "P2b-decomposition (WP-C)",
        "seeds": SEEDS,
        "headline_config": {"n_bins": HEADLINE["n_bins"],
                            "hi_days": HEADLINE["hi"] / MS_PER_DAY,
                            "min_events": HEADLINE["min_events"]},
        "definitions": {
            "SHAPE": "per-half level removed: H_a - H_1 within ia six and "
                     "within cd six (plan WP-C task 1)",
            "SPEC_T_MINUS_H0_primary": "both H_0 columns (indices 0, 6)",
            "sensitivity": ["minus ia only", "minus cd only"],
            "note": "bitacora/04 did not specify which H_0 columns its "
                    "'SPEC_T minus H_0' arm removed; all three variants are "
                    "reported so the original computation's reading is "
                    "identifiable",
        },
        "n": int(len(y)),
        "majority_baseline": float(max(y.mean(), 1 - y.mean())),
        "n_excluded_bot": bl["n_excluded_bot"],
        "n_excluded_human": bl["n_excluded_human"],
        "arms": res,
        "comparisons": comparisons,
        "overflow_mass": ovf,
        "reconciliation_vs_bitacora04": reconciliation,
        "fidelity_gate": fidelity,
    }
    path = RESULTS / "p2b_decomposition.json"
    path.write_text(json.dumps(report, indent=1))

    print("\n" + "=" * 72)
    print("WP-C — P2 QUALIFICATION ARMS, BACKFILLED")
    print("=" * 72)
    for k, v in reconciliation.items():
        flag = "  <-- FINDING" if v["finding"] else ""
        print(f"{k:<28} quoted {v['quoted']:+.4f}   measured "
              f"{v['measured']:+.4f}{flag}")
    print(f"\nfidelity gate vs {P2_JSON.name}: "
          f"{'PASS' if fidelity['pass'] else 'FAIL'} "
          f"(max |diff| {fidelity['max_abs_diff']:.2e})")
    print(f"json -> {path}")


if __name__ == "__main__":
    main()
