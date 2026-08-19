#!/usr/bin/env python
"""P2 -- the temporal front, and the test of H1.

H1 as amended (bitacora/02_h1_amendment.md) has two clauses, both required:

    (i)   AUC(SPEC_T + COUNT) - AUC(COUNT alone)  > 0.02
    (ii)  AUC(SPEC_T)         - AUC(SHAN alone)   > 0.02

each by paired Wilcoxon over >=10 seeds at p < 0.05. COUNT alone scores AUC
0.939 on this corpus, so clause (i) is the harder floor. Clause (ii) carries a
standing warning from bitacora 03 sect. 4: on synthetic data engineered to have
exactly H1's mechanism the gain stayed within +-0.014 across five difficulty
settings.

Order of work is fixed by the datasaurus rule -- render, then sweep the
invented parameters, then quote a number:

    G1  per-class alpha-curves with error bands; per-order separation; every
        order read against COUNT -- the objects, not statistics of them
    G3  the log-binning grid (n_bins, hi) and min_events swept as CLASSIFIER
        parameters, not just as histogram cosmetics. bitacora 03 sect. 5
        item 1 names this as the leading explanation for the synthetic flatness
    G4  partial correlation of each order with the label GIVEN count -- the
        process known to contain the confound

Usage:
    python scripts/run_p2_temporal.py [--quiet] [--seeds N] [--fast]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Must precede numpy/sklearn. HistGradientBoosting's OpenMP pool thrashes badly
# on this machine at these problem sizes: 5 folds of 4,770 x 12 took 115.6 s
# with the default pool and 0.87 s pinned to one thread -- a 133x difference,
# entirely scheduling overhead. Pinned here rather than left to the environment
# so the run time is a property of the script, not of the shell it was launched
# from. Determinism (rule S2.6) is unaffected either way.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import Config, FIGURES, RESULTS, DATA_PROCESSED
from renyiext.events import load_events_cached
from renyiext.features import temporal_blocks, MS_PER_DAY
from renyiext.spectrum import spectrum_labels

warnings.filterwarnings("ignore", category=UserWarning)

CACHE = DATA_PROCESSED / "cresci_events_d9.npz"

ARMS = {
    "COUNT":            ("COUNT",),
    "BURST":            ("BURST",),
    "SHAN":             ("SHAN",),
    "SPEC_T":           ("SPEC_T",),
    "COUNT+BURST":      ("COUNT", "BURST"),
    "COUNT+SHAN":       ("COUNT", "SHAN"),
    "COUNT+SPEC_T":     ("COUNT", "SPEC_T"),
}


def tpr_at_fpr(y, s, target=0.01):
    """TPR at FPR = 1% -- the deployment regime (docs/02-PROTOCOL.md sect. 6)."""
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target, fpr, tpr))


def eval_arm(X, y, seed, model="hgb", n_folds=5):
    """One arm, one seed: 5-fold stratified CV, out-of-fold scores."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        if model == "hgb":
            clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200,
                                                 early_stopping=False)
        else:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=5000, random_state=seed))
        clf.fit(X[tr], y[tr])          # scaler fitted on the training fold only
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return {
        "auc": float(roc_auc_score(y, oof)),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, oof),
        "macro_f1": float(f1_score(y, (oof > 0.5).astype(int), average="macro")),
        "accuracy": float(accuracy_score(y, (oof > 0.5).astype(int))),
    }


def run_arms(blocks, y, seeds, model="hgb", quiet=False):
    out = {}
    for name, parts in ARMS.items():
        X = np.hstack([blocks[p] for p in parts])
        per_seed = [eval_arm(X, y, s, model) for s in seeds]
        out[name] = {"n_features": int(X.shape[1]),
                     "auc": [r["auc"] for r in per_seed],
                     "tpr01": [r["tpr_at_1pct_fpr"] for r in per_seed],
                     "macro_f1": [r["macro_f1"] for r in per_seed],
                     "accuracy": [r["accuracy"] for r in per_seed]}
        if not quiet:
            a = np.array(out[name]["auc"])
            print(f"    {name:<14} AUC {a.mean():.4f} +- {a.std():.4f}", flush=True)
    return out


def paired(a, b):
    """Paired Wilcoxon over seeds, in the repository's standard format."""
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


def gate1_render(bl, quiet=False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    y = bl["y"].astype(bool)
    spec = bl["blocks"]["SPEC_T"]
    lab = spectrum_labels()
    names = [f"{c}_ia" for c in lab] + [f"{c}_cd" for c in lab]
    xs = np.arange(6)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    for j, (sl, title) in enumerate(((slice(0, 6), "inter-arrival"),
                                     (slice(6, 12), "hour-of-day"))):
        a = ax[0, j]
        for mask, nm, c in ((~y, "human", "#2a9d8f"), (y, "bot", "#e76f51")):
            m, s = spec[mask, sl].mean(axis=0), spec[mask, sl].std(axis=0)
            a.plot(xs, m, "o-", color=c, lw=2, label=nm)
            a.fill_between(xs, m - s, m + s, color=c, alpha=0.18)
        a.set_xticks(xs); a.set_xticklabels(lab, rotation=30)
        a.set_title(f"Rényi spectrum — {title} (mean ± 1 SD)")
        a.set_ylabel("bits"); a.legend()

    a = ax[1, 0]
    seps = [abs(roc_auc_score(y, spec[:, k]) - 0.5) * 2 for k in range(12)]
    a.bar(np.arange(12), seps, color=["#3b6ea5"] * 6 + ["#9b5de5"] * 6)
    a.axhline(2 * abs(roc_auc_score(y, bl["blocks"]["COUNT"][:, 0]) - 0.5),
              color="crimson", ls="--", lw=2, label="COUNT alone")
    a.set_xticks(np.arange(12)); a.set_xticklabels(names, rotation=60, fontsize=7)
    a.set_ylabel("|2·AUC − 1| univariate")
    a.set_title("Per-order separation, against the COUNT floor")
    a.legend(fontsize=8)

    a = ax[1, 1]
    cnt = bl["blocks"]["COUNT"][:, 0]
    rho = [abs(np.corrcoef(cnt, spec[:, k])[0, 1]) for k in range(12)]
    a.bar(np.arange(12), rho, color=["#3b6ea5"] * 6 + ["#9b5de5"] * 6)
    a.set_xticks(np.arange(12)); a.set_xticklabels(names, rotation=60, fontsize=7)
    a.set_ylim(0, 1); a.set_ylabel("|corr| with log event count")
    a.set_title("How much of each order IS volume (R1)")

    fig.tight_layout()
    fig.savefig(FIGURES / "p2_g1_spectrum.png", dpi=130)
    plt.close(fig)
    if not quiet:
        print(f"  [G1] rendered -> {FIGURES / 'p2_g1_spectrum.png'}")
    return {"per_order_sep": seps, "per_order_corr_with_count": rho,
            "order_names": names}


def gate3_binning_sweep(ev, seeds, quiet=False):
    """bitacora 03 sect. 5 item 1: the grid may destroy the tail before the
    spectrum sees it. Swept here for AUC, not merely for the histogram."""
    hi_default = 400 * MS_PER_DAY
    grid = [{"n_bins": n, "hi": hi_default, "min_events": 5}
            for n in (8, 12, 16, 24, 32, 48)]
    grid += [{"n_bins": 24, "hi": d * MS_PER_DAY, "min_events": 5}
             for d in (30, 100, 400, 1000)]
    grid += [{"n_bins": 24, "hi": hi_default, "min_events": m}
             for m in (2, 5, 10, 20)]

    rows = []
    for g in grid:
        bl = temporal_blocks(ev, n_bins=g["n_bins"], hi=g["hi"],
                             min_events=g["min_events"])
        y, b = bl["y"], bl["blocks"]
        sp = [eval_arm(b["SPEC_T"], y, s)["auc"] for s in seeds]
        sh = [eval_arm(b["SHAN"], y, s)["auc"] for s in seeds]
        ct = [eval_arm(b["COUNT"], y, s)["auc"] for s in seeds]
        cs = [eval_arm(np.hstack([b["COUNT"], b["SPEC_T"]]), y, s)["auc"]
              for s in seeds]
        rows.append({"n_bins": g["n_bins"], "hi_days": g["hi"] / MS_PER_DAY,
                     "min_events": g["min_events"], "n": int(len(y)),
                     "auc_spec": float(np.mean(sp)), "auc_shan": float(np.mean(sh)),
                     "auc_count": float(np.mean(ct)),
                     "auc_count_spec": float(np.mean(cs)),
                     "clause_i": float(np.mean(cs) - np.mean(ct)),
                     "clause_ii": float(np.mean(sp) - np.mean(sh))})
        if not quiet:
            r = rows[-1]
            print(f"    bins={r['n_bins']:<3} hi={r['hi_days']:<6.0f}d "
                  f"min_ev={r['min_events']:<3} n={r['n']:<5} "
                  f"(i) {r['clause_i']:+.4f}  (ii) {r['clause_ii']:+.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = list(range(42, 42 + args.seeds))
    sweep_seeds = seeds[:3] if args.fast else seeds

    print("loading events ...", flush=True)
    ev = load_events_cached(CACHE)
    print(" ", ev, flush=True)

    print("\n[G3] binning sweep (invented parameters, swept for AUC)")
    sweep = gate3_binning_sweep(ev, sweep_seeds, args.quiet)

    # Headline uses the protocol default, NOT the sweep argmax: choosing the
    # argmax would be selecting on the outcome. The sweep is reported in full.
    head = {"n_bins": 24, "hi": 400 * MS_PER_DAY, "min_events": 5}
    bl = temporal_blocks(ev, **head)
    y = bl["y"]
    majority = float(max(y.mean(), 1 - y.mean()))

    print(f"\n[G1] render, n={len(y)} (bot {bl['n_kept_bot']}, "
          f"human {bl['n_kept_human']}; excluded bot {bl['n_excluded_bot']}, "
          f"human {bl['n_excluded_human']})")
    g1 = gate1_render(bl, args.quiet)

    print(f"\n[arms] {len(seeds)} seeds, 5-fold stratified CV, HGB")
    res = run_arms(bl["blocks"], y, seeds, "hgb", args.quiet)

    clause_i = paired(res["COUNT+SPEC_T"]["auc"], res["COUNT"]["auc"])
    clause_ii = paired(res["SPEC_T"]["auc"], res["SHAN"]["auc"])
    vs_burst = paired(res["COUNT+SPEC_T"]["auc"], res["COUNT+BURST"]["auc"])

    cnt = bl["blocks"]["COUNT"][:, 0]
    spec = bl["blocks"]["SPEC_T"]
    names = g1["order_names"]

    def partial_corr(x, yv, z):
        rx = x - np.polyval(np.polyfit(z, x, 1), z)
        ry = yv - np.polyval(np.polyfit(z, yv, 1), z)
        return float(np.corrcoef(rx, ry)[0, 1])

    partials = {names[k]: {
        "raw": float(np.corrcoef(spec[:, k], y)[0, 1]),
        "given_count": partial_corr(spec[:, k], y.astype(float), cnt)}
        for k in range(12)}

    report = {
        "phase": "P2", "seeds": seeds,
        "headline_config": {"n_bins": head["n_bins"],
                            "hi_days": head["hi"] / MS_PER_DAY,
                            "min_events": head["min_events"]},
        "n": int(len(y)), "majority_baseline": majority,
        "n_kept_bot": bl["n_kept_bot"], "n_kept_human": bl["n_kept_human"],
        "n_excluded_bot": bl["n_excluded_bot"],
        "n_excluded_human": bl["n_excluded_human"],
        "arms": res, "sweep": sweep,
        "H1_clause_i_count_spec_vs_count": clause_i,
        "H1_clause_ii_spec_vs_shannon": clause_ii,
        "vs_burstiness_floor": vs_burst,
        "partial_correlations": partials, "per_order": g1,
        "G2_pass": bool(clause_i["clears_floor"] and clause_ii["clears_floor"]),
    }
    (RESULTS / "p2_temporal.json").write_text(json.dumps(report, indent=1))

    print("\n" + "=" * 74)
    print("P2 — TEMPORAL FRONT, TEST OF H1")
    print("=" * 74)
    print(f"n = {len(y)}  (bot {bl['n_kept_bot']}, human {bl['n_kept_human']})   "
          f"majority baseline {majority:.4f}")
    print(f"excluded at min_events={head['min_events']}: "
          f"bot {bl['n_excluded_bot']}, human {bl['n_excluded_human']}")
    print(f"\n{'arm':<15}{'dim':>4}{'AUC':>10}{'±SD':>8}{'TPR@1%':>9}"
          f"{'mF1':>8}{'acc':>8}")
    for k, v in res.items():
        a, t = np.array(v["auc"]), np.array(v["tpr01"])
        print(f"{k:<15}{v['n_features']:>4}{a.mean():>10.4f}{a.std():>8.4f}"
              f"{t.mean():>9.3f}{np.mean(v['macro_f1']):>8.3f}"
              f"{np.mean(v['accuracy']):>8.3f}")

    print(f"\n{'H1 clause':<44}{'delta':>9}{'wins':>8}{'p':>9}  verdict")
    for nm, c in (("(i)  COUNT+SPEC_T  vs  COUNT alone", clause_i),
                  ("(ii) SPEC_T        vs  SHAN alone", clause_ii),
                  ("     COUNT+SPEC_T  vs  COUNT+BURST", vs_burst)):
        print(f"{nm:<44}{c['mean_diff']:>+9.4f}{c['wins']:>8}{c['p']:>9.4f}"
              f"  {'CLEARS' if c['clears_floor'] else 'fails'}")

    print("\npartial correlation with label, given log count (G4):")
    for k, v in partials.items():
        print(f"  {k:<10} raw {v['raw']:+.3f}   |   given count {v['given_count']:+.3f}")

    print(f"\nGATE G2 (H1): {'PASS' if report['G2_pass'] else 'FAIL'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
