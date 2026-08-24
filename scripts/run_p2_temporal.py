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

WP-E (2026-08-24): all evaluation logic lives in ``renyiext.evaluate``
(per-fold ``tpr01_foldmean`` beside the pooled statistic, review C3; the D2
noise-padded dimensionality control with its pre-registered failure
semantics, review C2; ``sigma_config`` beside every floor verdict, review
D4). The retroactive dim-matched rows are computed here:

    clause (ii) reading       SPEC_T          vs SHAN+NOISE(10)
    burstiness verdict        COUNT+SPEC_T    vs COUNT+BURST+NOISE(9)

P2's registered gate verdict stands as gated; the matched rows attach the
equal-dimension reading to it, and whichever interpretation rule fires is
EXECUTED (downgrade recorded in HANDOFF/FINDINGS), never merely noted.

Regression gate: before the JSON is rewritten, every legacy array
(AUC / pooled TPR / macro-F1 / accuracy per arm, sweep values, clause
deltas, baselines) is compared elementwise against the artefact on disk;
tolerance 1e-4 (acceptance: unchanged to 4 decimals). Any other drift is a
bug -- find it before committing, never tune to it.

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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import Config, FIGURES, RESULTS, DATA_PROCESSED
from renyiext.events import load_events_cached
from renyiext.features import temporal_blocks, MS_PER_DAY
from renyiext.spectrum import spectrum_labels
from renyiext.evaluate import (eval_arm, run_arms, paired, partial_corr,
                               noise_padding, interpret_dim_matched,
                               sigma_config)

warnings.filterwarnings("ignore", category=UserWarning)

CACHE = DATA_PROCESSED / "cresci_events_d9.npz"
OUT_JSON = RESULTS / "p2_temporal.json"
GATE_TOL = 1e-4

ARMS = {
    "COUNT":            ("COUNT",),
    "BURST":            ("BURST",),
    "SHAN":             ("SHAN",),
    "SPEC_T":           ("SPEC_T",),
    "COUNT+BURST":      ("COUNT", "BURST"),
    "COUNT+SHAN":       ("COUNT", "SHAN"),
    "COUNT+SPEC_T":     ("COUNT", "SPEC_T"),
}

# Retroactive dim-matched controls (plan WP-E task 3): (floor blocks,
# family blocks). k and the arm name are derived, never chosen; arm_index
# is the D2 salt, assigned by this tuple's fixed order.
DIM_MATCHED = (
    (("SHAN",),            ("SPEC_T",)),          # clause (ii) reading
    (("COUNT", "BURST"),   ("COUNT", "SPEC_T")),  # burstiness verdict
)


def materialize(blocks, parts):
    return np.hstack([blocks[p] for p in parts])


def dim_matched_specs(blocks):
    """Build the D2 padded-floor arms and their G3 config echo."""
    specs, echo = {}, {}
    for i, (floor_parts, fam_parts) in enumerate(DIM_MATCHED):
        X_floor = materialize(blocks, floor_parts)
        X_fam = materialize(blocks, fam_parts)
        k = int(X_fam.shape[1] - X_floor.shape[1])
        name = "+".join(floor_parts) + f"+NOISE({k})"
        specs[name] = (lambda s, Xf=X_floor, k=k, i=i:
                       noise_padding(Xf, k, s, i))
        echo[name] = {"floor_blocks": list(floor_parts),
                      "family_blocks": list(fam_parts),
                      "k": k, "arm_index": i,
                      "rng": "default_rng(seed*1000 + arm_index)",
                      "k_rule": "dim(family) - dim(floor)"}
    return specs, echo


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
    spectrum sees it. Swept here for AUC, not merely for the histogram.

    WP-E: each row also carries ``auc_count_burst`` so the burstiness
    verdict's ``sigma_config`` (plan §8 D3) spans the full published sweep.
    """
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
        cs = [eval_arm(materialize(b, ("COUNT", "SPEC_T")), y, s)["auc"]
              for s in seeds]
        cb = [eval_arm(materialize(b, ("COUNT", "BURST")), y, s)["auc"]
              for s in seeds]
        rows.append({"n_bins": g["n_bins"], "hi_days": g["hi"] / MS_PER_DAY,
                     "min_events": g["min_events"], "n": int(len(y)),
                     "auc_spec": float(np.mean(sp)), "auc_shan": float(np.mean(sh)),
                     "auc_count": float(np.mean(ct)),
                     "auc_count_spec": float(np.mean(cs)),
                     "auc_count_burst": float(np.mean(cb)),
                     "clause_i": float(np.mean(cs) - np.mean(ct)),
                     "clause_ii": float(np.mean(sp) - np.mean(sh))})
        if not quiet:
            r = rows[-1]
            print(f"    bins={r['n_bins']:<3} hi={r['hi_days']:<6.0f}d "
                  f"min_ev={r['min_events']:<3} n={r['n']:<5} "
                  f"(i) {r['clause_i']:+.4f}  (ii) {r['clause_ii']:+.4f}", flush=True)
    return rows


def regression_gate(new_report, path=OUT_JSON, tol=GATE_TOL):
    """Elementwise gate of every legacy number against the artefact on disk
    (plan WP-E task 2). Only additive structure may differ; any value drift
    beyond ``tol`` raises BEFORE the artefact is overwritten. Returns the
    gate summary for the log (never stored in the JSON -- storing it would
    make consecutive runs differ, breaking S2.6)."""
    if not path.exists():
        print(f"  [gate] {path.name} absent -- first run, nothing to gate")
        return None
    old = json.loads(path.read_text())
    if old.get("seeds") != new_report["seeds"]:
        print("  [gate] SKIPPED -- seed set differs from the stored artefact")
        return None

    worst, where = 0.0, ""

    def cmp(ov, nv, tag):
        nonlocal worst, where
        d = abs(float(ov) - float(nv))
        if d > worst:
            worst, where = d, tag

    for arm, ov in old["arms"].items():
        nv = new_report["arms"][arm]
        assert nv["n_features"] == ov["n_features"], f"{arm}: dim drifted"
        for metric in ("auc", "tpr01", "macro_f1", "accuracy",
                       "tpr01_foldmean"):
            if metric in ov and metric in nv:
                for i, (a, b) in enumerate(zip(ov[metric], nv[metric])):
                    cmp(a, b, f"arms.{arm}.{metric}[{i}]")

    for i, (orow, nrow) in enumerate(zip(old["sweep"], new_report["sweep"])):
        for key, ov in orow.items():
            if isinstance(ov, (int, float)) and key in nrow:
                cmp(ov, nrow[key], f"sweep[{i}].{key}")

    for key in ("H1_clause_i_count_spec_vs_count",
                "H1_clause_ii_spec_vs_shannon", "vs_burstiness_floor"):
        for stat in ("mean_diff", "std_diff", "p"):
            cmp(old[key][stat], new_report[key][stat], f"{key}.{stat}")

    for key in ("n", "majority_baseline", "n_kept_bot", "n_kept_human",
                "n_excluded_bot", "n_excluded_human"):
        cmp(old[key], new_report[key], key)
    assert old["G2_pass"] == new_report["G2_pass"], "G2_pass flipped"

    if worst > tol:
        raise AssertionError(
            f"REGRESSION GATE FAILED: max |diff| {worst:.3e} > {tol:g} at "
            f"{where} vs {path} -- pipeline drift; find the bug before "
            f"committing, do not overwrite the artefact")
    summary = {"max_abs_diff": worst, "tol": tol, "where": where or "(all zero)",
               "pass": True}
    print(f"  [gate] regression vs {path.name}: PASS "
          f"(max |diff| {worst:.2e} <= {tol:g})")
    return summary


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
    legacy_arms = {name: materialize(bl["blocks"], parts)
                   for name, parts in ARMS.items()}
    res = run_arms(legacy_arms, y, seeds, "hgb", args.quiet)

    print("[dim-matched floors] plan §8 D2 noise-padded controls")
    dm_specs, dm_echo = dim_matched_specs(bl["blocks"])
    res.update(run_arms(dm_specs, y, seeds, "hgb", args.quiet))

    clause_i = paired(res["COUNT+SPEC_T"]["auc"], res["COUNT"]["auc"])
    clause_ii = paired(res["SPEC_T"]["auc"], res["SHAN"]["auc"])
    vs_burst = paired(res["COUNT+SPEC_T"]["auc"], res["COUNT+BURST"]["auc"])

    # sigma_config (§8 D3): population SD of each delta across the full
    # published sweep, beside the seed SD the paired stats already carry.
    clause_i["sigma_config"] = sigma_config([r["clause_i"] for r in sweep])
    clause_ii["sigma_config"] = sigma_config([r["clause_ii"] for r in sweep])
    vs_burst["sigma_config"] = sigma_config(
        [r["auc_count_spec"] - r["auc_count_burst"] for r in sweep])

    shan_n_name = next(iter(dm_echo))            # "SHAN+NOISE(10)"
    cbn_name = list(dm_echo)[1]                  # "COUNT+BURST+NOISE(9)"
    clause_ii_dm = paired(res["SPEC_T"]["auc"], res[shan_n_name]["auc"])
    burst_dm = paired(res["COUNT+SPEC_T"]["auc"], res[cbn_name]["auc"])
    clause_ii_dm["sigma_config"] = clause_ii["sigma_config"]
    burst_dm["sigma_config"] = vs_burst["sigma_config"]
    clause_ii_dm["verdict"] = interpret_dim_matched(clause_ii_dm["mean_diff"],
                                                    clause_ii_dm["significant"])
    burst_dm["verdict"] = interpret_dim_matched(burst_dm["mean_diff"],
                                                burst_dm["significant"])

    cnt = bl["blocks"]["COUNT"][:, 0]
    spec = bl["blocks"]["SPEC_T"]
    names = g1["order_names"]

    partials = {names[k]: {
        "raw": float(np.corrcoef(spec[:, k], y)[0, 1]),
        "given_count": partial_corr(spec[:, k], y.astype(float), cnt)}
        for k in range(12)}

    dim_matched_block = {
        "definition": "plan §8 D2: X_noisy = [X_floor || N], "
                      "N ~ default_rng(seed*1000 + arm_index), "
                      "k = dim(family) - dim(floor)",
        "interpretation_rules":
            "pre-registered, plan WP-E task 3 [rev1]; binding: delta <= 0 -> "
            "confounded_dimensionality (downgrade EXECUTED in HANDOFF/"
            "FINDINGS); 0 < delta < 0.02 -> real_but_subfloor_not_claimable; "
            "delta >= 0.02 and significant -> supports_clause",
        "knobs": dm_echo,
        "clause_ii_SPEC_T_vs_SHAN_NOISE": {
            **clause_ii_dm,
            "family": "SPEC_T", "floor_arm": shan_n_name,
            "family_auc_mean": float(np.mean(res["SPEC_T"]["auc"])),
            "floor_arm_auc_mean": float(np.mean(res[shan_n_name]["auc"]))},
        "burstiness_COUNT_SPEC_T_vs_COUNT_BURST_NOISE": {
            **burst_dm,
            "family": "COUNT+SPEC_T", "floor_arm": cbn_name,
            "family_auc_mean": float(np.mean(res["COUNT+SPEC_T"]["auc"])),
            "floor_arm_auc_mean": float(np.mean(res[cbn_name]["auc"]))},
    }

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
        "dim_matched": dim_matched_block,
        "partial_correlations": partials, "per_order": g1,
        "G2_pass": bool(clause_i["clears_floor"] and clause_ii["clears_floor"]),
    }

    regression_gate(report)
    OUT_JSON.write_text(json.dumps(report, indent=1))

    print("\n" + "=" * 74)
    print("P2 — TEMPORAL FRONT, TEST OF H1")
    print("=" * 74)
    print(f"n = {len(y)}  (bot {bl['n_kept_bot']}, human {bl['n_kept_human']})   "
          f"majority baseline {majority:.4f}")
    print(f"excluded at min_events={head['min_events']}: "
          f"bot {bl['n_excluded_bot']}, human {bl['n_excluded_human']}")
    print(f"\n{'arm':<22}{'dim':>4}{'AUC':>10}{'±SD':>8}{'TPR@1%':>9}"
          f"{'TPRfold':>9}{'mF1':>7}{'acc':>7}")
    for k, v in res.items():
        a, t = np.array(v["auc"]), np.array(v["tpr01"])
        tf = np.array(v["tpr01_foldmean"])
        print(f"{k:<22}{v['n_features']:>4}{a.mean():>10.4f}{a.std():>8.4f}"
              f"{t.mean():>9.3f}{tf.mean():>9.3f}"
              f"{np.mean(v['macro_f1']):>7.3f}{np.mean(v['accuracy']):>7.3f}")

    print(f"\n{'comparison':<44}{'delta':>9}{'wins':>8}{'p':>9}"
          f"{'sig_cfg':>9}  verdict")
    for nm, c in (("(i)  COUNT+SPEC_T  vs  COUNT alone", clause_i),
                  ("(ii) SPEC_T        vs  SHAN alone", clause_ii),
                  ("     COUNT+SPEC_T  vs  COUNT+BURST", vs_burst)):
        print(f"{nm:<44}{c['mean_diff']:>+9.4f}{c['wins']:>8}{c['p']:>9.4f}"
              f"{c['sigma_config']:>9.4f}"
              f"  {'CLEARS' if c['clears_floor'] else 'fails'}")
    for nm, c in (("(ii) matched: SPEC_T vs SHAN+NOISE", clause_ii_dm),
                  ("     matched: CS vs CB+NOISE", burst_dm)):
        print(f"{nm:<44}{c['mean_diff']:>+9.4f}{c['wins']:>8}{c['p']:>9.4f}"
              f"{c['sigma_config']:>9.4f}  {c['verdict']}")

    print("\npartial correlation with label, given log count (G4):")
    for k, v in partials.items():
        print(f"  {k:<10} raw {v['raw']:+.3f}   |   given count {v['given_count']:+.3f}")

    print(f"\nGATE G2 (H1): {'PASS' if report['G2_pass'] else 'FAIL'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
