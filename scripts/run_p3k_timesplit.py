#!/usr/bin/env python
"""WP-K -- within-corpus temporal-split generalisation axis (plan WP-K).

A second generalisation axis with fixed labels and a shifted era, enabled
uniquely by the snowflake decode: train on accounts whose first decoded
event predates the pre-registered boundary (2012-07-01T00:00Z), transfer to
the newer accounts. This previews H4 **without** schema mismatch -- same
corpus, same features, only the era shifts.

Split (pre-registered v1.0): account -> train iff first decoded event <
boundary. Sizes, per-side class balance, boundary-spanning fraction and era
histograms reported (G1); disjointness asserted elementwise on id sets in
code AND recorded in the JSON (G2); boundary sensitivity at 2012-01-01 and
2013-01-01 recorded as split statistics (G3).

Degradation per family, plan §8 D6: R = 20 seeded draws -- fit on 80 % of
the train partition, within-estimate on the held-out 20 %, transfer-estimate
on the FULL test partition; delta_r = within_r - transfer_r; mean +/- SD
over r; plus paired bootstrap over test users (B = 1000, SAME resamples
across families) for the transfer term. Scalers: HGB is scale-invariant and
no LR head is used here, so no scaler is fitted at all -- the leakage rule
(scalers fitted train-side) is satisfied vacuously and stated.

Families: META-lite (+ a no-age variant for the R8 guard -- age is the field
mechanically tied to calendar era), COUNT, BURST, SHAN, SPEC_T, SHAPE,
TAIL, SURV, TAIL+SURV (WP-J), SPEC_B_ALPHA (WP-H), SPEC_X_WORD / SPEC_X_CHAR
/ SPEC_X (WP-I). All features are computed on each account's FULL timeline;
only the account ERA shifts across the split.

Interpretation guard (R8 logic): if delta_META - delta_SPEC-family > 0.05,
the pair is recorded as direct H4-preview evidence, labelled **preview**,
never H4.

G4 null: within-era shuffle of assignment (random halves of the same
population, same D6 machinery) drives delta -> 0. Expected silent BY
CONSTRUCTION: randomly drawn halves of one population are exchangeable --
there is no covariate shift to degrade against -- so any nonzero delta
would measure sampling noise, not shift; its measured size calibrates the
reading of the real deltas.

Usage:
    python scripts/run_p3k_timesplit.py [--quiet]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import DATA_PROCESSED, FIGURES, RESULTS
from renyiext.events import load_events_cached, load_cresci_text_side
from renyiext.features import temporal_blocks, MS_PER_DAY
from renyiext.behaviour import alphabet_spectrum
from renyiext.textfront import account_text_features
from renyiext.tailstats import tail_features

warnings.filterwarnings("ignore", category=UserWarning)

CACHE = DATA_PROCESSED / "cresci_events_d9.npz"
OUT_JSON = RESULTS / "p3k_timesplit.json"
BOUNDARY_MS = int(datetime(2012, 7, 1, tzinfo=timezone.utc).timestamp()
                  * 1000)
BOUNDARY_ALT = {"2012-01-01": datetime(2012, 1, 1, tzinfo=timezone.utc),
                "2013-01-01": datetime(2013, 1, 1, tzinfo=timezone.utc)}
R_DRAWS = 20
B_BOOT = 1000
RNG_DRAWS, RNG_BOOT = 42, 1042          # distinct streams (§8 preamble)
BOT_C, HUM_C = "#e76f51", "#2a9d8f"


def auc(y, s):
    return float(roc_auc_score(y, s))


def d6_family(X_tr, y_tr, X_te, y_te, rng):
    """One family's D6 run: R draws of within/transfer + full-train
    transfer scores for the shared bootstrap."""
    n = len(y_tr)
    rng = np.random.default_rng(rng)
    within, transfer = [], []
    for _ in range(R_DRAWS):
        perm = rng.permutation(n)
        m = int(0.8 * n)
        tr, ho = perm[:m], perm[m:]
        clf = _fit(X_tr[tr], y_tr[tr])
        within.append(auc(y_tr[ho], clf.predict_proba(X_tr[ho])[:, 1]))
        transfer.append(auc(y_te, clf.predict_proba(X_te)[:, 1]))
    full = _fit(X_tr, y_tr)
    scores = full.predict_proba(X_te)[:, 1]
    return (np.array(within), np.array(transfer), scores)


def _fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=42, max_iter=200,
                                          early_stopping=False).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("loading events ...", flush=True)
    ev = load_events_cached(CACHE)
    print(" ", ev, flush=True)

    counts = ev.counts()
    kept = counts >= 5
    idx = np.where(kept)[0]
    y = (ev.labels[idx] == 1).astype(int)

    bl = temporal_blocks(ev, n_bins=24, hi=400 * MS_PER_DAY, min_events=5)
    assert np.array_equal(idx, bl["index"])
    fid = float(np.abs(np.log1p(counts[idx]) -
                       bl["blocks"]["COUNT"][:, 0]).max())
    assert fid == 0.0

    print("loading text/meta side + building all families ...", flush=True)
    t0 = time.time()
    meta, username, seq = load_cresci_text_side(
        ev, kept, int(datetime(2015, 1, 1, tzinfo=timezone.utc)
                      .timestamp() * 1000))

    # ---- the feature table: one column group per family --------------------
    b = bl["blocks"]
    shape = b["SPEC_T"].copy()
    shape[:, :6] -= shape[:, :6][:, [2]]
    shape[:, 6:] -= shape[:, 6:][:, [2]]
    tails = np.zeros((len(idx), 1)); survs = np.zeros((len(idx), 3))
    spec_ba = np.zeros((len(idx), 6))
    spec_w = np.zeros((len(idx), 6)); spec_c = np.zeros((len(idx), 6))
    for j, i in enumerate(idx):
        ts, _ = ev.events_of(int(i))
        a, _, _, sv = tail_features(ts)
        tails[j, 0] = a; survs[j] = sv
        types = np.array([t for t, _ in seq[j]], dtype=np.int64)
        spec_ba[j] = alphabet_spectrum(types, collapse=True)
        texts = [t for _, t in seq[j]]
        sw, sc, _ = account_text_features(texts)
        spec_w[j] = sw; spec_c[j] = sc
    print(f"  features built in {time.time()-t0:.0f}s", flush=True)

    logc = b["COUNT"]
    FAM = {
        "META": meta, "META_no_age": meta[:, :3],
        "COUNT": logc, "BURST": b["BURST"], "SHAN": b["SHAN"],
        "SPEC_T": b["SPEC_T"], "SHAPE": shape,
        "TAIL": tails, "SURV": survs,
        "TAIL+SURV": np.hstack([tails, survs]),
        "SPEC_B_ALPHA": spec_ba,
        "SPEC_X_WORD": spec_w, "SPEC_X_CHAR": spec_c,
        "SPEC_X": np.hstack([spec_w, spec_c]),
    }

    # ---- split (pre-registered v1.0) ----------------------------------------
    first_ms = np.array([ev.events_of(int(i))[0][0] for i in idx])
    train_m = first_ms < BOUNDARY_MS
    tr, te = np.where(train_m)[0], np.where(~train_m)[0]

    # G2: disjointness on user-id sets, elementwise
    ids_tr = set(map(str, ev.user_ids[idx[tr]]))
    ids_te = set(map(str, ev.user_ids[idx[te]]))
    assert not (ids_tr & ids_te), "split leaked: overlapping user ids"
    disjointness = {"n_intersect": len(ids_tr & ids_te),
                    "n_train": len(ids_tr), "n_test": len(ids_te),
                    "asserted": True}

    def split_stats(boundary_ms):
        m = first_ms < boundary_ms
        a, c = np.where(m)[0], np.where(~m)[0]
        ya, yc = y[a], y[c]
        span = ((first_ms < boundary_ms) &
                (np.array([ev.events_of(int(i))[0][-1] for i in idx])
                 >= boundary_ms))
        return {"n_train": int(m.sum()), "n_test": int((~m).sum()),
                "balance_train_bot": float(ya.mean()),
                "balance_test_bot": float(yc.mean()),
                "spanning_boundary": int(span.sum()),
                "spanning_fraction": float(span.mean())}

    stats = {"2012-07-01 (registered)": split_stats(BOUNDARY_MS)}
    for nm, dt in BOUNDARY_ALT.items():
        stats[nm] = split_stats(int(dt.timestamp() * 1000))
    if not args.quiet:
        s = stats["2012-07-01 (registered)"]
        print(f"[split] train {s['n_train']} (bot {s['balance_train_bot']:.4f}"
              f") / test {s['n_test']} (bot {s['balance_test_bot']:.4f}); "
              f"spanning {s['spanning_boundary']} "
              f"({100*s['spanning_fraction']:.2f}%)")

    Xtr_y, Xte_y = y[tr], y[te]

    # ---- D6 per family -------------------------------------------------------
    fams_out, boot_scores = {}, {}
    boot_rng = np.random.default_rng(RNG_BOOT)
    boot_idx = boot_rng.integers(0, len(te), size=(B_BOOT, len(te)))
    draw_rng = np.random.default_rng(RNG_DRAWS)
    for name, X in FAM.items():
        Xtr, Xte = X[tr], X[te]
        within, transfer, scores = d6_family(Xtr, Xtr_y, Xte, Xte_y,
                                             RNG_DRAWS)
        boots = np.array([auc(Xte_y[bi], scores[bi]) for bi in boot_idx])
        ci = [float(np.percentile(boots, 2.5)),
              float(np.percentile(boots, 97.5))]
        d = within - transfer
        fams_out[name] = {
            "dim": int(np.asarray(X).shape[1]),
            "within_mean": float(within.mean()),
            "within_sd": float(within.std()),
            "transfer_mean": float(transfer.mean()),
            "transfer_sd": float(transfer.std()),
            "delta_mean": float(d.mean()), "delta_sd": float(d.std()),
            "transfer_full_fit_auc": auc(Xte_y, scores),
            "transfer_ci95_boot": ci,
            "majority_baseline_train": float(max(Xtr_y.mean(),
                                                 1 - Xtr_y.mean())),
            "majority_baseline_test": float(max(Xte_y.mean(),
                                                1 - Xte_y.mean())),
        }
        boot_scores[name] = scores
        if not args.quiet:
            v = fams_out[name]
            print(f"  {name:<14} within {v['within_mean']:.4f} "
                  f"transfer {v['transfer_mean']:.4f} "
                  f"(CI {ci[0]:.4f}-{ci[1]:.4f})  Δ {v['delta_mean']:+.4f}")

    # ---- R8 guard + preview pairs -------------------------------------------
    d_meta = fams_out["META"]["delta_mean"]
    preview_pairs = []
    for name, v in fams_out.items():
        if name.startswith(("META", "COUNT")):
            continue
        gap = d_meta - v["delta_mean"]
        preview_pairs.append({"pair": f"META vs {name}", "delta_gap": gap,
                              "preview_fired": bool(gap > 0.05)})
    r8 = {"meta_full_delta": d_meta,
          "meta_no_age_delta": fams_out["META_no_age"]["delta_mean"],
          "note": "age is the META field mechanically tied to calendar era; "
                  "the no-age variant is the era-guard reading",
          "preview_pairs": sorted(preview_pairs,
                                  key=lambda p: -p["delta_gap"])}
    r8["any_preview_fired"] = bool(any(p["preview_fired"]
                                       for p in r8["preview_pairs"]))

    # ---- G4 null: within-era shuffled assignment ----------------------------
    null_rng = np.random.default_rng(77)
    perm = null_rng.permutation(len(y))
    half = len(y) // 2
    ptr, pte = perm[:half], perm[half:]
    null_out = {}
    for name, X in FAM.items():
        within, transfer, _ = d6_family(X[ptr], y[ptr], X[pte], y[pte],
                                        77)
        null_out[name] = float((within - transfer).mean())
    g4 = {
        "design": "random halves of the SAME population, identical D6 "
                  "machinery (R = 20)",
        "delta_mean_per_family": null_out,
        "max_abs_delta": float(max(abs(v) for v in null_out.values())),
        "expected_silent_because":
            "randomly drawn halves of one population are exchangeable: "
            "there is no covariate shift to degrade against, so delta "
            "measures only draw noise; its measured size calibrates the "
            "real deltas above",
    }
    if not args.quiet:
        print(f"[G4] shuffled-assignment null: max |Δ| "
              f"{g4['max_abs_delta']:.4f} (expected silent)")

    # ---- figures --------------------------------------------------------------
    years = first_ms.astype("datetime64[ms]").astype(datetime)
    yr = np.array([t.year for t in years])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    bins = np.arange(yr.min(), yr.max() + 2) - 0.5
    for mask, nm, col in ((~train_m, "test era", HUM_C),
                          (train_m, "train era", BOT_C)):
        ax.hist(yr[mask], bins=bins, alpha=0.65, color=col, label=nm)
    ax.axvline(2012.5, color="black", ls="--", lw=2)
    ax.text(2012.55, ax.get_ylim()[1] * 0.9,
            "boundary 2012-07-01\n(pre-registered)", fontsize=9)
    ax.set_xlabel("year of account's first decoded event")
    ax.set_ylabel("accounts")
    ax.set_title(f"WP-K — era split (G1): train {stats['2012-07-01 (registered)']['n_train']} "
                 f"/ test {stats['2012-07-01 (registered)']['n_test']}, "
                 f"spanning {stats['2012-07-01 (registered)']['spanning_fraction']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "p3k_era_split.png", dpi=130)
    plt.close(fig)
    if not args.quiet:
        print(f"  [G1] rendered -> {FIGURES / 'p3k_era_split.png'}")

    order = sorted(FAM, key=lambda f: fams_out[f]["delta_mean"])
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(order) + 2.2))
    ys = np.arange(len(order))
    vals = [fams_out[f]["delta_mean"] for f in order]
    errs = [fams_out[f]["delta_sd"] for f in order]
    cols = [BOT_C if f == "META" else "#3b6ea5" for f in order]
    ax.barh(ys, vals, xerr=errs, color=cols, alpha=0.85)
    for yi, f in zip(ys, order):
        lo, hi = fams_out[f]["transfer_ci95_boot"]
        ax.plot([lo - fams_out[f]["within_mean"],
                 hi - fams_out[f]["within_mean"]],
                [yi, yi], color="black", lw=1.4)
    ax.axvline(0.0, color="grey", lw=0.9)
    ax.axvline(0.05, color="crimson", ls="--", lw=1.5,
               label="preview threshold 0.05 (Δ_META − Δ_family)")
    ax.set_yticks(ys); ax.set_yticklabels(order, fontsize=8.5)
    ax.set_xlabel("degradation Δ = within − transfer (era shift)")
    ax.set_title("WP-K — degradation per family with D6 SDs (bars) and "
                 "bootstrap CI of transfer (whiskers)\n"
                 "[preview-labelled evidence — not H4]", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "p3k_degradation.png", dpi=130)
    plt.close(fig)
    if not args.quiet:
        print(f"  rendered -> {FIGURES / 'p3k_degradation.png'}")

    report = {
        "phase": "P3k-timesplit (WP-K)",
        "exploratory_label": "all evidence PREVIEW-labelled, never H4",
        "split_definition": "account -> train iff first decoded event < "
                            "2012-07-01T00:00Z (pre-registered v1.0)",
        "randomness": {"d6_draws": f"default_rng({RNG_DRAWS}), R={R_DRAWS}",
                       "bootstrap": f"default_rng({RNG_BOOT}), B={B_BOOT}, "
                                    "same resamples across families",
                       "g4_null": "default_rng(77)"},
        "feature_config": {"min_events": 5, "n_kept": int(len(idx)),
                           "note": "features on FULL timelines; only era "
                                   "shifts across the split"},
        "split_stats": stats,
        "disjointness": disjointness,
        "families": fams_out,
        "r8_guard": r8,
        "g4_null": g4,
        "fidelity": {"kept_index_matches_temporal": True,
                     "count_block_max_abs_diff": fid},
    }
    OUT_JSON.write_text(json.dumps(report, indent=1))

    print("\n" + "=" * 88)
    print("WP-K — TEMPORAL-SPLIT GENERALISATION  [preview-labelled]")
    print("=" * 88)
    print(f"{'family':<15}{'dim':>4}{'within':>9}{'transfer':>10}"
          f"{'Δ':>9}{'CI95 transfer':>18}")
    for name in order:
        v = fams_out[name]
        ci = v["transfer_ci95_boot"]
        print(f"{name:<15}{v['dim']:>4}{v['within_mean']:>9.4f}"
              f"{v['transfer_mean']:>10.4f}{v['delta_mean']:>+9.4f}"
              f"   [{ci[0]:.4f},{ci[1]:.4f}]")
    fired = [p["pair"] for p in r8["preview_pairs"] if p["preview_fired"]]
    print(f"\npreview pairs fired (>0.05): {fired or 'none'}")
    print(f"G4 null max |Δ|: {g4['max_abs_delta']:.4f}")
    print(f"json -> {OUT_JSON}")
    print("=" * 88)


if __name__ == "__main__":
    main()
