#!/usr/bin/env python
"""WP-B -- TwiBot-20 preflight: the volume landscape, and the H4 framing.

PLAN-02-ext-research.md sect. 5 WP-B. Descriptive statistics plus ONE model
(univariate volume AUC); its job is to select, before any front is built,
whether H4 runs as pre-registered or in the amended H4' form (incremental over
each corpus's own volume feature).

Pre-registered branch (plan WP-B task 2): AUC(statuses alone) >= 0.85 on
TwiBot-20  ==>  H4' fires; else H4 stands as chartered.

What this script measures, from artefacts already on disk:

  TB20  data/processed/ext/twibot20_preflight_v1.npz -- BotRGCN's tensor set
        (converted once from data/raw/bot/twibot-20/*.pt via torch; labels
        occupy the first 11,826 rows per botsage/data.py's documented
        convention). Five properties ALREADY z-scored, verified here (G3):
        [followers, active_days, screen_name_length, friends, statuses].
  Cresci  raw public_metrics from node.json for the labelled users, cached to
        data/processed/ext/cresci_meta_v1.npz on first run.

Known limitations, recorded rather than hidden:

  * Per-user TWEET COUNTS are not recoverable for TwiBot-20 (tweets_tensor.pt
    is dense pooled embeddings; every row non-zero), so the retention curve is
    computed on statuses deciles, not tweet-count cutoffs, and "tweet-text
    availability" is uninformative by construction (measured: all rows
    non-zero).
  * The five properties arrive z-scored, so absolute scales are gone; AUC is
    affine-invariant and therefore unaffected (stated where used). Raw-scale
    overlays are possible only against Cresci's RAW profile fields.

Usage:
    python scripts/run_p6b_tb20_preflight.py [--quiet]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import DATA_RAW, DATA_PROCESSED, FIGURES, RESULTS

MS_PER_DAY = 86_400_000.0
AGE_REFERENCE_ISO = "2020-01-01T00:00:00+00:00"   # fixed, printed (G3)
AGE_REF_MS = int(datetime.fromisoformat(AGE_REFERENCE_ISO).timestamp() * 1000)

# D8 overlap mapping: source field -> target column index in the TB20 tensor.
OVERLAP = [("followers_count", 0, "followers"),
           ("following_count", 3, "friends"),
           ("tweet_count",     4, "statuses"),
           ("account_age_days", 1, "active_days")]
DROPPED = {"cresci": ["listed_count (favourites stand-in)"],
           "twibot20": ["screen_name_length"]}
VOLUME_COL = 4          # 'statuses' -- named before any AUC is computed
VOLUME_BRANCH_THRESHOLD = 0.85
CRESCI_FMT = "%a %b %d %H:%M:%S %z %Y"


def load_cresci_meta() -> tuple[np.ndarray, np.ndarray]:
    """Raw [tweet_count, followers, following, listed, age_days] + labels."""
    cache = DATA_PROCESSED / "cresci_meta_v1.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["meta"], z["labels"]

    base = DATA_RAW / "bot" / "cresci-2015"
    labels_by_id: dict[str, int] = {}
    with open(base / "label.csv") as fh:
        next(fh)
        for line in fh:
            uid, lab = line.rstrip("\n").split(",")[:2]
            labels_by_id[uid] = 1 if lab.strip() == "bot" else 0

    meta, labels = [], []
    ref = datetime.fromisoformat("2013-06-06T00:00:00+00:00").timestamp() * 1000
    with open(base / "node.json") as fh:
        nodes = json.load(fh)          # same access pattern as events.py
    for obj in nodes:
        oid = obj.get("id", "")
        if not oid.startswith("u") or oid not in labels_by_id:
            continue
        pm = obj.get("public_metrics") or {}
        created = obj.get("created_at")
        try:
            born = int(datetime.strptime(created, CRESCI_FMT)
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
        except (TypeError, ValueError):
            born = None
        age = (ref - born) / MS_PER_DAY if born is not None else np.nan
        meta.append([
            float(pm.get("tweet_count") or 0),
            float(pm.get("followers_count") or 0),
            float(pm.get("following_count") or 0),
            float(pm.get("listed_count") or 0),
            float(age),
        ])
        labels.append(labels_by_id[oid])
    del nodes
    meta = np.asarray(meta, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, meta=meta, labels=labels)
    return meta, labels


def auc_cv(X, y, seeds):
    out = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            clf = HistGradientBoostingClassifier(random_state=s, max_iter=200,
                                                 early_stopping=False)
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        out.append(roc_auc_score(y, oof))
    return float(np.mean(out)), float(np.std(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    seeds = list(range(42, 52))

    z = np.load(DATA_PROCESSED / "twibot20_preflight_v1.npz")
    P, y_tb, names = z["props_labelled"], z["labels"], list(z["prop_names"])
    corpus_mu, corpus_sd = z["corpus_col_mean"], z["corpus_col_sd"]
    meta_cr, y_cr = load_cresci_meta()

    n_tb, n_cr = len(y_tb), len(y_cr)
    maj_tb = float(max(y_tb.mean(), 1 - y_tb.mean()))
    maj_cr = float(max(y_cr.mean(), 1 - y_cr.mean()))

    # ---- G3: verify the "already z-scored" claim at the level it was made
    # (the corpus it was standardised over), and report labelled-subset drift.
    col_mu, col_sd = P.mean(0), P.std(0)
    zscore_claim_verified = bool(np.all(np.abs(corpus_mu) < 0.01)
                                 and np.all(np.abs(corpus_sd - 1) < 0.01))

    # ---- univariate AUCs (affine-invariant; z-scoring cannot move them) --
    prop_auc = {}
    for j, nm in enumerate(names):
        m, s = auc_cv(P[:, [j]], y_tb, seeds)
        prop_auc[nm] = {"auc": m, "sd": s}
    vol_auc, vol_sd = prop_auc[names[VOLUME_COL]]["auc"], \
        prop_auc[names[VOLUME_COL]]["sd"]
    fired = bool(vol_auc >= VOLUME_BRANCH_THRESHOLD)

    # ---- retention on statuses deciles (tweet counts unavailable) --------
    qs = np.quantile(P[:, VOLUME_COL], np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(qs, P[:, VOLUME_COL], side="right") - 1, 0, 9)
    ret_decile = {f"d{i}": {
        "bot_kept_frac": float((y_tb[(dec == i)] == 1).mean()),
        "n": int((dec == i).sum())} for i in range(10)}

    # ---- alignment dry-run (D8/R8): source scaler onto z-scored target ---
    src_idx = [{"followers_count": 1, "following_count": 2,
                "tweet_count": 0, "account_age_days": 4}[f] for f, _, _ in OVERLAP]
    scaler = StandardScaler().fit(meta_cr[:, src_idx])
    tgt = P[:, [j for _, j, _ in OVERLAP]]
    aligned = scaler.transform(tgt)
    offsets = {}
    for k, (sf, _, tf) in enumerate(OVERLAP):
        offsets[f"{sf} -> {tf}"] = {
            "cresci_mu": float(scaler.mean_[k]),
            "cresci_sd": float(scaler.scale_[k]),
            "tb20_post_align_mean": float(aligned[:, k].mean()),
            "tb20_post_align_sd": float(aligned[:, k].std()),
        }

    # ---- R8 made measurable: naive vs marginal-recalibrated alignment ----
    # (a) strict D8: source scaler onto target -- the transfer a referee gets
    #     if no recalibration is allowed;
    # (b) upper bound that leaks ONLY label-free marginals (target mean/sd per
    #     field), isolating pure-scale artefact from genuine shift.
    # Diagnostic only; labelled exploratory in the JSON.
    diag = {}
    for tag, X4 in (("naive_source_scaler", aligned),
                    ("marginal_recalibrated",
                     StandardScaler().fit_transform(tgt))):
        m, s = auc_cv(X4, y_tb, seeds)
        diag[tag] = {"auc_4field_meta_on_tb20": m, "sd": s,
                     "post_align_col_sds": [float(v) for v in X4.std(0)]}

    # ---- figures ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    a = axes[0]
    bins = np.linspace(-4, 8, 61)
    for mask, nm, c in ((y_tb == 0, "human", "#2a9d8f"),
                        (y_tb == 1, "bot", "#e76f51")):
        a.hist(P[mask, VOLUME_COL], bins=bins, histtype="step", lw=2,
               density=True, color=c, label=nm)
    a.set_title(f"TwiBot-20 statuses (z) — AUC alone "
                f"{vol_auc:.3f}", loc="left")
    a.set_xlabel("statuses_count (corpus z-score)"); a.legend(fontsize=8)

    a = axes[1]
    vols = meta_cr[:, 0]
    bins = np.logspace(0, np.log10(max(vols.max(), 1)), 60)
    for mask, nm, c in ((y_cr == 0, "human", "#2a9d8f"),
                        (y_cr == 1, "bot", "#e76f51")):
        a.hist(np.maximum(vols[mask], 0.5), bins=bins, histtype="step",
               lw=2, density=True, color=c, label=nm)
    a.set_xscale("log")
    a.set_title("Cresci-2015 RAW tweet_count — no TB20 counterpart at raw "
                "scale", loc="left", fontsize=9)
    a.set_xlabel("profile tweet_count (raw)"); a.legend(fontsize=8)

    a = axes[2]
    xs = np.arange(10)
    frac_bot = [ret_decile[f"d{i}"]["bot_kept_frac"] for i in range(10)]
    a.bar(xs, frac_bot, color="#3b6ea5")
    a.axhline(maj_tb, color="crimson", ls="--", lw=1.5,
              label=f"majority {maj_tb:.3f}")
    a.set_xticks(xs); a.set_xlabel("statuses decile (low → high)")
    a.set_ylabel("bot fraction"); a.set_title("Bot share by volume decile",
                                              loc="left")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "p6b_tb20_volume.png", dpi=130)
    plt.close(fig)

    report = {
        "phase": "P6-preflight (WP-B)",
        "seeds": seeds,
        "twibot20": {
            "n_labelled": n_tb, "n_bot": int(y_tb.sum()),
            "n_human": int((y_tb == 0).sum()),
            "majority_baseline": maj_tb,
            "prop_names": names,
            "corpus_col_mean": [float(v) for v in corpus_mu],
            "corpus_col_sd": [float(v) for v in corpus_sd],
            "labelled_subset_col_mean": [float(v) for v in col_mu],
            "labelled_subset_col_sd": [float(v) for v in col_sd],
            "zscore_claim_verified": zscore_claim_verified,
            "zscore_note": "verified against the full 229580-user corpus the "
                           "standardisation was fit on; the labelled subset "
                           "drifts (heavy-tailed slice), which is itself "
                           "reported",
            "per_property_auc": prop_auc,
            "volume_column": names[VOLUME_COL],
            "volume_auc": vol_auc, "volume_auc_sd": vol_sd,
            "retention_on_status_deciles": ret_decile,
            "text_availability": {
                "derivable": False,
                "note": "tweets_tensor.pt is dense pooled embeddings; all "
                        "229580 rows non-zero, so availability is 1 by "
                        "construction and per-user tweet counts are lost"},
        },
        "cresci2015": {"n_labelled": n_cr, "n_bot": int(y_cr.sum()),
                       "majority_baseline": maj_cr},
        "alignment_dry_run": {"overlap": [list(o) for o in OVERLAP],
                              "dropped": DROPPED,
                              "age_reference_iso": AGE_REFERENCE_ISO,
                              "offsets": offsets,
                              "r8_diagnostic_auc": diag},
        "branch": {"rule": "AUC(statuses alone) >= 0.85 -> H4' (incremental "
                           "over each corpus's own volume)",
                   "threshold": VOLUME_BRANCH_THRESHOLD,
                   "value": vol_auc, "fired_h4_prime": fired},
    }
    path = RESULTS / "p6b_tb20_preflight.json"
    path.write_text(json.dumps(report, indent=1))

    if not args.quiet:
        print(f"TB20 labelled {n_tb} (bot {int(y_tb.sum())}, majority "
              f"{maj_tb:.4f}) | z-score verified vs corpus: "
              f"{zscore_claim_verified}")
        for nm, d in prop_auc.items():
            star = "*" if nm == names[VOLUME_COL] else ""
            print(f"  AUC {nm:<18} {d['auc']:.4f} ± {d['sd']:.4f} {star}")
        for tag, d in diag.items():
            print(f"  R8 diag {tag:<24} 4-field META AUC {d['auc_4field_meta_on_tb20']:.4f}"
                  f"  col sds {[round(v,5) for v in d['post_align_col_sds']]}")
        print(f"BRANCH: {'H4 prime FIRES' if fired else 'H4 as chartered'} "
              f"(threshold {VOLUME_BRANCH_THRESHOLD}, value {vol_auc:.4f})")
    print(f"\njson -> {path}\nfigure -> {FIGURES}/p6b_tb20_volume.png")


if __name__ == "__main__":
    main()
