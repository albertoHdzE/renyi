#!/usr/bin/env python
"""WP-N -- Protocol C execution: H4, the primary claim (plan WP-N).

SCOPED BY THE TARGET-MODALITY AUDIT (recorded in the JSON, bitacora 19):
the only TwiBot-20 artefact obtainable is BotRGCN's tensor set
(data/raw/bot/twibot-20/), whose tweets_tensor.pt is dense pooled 768-d
embeddings -- no timestamps, no post types, no raw text. Raw TwiBot-20 is
gated by its authors (github.com/bunsenfeng/twibot-20: "Due to privacy
issues, we are not directly posting the dataset"), and the open TwiBot-22
user.json carries profiles only. Consequence, audited elementwise in-script:

  commensurable across corpora : META_aligned(4), VOL_PROFILE(1)
  uncomputable on the target   : COUNT(decoded), BURST, SHAN, SPEC_T,
                                 SHAPE, TAIL, SURV, TAIL+SURV,
                                 SPEC_B_ALPHA, SPEC_X_WORD/CHAR

so H4 executes on the side that CAN run: the degradation estimator (D6) for
the metadata family and its floors, with every pre-registered control --
R8 alignment-with/without, dim-matched arms, sanity pseudo-transfer null,
majority baselines, TPR@1%FPR. The delta_ours half of H4 is recorded
UNTESTABLE_PENDING_DATA, not passed, not failed.

D6 machinery is IDENTICAL to the committed WP-K producer
(scripts/run_p3k_timesplit.py): R = 20 draws from default_rng(42) -- fit on
80 % of the source labelled accounts, within-estimate on the held-out 20 %,
transfer-estimate on the FULL labelled target; delta_r = within_r -
transfer_r; mean +/- SD over r; paired bootstrap over target users
(B = 1000, default_rng(1042), same resamples across families).

Transform variants (R8, bitacora 07 sect. 2 -- open item 7):
  naive : source scaler mu/sigma applied to the SHIPPED z-scored columns
          (strict protocol sect. 4: standardisation fitted on source only)
          -- the charter-faithful arm;
  recal : StandardScaler fitted on the target's labelled columns (label-free
          target marginals) -- the same definition as the WP-B diagnostic,
          kept comparable with it. The corpus-reconstruction route is
          provably a no-op here (the shipped columns ARE corpus-z-scored),
          so it would reproduce naive; recorded rather than run.
  Both variants are reported for both heads. Within a variant HGB is
  insensitive to the affine choice of source-side scaling, but ACROSS
  variants the target's units relative to the source-fitted thresholds
  differ, so naive/recal HGB differences are meaningful (they are the R8
  measurement). HGB primary verdicts do not depend on the choice if the
  verdict agrees across both (preflight cross-check under a different
  design -- CV on target: META 0.7864 naive vs 0.7859 recalibrated,
  results/p6b_tb20_preflight.json).

Usage:
    python scripts/run_p6n_transfer.py [--quiet]
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import DATA_PROCESSED, DATA_RAW, FIGURES, RESULTS
from renyiext.evaluate import tpr_at_fpr

OUT_JSON = RESULTS / "p6n_transfer.json"
R_DRAWS = 20
B_BOOT = 1000
RNG_DRAWS, RNG_BOOT, RNG_NULL = 42, 1042, 77       # distinct streams (§8)
NOISE_SALT_SRC, NOISE_SALT_TGT = 101, 102          # declared sequence, printed
H4_GAP_THRESHOLD = 0.05                             # charter sect. 3, H4
BOT_C, HUM_C, FAM_C = "#e76f51", "#2a9d8f", "#3b6ea5"

# D8 overlap mapping -- VERBATIM from run_p6b_tb20_preflight.py so both
# artefacts describe the same alignment (G2/G3: names, indices, drops).
OVERLAP = [("followers_count", 0, "followers"),
           ("following_count", 3, "friends"),
           ("tweet_count",     4, "statuses"),
           ("account_age_days", 1, "active_days")]
SRC_IDX = [{"followers_count": 1, "following_count": 2,
            "tweet_count": 0, "account_age_days": 4}[f] for f, _, _ in OVERLAP]
DROPPED = {"cresci": ["listed_count (favourites stand-in)"],
           "twibot20": ["screen_name_length"]}
AGE_NOTE = (
    "Cresci age_days is cached against ref 2013-06-06 "
    "(cresci_meta_v1.npz builder); TB20 active_days' reference date is "
    "undocumented in the tensor release. A constant per-feature offset "
    "cannot move HGB splits; for LR it is absorbed by the recal variant's "
    "affine map only approximately. Recorded as a named component of the "
    "META transfer caveat.")

# Target-modality audit measured interactively on 2025-08-25 (torch session;
# commands quoted in bitacora 19 sect. 1). Re-measured live when torch is
# importable; the constants below are what the JSON falls back to otherwise.
AUDIT_MEASURED = {
    "date_measured": "2025-08-25",
    "tweets_tensor_shape": [229580, 768],
    "tweets_tensor_dtype": "float32",
    "tweets_tensor_nonzero_row_fraction": 1.0,
    "tweets_tensor_row_norm_percentiles_0_25_50_75_100":
        [9.29, 10.93, 11.18, 11.40, 13.62],
    "label_pt_shape": [11826],
    "label_pt_counts_human_bot": [5237, 6589],
}
RAW_GATING_QUOTE = ("github.com/bunsenfeng/twibot-20 README: 'Due to privacy "
                    "issues, we are not directly posting the dataset. If you "
                    "are interested in using the dataset, please contact "
                    "shangbin at cs.washington.edu'")


def load_cresci_meta():
    """Verbatim contract with run_p6b_tb20_preflight.load_cresci_meta: reads
    the same cache (data/processed/ext/cresci_meta_v1.npz), which exists, so
    both producers consume byte-identical arrays."""
    cache = DATA_PROCESSED / "cresci_meta_v1.npz"
    z = np.load(cache, allow_pickle=False)
    return z["meta"], z["labels"]


def _fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=42, max_iter=200,
                                          early_stopping=False).fit(X, y)


def _fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000)).fit(X, y)


def d6_family(X_tr, y_tr, X_te, y_te, boot_idx, head="hgb"):
    """Identical machinery to run_p3k_timesplit.d6_family (WP-K), plus an
    optional LR head and the metrics protocol sect. 6 asks to report."""
    fitter = _fit if head == "hgb" else _fit_lr
    n = len(y_tr)
    rng = np.random.default_rng(RNG_DRAWS)
    within, transfer = [], []
    for _ in range(R_DRAWS):
        perm = rng.permutation(n)
        m = int(0.8 * n)
        tr, ho = perm[:m], perm[m:]
        clf = fitter(X_tr[tr], y_tr[tr])
        s_ho = clf.predict_proba(X_tr[ho])[:, 1]
        s_te = clf.predict_proba(X_te)[:, 1]
        within.append(roc_auc_score(y_tr[ho], s_ho))
        transfer.append(roc_auc_score(y_te, s_te))
    full = fitter(X_tr, y_tr)
    scores = full.predict_proba(X_te)[:, 1]
    boots = np.array([roc_auc_score(y_te[bi], scores[bi]) for bi in boot_idx])
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    d = np.asarray(within) - np.asarray(transfer)
    pred = (scores > 0.5).astype(int)
    return {
        "dim": int(np.asarray(X_tr).shape[1]),
        "within_mean": float(np.mean(within)),
        "within_sd": float(np.std(within)),
        "transfer_mean": float(np.mean(transfer)),
        "transfer_sd": float(np.std(transfer)),
        "delta_mean": float(d.mean()), "delta_sd": float(d.std()),
        "transfer_full_fit_auc": float(roc_auc_score(y_te, scores)),
        "transfer_ci95_boot": ci,
        "transfer_tpr_at_1pct_fpr": tpr_at_fpr(y_te, scores),
        "transfer_macro_f1": float(f1_score(y_te, pred, average="macro")),
        "transfer_accuracy": float(accuracy_score(y_te, pred)),
        "majority_baseline_train": float(max(y_tr.mean(), 1 - y_tr.mean())),
        "majority_baseline_test": float(max(y_te.mean(), 1 - y_te.mean())),
        "within_per_draw": [float(v) for v in within],
        "transfer_per_draw": [float(v) for v in transfer],
    }, scores


def audit_modality() -> dict:
    """The scoping finding, elementwise. Live re-measurement when torch is
    available; else the dated interactive measurement stands (command in
    bitacora 19)."""
    tb_dir = DATA_RAW / "bot" / "twibot-20"
    files = {f.name: f.stat().st_size for f in sorted(tb_dir.glob("*"))}
    measured = dict(AUDIT_MEASURED)
    try:
        import torch  # noqa: F401  -- optional; ext venv does not ship it
        t = torch.load(tb_dir / "tweets_tensor.pt", map_location="cpu",
                       weights_only=False).numpy()
        tn = t.sum(1) != 0
        measured = {
            "date_measured": "live",
            "tweets_tensor_shape": list(t.shape),
            "tweets_tensor_dtype": str(t.dtype),
            "tweets_tensor_nonzero_row_fraction": float(tn.mean()),
            "tweets_tensor_row_norm_percentiles_0_25_50_75_100":
                [float(v) for v in np.percentile(
                    np.linalg.norm(t, axis=1), [0, 25, 50, 75, 100])],
            "label_pt_shape": list(torch.load(tb_dir / "label.pt",
                                              map_location="cpu",
                                              weights_only=False).shape),
            "label_pt_counts_human_bot": [
                int(v) for v in np.unique(torch.load(
                    tb_dir / "label.pt", map_location="cpu",
                    weights_only=False).numpy(), return_counts=True)[1]],
        }
    except ImportError:
        measured["fallback"] = ("torch unavailable in the plan env; the "
                                "dated interactive measurement stands")
    uncomputable = ["COUNT(decoded)", "BURST", "SHAN", "SPEC_T", "SHAPE",
                    "TAIL", "SURV", "TAIL+SURV", "SPEC_B_ALPHA",
                    "SPEC_X_WORD", "SPEC_X_CHAR", "SPEC_X"]
    return {
        "verdict": ("H4's delta_ours side is UNTESTABLE on the available "
                    "TwiBot-20 artefact: every spectral/temporal/textual "
                    "family needs event timestamps, post types or raw text; "
                    "the artefact carries five z-scored profile scalars and "
                    "dense pooled embeddings only."),
        "files_present_bytes": files,
        "expected_files": ["edge_index.pt", "label.pt",
                           "num_properties_tensor.pt", "split_new.json",
                           "tweets_tensor.pt"],
        "measured": measured,
        "raw_dataset_availability": RAW_GATING_QUOTE,
        "twibot22_user_json_checked": (
            "data/raw/bot/twibot-22/user.json (Zenodo open file) holds "
            "profiles only -- no tweets field, no timestamps; inspected "
            "2025-08-25, keys enumerated in bitacora 19"),
        "commensurable_families": ["META_aligned(4)", "VOL_PROFILE(1)"],
        "uncomputable_on_target": uncomputable,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ---- source and target sides -------------------------------------------
    meta_cr, y_cr = load_cresci_meta()
    z = np.load(DATA_PROCESSED / "twibot20_preflight_v1.npz")
    P_tb = z["props_labelled"]                      # labelled rows, z-scored
    y_tb = z["labels"].astype(int)
    names_tb = list(z["prop_names"])

    X_src = meta_cr[:, SRC_IDX]                     # raw-scale four fields
    tgt_cols = [j for _, j, _ in OVERLAP]
    assert [names_tb[j] for j in tgt_cols] == \
        ["followers", "friends", "statuses", "active_days"]
    P_ship = P_tb[:, tgt_cols]

    scaler = StandardScaler().fit(X_src)
    mu_s, sd_s = scaler.mean_, scaler.scale_
    tgt_naive = (P_ship - mu_s) / sd_s
    # Marginal-recalibrated variant, SAME definition as the WP-B preflight
    # diagnostic: StandardScaler fitted on the target's labelled columns
    # (label-free target marginals). Note the corpus-reconstruction route
    # (z*sigma_corpus + mu_corpus, then source stats) is provably a no-op
    # here -- corpus mu ~ 0 and sigma ~ 1 are exactly the verified z-scoring
    # -- so it would reproduce naive to float drift; recorded in the JSON.
    tgt_recal = StandardScaler().fit_transform(P_ship)
    assert np.isfinite(X_src).all() and np.isfinite(tgt_naive).all()

    y_src = y_cr.astype(int)
    n_src, n_tgt = len(y_src), len(y_tb)

    # All heads consume matrices on the SOURCE scale (protocol sect. 4:
    # standardisation fitted on source only). HGB is invariant to the affine
    # map; for LR this makes the pipeline's inner StandardScaler a near no-op
    # fitted on source training folds, so the naive/recal contrast measures
    # the target transform and nothing else.
    X_src_std = scaler.transform(X_src)
    vol_s = X_src_std[:, [2]]                    # tweet_count column

    def pad(base_vol_t):
        rng_s = np.random.default_rng(RNG_DRAWS * 1000 + NOISE_SALT_SRC)
        rng_t = np.random.default_rng(RNG_DRAWS * 1000 + NOISE_SALT_TGT)
        return (np.hstack([vol_s, rng_s.standard_normal((n_src, 3))]),
                np.hstack([base_vol_t, rng_t.standard_normal((n_tgt, 3))]))

    src_pad_naive, tgt_pad_naive = pad(tgt_naive[:, [2]])
    src_pad_recal, tgt_pad_recal = pad(tgt_recal[:, [2]])

    ARMS = {
        f"META_aligned[naive]": (X_src_std, tgt_naive),
        f"META_aligned[recal]": (X_src_std, tgt_recal),
        "VOL_PROFILE[naive]": (vol_s, tgt_naive[:, [2]]),
        "VOL_PROFILE[recal]": (vol_s, tgt_recal[:, [2]]),
        "VOL_PROFILE+NOISE(3)[naive]": (src_pad_naive, tgt_pad_naive),
        "VOL_PROFILE+NOISE(3)[recal]": (src_pad_recal, tgt_pad_recal),
    }

    boot_rng = np.random.default_rng(RNG_BOOT)
    boot_idx = boot_rng.integers(0, n_tgt, size=(B_BOOT, n_tgt))

    fams_out, scores_by_arm = {}, {}
    for name, (Xs, Xt) in ARMS.items():
        for head in ("hgb", "lr"):
            key = f"{name}|{head}"
            out, scores = d6_family(Xs, y_src, Xt, y_tb, boot_idx, head=head)
            fams_out[key] = out
            scores_by_arm[key] = scores
            if not args.quiet:
                print(f"  {key:<38} within {out['within_mean']:.4f} "
                      f"transfer {out['transfer_mean']:.4f} "
                      f"(CI {out['transfer_ci95_boot'][0]:.4f}-"
                      f"{out['transfer_ci95_boot'][1]:.4f})  "
                      f"Δ {out['delta_mean']:+.4f}", flush=True)

    # ---- R8 table: the alignment control, both heads ------------------------
    r8 = {}
    for head in ("hgb", "lr"):
        nk = f"META_aligned[naive]|{head}"
        rk = f"META_aligned[recal]|{head}"
        gap = abs(fams_out[nk]["delta_mean"] - fams_out[rk]["delta_mean"])
        r8[head] = {
            "meta_naive_transfer_auc": fams_out[nk]["transfer_mean"],
            "meta_recal_transfer_auc": fams_out[rk]["transfer_mean"],
            "delta_gap_abs": float(gap),
            "verdict": ("effect present under BOTH variants -- not a "
                        "transform artefact" if
                        (fams_out[nk]["delta_mean"] > 0.02 and
                         fams_out[rk]["delta_mean"] > 0.02)
                        else "variant-dependent -- treat as suspect"),
            "artefact_rule": ("if the effect exists only under one variant, "
                              "it is an artefact of that variant's "
                              "transform (protocol sect. 4 / plan WP-N "
                              "task 3)"),
            "preflight_crosscheck": {
                "note": ("WP-B diagnostic, 5-fold CV on target, not the "
                         "same estimator -- cited for consistency only"),
                "naive_source_scaler": 0.7864,
                "marginal_recalibrated": 0.7859,
                "source": "results/p6b_tb20_preflight.json"},
        }
    r8["post_transform_col_sds"] = {
        "naive": [float(v) for v in tgt_naive.std(0)],
        "recal": [float(v) for v in tgt_recal.std(0)],
    }

    # ---- H4 block: the half that can run ------------------------------------
    d_meta = {h: fams_out[f"META_aligned[naive]|{h}"]["delta_mean"]
              for h in ("hgb", "lr")}
    ci_meta = fams_out[f"META_aligned[naive]|hgb"]["transfer_ci95_boot"]
    h4 = {
        "framing": "H4 as chartered (WP-B amendment did not fire; bitacora 07)",
        "gap_rule": f"delta_METADATA - delta_OURS > {H4_GAP_THRESHOLD} "
                    "over >= 10 seeds (charter sect. 3)",
        "evaluable": False,
        "why_not": ("no OURS family is computable on the available target "
                    "(modality audit); the gap's second term does not exist"),
        "metadata_side_measured": {
            "delta_META_hgb": d_meta["hgb"], "delta_META_lr": d_meta["lr"],
            "transfer_ci95_boot_hgb": ci_meta,
            "reading": ("whether metadata degrades on transfer -- one "
                        "inequality's side, not the claim")},
        "preview_pairs": ("none computable -- recorded rather than omitted; "
                          "WP-K's within-Cresci axis remains the only "
                          "preview evidence (max SPEC gap +0.0385)"),
        "status": "UNTESTABLE_PENDING_DATA",
        "what_would_complete_it": [
            "obtain raw TwiBot-20 from the authors (gated; contact in the "
            "modality audit) -- timestamps unlock SPEC_T/SHAPE/BURST/SHAN/"
            "TAIL/SURV/COUNT, text unlocks SPEC_X, post types unlock "
            "SPEC_B_ALPHA",
            "or a pre-registered amendment naming a different target corpus "
            "on which these modalities exist",
        ],
    }

    # ---- G4 sanity null: pseudo-transfer within Cresci ----------------------
    # Plan says "split by account id hash"; the meta cache carries no ids, so
    # the WP-K null machinery (exchangeable random halves, default_rng(77))
    # is used instead -- the conservative-equivalent reading, recorded in
    # bitacora 19 sect. 4.
    null_rng = np.random.default_rng(RNG_NULL)
    perm = null_rng.permutation(n_src)
    half = n_src // 2
    pa, pb = perm[:half], perm[half:]
    boot_idx_null = np.random.default_rng(RNG_BOOT).integers(
        0, len(pb), size=(B_BOOT, len(pb)))
    null_out = {}
    for head in ("hgb", "lr"):
        out, _ = d6_family(X_src_std[pa], y_src[pa], X_src_std[pb], y_src[pb],
                           boot_idx_null, head=head)
        null_out[head] = {"delta_mean": out["delta_mean"],
                          "within_mean": out["within_mean"],
                          "transfer_mean": out["transfer_mean"]}
    g4 = {
        "design": ("random halves of the SAME population (Cresci labelled), "
                   "identical D6 machinery; plan said 'account id hash', the "
                   "meta cache has no ids, so WP-K's permutation halves are "
                   "used -- deviation recorded"),
        "delta_mean_per_head": {h: null_out[h]["delta_mean"]
                                for h in ("hgb", "lr")},
        "max_abs_delta": float(max(abs(null_out[h]["delta_mean"])
                                   for h in ("hgb", "lr"))),
        "expected_silent_because":
            "exchangeable halves have no covariate shift; delta measures "
            "draw noise only, calibrating the real deltas above",
    }
    if not args.quiet:
        print(f"[G4] pseudo-transfer null max |Δ| {g4['max_abs_delta']:.4f}")

    # ---- composition / era-shift caveats (quantified, not waved at) --------
    composition = {
        "source": {"n_labelled": n_src, "n_bot": int(y_src.sum()),
                   "balance_bot": float(y_src.mean()),
                   "majority_baseline": float(max(y_src.mean(),
                                                  1 - y_src.mean()))},
        "target": {"n_labelled": n_tgt, "n_bot": int(y_tb.sum()),
                   "balance_bot": float(y_tb.mean()),
                   "majority_baseline": float(max(y_tb.mean(),
                                                  1 - y_tb.mean()))},
        "balance_shift_pp": float((y_tb.mean() - y_src.mean()) * 100),
        "era_statement": (
            "a 2011-13 fake-follower/social-spam population and a 2020 "
            "diverse-domain population are different bot populations; the "
            "claim this artefact can ever test is feature-family robustness "
            "under covariate shift, not bot identity (charter scope note)"),
        "label_source_heterogeneity": (
            "unrecoverable: the tensor release carries a single binary "
            "label column; TwiBot-20's per-source bot categories ship only "
            "in the gated raw release"),
        "age_reference_note": AGE_NOTE,
    }

    # ---- figures -------------------------------------------------------------
    show = [k for k in fams_out if k.endswith("|hgb")]
    order = sorted(show, key=lambda k: fams_out[k]["delta_mean"])
    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(order) + 2.4))
    ys = np.arange(len(order))
    vals = [fams_out[k]["delta_mean"] for k in order]
    errs = [fams_out[k]["delta_sd"] for k in order]
    cols = [FAM_C if "NOISE" in k else BOT_C if "META" in k else "#8a8a86"
            for k in order]
    ax.barh(ys, vals, xerr=errs, color=cols, alpha=0.85)
    for yi, k in zip(ys, order):
        w = fams_out[k]["transfer_ci95_boot"]
        ax.plot([w[0] - fams_out[k]["within_mean"],
                 w[1] - fams_out[k]["within_mean"]], [yi, yi],
                color="black", lw=1.4)
    ax.axvline(0.0, color="grey", lw=0.9)
    ax.axvline(g4["max_abs_delta"], color="#8a8a86", ls=":", lw=1.2)
    ax.axvline(-g4["max_abs_delta"], color="#8a8a86", ls=":", lw=1.2,
               label=f"G4 null band ±{g4['max_abs_delta']:.4f}")
    ax.set_yticks(ys)
    ax.set_yticklabels([k.replace("[naive]", "").replace("[recal]", " [r]")
                        for k in order], fontsize=8.5)
    ax.set_xlabel("degradation Δ = within − transfer (Cresci → TwiBot-20)")
    ax.set_title("WP-N — Protocol C, scoped to the commensurable families\n"
                 "[H4 UNTESTABLE_PENDING_DATA — delta_ours has no target-side "
                 "family]", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "p6n_degradation.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for a, head in zip(axes, ("hgb", "lr")):
        xs = np.arange(2)
        nv = [fams_out[f"META_aligned[naive]|{head}"]["transfer_mean"],
              fams_out[f"VOL_PROFILE[naive]|{head}"]["transfer_mean"]]
        rv = [fams_out[f"META_aligned[recal]|{head}"]["transfer_mean"],
              fams_out[f"VOL_PROFILE[recal]|{head}"]["transfer_mean"]]
        a.bar(xs - 0.2, nv, 0.4, color=BOT_C, label="naive (strict D8)")
        a.bar(xs + 0.2, rv, 0.4, color=FAM_C, label="marginal-recalibrated")
        a.set_xticks(xs)
        a.set_xticklabels(["META_aligned(4)", "VOL_PROFILE(1)"], fontsize=9)
        a.set_ylim(0.5, 1.0)
        a.set_title(f"{head.upper()} transfer AUC"
                    + (" (scale-invariant head)" if head == "hgb" else ""),
                    loc="left", fontsize=10)
        a.legend(fontsize=8)
    axes[0].set_ylabel("AUC on TwiBot-20 labelled")
    fig.suptitle("R8 alignment control — with (recal) and without (naive) "
                 "marginal recalibration", y=1.02, fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "p6n_alignment.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for a, key, c in ((axes[0], "META_aligned[naive]|hgb", BOT_C),
                      (axes[1], "VOL_PROFILE[naive]|hgb", "#8a8a86")):
        s = scores_by_arm[key]
        bins = np.linspace(0, 1, 41)
        for m, nm, cc in ((y_tb == 0, "human", HUM_C), (y_tb == 1, "bot", c)):
            a.hist(s[m], bins=bins, histtype="step", lw=2, density=True,
                   color=cc, label=nm)
        a.set_title(f"transferred scores — {key.split('|')[0]} "
                    f"(AUC {roc_auc_score(y_tb, s):.4f})", loc="left",
                    fontsize=9.5)
        a.set_xlabel("predict_proba"); a.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("G1 — the objects behind the bars: full-fit transferred "
                 "score distributions", y=1.03, fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "p6n_score_distributions.png", dpi=130)
    plt.close(fig)

    # ---- report ---------------------------------------------------------------
    census = {
        "d6_runs": len(fams_out),
        "comparisons": ("8 family-variant x head D6 runs; 2 R8 pairings; "
                        "2 null heads; no hypothesis test fired (H4 not "
                        "evaluable); nothing exploratory claimed"),
    }
    report = {
        "phase": "P6-transfer (WP-N)",
        "scoping": audit_modality(),
        "framing": h4,
        "config_echo": {
            "seeds": "D6 draw stream default_rng(42), R=20; HGB/LR "
                     "random_state=42 (WP-K convention)",
            "bootstrap": f"default_rng({RNG_BOOT}), B={B_BOOT}, shared "
                         "resamples across families",
            "g4_null_rng": f"default_rng({RNG_NULL})",
            "noise_salts": {"src": NOISE_SALT_SRC, "tgt": NOISE_SALT_TGT,
                            "formula": "default_rng(RNG_DRAWS*1000 + salt), "
                                       "one fixed draw per side per padded "
                                       "arm (the D6 stream, not the seed, "
                                       "carries the resampling)"},
            "overlap_mapping": [list(o) for o in OVERLAP],
            "dropped_fields": DROPPED,
            "transform_variants": {
                "naive": "(P_shipped - mu_src) / sigma_src -- strict "
                         "protocol sect. 4; the charter-faithful arm",
                "target_marginal_recalibrated":
                    "StandardScaler fitted on the target's labelled columns "
                    "(label-free target marginals; same definition as the "
                    "WP-B diagnostic, so its cross-check stays comparable)",
                "why_not_corpus_reconstruction": (
                    "z*sigma_corpus + mu_corpus with mu~0, sigma~1 (the "
                    "verified z-scoring) then source stats reproduces naive "
                    "to float drift -- measured: corpus_col_mean |.|<0.01, "
                    "corpus_col_sd 1.000-1.002"),
                "invariance_note": (
                    "Within a variant, HGB is insensitive to the choice of "
                    "source-side scaling (affine maps preserve splits). "
                    "Across variants the target sits in different units "
                    "relative to source-fitted thresholds, so naive/recal "
                    "HGB differences are meaningful: they measure the "
                    "marginal mismatch through the model. Single-feature "
                    "LR AUCs coincide across variants because a positive-"
                    "slope affine map preserves the ranking of a monotone "
                    "score; multi-feature LR does not"),
            },
            "classifier": "HGB(max_iter=200, early_stopping=False, rs=42); "
                          "LR = StandardScaler+LogisticRegression(5000) "
                          "(inner scaler fitted on source training folds)",
            "h4_gap_threshold": H4_GAP_THRESHOLD,
            "effect_floor_untouched": 0.02,
        },
        "composition_and_era_caveats": composition,
        "alignment_r8": r8,
        "families": fams_out,
        "sanity_null_g4": g4,
        "multiple_comparisons": census,
    }
    OUT_JSON.write_text(json.dumps(report, indent=1))

    print("\n" + "=" * 88)
    print("WP-N — PROTOCOL C (SCOPED): the metadata side of H4  "
          "[UNTESTABLE_PENDING_DATA]")
    print("=" * 88)
    print(f"{'arm|head':<38}{'within':>9}{'transfer':>10}{'Δ':>9}"
          f"{'CI95 transfer':>18}{'tpr@1%':>8}")
    for k in sorted(fams_out):
        v = fams_out[k]
        ci = v["transfer_ci95_boot"]
        print(f"{k:<38}{v['within_mean']:>9.4f}{v['transfer_mean']:>10.4f}"
              f"{v['delta_mean']:>+9.4f}   [{ci[0]:.4f},{ci[1]:.4f}]"
              f"{v['transfer_tpr_at_1pct_fpr']:>8.4f}")
    print(f"\nG4 pseudo-transfer null max |Δ|: {g4['max_abs_delta']:.4f}")
    print(f"H4 status: {h4['status']} — {h4['why_not']}")
    print(f"json -> {OUT_JSON}")
    print("=" * 88)


if __name__ == "__main__":
    main()
