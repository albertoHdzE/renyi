#!/usr/bin/env python
"""Amendment H4-T (bitacora 23): the TEXT front sits the transfer exam.

Fit Cresci-2015 -> test TwiBot-20 (raw labelled release,
data/raw/bot/twibot-20-raw/, PRELIMINARY Kaggle copy -- provenance and
re-derivation obligation in bitacora 23 sect. 1).

Registered scope: families SPEC_X_WORD(6) / SPEC_X_CHAR(6) / SPEC_X(12)
(WP-I definitions verbatim via renyiext.textfront) vs floors
META_raw(4) / VOL_PROFILE(1) / TOKENS(1) / SHAN slices (+D2 noise pads).
Temporal families and SPEC_B_ALPHA are OUT by benchmark construction
(bitacora 22 sect. 2) -- recorded, not silently dropped. COUNT is replaced,
namedly, by VOL_PROFILE/TOKENS (protocol sect. 4).

Primary verdict: delta_META - delta_SPEC_X > 0.05 (charter threshold),
D6 estimator identical to WP-K/N (R=20 default_rng(42), B=1000
default_rng(1042) shared resamples, G4 null default_rng(77), HGB rs=42).
Window control for the target's <=200-text cap: headline (source FULL
timeline) vs equal-cap (source capped to its 200 most recent texts --
sequences are time-sorted so the cap is exact); sigma_config across the
two configs beside every verdict. Age reference 2020-01-01T00:00Z on BOTH
corpora (retires the bitacora-19 mismatch caveat). R8 transforms: at raw
target scale naive=(raw-mu_src)/sigma_src is the faithful map and
recal=StandardScaler(target marginals) is expected to be moot -- both run
for META_raw so the record shows it (bitacora 22 sect. 3c); spectra and
log-count arms are definition-commensurable and run once.

Population note (implementation of amendment sect. 3's exclusion rule):
the FAMILY-vs-FLOOR table and the primary gap run on the token-positive
common population (paired); META/VOL are additionally reported on the
FULL labelled populations of both sides as context rows. Both gap
readings land in the JSON; sigma_config spans them.

Usage:
    python scripts/run_p6n_transfer_text.py [--quiet]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import DATA_RAW, DATA_PROCESSED, FIGURES, RESULTS
from renyiext.evaluate import tpr_at_fpr, interpret_dim_matched
from renyiext.events import load_events_cached, load_cresci_text_side, \
    load_tb20_text_side
from renyiext.textfront import account_text_features

OUT_JSON = RESULTS / "p6n_transfer_text.json"
CACHE = DATA_PROCESSED / "cresci_events_d9.npz"
AGE_REF_MS = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
R_DRAWS, B_BOOT = 20, 1000
RNG_DRAWS, RNG_BOOT, RNG_NULL = 42, 1042, 77
FLOOR_SALTS = {                      # declared sequence, printed (G3)
    "SHAN_WORD+NOISE(5)": 103, "SHAN_CHAR+NOISE(5)": 104,
    "SHAN_X+NOISE(10)": 105, "META+NOISE(2)": 106, "META+NOISE(8)": 107,
}
H4_GAP = 0.05                        # charter threshold, inherited
CAP_TARGET_TEXTS = 200               # the release's own per-user cap
BOT_C, HUM_C, FAM_C, FLR_C = "#e76f51", "#2a9d8f", "#3b6ea5", "#8a8a86"
PRELIMINARY = True


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fit(X, y, head):
    if head == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            random_state=42, max_iter=200, early_stopping=False).fit(X, y)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000)).fit(X, y)


def d6_family(X_tr, y_tr, X_te, y_te, boot_idx, head="hgb"):
    """Identical machinery to run_p3k_timesplit.d6_family / the WP-N
    producer (R draws, shared-stream partitions, paired bootstrap)."""
    rng = np.random.default_rng(RNG_DRAWS)
    n = len(y_tr)
    within, transfer = [], []
    for _ in range(R_DRAWS):
        perm = rng.permutation(n)
        m = int(0.8 * n)
        tr, ho = perm[:m], perm[m:]
        clf = _fit(X_tr[tr], y_tr[tr], head)
        within.append(roc_auc_score(y_tr[ho],
                                    clf.predict_proba(X_tr[ho])[:, 1]))
        transfer.append(roc_auc_score(y_te,
                                      clf.predict_proba(X_te)[:, 1]))
    scores = _fit(X_tr, y_tr, head).predict_proba(X_te)[:, 1]
    boots = np.array([roc_auc_score(y_te[bi], scores[bi]) for bi in boot_idx])
    w = np.asarray(within)
    t = np.asarray(transfer)
    pred = (scores > 0.5).astype(int)
    return {
        "dim": int(np.asarray(X_tr).shape[1]),
        "within_mean": float(w.mean()), "within_sd": float(w.std()),
        "transfer_mean": float(t.mean()), "transfer_sd": float(t.std()),
        "delta_mean": float((w - t).mean()), "delta_sd": float((w - t).std()),
        "delta_per_draw": [float(v) for v in (w - t)],
        "within_per_draw": [float(v) for v in w],
        "transfer_per_draw": [float(v) for v in t],
        "transfer_full_fit_auc": float(roc_auc_score(y_te, scores)),
        "transfer_ci95_boot": [float(np.percentile(boots, 2.5)),
                               float(np.percentile(boots, 97.5))],
        "transfer_tpr_at_1pct_fpr": tpr_at_fpr(y_te, scores),
        "transfer_macro_f1": float(f1_score(y_te, pred, average="macro")),
        "transfer_accuracy": float(accuracy_score(y_te, pred)),
        "majority_baseline_train": float(max(y_tr.mean(), 1 - y_tr.mean())),
        "majority_baseline_test": float(max(y_te.mean(), 1 - y_te.mean())),
    }, scores


def spectra_matrix(text_lists):
    n = len(text_lists)
    sw = np.zeros((n, 6))
    sc = np.zeros((n, 6))
    nt = np.zeros(n, dtype=np.int64)
    for j, texts in enumerate(text_lists):
        sw[j], sc[j], nt[j] = account_text_features(texts)
    return sw, sc, nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    t00 = time.time()

    # ---------------- provenance fingerprints --------------------------------
    raw_dir = DATA_RAW / "bot" / "twibot-20-raw"
    hashes = {f.name: sha256_of(f) for f in sorted(raw_dir.glob("*.json"))}

    # ---------------- SOURCE side --------------------------------------------
    ev = load_events_cached(CACHE)
    counts = ev.counts()
    kept = counts >= 5
    idx = np.where(kept)[0]
    y_cr_full = (ev.labels[idx] == 1).astype(int)
    meta_cr_full, _, seq = load_cresci_text_side(ev, kept, AGE_REF_MS)
    if not args.quiet:
        print(f"[source] kept {len(idx)} | bot share "
              f"{y_cr_full.mean():.4f}", flush=True)

    print("[source] spectra, full timeline ...", flush=True)
    t0 = time.time()
    texts_cr_full = [[t for _, t in s] for s in seq]
    sw_cr, sc_cr, nt_cr = spectra_matrix(texts_cr_full)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    print("[source] spectra, equal-cap (200 most recent texts) ...",
          flush=True)
    texts_cr_cap = [s[-CAP_TARGET_TEXTS:] for s in texts_cr_full]
    sw_cr_cap, sc_cr_cap, nt_cr_cap = spectra_matrix(texts_cr_cap)
    del seq

    # ---------------- TARGET side ---------------------------------------------
    print("[target] raw release -> features ...", flush=True)
    t0 = time.time()
    meta_tb_full, y_tb_full, texts_tb_full = load_tb20_text_side(AGE_REF_MS)
    sw_tb, sc_tb, nt_tb = spectra_matrix(texts_tb_full)
    print(f"  {len(y_tb_full)} users in {time.time()-t0:.0f}s", flush=True)

    # ---------------- exclusions / populations --------------------------------
    pos_cr = nt_cr > 0
    pos_tb = nt_tb > 0
    excl = {
        "source_zero_token": {"bot": int(((nt_cr == 0) & (y_cr_full == 1)).sum()),
                              "human": int(((nt_cr == 0) & (y_cr_full == 0)).sum())},
        "target_missing_or_empty_tweet_field": {
            "bot": int(sum(1 for j in range(len(y_tb_full))
                           if not texts_tb_full[j] and y_tb_full[j] == 1)),
            "human": int(sum(1 for j in range(len(y_tb_full))
                             if not texts_tb_full[j] and y_tb_full[j] == 0))},
        "target_zero_token": {"bot": int(((nt_tb == 0) & (y_tb_full == 1)).sum()),
                              "human": int(((nt_tb == 0) & (y_tb_full == 0)).sum())},
    }
    src = np.where(pos_cr)[0]
    tgt = np.where(pos_tb)[0]
    y_src, y_tgt = y_cr_full[src], y_tb_full[tgt]

    def pop_stats(y):
        return {"n": int(len(y)), "n_bot": int(y.sum()),
                "balance_bot": float(y.mean()),
                "majority_baseline": float(max(y.mean(), 1 - y.mean()))}

    composition = {
        "population_note": "family-vs-floor table and primary gap run on "
                           "the token-positive population (paired); "
                           "META/VOL additionally reported on the FULL "
                           "populations as context (amendment sect. 3)",
        "token_positive_source": pop_stats(y_src),
        "token_positive_target": pop_stats(y_tgt),
        "full_source": pop_stats(y_cr_full),
        "full_target": pop_stats(y_tb_full),
        "balance_shift_pp": float((y_tgt.mean() - y_src.mean()) * 100),
        "age_reference_iso": "2020-01-01T00:00:00+00:00 (both corpora)",
        "era_statement": "2011-13 fake-follower/social-spam bots vs a 2020 "
                         "diverse-domain population: the claim tested is "
                         "feature-family robustness, not bot identity",
        "provenance": {
            "preliminary": PRELIMINARY,
            "origin": "third-party Kaggle re-upload marvinvanbo/twibot-20; "
                      "owner decision recorded in bitacora 23 sect. 1; "
                      "permission e-mail in preparation",
            "sha256": hashes,
            "re_derivation_obligation": "regenerate every number on the "
                                        "authorized copy and compare "
                                        "elementwise; differences are a "
                                        "finding about the copy"},
    }

    # ---------------- matrices -------------------------------------------------
    log_vol_cr = np.log1p(meta_cr_full[:, 2])[:, None]
    log_vol_tb = np.log1p(meta_tb_full[:, 2])[:, None]
    nan_age_cr = int((meta_cr_full[:, 3] != meta_cr_full[:, 3]).sum())
    nan_age_tb = int((meta_tb_full[:, 3] != meta_tb_full[:, 3]).sum())

    M_cr = meta_cr_full[src]
    M_tb = meta_tb_full[tgt]
    scaler_meta = StandardScaler().fit(M_cr)
    tb_naive = (M_tb - scaler_meta.mean_) / scaler_meta.scale_
    tb_recal = StandardScaler().fit_transform(M_tb)

    V_cr = log_vol_cr[src][:, :]
    V_tb = log_vol_tb[tgt]
    T_cr = np.log1p(nt_cr[src])[:, None]
    T_tb = np.log1p(nt_tb[tgt])[:, None]

    def padded(base_src, base_tgt, k, salt):
        rs = np.random.default_rng(RNG_DRAWS * 1000 + salt)
        rt = np.random.default_rng(RNG_DRAWS * 1000 + salt + 1000)
        return (np.hstack([base_src, rs.standard_normal((len(base_src), k))]),
                np.hstack([base_tgt, rt.standard_normal((len(base_tgt), k))]))

    SW_cr, SC_cr = sw_cr[src], sc_cr[src]
    SW_tb, SC_tb = sw_tb[tgt], sc_tb[tgt]
    X_SPEC_X = (np.hstack([SW_cr, SC_cr]), np.hstack([SW_tb, SC_tb]))
    H1W_cr, H1W_tb = SW_cr[:, [2]], SW_tb[:, [2]]
    H1C_cr, H1C_tb = SC_cr[:, [2]], SC_tb[:, [2]]
    HX_cr, HX_tb = np.hstack([H1W_cr, H1C_cr]), np.hstack([H1W_tb, H1C_tb])

    ARMS = {}
    ARMS["META_raw[naive]"] = (scaler_meta.transform(M_cr), tb_naive)
    ARMS["META_raw[recal]"] = (scaler_meta.transform(M_cr), tb_recal)
    ARMS["VOL_PROFILE"] = (V_cr, V_tb)
    ARMS["TOKENS"] = (T_cr, T_tb)
    ARMS["SPEC_X_WORD"] = (SW_cr, SW_tb)
    ARMS["SPEC_X_CHAR"] = (SC_cr, SC_tb)
    ARMS["SPEC_X"] = X_SPEC_X
    ARMS["SHAN_WORD+NOISE(5)"] = padded(H1W_cr, H1W_tb, 5,
                                        FLOOR_SALTS["SHAN_WORD+NOISE(5)"])
    ARMS["SHAN_CHAR+NOISE(5)"] = padded(H1C_cr, H1C_tb, 5,
                                        FLOOR_SALTS["SHAN_CHAR+NOISE(5)"])
    ARMS["SHAN_X+NOISE(10)"] = padded(HX_cr, HX_tb, 10,
                                      FLOOR_SALTS["SHAN_X+NOISE(10)"])
    ARMS["META+NOISE(2)"] = padded(scaler_meta.transform(M_cr), tb_naive, 2,
                                   FLOOR_SALTS["META+NOISE(2)"])
    ARMS["META+NOISE(8)"] = padded(scaler_meta.transform(M_cr), tb_naive, 8,
                                   FLOOR_SALTS["META+NOISE(8)"])

    n_tgt_pos = len(tgt)
    boot_idx = np.random.default_rng(RNG_BOOT).integers(
        0, n_tgt_pos, size=(B_BOOT, n_tgt_pos))

    fams_out, scores_by_arm = {}, {}
    for name, (Xs, Xt) in ARMS.items():
        for head in ("hgb", "lr"):
            key = f"{name}|{head}"
            out, scores = d6_family(Xs, y_src, Xt, y_tgt, boot_idx, head)
            fams_out[key] = out
            scores_by_arm[key] = scores
            if not args.quiet:
                ci = out["transfer_ci95_boot"]
                print(f"  {key:<28} within {out['within_mean']:.4f} "
                      f"transfer {out['transfer_mean']:.4f} "
                      f"(CI {ci[0]:.4f}-{ci[1]:.4f}) "
                      f"d {out['delta_mean']:+.4f}", flush=True)

    # context rows: META/VOL on the FULL populations (unpaired with the
    # table above; different test sizes get their own bootstrap stream)
    boot_full = np.random.default_rng(RNG_BOOT + 1).integers(
        0, len(y_tb_full), size=(B_BOOT, len(y_tb_full)))
    scaler_full = StandardScaler().fit(meta_cr_full[np.isfinite(
        meta_cr_full).all(1)])
    cr_ok = np.where(np.isfinite(meta_cr_full).all(1))[0]
    tb_ok = np.where(np.isfinite(meta_tb_full).all(1))[0]
    context = {}
    for nm, (Xs, Xt) in {
            "META_raw[naive]": (
                scaler_full.transform(meta_cr_full[cr_ok]),
                (meta_tb_full[tb_ok] - scaler_full.mean_)
                / scaler_full.scale_),
            "VOL_PROFILE": (log_vol_cr[cr_ok], log_vol_tb[tb_ok])}.items():
        for head in ("hgb",):
            key = f"{nm}|{head}"
            out, _sc = d6_family(Xs, y_cr_full[cr_ok], Xt,
                                 y_tb_full[tb_ok], boot_full, head)
            context[key] = out
            fams_out[key + " [context:full-pops]"] = out

    # ---------------- primary gap + floor comparisons --------------------------
    def dd(arm_key):
        return fams_out[arm_key]["delta_per_draw"]

    def paired_block(a_key, b_key):
        da, db = dd(a_key), dd(b_key)
        diffs = [x - y_ for x, y_ in zip(db, da)]      # family minus floor
        try:
            p = float(wilcoxon(da, db).pvalue)
        except ValueError:
            p = 1.0
        md = float(np.mean(diffs))
        return {"family": a_key.split("|")[0], "floor": b_key.split("|")[0],
                "head": a_key.split("|")[1],
                "mean_diff": md, "std_diff": float(np.std(diffs)),
                "wins": f"{sum(d > 0 for d in diffs)}/{len(diffs)}", "p": p,
                "verdict": interpret_dim_matched(md, p < 0.05)}

    comparisons = []
    for head in ("hgb", "lr"):
        comparisons.append(paired_block(f"SPEC_X_WORD|{head}",
                                        f"SHAN_WORD+NOISE(5)|{head}"))
        comparisons.append(paired_block(f"SPEC_X_CHAR|{head}",
                                        f"SHAN_CHAR+NOISE(5)|{head}"))
        comparisons.append(paired_block(f"SPEC_X|{head}",
                                        f"SHAN_X+NOISE(10)|{head}"))
        comparisons.append(paired_block(f"SPEC_X_WORD|{head}",
                                        f"META+NOISE(2)|{head}"))
        comparisons.append(paired_block(f"SPEC_X_CHAR|{head}",
                                        f"META+NOISE(2)|{head}"))
        comparisons.append(paired_block(f"SPEC_X|{head}",
                                        f"META+NOISE(8)|{head}"))

    def holm(pvals):
        m = len(pvals)
        order = np.argsort(pvals)
        adj = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            v = (m - rank) * pvals[i]
            running = max(running, v)
            adj[i] = min(1.0, running)
        return adj

    for head in ("hgb", "lr"):
        sel = [c for c in comparisons if c["head"] == head]
        adj = holm([c["p"] for c in sel])
        for c, a in zip(sel, adj):
            c["p_holm_adjusted"] = float(a)
            c["significant_holm_0.05"] = bool(a < 0.05)

    d_meta = dd("META_raw[naive]|hgb")
    gaps = {}
    for fam in ("SPEC_X_WORD", "SPEC_X_CHAR", "SPEC_X"):
        dfam = dd(f"{fam}|hgb")
        g = [m - f for m, f in zip(d_meta, dfam)]
        gaps[fam] = {
            "gap_mean": float(np.mean(g)),
            "draws_gap_gt_threshold": f"{sum(x > H4_GAP for x in g)}/20",
            "note": "per-draw paired; threshold charter 0.05"}

    # ---------------- G4 sanity null -------------------------------------------
    null_rng = np.random.default_rng(RNG_NULL)
    perm = null_rng.permutation(len(y_src))
    half = len(y_src) // 2
    pa, pb = perm[:half], perm[half:]
    boot_null = np.random.default_rng(RNG_BOOT).integers(
        0, len(pb), size=(B_BOOT, len(pb)))
    null_out = {}
    for nm in ("META_raw[naive]", "VOL_PROFILE", "TOKENS",
               "SPEC_X_WORD", "SPEC_X_CHAR", "SPEC_X"):
        Xs, Xt = ARMS[nm]
        for head in ("hgb", "lr"):
            out, _sc = d6_family(Xs[pa], y_src[pa], Xs[pb], y_src[pb],
                                 boot_null, head)
            null_out[f"{nm}|{head}"] = float(out["delta_mean"])
    g4 = {"design": "exchangeable halves of the token-positive source "
                    "population, identical D6 machinery",
          "delta_mean": null_out,
          "max_abs_delta": float(max(abs(v) for v in null_out.values())),
          "expected_silent_because": "no covariate shift between halves; "
                                     "delta measures draw noise only"}

    # ---------------- figures ---------------------------------------------------
    show = [k for k in fams_out if k.endswith("|hgb")
            and "[context" not in k]
    order = sorted(show, key=lambda k: fams_out[k]["delta_mean"])
    fig, ax = plt.subplots(figsize=(10.8, 0.42 * len(order) + 2.6))
    ys = np.arange(len(order))
    vals = [fams_out[k]["delta_mean"] for k in order]
    errs = [fams_out[k]["delta_sd"] for k in order]
    cols = [FAM_C if "SPEC_X" in k else BOT_C if "META" in k else FLR_C
            for k in order]
    ax.barh(ys, vals, xerr=errs, color=cols, alpha=0.85)
    for yi, k in zip(ys, order):
        w = fams_out[k]["transfer_ci95_boot"]
        wm = fams_out[k]["within_mean"]
        ax.plot([wm - w[1], wm - w[0]], [yi, yi], color="black", lw=1.4)
    ax.axvline(0.0, color="grey", lw=0.9)
    ax.axvline(g4["max_abs_delta"], color="grey", ls=":", lw=1.2)
    ax.axvline(-g4["max_abs_delta"], color="grey", ls=":", lw=1.2,
               label=f"G4 null band ±{g4['max_abs_delta']:.4f}")
    ax.set_yticks(ys); ax.set_yticklabels(order, fontsize=8.5)
    ax.set_xlabel("degradation Δ = within − transfer (Cresci → TwiBot-20)")
    ax.set_title("H4-T (preliminary) — degradation per arm, hgb\n"
                 "text spectra vs metadata floors under D6",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "p6nt_degradation.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for a, (nts, yy, nm) in zip(axes, (
            (nt_cr, y_cr_full, "Cresci-2015 (kept, full timeline)"),
            (nt_tb, y_tb_full, "TwiBot-20 (release cap 200 texts)"))):
        bins = np.arange(0, 2100, 25)
        for m, lab, c in ((yy == 0, "human", HUM_C), (yy == 1, "bot", BOT_C)):
            a.hist(np.clip(nts[m], 0, 2050), bins=bins, histtype="step",
                   lw=2, density=True, color=c, label=lab)
        a.axvline(CAP_TARGET_TEXTS, color="black", ls="--", lw=1.2)
        a.set_title(nm, loc="left", fontsize=10)
        a.set_xlabel("texts per account"); a.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("G1/G2 — the window both sides actually observed "
                 "(dashed = target release cap)", y=1.02, fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "p6nt_length_distributions.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for a, key, c in ((axes[0], "SPEC_X_CHAR|hgb", FAM_C),
                      (axes[1], "META_raw[naive]|hgb", BOT_C)):
        s = scores_by_arm[key]
        bins = np.linspace(0, 1, 41)
        for m, nm2, cc in ((y_tgt == 0, "human", HUM_C),
                           (y_tgt == 1, "bot", c)):
            a.hist(s[m], bins=bins, histtype="step", lw=2, density=True,
                   color=cc, label=nm2)
        a.set_title(f"transferred scores — {key.split('|')[0]} "
                    f"(AUC {roc_auc_score(y_tgt, s):.4f})", loc="left",
                    fontsize=9.5)
        a.set_xlabel("predict_proba"); a.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.text(0.5, 0.955,
             "G1 — transferred score distributions, token-positive target "
             "(PRELIMINARY: Kaggle copy)", ha="center", fontsize=10.5,
             fontweight="bold")
    fig.savefig(FIGURES / "p6nt_score_distributions.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # ---------------- report -----------------------------------------------------
    report = {
        "phase": "P6-transfer-text (amendment H4-T, bitacora 23)",
        "preliminary": PRELIMINARY,
        "provenance_and_hashes": composition.pop("provenance"),
        "composition": composition,
        "config_echo": {
            "estimator": "D6 identical to WP-K/N: R=20 default_rng(42), "
                         f"B={B_BOOT} default_rng({RNG_BOOT}) shared "
                         "resamples, HGB rs=42, LR secondary",
            "g4_null_rng": RNG_NULL,
            "floor_salts": FLOOR_SALTS,
            "feature_defs": "renyiext.textfront.account_text_features "
                            "(WP-I, \\w+ Unicode, no lowercasing, char "
                            "counts summed per tweet; orders "
                            "{0,.5,1,2,4,inf}, base 2)",
            "cap_texts_target": CAP_TARGET_TEXTS,
            "equal_cap_config": "source capped to its 200 most recent "
                                "texts (time-sorted sequences); see "
                                "equal_cap block below",
            "age_reference_both_corpora": "2020-01-01T00:00Z",
            "nan_age_exclusions": {"source": nan_age_cr,
                                   "target": nan_age_tb},
            "r8_note": "raw-scale target makes naive the faithful map and "
                       "recal expectedly moot; both still run for "
                       "META_raw (bitacora 22 sect. 3c)",
            "spectra_variant_note": "spectra/log-count arms are "
                                    "definition-commensurable and run once",
            "h4_gap_threshold": H4_GAP,
        },
        "equal_cap_block": {},
        "families": fams_out,
        "comparisons": comparisons,
        "primary_gaps_hgb_headline": gaps,
        "sanity_null_g4": g4,
        "multiple_comparisons": {
            "census": "12 D6 runs (headline) + 4 context + equal-cap "
                      "block + 6 null runs; Holm family = the 6 "
                      "SPEC_X-vs-floor comparisons per head",
            "holm_applied_to": "comparisons[] with p_holm_adjusted",
        },
    }

    # equal-cap config: rerun families + META + Shannon floors on capped
    # source spectra (registered window control)
    sw_c, sc_c = sw_cr_cap[src], sc_cr_cap[src]
    nt_c = nt_cr_cap[src]
    eq = {}
    cap_arms = {
        "SPEC_X_WORD": (sw_c, SW_tb),
        "SPEC_X_CHAR": (sc_c, SC_tb),
        "SPEC_X": (np.hstack([sw_c, sc_c]),
                   np.hstack([SW_tb, SC_tb])),
        "META_raw[naive]": (scaler_meta.transform(M_cr), tb_naive),
        "SHAN_X+NOISE(10)": None,
    }
    hx_c = np.hstack([sw_c[:, [2]], sc_c[:, [2]]])
    cap_arms["SHAN_X+NOISE(10)"] = padded(hx_c, HX_tb, 10,
                                          FLOOR_SALTS["SHAN_X+NOISE(10)"])
    for nm, (Xs, Xt) in cap_arms.items():
        for head in ("hgb", "lr"):
            out, _sc = d6_family(Xs, y_src, Xt, y_tgt, boot_idx, head)
            eq[f"{nm}|{head}"] = {"delta_mean": out["delta_mean"],
                                  "delta_sd": out["delta_sd"]}
    sig = {}
    for nm in ("SPEC_X_WORD", "SPEC_X_CHAR", "SPEC_X", "META_raw[naive]"):
        full = fams_out[f"{nm}|hgb"]["delta_mean"]
        capped = eq[f"{nm}|hgb"]["delta_mean"]
        sig[nm] = {"delta_full_timeline": full, "delta_equal_cap": capped,
                   "abs_shift": abs(full - capped)}
    report["equal_cap_block"] = {
        "runs": eq,
        "sigma_config_across_configs_hgb": sig,
        "note": "sigma_config (D3): how much each verdict depends on the "
                "observed-window config; reported beside the primary gap"}

    OUT_JSON.write_text(json.dumps(report, indent=1))

    print("\n" + "=" * 88)
    print("H4-T — TEXT FRONT SITS THE TRANSFER EXAM  [PRELIMINARY]")
    print("=" * 88)
    print(f"{'arm|head':<28}{'within':>9}{'transfer':>10}"
          f"{'delta':>9}{'CI95':>18}{'acc':>8}{'maj':>7}")
    for k in sorted(fams_out):
        v = fams_out[k]
        ci = v["transfer_ci95_boot"]
        print(f"{k:<28}{v['within_mean']:>9.4f}{v['transfer_mean']:>10.4f}"
              f"{v['delta_mean']:>+9.4f}   [{ci[0]:.4f},{ci[1]:.4f}]"
              f"{v['transfer_accuracy']:>8.4f}"
              f"{v['majority_baseline_test']:>7.4f}")
    print(f"\nprimary gaps vs META (hgb, paired): "
          f"{ {k: round(v['gap_mean'], 4) for k, v in gaps.items()} }")
    print(f"G4 null max |d|: {g4['max_abs_delta']:.4f}")
    print(f"json -> {OUT_JSON}")
    print("=" * 88)


if __name__ == "__main__":
    main()
