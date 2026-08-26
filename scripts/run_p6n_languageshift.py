#!/usr/bin/env python
"""Amendment LS-1 (bitacora 24): is the char-spectrum inversion a
corpus-language effect?

F16's named suspect, adjudicated exactly as pre-registered BEFORE any
language computation (registration = bitacora 24 sects. 1-7, committed
first). H4-T's failure stands regardless of this outcome; PRELIMINARY
provenance (Kaggle copy) and the re-derivation obligation carry over.

Arms (registration sect. 5):
  A  census: langid label distribution per corpus x class on the
     token-positive population + per-account EN-tweet share histograms
     (every tweet, first 200 chars -- measured cost made a subsample
     knob unnecessary, so none was invented).
  B  PRIMARY: SPEC_X_CHAR transfer re-run (D6 byte-identical to WP-K/N/
     H4-T) on the full token-positive population [reference] vs the
     EN-intersect-EN restricted population; context arms META_raw[naive],
     SHAN_CHAR+NOISE(5) (salt 104), SPEC_X_WORD on both populations.
     Sensitivities on the primary arm only: equal-cap source config and
     the 10k account-concat budget.
  B2 directionality placebo: EN-restricted SOURCE -> FULL target.
  C  local restoration (exploratory): within-TB20 stratified 5-fold AUC of
     char spectra restricted vs full (renyiext.evaluate, seeds 42..51).
  D  G4 permutation-half null through BOTH pipelines.

Verdict rules are sect. 6 verbatim: bar 0.05 (charter threshold
inherited), >=16/20 draws, repair conjunct (restricted transfer_mean >
0.5 AND its CI95 low > 0.5); P1 bars 0.20 share points; combined
vocabulary SUPPORTED / PARTIAL-MECHANISM / PARTIAL-EFFECT / REFUTED.
Either outcome lands in FINDINGS F17 with the weight of its label.

Fidelity gates (datasaurus G2: compare objects elementwise): the eight
full-reference D6 arms must reproduce the committed p6n_transfer_text.json
rows (max |diff| < 1e-12) and the full-pipeline G4 null must reproduce
its committed values for the rerun arms. A breach ABORTS the run --
drifted machinery would make delta_inv meaningless.

Usage:
    python scripts/run_p6n_languageshift.py [--quiet]
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import DATA_RAW, DATA_PROCESSED, FIGURES, RESULTS
from renyiext.evaluate import paired as seed_paired, run_arms
from renyiext.events import load_events_cached, load_cresci_text_side, \
    load_tb20_text_side
from renyiext.lid import assert_deterministic, classify_account, \
    tool_echo, tweet_en_stats
from renyiext.textfront import account_text_features

OUT_JSON = RESULTS / "p6n_ls1.json"
H4T_JSON = RESULTS / "p6n_transfer_text.json"
CACHE = DATA_PROCESSED / "cresci_events_d9.npz"
AGE_REF_MS = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
R_DRAWS, B_BOOT = 20, 1000
RNG_DRAWS, RNG_BOOT, RNG_NULL = 42, 1042, 77
SALT_SHAN_CHAR = 104                  # the arm's declared salt (bitacora 23)
INV_BAR = 0.05                        # charter threshold, inherited
P1_BAR = 0.20                         # share points
REPAIR_DRAWS_MIN = 16                 # of 20
CAP_TARGET_TEXTS = 200                # release cap (equal-cap config)
BUDGET_ALT = 10_000                   # registered budget sensitivity
SEEDS_LOCAL = list(range(42, 52))     # producer seed convention
FID_ATOL = 1e-12
PRELIMINARY = True
BOT_C, HUM_C, FAM_C = "#e76f51", "#2a9d8f", "#3b6ea5"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fit(X, y, head):
    if head == "hgb":
        return HistGradientBoostingClassifier(
            random_state=42, max_iter=200, early_stopping=False).fit(X, y)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000)).fit(X, y)


def d6_family(X_tr, y_tr, X_te, y_te, boot_idx, head="hgb"):
    """Byte-identical D6 machinery to the WP-K/N/H4-T producers."""
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
    return {
        "dim": int(np.asarray(X_tr).shape[1]),
        "within_mean": float(w.mean()), "within_sd": float(w.std()),
        "transfer_mean": float(t.mean()), "transfer_sd": float(t.std()),
        "delta_mean": float((w - t).mean()), "delta_sd": float((w - t).std()),
        "delta_per_draw": [float(v) for v in (w - t)],
        "transfer_per_draw": [float(v) for v in t],
        "transfer_full_fit_auc": float(roc_auc_score(y_te, scores)),
        "transfer_ci95_boot": [float(np.percentile(boots, 2.5)),
                               float(np.percentile(boots, 97.5))],
    }


def spectra_matrix(text_lists):
    n = len(text_lists)
    sw = np.zeros((n, 6))
    sc = np.zeros((n, 6))
    nt = np.zeros(n, dtype=np.int64)
    for j, texts in enumerate(text_lists):
        sw[j], sc[j], nt[j] = account_text_features(texts)
    return sw, sc, nt


def padded(base_src, base_tgt, k, salt):
    rs = np.random.default_rng(RNG_DRAWS * 1000 + salt)
    rt = np.random.default_rng(RNG_DRAWS * 1000 + salt + 1000)
    return (np.hstack([base_src, rs.standard_normal((len(base_src), k))]),
            np.hstack([base_tgt, rt.standard_normal((len(base_tgt), k))]))


def pop_stats(y):
    return {"n": int(len(y)), "n_bot": int(np.asarray(y).sum()),
            "balance_bot": float(np.mean(y)),
            "majority_baseline": float(max(np.mean(y), 1 - np.mean(y)))}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    assert_deterministic()
    lid_echo = tool_echo()

    # ---------------- provenance fingerprints --------------------------------
    raw_dir = DATA_RAW / "bot" / "twibot-20-raw"
    hashes = {f.name: sha256_of(f) for f in sorted(raw_dir.glob("*.json"))}

    # ---------------- sides (identical reads to H4-T) -------------------------
    ev = load_events_cached(CACHE)
    counts = ev.counts()
    kept = counts >= 5
    idx = np.where(kept)[0]
    y_cr_full = (ev.labels[idx] == 1).astype(int)
    meta_cr_full, _, seq = load_cresci_text_side(ev, kept, AGE_REF_MS)
    texts_cr_full = [[t for _, t in s] for s in seq]
    del seq
    if not args.quiet:
        print(f"[source] kept {len(idx)} | bot share "
              f"{y_cr_full.mean():.4f}", flush=True)

    sw_cr, sc_cr, nt_cr = spectra_matrix(texts_cr_full)

    meta_tb_full, y_tb_full, texts_tb_full = load_tb20_text_side(AGE_REF_MS)
    sw_tb, sc_tb, nt_tb = spectra_matrix(texts_tb_full)

    pos_cr = nt_cr > 0
    pos_tb = nt_tb > 0
    src = np.where(pos_cr)[0]
    tgt = np.where(pos_tb)[0]
    y_src, y_tgt = y_cr_full[src], y_tb_full[tgt]

    # ---------------- LID pass, account level ---------------------------------
    print("[lid] account-level classification ...", flush=True)
    t0 = time.time()
    src_lang_conf = [classify_account(texts_cr_full[j]) for j in src]
    tgt_lang_conf = [classify_account(texts_tb_full[j]) for j in tgt]
    en_src = np.array([l == "en" for l, _ in src_lang_conf])
    en_tgt = np.array([l == "en" for l, _ in tgt_lang_conf])
    if not args.quiet:
        print(f"  done in {time.time()-t0:.0f}s | EN share src "
              f"{en_src.mean():.4f} tgt {en_tgt.mean():.4f}", flush=True)

    # budget-sensitivity pass (10k chars), registered sect. 3 sweep
    langs10_src = [classify_account(texts_cr_full[j], BUDGET_ALT)
                   for j in src]
    langs10_tgt = [classify_account(texts_tb_full[j], BUDGET_ALT)
                   for j in tgt]
    en_src10 = np.array([l == "en" for l, _ in langs10_src])
    en_tgt10 = np.array([l == "en" for l, _ in langs10_tgt])

    # ---------------- arm A: census (token-positive population) ---------------
    census_labels = {}
    for corpus, langs, yy in (("cresci15", src_lang_conf, y_src),
                              ("twibot20", tgt_lang_conf, y_tgt)):
        for cls, cname in ((1, "bot"), (0, "human")):
            mask = yy == cls
            census_labels[f"{corpus}|{cname}"] = dict(
                Counter(l for (l, _), m in zip(langs, mask) if m))

    def eshare(key):
        c = census_labels[key]
        return c.get("en", 0) / sum(c.values())

    e_c15_b, e_c15_h = eshare("cresci15|bot"), eshare("cresci15|human")
    e_tb_b, e_tb_h = eshare("twibot20|bot"), eshare("twibot20|human")

    # per-tweet secondary: EVERY tweet's first 200 chars
    print("[lid] per-tweet secondary (all tweets, first 200 chars) ...",
          flush=True)
    tweet_share = {}
    for corpus, pop_idx, yy in (("cresci15", src, y_src),
                                ("twibot20", tgt, y_tgt)):
        texts_pop = texts_cr_full if corpus == "cresci15" else texts_tb_full
        for cls, cname in ((1, "bot"), (0, "human")):
            t0 = time.time()
            shares = []
            for j in pop_idx[yy == cls]:
                st = tweet_en_stats(texts_pop[j])
                if st["n_tweets"]:
                    shares.append(st["en_share"])
            tweet_share[f"{corpus}|{cname}"] = shares
            if not args.quiet:
                print(f"  {corpus}|{cname}: {len(shares)} accounts "
                      f"in {time.time()-t0:.0f}s", flush=True)

    # ---------------- restricted populations -----------------------------------
    rs = src[en_src]
    rt = tgt[en_tgt]
    y_rsrc, y_rtgt = y_src[en_src], y_tgt[en_tgt]
    excl_restricted = {}
    for side, en_mask, yy in (("source", en_src, y_src),
                              ("target", en_tgt, y_tgt)):
        for cls, cname in ((1, "bot"), (0, "human")):
            excl_restricted[f"{side}_non_en_{cname}"] = int(
                (~en_mask & (yy == cls)).sum())

    # ---------------- matrices --------------------------------------------------
    M_cr = meta_cr_full[src]
    M_tb = meta_tb_full[tgt]
    scaler_meta = StandardScaler().fit(M_cr)
    tb_naive = (M_tb - scaler_meta.mean_) / scaler_meta.scale_

    SC_cr_s, SC_tb_t = sc_cr[src], sc_tb[tgt]
    SW_cr_s, SW_tb_t = sw_cr[src], sw_tb[tgt]
    H1C_cr, H1C_tb = SC_cr_s[:, [2]], SC_tb_t[:, [2]]
    SC_cr_r, SC_tb_r = SC_cr_s[en_src], SC_tb_t[en_tgt]
    SW_cr_r, SW_tb_r = SW_cr_s[en_src], SW_tb_t[en_tgt]
    M_cr_r, tb_naive_r = scaler_meta.transform(M_cr)[en_src], tb_naive[en_tgt]
    H1C_cr_r, H1C_tb_r = H1C_cr[en_src], H1C_tb[en_tgt]

    boot_full = np.random.default_rng(RNG_BOOT).integers(
        0, len(y_tgt), size=(B_BOOT, len(y_tgt)))
    boot_rest = np.random.default_rng(RNG_BOOT).integers(
        0, len(y_rtgt), size=(B_BOOT, len(y_rtgt)))

    def build_arms(sc_s, sc_t, sw_s, sw_t, m_s, m_t, h1_s, h1_t):
        return {
            "SPEC_X_CHAR": (sc_s, sc_t),
            "META_raw[naive]": (m_s, m_t),
            "SHAN_CHAR+NOISE(5)": padded(h1_s, h1_t, 5, SALT_SHAN_CHAR),
            "SPEC_X_WORD": (sw_s, sw_t),
        }

    ARMS_FULL = build_arms(SC_cr_s, SC_tb_t, SW_cr_s, SW_tb_t,
                           scaler_meta.transform(M_cr), tb_naive,
                           H1C_cr, H1C_tb)
    ARMS_REST = build_arms(SC_cr_r, SC_tb_r, SW_cr_r, SW_tb_r,
                           M_cr_r, tb_naive_r, H1C_cr_r, H1C_tb_r)

    # ---------------- arm B: D6 runs --------------------------------------------
    fams_full, fams_rest = {}, {}
    for nm, (Xs, Xt) in ARMS_FULL.items():
        for head in ("hgb", "lr"):
            key = f"{nm}|{head}"
            fams_full[key] = d6_family(Xs, y_src, Xt, y_tgt, boot_full, head)
            if not args.quiet:
                print(f"  FULL {key:<26} transfer "
                      f"{fams_full[key]['transfer_mean']:.4f}", flush=True)
    for nm, (Xs, Xt) in ARMS_REST.items():
        for head in ("hgb", "lr"):
            key = f"{nm}|{head}"
            fams_rest[key] = d6_family(Xs, y_rsrc, Xt, y_rtgt,
                                       boot_rest, head)
            if not args.quiet:
                print(f"  REST {key:<26} transfer "
                      f"{fams_rest[key]['transfer_mean']:.4f}", flush=True)

    # ---------------- fidelity gate vs committed H4-T artefact ------------------
    committed = json.loads(H4T_JSON.read_text())
    fid_rows, worst = {}, 0.0
    for key in fams_full:
        ref = committed["families"][key]
        diffs = [abs(fams_full[key][f] - ref[f])
                 for f in ("within_mean", "transfer_mean", "delta_mean",
                           "delta_sd", "transfer_full_fit_auc")]
        diffs += [abs(a - b) for a, b in
                  zip(fams_full[key]["transfer_ci95_boot"],
                      ref["transfer_ci95_boot"])]
        fid_rows[key] = max(diffs)
        worst = max(worst, fid_rows[key])
    if worst >= FID_ATOL:
        print("FIDELITY GATE BREACH vs p6n_transfer_text.json:", fid_rows)
        sys.exit(1)

    # ---------------- primary quantity (sect. 6) ---------------------------------
    def delta_inv(head):
        full = fams_full[f"SPEC_X_CHAR|{head}"]
        rest = fams_rest[f"SPEC_X_CHAR|{head}"]
        per_draw = [f - r for f, r in zip(full["transfer_per_draw"],
                                          rest["transfer_per_draw"])]
        ci_lo = rest["transfer_ci95_boot"][0]
        repair = bool(rest["transfer_mean"] > 0.5 and ci_lo > 0.5)
        draws_gt = sum(x > INV_BAR for x in per_draw)
        passed = bool(float(np.mean(per_draw)) > INV_BAR
                      and draws_gt >= REPAIR_DRAWS_MIN and repair)
        return {
            "delta_inv_mean": float(np.mean(per_draw)),
            "delta_inv_sd": float(np.std(per_draw)),
            "draws_gt_bar": f"{draws_gt}/{R_DRAWS}",
            "bar": INV_BAR,
            "full_transfer_mean": full["transfer_mean"],
            "restricted_transfer_mean": rest["transfer_mean"],
            "repair_conjunct": {
                "restricted_transfer_mean": rest["transfer_mean"],
                "restricted_ci95_low": ci_lo,
                "holds": repair},
            "verdict_primary": ("PASS" if passed
                                else "FAIL_bar_or_draws_or_repair"),
            "pairing_note": "draws are index-paired across two D6 blocks; "
                            "partition streams differ because the "
                            "population sizes differ, so the 20 differences "
                            "are an independent-draws ensemble read through "
                            "its mean and exceedance count",
        }

    inv_hgb = delta_inv("hgb")
    inv_lr = delta_inv("lr")

    # ---------------- P1 composition corroboration -------------------------------
    p1_i = float(e_c15_b - e_c15_h)
    p1_ii = float(min(e_tb_b, e_tb_h) - e_c15_h)
    verdict_p1 = bool(p1_i > P1_BAR and p1_ii > P1_BAR)

    vp = inv_hgb["verdict_primary"] == "PASS"
    combined = (("SUPPORTED" if verdict_p1 else "PARTIAL-MECHANISM") if vp
                else ("PARTIAL-EFFECT" if verdict_p1 else "REFUTED"))

    # ---------------- B2 directionality placebo -----------------------------------
    b2 = {}
    for head in ("hgb", "lr"):
        b2[f"SPEC_X_CHAR|{head}"] = d6_family(
            SC_cr_r, y_rsrc, SC_tb_t, y_tgt, boot_full, head)

    # ---------------- sensitivities (primary arm only) -----------------------------
    rs10, rt10 = src[en_src10], tgt[en_tgt10]
    y_rsrc10, y_rtgt10 = y_src[en_src10], y_tgt[en_tgt10]
    boot_rest10 = np.random.default_rng(RNG_BOOT).integers(
        0, len(y_rtgt10), size=(B_BOOT, len(y_rtgt10)))
    budget_row = {}
    for head in ("hgb", "lr"):
        budget_row[f"SPEC_X_CHAR|{head}"] = d6_family(
            SC_cr_s[en_src10], y_rsrc10, SC_tb_t[en_tgt10], y_rtgt10,
            boot_rest10, head)
    inv10_mean = float(np.mean([
        f - r for f, r in zip(fams_full["SPEC_X_CHAR|hgb"]["transfer_per_draw"],
                              budget_row["SPEC_X_CHAR|hgb"]["transfer_per_draw"])]))

    if not args.quiet:
        print("[spectra] equal-cap source ...", flush=True)
    texts_cr_cap = [s[-CAP_TARGET_TEXTS:] for s in texts_cr_full]
    _, sc_cr_cap, _ = spectra_matrix(texts_cr_cap)
    SC_cap_s = sc_cr_cap[src]
    eqcap = {
        "SPEC_X_CHAR|hgb": d6_family(SC_cap_s, y_src, SC_tb_t, y_tgt,
                                     boot_full, "hgb"),
        "SPEC_X_CHAR_REST|hgb": d6_family(SC_cap_s[en_src], y_rsrc,
                                          SC_tb_t[en_tgt], y_rtgt,
                                          boot_rest, "hgb"),
    }
    inv_cap_mean = float(np.mean([
        f - r for f, r in zip(eqcap["SPEC_X_CHAR|hgb"]["transfer_per_draw"],
                              eqcap["SPEC_X_CHAR_REST|hgb"]["transfer_per_draw"])]))

    # ---------------- arm C: local restoration (exploratory) ------------------------
    local = {}
    for pop, Xt, yt in (("full", SC_tb_t, y_tgt),
                        ("restricted", SC_tb_r, y_rtgt)):
        local[pop] = run_arms({
            "SPEC_X_CHAR": Xt,
            "H1_CHAR": Xt[:, [2]],
            "HINF_CHAR": Xt[:, [5]],
        }, yt, SEEDS_LOCAL, quiet=True)
    c_pairs = {}
    for fam in ("SPEC_X_CHAR", "H1_CHAR", "HINF_CHAR"):
        c_pairs[fam] = seed_paired(local["restricted"][fam]["auc"],
                                   local["full"][fam]["auc"])
        c_pairs[fam]["p_holm_adjusted"] = None
    adj = holm([c_pairs[f]["p"] for f in ("SPEC_X_CHAR", "H1_CHAR",
                                          "HINF_CHAR")])
    for f, a in zip(("SPEC_X_CHAR", "H1_CHAR", "HINF_CHAR"), adj):
        c_pairs[f]["p_holm_adjusted"] = float(a)

    # ---------------- arm D: G4 nulls through both pipelines ------------------------
    def g4_null(mats_by_arm, y_tr):
        null_rng = np.random.default_rng(RNG_NULL)
        perm = null_rng.permutation(len(y_tr))
        half = len(y_tr) // 2
        pa, pb = perm[:half], perm[half:]
        boot_null = np.random.default_rng(RNG_BOOT).integers(
            0, len(pb), size=(B_BOOT, len(pb)))
        out = {}
        for nm in ("META_raw[naive]", "SPEC_X_CHAR"):
            Xs = mats_by_arm[nm]
            for head in ("hgb", "lr"):
                o = d6_family(Xs[pa], y_tr[pa], Xs[pb], y_tr[pb],
                              boot_null, head)
                out[f"{nm}|{head}"] = float(o["delta_mean"])
        return out

    g4_full_pipe = g4_null({k: ARMS_FULL[k][0] for k in
                            ("META_raw[naive]", "SPEC_X_CHAR")}, y_src)
    g4_rest_pipe = g4_null({k: ARMS_REST[k][0] for k in
                            ("META_raw[naive]", "SPEC_X_CHAR")}, y_rsrc)

    null_worst = 0.0
    for key in ("META_raw[naive]|hgb", "META_raw[naive]|lr",
                "SPEC_X_CHAR|hgb", "SPEC_X_CHAR|lr"):
        null_worst = max(null_worst, abs(
            g4_full_pipe[key]
            - committed["sanity_null_g4"]["delta_mean"][key]))
    if null_worst >= FID_ATOL:
        print("NULL FIDELITY GATE BREACH:", g4_full_pipe)
        sys.exit(1)

    # ---------------- figures ---------------------------------------------------------
    LANG_ORDER = sorted({l for c in census_labels.values() for l in c},
                        key=lambda l: (-sum(census_labels[c].get(l, 0)
                                            for c in census_labels), l))
    show = LANG_ORDER[:10]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, corpus, title in (
            (axes[0], "cresci15", "Cresci-2015 (token-positive)"),
            (axes[1], "twibot20", "TwiBot-20 (token-positive) — "
             "PRELIMINARY Kaggle copy")):
        x = np.arange(len(show))
        w = 0.38
        for k, (cls, col) in enumerate((("bot", BOT_C),
                                        ("human", HUM_C))):
            c = census_labels[f"{corpus}|{cls}"]
            n_tot = sum(c.values())
            shares = [c.get(l, 0) / n_tot for l in show]
            ax.bar(x + (k - 0.5) * w, shares, width=w, color=col,
                   alpha=0.88, label=cls)
        ax.set_xticks(x)
        ax.set_xticklabels(show, rotation=45, fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("share of accounts")
    fig.suptitle("LS-1 arm A — the object: language-label distribution at "
                 "account grain (langid forced-choice, 16-lang set)",
                 y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "p6nls_language_census.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins_ts = np.linspace(0, 1, 21)
    for ax, corpus, title in (
            (axes[0], "cresci15", "Cresci-2015"),
            (axes[1], "twibot20", "TwiBot-20 (PRELIMINARY)")):
        for cls, col in (("human", HUM_C), ("bot", BOT_C)):
            arr = np.asarray(tweet_share[f"{corpus}|{cls}"], dtype=float)
            ax.hist(arr[~np.isnan(arr)], bins=bins_ts, histtype="step",
                    lw=2, density=True, color=col, label=cls)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("per-account EN-tweet share")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("LS-1 arm A secondary — per-tweet EN-share distributions "
                 "(every tweet's first 200 chars)", y=1.02, fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "p6nls_tweet_share.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    bins_sc = np.linspace(0, 1, 41)
    clf = HistGradientBoostingClassifier(
        random_state=42, max_iter=200, early_stopping=False).fit(
            ARMS_FULL["SPEC_X_CHAR"][0], y_src)
    scores_full = clf.predict_proba(SC_tb_t)[:, 1]
    clf_r = HistGradientBoostingClassifier(
        random_state=42, max_iter=200, early_stopping=False).fit(
            ARMS_REST["SPEC_X_CHAR"][0], y_rsrc)
    scores_rest = clf_r.predict_proba(SC_tb_r)[:, 1]
    for ax, scores, yy, sub in (
            (axes[0], scores_full, y_tgt,
             f"full population (AUC {roc_auc_score(y_tgt, scores_full):.4f})"),
            (axes[1], scores_rest, y_rtgt,
             f"EN∩EN population (AUC {roc_auc_score(y_rtgt, scores_rest):.4f})")):
        for m, nm2, cc in ((yy == 0, "human", HUM_C),
                           (yy == 1, "bot", FAM_C)):
            ax.hist(scores[m], bins=bins_sc, histtype="step", lw=2,
                    density=True, color=cc, label=nm2)
        ax.set_title(f"transferred SPEC_XCHAR scores — {sub}",
                     loc="left", fontsize=9)
        ax.set_xlabel("predict_proba")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    a = axes[2]
    vals_b = [fams_full["SPEC_X_CHAR|hgb"]["within_mean"],
              fams_full["SPEC_X_CHAR|hgb"]["transfer_mean"],
              fams_rest["SPEC_X_CHAR|hgb"]["within_mean"],
              fams_rest["SPEC_X_CHAR|hgb"]["transfer_mean"]]
    a.bar(range(4), vals_b,
          color=["#bbbbbb", FAM_C, "#bbbbbb", FAM_C], alpha=0.9)
    a.axhline(0.5, color="black", ls="--", lw=1.1, label="chance")
    band = max(max(abs(v) for v in g4_full_pipe.values()),
               max(abs(v) for v in g4_rest_pipe.values()))
    a.axhspan(0.5 - band, 0.5 + band, color="grey", alpha=0.18,
              label=f"G4 band ±{band:.4f}")
    a.set_xticks(range(4))
    a.set_xticklabels(["within\n(full)", "transfer\n(full)",
                       "within\n(EN∩EN)", "transfer\n(EN∩EN)"], fontsize=8)
    a.set_ylim(0.4, 1.02)
    a.set_title("the inversion and its repair (hgb)", loc="left",
                fontsize=9)
    a.legend(fontsize=8)
    fig.suptitle("LS-1 arm B — Δ_inv adjudication (PRELIMINARY: Kaggle "
                 "copy)", y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "p6nls_inversion.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # ---------------- report ------------------------------------------------------------
    report = {
        "phase": "P6-language-shift (amendment LS-1, bitacora 24)",
        "preliminary": PRELIMINARY,
        "provenance_and_hashes": {
            "preliminary": PRELIMINARY,
            "origin": "third-party Kaggle re-upload marvinvanbo/twibot-20; "
                      "owner decision recorded in bitacora 23 sect. 1; "
                      "inherits the re-derivation obligation on the "
                      "authorized copy",
            "sha256": hashes,
        },
        "lid_tool_echo": lid_echo,
        "composition": {
            "token_positive_source": pop_stats(y_src),
            "token_positive_target": pop_stats(y_tgt),
            "restricted_source": pop_stats(y_rsrc),
            "restricted_target": pop_stats(y_rtgt),
            "exclusions_restriction": excl_restricted,
            "budget10k_restricted_target_n": int(len(y_rtgt10)),
            "age_reference_iso": "2020-01-01T00:00:00+00:00 (both corpora)",
        },
        "census": {
            "population": "token-positive (the population every arm runs on)",
            "label_distributions": census_labels,
            "en_shares": {"cresci15|bot": e_c15_b,
                          "cresci15|human": e_c15_h,
                          "twibot20|bot": e_tb_b,
                          "twibot20|human": e_tb_h},
            "tweet_share_stats": {
                k: {"mean": (float(np.nanmean(v)) if v else None),
                    "median": (float(np.nanmedian(v)) if v else None),
                    "n_accounts": len(v)}
                for k, v in tweet_share.items()},
        },
        "fidelity_gate_vs_h4t": {
            "artefact": "results/p6n_transfer_text.json",
            "fields": ["within_mean", "transfer_mean", "delta_mean",
                       "delta_sd", "transfer_full_fit_auc",
                       "transfer_ci95_boot[2]"],
            "atol": FID_ATOL,
            "max_abs_diff_per_arm": fid_rows,
            "max_abs_diff": worst,
            "passed": bool(worst < FID_ATOL),
        },
        "families_full_reference": fams_full,
        "families_restricted": fams_rest,
        "primary_delta_inv": {
            "rule": "sect. 6 verbatim: mean > 0.05 AND >=16/20 draws AND "
                    "repair (restricted transfer > 0.5 with CI low > 0.5)",
            "hgb": inv_hgb,
            "lr": inv_lr,
        },
        "p1_composition": {
            "e_c15_bot": e_c15_b, "e_c15_human": e_c15_h,
            "e_tb20_bot": e_tb_b, "e_tb20_human": e_tb_h,
            "predicate_i_cresci_gap": p1_i,
            "predicate_ii_target_margin": p1_ii,
            "bar": P1_BAR,
            "verdict_p1": verdict_p1,
        },
        "combined_verdict": combined,
        "b2_placebo_en_source_to_full_target": b2,
        "budget_sensitivity_10k": {
            "runs": budget_row,
            "delta_inv_mean_hgb": inv10_mean,
            "abs_shift_vs_primary_hgb": abs(inv10_mean
                                            - inv_hgb["delta_inv_mean"]),
        },
        "equal_cap_sensitivity": {
            "runs": {k: {"delta_mean": v["delta_mean"],
                         "transfer_mean": v["transfer_mean"]}
                     for k, v in eqcap.items()},
            "delta_inv_mean_hgb": inv_cap_mean,
            "abs_shift_vs_primary_hgb": abs(inv_cap_mean
                                            - inv_hgb["delta_inv_mean"]),
        },
        "c_local_restoration": {
            "design": "within-TB20 stratified 5-fold, seeds 42..51, hgb",
            "runs": local,
            "restricted_minus_full": c_pairs,
            "note": "exploratory family, Holm across the three comparisons; "
                    "seed pairing gives the 10-seed p-floor 0.001953; "
                    "'clears_floor' keys inherit evaluate.paired's 0.02 "
                    "AUC floor semantics (not the P1 share bar)",
        },
        "sanity_null_g4": {
            "design": "exchangeable halves of the respective SOURCE "
                      "population, identical D6 machinery",
            "full_pipeline": g4_full_pipe,
            "restricted_pipeline": g4_rest_pipe,
            "committed_fidelity_max_abs_diff": null_worst,
            "expected_silent_because": "no covariate shift between halves",
        },
        "multiple_comparisons": {
            "gated_set": ["delta_inv conjuncts (3)", "P1 predicate (i)",
                          "P1 predicate (ii)"],
            "exploratory_holm_family": ["C-arm SPEC_X_CHAR",
                                        "C-arm H1_CHAR", "C-arm HINF_CHAR"],
            "context_ungated": ["b2 placebo", "budget 10k row",
                                "equal-cap row"],
        },
        "figures": ["p6nls_language_census.png", "p6nls_tweet_share.png",
                    "p6nls_inversion.png"],
    }

    OUT_JSON.write_text(json.dumps(report, indent=1))

    band_all = max(abs(v) for v in
                   list(g4_full_pipe.values()) + list(g4_rest_pipe.values()))
    print("\n" + "=" * 88)
    print("LS-1 — IS THE CHAR-SPECTRUM INVERSION A CORPUS-LANGUAGE EFFECT? "
          "[PRELIMINARY]")
    print("=" * 88)
    print(f"census EN shares   C15 bot/hum {e_c15_b:.4f}/{e_c15_h:.4f}   "
          f"TB20 bot/hum {e_tb_b:.4f}/{e_tb_h:.4f}")
    print(f"P1   gap(i) {p1_i:+.4f}  margin(ii) {p1_ii:+.4f}  "
          f"bar {P1_BAR:.2f}  -> {'PASS' if verdict_p1 else 'FAIL'}")
    print(f"B    delta_inv (hgb) {inv_hgb['delta_inv_mean']:+.4f} "
          f"[{inv_hgb['draws_gt_bar']} draws > {INV_BAR}]  repair "
          f"{inv_hgb['repair_conjunct']['restricted_transfer_mean']:.4f} "
          f"(CI lo {inv_hgb['repair_conjunct']['restricted_ci95_low']:.4f})"
          f"  -> {inv_hgb['verdict_primary']}")
    print(f"B2   placebo (EN-source -> FULL-target) transfer hgb "
          f"{b2['SPEC_X_CHAR|hgb']['transfer_mean']:.4f}")
    print(f"sens equal-cap |d_inv shift| "
          f"{abs(inv_cap_mean - inv_hgb['delta_inv_mean']):.4f}   "
          f"budget-10k |shift| "
          f"{abs(inv10_mean - inv_hgb['delta_inv_mean']):.4f}")
    print(f"C    local char separability REST-full "
          f"{c_pairs['SPEC_X_CHAR']['mean_diff']:+.4f} "
          f"(holm p {c_pairs['SPEC_X_CHAR']['p_holm_adjusted']:.4f})")
    print(f"D    G4 null max|d| full "
          f"{max(abs(v) for v in g4_full_pipe.values()):.4f} / "
          f"rest {max(abs(v) for v in g4_rest_pipe.values()):.4f}")
    print(f"fidelity vs H4-T artefact max |diff| {worst:.2e} (PASS)")
    print(f"\nCOMBINED VERDICT: {combined}")
    print(f"json -> {OUT_JSON}   (band {band_all:.4f})")
    print("=" * 88)


if __name__ == "__main__":
    main()
