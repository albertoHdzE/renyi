#!/usr/bin/env python
"""WP-I -- text front SPEC_X (phase P3b; plan WP-I).

EXPLORATORY under H2 (which WP-H passed), Holm-corrected within the SPEC_X
family; every number in this artefact carries the exploratory label.

Blocks (``renyiext.textfront``), on raw uncleaned text (D4):

    SPEC_X_WORD   6-vector spectrum of the per-account word-frequency
                  distribution; tokens = ``\\w+`` over Unicode, NO
                  lowercasing (casing is signal)
    SPEC_X_CHAR   6-vector spectrum of the per-account raw-character
                  frequency distribution (per-tweet counts summed, no
                  separators injected)

Length control (R2, amended form): token count is a covariate exactly like
event count -- ``rho(H_alpha, label | log tokens)`` reported for all 12
orders (plus given log count as context); length distributions rendered per
class; sensitivity arm restricted to accounts with >= 512 tokens; no fixed-n
subsampling (D3').

Arms: COUNT, TOKENS, META-lite (botsage recipe, age vs 2015-01-01), Shannon
slices, SPEC_X_WORD/SPEC_X_CHAR/SPEC_X alone and over COUNT/TOKENS. Floors
get the WP-E dim-matched noise arms (salts by declaration order; binding
interpretation rules). Gated unmatched verdicts recorded beside them (D10).

Census (G3): the token regex and no-lowercase rule are printed; the
with/without-URL variant is COUNTED -- URL prevalence per class and a
url-stripped SPEC_X_WORD sensitivity arm; the headline keeps raw text.

sigma_config: this WP's only published config axis is the >= 512-token
sensitivity split; the comparison deltas are reported under both and their
population SD (ddof = 0) recorded where both sides exist. Holm: step-down
over the family of all paired comparisons in this artefact (gated +
dim-matched), membership and adjustments documented.

Usage:
    python scripts/run_p3i_textfront.py [--quiet] [--seeds N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import DATA_PROCESSED, FIGURES, RESULTS
from renyiext.events import load_events_cached, load_cresci_text_side
from renyiext.features import temporal_blocks, MS_PER_DAY
from renyiext.textfront import (WORD_RE, URL_RE, word_tokens, strip_urls,
                                account_text_features)
from renyiext.evaluate import (eval_arm, run_arms, paired, partial_corr,
                               noise_padding, interpret_dim_matched,
                               sigma_config)

warnings.filterwarnings("ignore", category=UserWarning)

CACHE = DATA_PROCESSED / "cresci_events_d9.npz"
OUT_JSON = RESULTS / "p3i_textfront.json"
AGE_REF_MS = int(datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp()
                 * 1000)
TOKEN_THRESHOLD = 512            # pre-registered sensitivity split
BOT_C, HUM_C = "#e76f51", "#2a9d8f"
ORDERS = ["H_0", "H_0.5", "H_1", "H_2", "H_4", "H_inf"]

GATED = (
    ("SPEC_X_WORD", "COUNT"), ("SPEC_X_WORD", "TOKENS"),
    ("SPEC_X_WORD", "META"), ("SPEC_X_WORD", "SHAN_WORD"),
    ("SPEC_X_CHAR", "COUNT"), ("SPEC_X_CHAR", "TOKENS"),
    ("SPEC_X_CHAR", "META"), ("SPEC_X_CHAR", "SHAN_CHAR"),
    ("SPEC_X", "SHAN_X"), ("SPEC_X", "META"), ("SPEC_X", "COUNT"),
    ("COUNT+SPEC_X", "COUNT"), ("TOKENS+SPEC_X", "TOKENS"),
)
DIM_MATCHED = (
    (("SHAN_WORD",), "SPEC_X_WORD"),                    # salt 0
    (("SHAN_CHAR",), "SPEC_X_CHAR"),                    # salt 1
    (("SHAN_X",), "SPEC_X"),                            # salt 2
    (("META",), "SPEC_X"),                              # salt 3
    (("COUNT", "SHAN_X"), "COUNT+SPEC_X"),              # salt 4
    (("TOKENS", "SHAN_X"), "TOKENS+SPEC_X"),            # salt 5
    (("META",), "SPEC_X_WORD"),                         # salt 6
)


def holm(pvals):
    """Step-down Holm adjustment; returns adjusted p (order preserved)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = list(range(42, 42 + args.seeds))

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
    print(f"[gate] kept index + COUNT vs temporal_blocks: max |diff| {fid:.2e}")

    print("loading corpus text + META-lite ...", flush=True)
    t0 = time.time()
    meta, username, seq = load_cresci_text_side(ev, kept, AGE_REF_MS)
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    # ---- features ----------------------------------------------------------
    n = len(seq)
    spec_w = np.zeros((n, 6)); spec_c = np.zeros((n, 6))
    spec_w_ns = np.zeros((n, 6))          # url-stripped census variant
    n_tokens = np.zeros(n, dtype=np.int64)
    cap = {"tweets": 0, "tweets_with_url": 0, "tokens": 0}
    texts_store = [None] * n
    for j, tweets in enumerate(seq):
        raw = [t for _, t in tweets]
        texts_store[j] = raw
        sw, sc, nt = account_text_features(raw)
        spec_w[j], spec_c[j], n_tokens[j] = sw, sc, nt
        spec_w_ns[j], _, _ = account_text_features(raw, strip=True)
        cap["tweets"] += len(raw)
        cap["tweets_with_url"] += sum(bool(URL_RE.search(t)) for t in raw)
        cap["tokens"] += nt
    logc = np.log1p(counts[idx])[:, None]
    logt = np.log1p(n_tokens)[:, None]
    h1_w, h1_c = spec_w[:, [2]], spec_c[:, [2]]

    zero_tok = n_tokens == 0
    excl = {"bot": int((zero_tok & (y == 1)).sum()),
            "human": int((zero_tok & (y == 0)).sum())}
    keep_tok = ~zero_tok
    sub = np.where(keep_tok)[0]
    ys, logcs, logts = y[sub], logc[sub], logt[sub]
    sws, scs = spec_w[sub], spec_c[sub]
    h1ws, h1cs = h1_w[sub], h1_c[sub]
    metas = meta[sub]

    capture = {
        "token_regex": r"\w+ (Unicode; NO lowercasing -- casing is signal, D4)",
        "char_rule": "every raw character of every kept tweet, per-tweet "
                     "counts summed, no separators injected",
        "url_variant": "census only; headline keeps raw text (D4)",
        "url_regex": r"https?://\S+",
        "tweets_seen": int(cap["tweets"]),
        "tweets_with_url": int(cap["tweets_with_url"]),
        "tweet_url_rate": float(cap["tweets_with_url"] / cap["tweets"]),
        "url_rate_by_class": {
            "bot": float(sum(bool(URL_RE.search(t))
                             for j in np.where(y == 1)[0]
                             for t in texts_store[j]) /
                         max(sum(len(texts_store[j])
                                 for j in np.where(y == 1)[0]), 1)),
            "human": float(sum(bool(URL_RE.search(t))
                               for j in np.where(y == 0)[0]
                               for t in texts_store[j]) /
                           max(sum(len(texts_store[j])
                                   for j in np.where(y == 0)[0]), 1))},
        "tokens_total": int(cap["tokens"]),
        "zero_token_exclusions": excl,
        "n_sample": int(len(sub)),
    }

    # ---- arms --------------------------------------------------------------
    arms = {
        "COUNT": logcs, "TOKENS": logts, "META": metas,
        "SHAN_WORD": h1ws, "SHAN_CHAR": h1cs,
        "SHAN_X": np.hstack([h1ws, h1cs]),
        "COUNT+SHAN_X": np.hstack([logcs, h1ws, h1cs]),
        "TOKENS+SHAN_X": np.hstack([logts, h1ws, h1cs]),
        "SPEC_X_WORD": sws, "SPEC_X_CHAR": scs,
        "SPEC_X": np.hstack([sws, scs]),
        "COUNT+SPEC_X": np.hstack([logcs, sws, scs]),
        "TOKENS+SPEC_X": np.hstack([logts, sws, scs]),
    }
    csw = np.hstack([logcs, sws, scs])
    tsx = np.hstack([logts, sws, scs])
    shx = np.hstack([h1ws, h1cs])
    noise = {
        "SHAN_WORD+NOISE(5)": (lambda s, X=h1ws: noise_padding(X, 5, s, 0)),
        "SHAN_CHAR+NOISE(5)": (lambda s, X=h1cs: noise_padding(X, 5, s, 1)),
        "SHAN_X+NOISE(10)": (lambda s, X=shx: noise_padding(X, 10, s, 2)),
        "META+NOISE(8)": (lambda s, X=metas: noise_padding(X, 8, s, 3)),
        "COUNT+SHAN_X+NOISE(10)": (lambda s, X=np.hstack([logcs, h1ws, h1cs]):
                                   noise_padding(X, 10, s, 4)),
        "TOKENS+SHAN_X+NOISE(10)": (lambda s, X=np.hstack([logts, h1ws,
                                                           h1cs]):
                                    noise_padding(X, 10, s, 5)),
        "META+NOISE(2)": (lambda s, X=metas: noise_padding(X, 2, s, 6)),
    }

    print("\n[arms] exploratory, Holm-corrected")
    res = run_arms(arms, ys, seeds, "hgb", args.quiet)
    res.update(run_arms(noise, ys, seeds, "hgb", args.quiet))

    comps = {}
    for fam, fl in GATED:
        c = paired(res[fam]["auc"], res[fl]["auc"])
        comps[f"{fam}_vs_{fl}"] = {"family": fam, "floor_arm": fl,
                                   "kind": "gated", **c}
    for floor_parts, fam in DIM_MATCHED:
        flname = "+".join(floor_parts)
        k = res[fam]["n_features"] - res[flname]["n_features"]
        fl = flname + f"+NOISE({k})"
        c = paired(res[fam]["auc"], res[fl]["auc"])
        c["verdict"] = interpret_dim_matched(c["mean_diff"], c["significant"])
        comps[f"{fam}_vs_{fl}"] = {"family": fam, "floor_arm": fl,
                                   "kind": "dim_matched", **c}

    # ---- Holm within the SPEC_X family -------------------------------------
    keys = list(comps)
    raw_p = [comps[k]["p"] for k in keys]
    adj = holm(raw_p)
    for k, a in zip(keys, adj):
        comps[k]["p_holm_adjusted"] = float(a)
        comps[k]["significant_holm_0.05"] = bool(a < 0.05)
    holm_block = {
        "family": "all paired comparisons in this artefact (gated + "
                  "dim-matched); every number exploratory",
        "n_comparisons": len(keys),
        "method": "Holm step-down, family-wise alpha 0.05",
        "surviving_holm": [k for k in keys
                           if comps[k]["significant_holm_0.05"]],
    }

    # ---- length control (R2 amended): all 12 orders -------------------------
    length_control = {}
    for blk, spec in (("word", sws), ("char", scs)):
        for k, o in enumerate(ORDERS):
            length_control[f"{o}_{blk}"] = {
                "raw": float(np.corrcoef(spec[:, k], ys)[0, 1]),
                "given_log_tokens": partial_corr(spec[:, k],
                                                 ys.astype(float),
                                                 logts[:, 0]),
                "given_log_count": partial_corr(spec[:, k],
                                                ys.astype(float),
                                                logcs[:, 0])}

    # ---- sensitivity: >= 512 tokens ----------------------------------------
    hi = n_tokens[sub] >= TOKEN_THRESHOLD
    sens = {"threshold_tokens": TOKEN_THRESHOLD,
            "survivors": {"bot": int((hi & (ys == 1)).sum()),
                          "human": int((hi & (ys == 0)).sum())},
            "note": "no fixed-n subsampling (D3'); threshold is the "
                    "registered sensitivity split"}
    if min(sens["survivors"].values()) >= 20:
        yh = ys[hi]
        sh = {
            "COUNT": float(np.mean([eval_arm(logcs[hi], yh, s)["auc"]
                                    for s in seeds])),
            "SPEC_X_WORD": float(np.mean([eval_arm(sws[hi], yh, s)["auc"]
                                          for s in seeds])),
            "SPEC_X_CHAR": float(np.mean([eval_arm(scs[hi], yh, s)["auc"]
                                          for s in seeds])),
            "SPEC_X": float(np.mean([eval_arm(
                np.hstack([sws[hi], scs[hi]]), yh, s)["auc"] for s in seeds])),
        }
        sh["SPEC_X_WORD_vs_COUNT"] = float(sh["SPEC_X_WORD"] - sh["COUNT"])
        sens["aucs_high_token_subsample"] = sh
        sens["sigma_config_SPEC_X_WORD_vs_COUNT"] = sigma_config(
            [comps["SPEC_X_WORD_vs_COUNT"]["mean_diff"],
             sh["SPEC_X_WORD_vs_COUNT"]])
    else:
        sens["evaluated"] = False
        sens["reason"] = "fewer than 20 accounts per class survive"

    # ---- URL-stripped census arm --------------------------------------------
    url_arm = {
        "SPEC_X_WORD_urlstripped": float(np.mean(
            [eval_arm(spec_w_ns[sub], ys, s)["auc"] for s in seeds])),
        "delta_vs_raw": None,
    }
    url_arm["delta_vs_raw"] = float(url_arm["SPEC_X_WORD_urlstripped"]
                                    - float(np.mean(res["SPEC_X_WORD"]["auc"])))

    # ---- figures ------------------------------------------------------------
    FIGURES.mkdir(parents=True, exist_ok=True)
    tok_pos = np.where(sub)[0]
    hi_w_bot = int(tok_pos[np.argmax(np.where(ys == 1, n_tokens[sub], -1))])
    hi_w_hum = int(tok_pos[np.argmax(np.where(ys == 0, n_tokens[sub], -1))])
    examples = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4))
    for c, (name, colour, row) in enumerate((("bot", BOT_C, hi_w_bot),
                                             ("human", HUM_C, hi_w_hum))):
        ax = axes[0, c]
        ax.axis("off")
        longest = max(texts_store[row], key=len)
        excerpt = longest[:400] + ("…" if len(longest) > 400 else "")
        wc = Counter(word_tokens(longest))
        top = wc.most_common(12)
        ax.text(0.0, 1.0, f"{name} {str(ev.user_ids[idx[row]])} — longest "
                f"tweet ({len(longest)} chars, RAW, uncleaned):",
                fontsize=9.5, color=colour, weight="bold", va="top",
                transform=ax.transAxes)
        ax.text(0.0, 0.88, repr(excerpt), fontsize=7.6, va="top",
                transform=ax.transAxes, wrap=True)
        ax2 = ax.inset_axes([0.0, 0.06, 1.0, 0.38])
        lbls = [w for w, _ in top][::-1]
        vals = [n for _, n in top][::-1]
        ax2.barh(range(len(vals)), vals, color=colour, alpha=0.8)
        ax2.set_yticks(range(len(vals)))
        ax2.set_yticklabels(lbls, fontsize=7)
        ax2.set_title("top tokens (raw, case-preserved)", fontsize=8.5)
        examples[name] = {"user_id": str(ev.user_ids[idx[row]]),
                          "n_tokens": int(n_tokens[row]),
                          "longest_tweet_chars": int(len(longest)),
                          "excerpt_recorded_chars": len(excerpt)}
    xs = np.arange(6)
    for k, (blk, spec) in enumerate((("WORD", sws), ("CHAR", scs))):
        a = axes[1, k]
        for mask, nm, col in ((ys == 0, "human", HUM_C),
                              (ys == 1, "bot", BOT_C)):
            m, s = spec[mask].mean(axis=0), spec[mask].std(axis=0)
            a.plot(xs, m, "o-", color=col, lw=2, label=nm)
            a.fill_between(xs, m - s, m + s, color=col, alpha=0.18)
        a.set_xticks(xs); a.set_xticklabels(ORDERS)
        a.set_ylabel("bits"); a.legend(fontsize=8)
        a.set_title(f"SPEC_X_{blk} — α-curves (mean ± 1 SD)", fontsize=10)
    fig.suptitle("WP-I — text-front objects (G1: raw text + token "
                 "histograms) and alpha-curves [EXPLORATORY]",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "p3i_spec_x_objects.png", dpi=130)
    plt.close(fig)
    if not args.quiet:
        print(f"  [G1] rendered -> {FIGURES / 'p3i_spec_x_objects.png'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.4))
    for mask, nm, col in ((ys == 0, "human", HUM_C), (ys == 1, "bot", BOT_C)):
        ax1.hist(np.log1p(n_tokens[sub][mask]), bins=50, alpha=0.6,
                 color=col, label=nm)
        med = np.median(n_tokens[sub][mask])
        ax1.axvline(np.log1p(med), color=col, ls=":", lw=1.5)
        ax1.text(np.log1p(med), ax1.get_ylim()[1] * 0.92,
                 f" {nm} med {med:.0f}", fontsize=8, color=col,
                 rotation=90, va="top")
    ax1.set_xlabel("log1p(tokens per account)")
    ax1.set_ylabel("accounts")
    ax1.set_title("Length distributions per class (R2 render)", fontsize=10)
    ax1.legend()
    ax2.axvline(TOKEN_THRESHOLD, color="crimson", ls="--", lw=1.5,
                label=f">= {TOKEN_THRESHOLD} tokens sensitivity")
    for mask, nm, col in ((ys == 0, "human", HUM_C), (ys == 1, "bot", BOT_C)):
        v = np.sort(n_tokens[sub][mask])
        ax2.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, label=nm)
    ax2.set_xscale("log")
    ax2.set_xlabel("tokens per account (log)"); ax2.set_ylabel("CDF")
    ax2.set_title("Survivor view of the sensitivity split", fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle("WP-I — length renders (G1/R2)", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "p3i_length.png", dpi=130)
    plt.close(fig)
    if not args.quiet:
        print(f"  [G1] rendered -> {FIGURES / 'p3i_length.png'}")

    # ---- JSON ---------------------------------------------------------------
    report = {
        "phase": "P3i-textfront (WP-I)",
        "exploratory": True,
        "seeds": seeds,
        "classifier": "HistGradientBoostingClassifier(max_iter=200, "
                      "early_stopping=False), StratifiedKFold(5)",
        "feature_config": {"min_events": 5, "n_kept": int(len(idx)),
                           "age_reference": "2015-01-01 (botsage recipe)",
                           "token_threshold_sensitivity": TOKEN_THRESHOLD},
        "capture": capture,
        "fidelity": {"kept_index_matches_temporal": True,
                     "count_block_max_abs_diff": fid},
        "arms": res,
        "comparisons": comps,
        "holm": holm_block,
        "length_control": length_control,
        "sensitivity_high_token": sens,
        "url_census_arm": url_arm,
        "examples": examples,
        "yule_k_note": "H2 on word frequency is the collision entropy "
                       "-log2 sum p^2 -- the functional Yule's K "
                       "normalises; a separating H2 is reported as a "
                       "Yule's-K-flavoured vocabulary concentration "
                       "(docs/03-PHASES.md P3 note)",
    }
    OUT_JSON.write_text(json.dumps(report, indent=1))

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 92)
    print("WP-I — TEXT FRONT SPEC_X  [EXPLORATORY, Holm-corrected]")
    print("=" * 92)
    print(f"{'arm':<24}{'dim':>4}{'AUC':>10}{'±SD':>8}{'TPR@1%':>9}"
          f"{'TPRfold':>9}")
    for k, v in res.items():
        a = np.array(v["auc"]); t = np.array(v["tpr01"])
        tf = np.array(v["tpr01_foldmean"])
        print(f"{k:<24}{v['n_features']:>4}{a.mean():>10.4f}{a.std():>8.4f}"
              f"{t.mean():>9.3f}{tf.mean():>9.3f}")
    print(f"\n{'comparison':<52}{'delta':>9}{'wins':>7}{'p':>9}"
          f"{'p_holm':>9}  verdict")
    for k in keys:
        c = comps[k]
        extra = f"  {c['verdict']}" if c["kind"] == "dim_matched" else ""
        print(f"{k:<52}{c['mean_diff']:>+9.4f}{c['wins']:>7}{c['p']:>9.4f}"
              f"{c['p_holm_adjusted']:>9.4f}{extra}")
    print("\nlength control (raw | given tokens | given count):")
    for o in ORDERS:
        for blk in ("word", "char"):
            v = length_control[f"{o}_{blk}"]
            print(f"  {o}_{blk:<5} {v['raw']:+.3f} | "
                  f"{v['given_log_tokens']:+.3f} | "
                  f"{v['given_log_count']:+.3f}")
    print(f"\n>= {TOKEN_THRESHOLD}-token sensitivity: survivors "
          f"{sens['survivors']}")
    if "aucs_high_token_subsample" in sens:
        print("  " + json.dumps(
            {k: round(v, 4) for k, v in
             sens["aucs_high_token_subsample"].items()}))
    print(f"URL census: {capture['tweet_url_rate']:.4f} of tweets carry a "
          f"URL (bot {capture['url_rate_by_class']['bot']:.4f} / human "
          f"{capture['url_rate_by_class']['human']:.4f}); stripped-arm delta "
          f"{url_arm['delta_vs_raw']:+.4f}")
    print(f"json -> {OUT_JSON}")
    print("=" * 92)


if __name__ == "__main__":
    main()
