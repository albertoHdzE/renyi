#!/usr/bin/env python
"""WP-H -- behavioural front SPEC_B (phase P3a; plan WP-H).

Blocks (``renyiext.behaviour``):

    SPEC_B_ALPHA     6-vector spectrum of the post-type distribution,
                     D11-collapsed to {original, reply, retweet}
                     (quote -> original; share-after-collapse reported;
                     the raw 4-symbol encoding is the published sensitivity
                     axis and stays the DNA encoding for P4)
    SPEC_B_MENTION   6-vector spectrum of the mention-target distribution
                     (registered rules: all ``@\\w+`` tokens; reply-leading
                     token dropped -- replied-to party != audience;
                     self-mentions dropped, case-insensitive; empty =>
                     account excluded from the block, counted per class)

Arms/floors (protocol §3 as amended; BURST is temporal-specific and is
substituted by the WP-E dim-matched noise arms -- documented): majority,
COUNT, META-lite (followers/following/statuses/age per botsage's read-only
recipe, age vs 2015-01-01), Shannon slices, COUNT+SHAN composites, and the
SPEC blocks alone and over COUNT. Every family-vs-floor pair with
dim(family) > dim(floor) gets its ``<FLOOR>+NOISE(k)`` control (§8 D2,
salts by declaration order); interpretation rules binding.

Gates: H2 directional (pre-registered) -- bots lower H_2 on the alphabet,
lower H_0 on mention-targets; one-sided Mann-Whitney p < 0.05 with
consistent signs; else docs/03-PHASES.md P3 rule (return to P1/P8'
reasoning; inconsistency = the volume/length detector firing).

Datasaurus: G1 -- example post-type sequence and mention list rendered above
the alpha-curves; G2 -- exclusions elementwise per class; G3 -- regex rules
printed with capture rates, D11 shares before/after, salts echoed; G4 --
marginal-preserving shuffle of post-type sequences run end-to-end, expected
silent (the spectrum is order-invariant, P6; separating world named in the
JSON, S4.1 logic).

sigma_config (§8 D3): population SD of each comparison's delta across the
published encoding axis {collapsed (headline), raw 4-symbol} for
alpha-sample rows; mention rows are encoding-invariant by construction
(sigma exactly 0).

Usage:
    python scripts/run_p3h_behaviour.py [--quiet] [--seeds N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import DATA_PROCESSED, FIGURES, RESULTS, DATA_RAW
from renyiext.events import load_events_cached, POST_TYPE
from renyiext.features import temporal_blocks, MS_PER_DAY
from renyiext.spectrum import spectrum, counts_to_probabilities, SPECTRUM_ALPHAS
from renyiext.behaviour import (ALPHABET3, MENTION_RE, collapse_post_types,
                                post_type_counts, alphabet_spectrum,
                                extract_mentions, mention_spectrum)
from renyiext.evaluate import (eval_arm, run_arms, paired, noise_padding,
                               interpret_dim_matched, sigma_config)

warnings.filterwarnings("ignore", category=UserWarning)

CACHE = DATA_PROCESSED / "cresci_events_d9.npz"
OUT_JSON = RESULTS / "p3h_behaviour.json"
AGE_REF_MS = int(datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
BOT_C, HUM_C = "#e76f51", "#2a9d8f"
SHUFFLE_SEED = 42
ORDERS = ["H_0", "H_0.5", "H_1", "H_2", "H_4", "H_inf"]

# (floor arms, family arm) per sample; salt = index within its tuple order.
DIM_MATCHED_ALPHA = (
    (("H1_alpha",), "SPEC_B_ALPHA"),
    (("META",), "SPEC_B_ALPHA"),
    (("COUNT", "H1_alpha"), "COUNT+SPEC_B_ALPHA"),
    (("COUNT", "META"), "COUNT+SPEC_B_ALPHA"),
)
DIM_MATCHED_MENTION = (
    (("H1_mention",), "SPEC_B_MENTION"),
    (("META",), "SPEC_B_MENTION"),
    (("SHAN_B",), "SPEC_B"),
    (("COUNT", "SHAN_B"), "COUNT+SPEC_B"),
    (("META",), "SPEC_B"),
)


def load_corpus_side(ev, kept):
    """Username, META-lite, and per-account (post_type, text) for kept
    accounts -- same access pattern as ``events.load_cresci_events``."""
    t0 = time.time()
    base = DATA_RAW / "bot" / "cresci-2015"
    with open(base / "edge.csv") as fh:
        header = next(fh).rstrip("\n").split(",")
        i_src, i_rel, i_tgt = (header.index("source_id"),
                               header.index("relation"),
                               header.index("target_id"))
        posts: dict[str, list[str]] = {}
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) <= i_tgt or p[i_rel] != "post":
                continue
            posts.setdefault(p[i_src], []).append(p[i_tgt])
    kept_ids = set(ev.user_ids[kept])
    with open(base / "node.json") as fh:
        nodes = json.load(fh)
    username, pm, born = {}, {}, {}
    texts = {}
    for obj in nodes:
        oid = obj.get("id", "")
        if oid.startswith("u") and oid in kept_ids:
            u = obj.get("public_metrics") or {}
            username[oid] = obj.get("username") or ""
            pm[oid] = (float(u.get("followers_count") or 0),
                       float(u.get("following_count") or 0),
                       float(u.get("tweet_count") or 0))
            try:
                born[oid] = int(datetime.strptime(
                    obj.get("created_at"), "%a %b %d %H:%M:%S %z %Y")
                    .timestamp() * 1000)
            except (TypeError, ValueError):
                born[oid] = None
        elif oid.startswith("t"):
            # all tweet texts transit through this dict (same footprint as
            # events.load_cresci_events); dropped after sequences are built
            texts[oid] = obj.get("text") or ""
    del nodes
    from renyiext.events import has_snowflake_form, decode_snowflake_array
    meta_rows, seq = [], []
    for i in np.where(kept)[0]:
        uid = str(ev.user_ids[i])
        f, g, tc = pm[uid]
        age = ((AGE_REF_MS - born[uid]) / MS_PER_DAY
               if born[uid] is not None else np.nan)
        meta_rows.append([f, g, tc, age])
        # rebuild the loader's exact per-account ordering (decode, stable
        # sort by time) so text aligns with the cached post_type slice --
        # asserted elementwise against the cache below
        raw = [tid for tid in posts.get(uid, [])
               if has_snowflake_form(int(tid.lstrip("t")))]
        ids = np.array([int(tid.lstrip("t")) for tid in raw], dtype=np.int64)
        ts = decode_snowflake_array(ids)
        order = np.argsort(ts, kind="stable")
        a, b = int(ev.offsets[i]), int(ev.offsets[i + 1])
        # ts[order] == the cache's time-sorted slice proves raw[order] is
        # exactly the loader's id order (same stable sort on the same ids)
        assert np.array_equal(ts[order], ev.ts_ms[a:b]), uid
        seq.append([(int(ev.post_type[a + k]), texts[raw[order[k]]])
                    for k in order])
    del texts
    print(f"  corpus text/meta loaded in {time.time()-t0:.0f}s", flush=True)
    return np.array(meta_rows, dtype=np.float64), username, seq


def build_features(seq, username, idx, ev):
    """Per-account spectra (both encodings), mention counts, capture stats."""
    n = len(seq)
    spec_a3 = np.zeros((n, 6)); spec_a4 = np.zeros((n, 6))
    h1_a3 = np.zeros(n); h1_a4 = np.zeros(n)
    mentions = [None] * n
    cap = {"tweets": 0, "with_token": 0, "leading_dropped": 0,
           "self_dropped": 0, "tokens": 0}
    counts4 = np.zeros((n, 4))
    for j, tweets in enumerate(seq):
        types = np.array([t for t, _ in tweets], dtype=np.int64)
        counts4[j] = post_type_counts(types, collapse=False)
        spec_a4[j] = alphabet_spectrum(types, collapse=False)
        spec_a3[j] = alphabet_spectrum(types, collapse=True)
        h1_a4[j] = spec_a4[j, 2]; h1_a3[j] = spec_a3[j, 2]
        uid = str(ev.user_ids[idx[j]]); own = username.get(uid, "")
        mc: Counter = Counter()
        for t, text in tweets:
            kept_toks, info = extract_mentions(text, t, own)
            cap["tweets"] += 1; cap["tokens"] += info["n_tokens"]
            cap["with_token"] += info["n_tokens"] > 0
            cap["leading_dropped"] += info["leading_dropped"]
            cap["self_dropped"] += info["n_self_dropped"]
            mc.update(kept_toks)
        mentions[j] = mc
    return (spec_a3, spec_a4, h1_a3, h1_a4, counts4, mentions, cap)


def alpha_curves(spec, y, path, title, examples=None, seq=None, types_seq=None,
                 quiet=False):
    """G1: objects above (example sequences), alpha-curves below."""
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.5)
    if types_seq is not None:
        for c, (name, colour) in enumerate((("bot", BOT_C), ("human", HUM_C))):
            ax = fig.add_subplot(gs[0, c])
            info = examples[name]
            s = np.array([t for t, _ in types_seq[info["row"]]])
            ax.plot(np.arange(len(s)), s, "|", ms=4, color=colour)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["orig", "reply", "retweet"], fontsize=8)
            ax.set_ylim(-0.3, 2.3)
            ax.set_title(f"{name} {info['uid']} — {info.get('label','')}"
                         f"post-type sequence ({len(s)} events)",
                         loc="left", fontsize=9.5)
            ax.set_xlabel("event #", fontsize=9)
    xs = np.arange(6)
    axc = fig.add_subplot(gs[1, :])
    for mask, nm, c in ((y == 0, "human", HUM_C), (y == 1, "bot", BOT_C)):
        m, s = spec[mask].mean(axis=0), spec[mask].std(axis=0)
        axc.plot(xs, m, "o-", color=c, lw=2, label=nm)
        axc.fill_between(xs, m - s, m + s, color=c, alpha=0.18)
    axc.set_xticks(xs); axc.set_xticklabels(ORDERS)
    axc.set_ylabel("bits"); axc.legend()
    axc.set_title(title, fontsize=10.5)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    if not quiet:
        print(f"  [G1] rendered -> {path}")


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

    # fidelity: same kept sample and COUNT block as the temporal producer
    bl = temporal_blocks(ev, n_bins=24, hi=400 * MS_PER_DAY, min_events=5)
    assert np.array_equal(idx, bl["index"]), "kept-sample drift vs temporal"
    fid_count = float(np.abs(np.log1p(counts[idx]) -
                             bl["blocks"]["COUNT"][:, 0]).max())
    print(f"[gate] kept index + COUNT vs temporal_blocks: max |diff| "
          f"{fid_count:.2e}")
    assert fid_count == 0.0

    print("loading corpus text + META-lite (node.json pass) ...")
    meta, username, seq = load_corpus_side(ev, kept)

    (spec_a3, spec_a4, h1_a3, h1_a4, counts4, mentions, cap) = \
        build_features(seq, username, idx, ev)

    # D11 shares (G3): quote share before collapse, after-collapse shares
    tot4 = counts4.sum(axis=0)
    share_before = {ALPHABET3[0] if k == 0 else
                    ("reply" if k == 1 else "retweet" if k == 2 else "quote"):
                    float(tot4[k] / tot4.sum()) for k in range(4)}
    collapsed_counts = counts4.copy()
    collapsed_counts[:, 0] += collapsed_counts[:, 3]
    collapsed_counts[:, 3] = 0
    tot3v = collapsed_counts.sum(axis=0)
    share_after = {ALPHABET3[k]: float(tot3v[k] / tot3v.sum())
                   for k in range(3)}

    # mention subsample + exclusions per class (G2)
    has_mention = np.array([len(m) > 0 for m in mentions])
    n_excl = {"bot": int((~has_mention & (y == 1)).sum()),
              "human": int((~has_mention & (y == 0)).sum())}
    mi = np.where(has_mention)[0]
    ym = y[mi]
    spec_m = np.array([mention_spectrum(mentions[j]) for j in mi])
    h1_m = spec_m[:, 2]

    capture = {
        "regex": r"@\w+",
        "rules": ["all @\\w+ tokens kept",
                  "reply-classified tweets: leading token dropped "
                  "(replied-to party != audience)",
                  "self-mentions dropped (case-insensitive username match)",
                  "retweet targets kept (only reply-leading is dropped)"],
        "tweets_seen": int(cap["tweets"]),
        "tweets_with_token": int(cap["with_token"]),
        "token_capture_rate": float(cap["with_token"] / cap["tweets"]),
        "reply_leading_dropped": int(cap["leading_dropped"]),
        "self_mentions_dropped": int(cap["self_dropped"]),
        "n_mentions_total": int(sum(len(m) for m in mentions)),
        "distinct_targets_total": int(len(set().union(
            *[set(m) for m in mentions if m]) if any(
            m for m in mentions) else set())),
        "mention_block_exclusions": n_excl,
        "n_mention_sample": int(len(mi)),
    }

    # ---- arms --------------------------------------------------------------
    LOGC = np.log1p(counts[idx])[:, None]
    META = meta
    ones = np.ones((len(idx), 1))

    def arms_alpha(coll: bool):
        sa = spec_a3 if coll else spec_a4
        h1 = h1_a3 if coll else h1_a4
        return {
            "COUNT": LOGC, "META": META,
            "COUNT+META": np.hstack([LOGC, META]),
            "H1_alpha": h1[:, None],
            "COUNT+H1_alpha": np.hstack([LOGC, h1[:, None]]),
            "SPEC_B_ALPHA": sa,
            "COUNT+SPEC_B_ALPHA": np.hstack([LOGC, sa]),
        }

    def noise_alpha(coll: bool):
        sa = spec_a3 if coll else spec_a4
        h1 = h1_a3 if coll else h1_a4
        csa = np.hstack([LOGC, sa])
        return {
            "H1_alpha+NOISE(5)": (lambda s, X=h1[:, None]:
                                  noise_padding(X, 5, s, 0)),
            "META+NOISE(2)": (lambda s, X=META: noise_padding(X, 2, s, 1)),
            "COUNT+H1_alpha+NOISE(5)": (lambda s, X=np.hstack([LOGC, h1[:,
                                        None]]): noise_padding(X, 5, s, 2)),
            "COUNT+META+NOISE(2)": (lambda s, X=np.hstack([LOGC, META]):
                                    noise_padding(X, 2, s, 3)),
            "_pairs": DIM_MATCHED_ALPHA,
            "_family": {"SPEC_B_ALPHA": sa, "COUNT+SPEC_B_ALPHA": csa},
        }

    LOGM = LOGC[mi]
    METAm = META[mi]
    arms_m = {
        "COUNT": LOGM, "META": METAm,
        "H1_mention": h1_m[:, None],
        "COUNT+H1_mention": np.hstack([LOGM, h1_m[:, None]]),
        "SHAN_B": np.hstack([h1_a3[mi][:, None], h1_m[:, None]]),
        "COUNT+SHAN_B": np.hstack([LOGM, h1_a3[mi][:, None], h1_m[:, None]]),
        "SPEC_B_MENTION": spec_m,
        "COUNT+SPEC_B_MENTION": np.hstack([LOGM, spec_m]),
        "SPEC_B": np.hstack([spec_a3[mi], spec_m]),
        "COUNT+SPEC_B": np.hstack([LOGM, spec_a3[mi], spec_m]),
    }
    n_mention = len(mi)
    noise_m = {
        "H1_mention+NOISE(5)": (lambda s, X=h1_m[:, None]:
                                noise_padding(X, 5, s, 4)),
        "META+NOISE(2)": (lambda s, X=METAm: noise_padding(X, 2, s, 5)),
        "SHAN_B+NOISE(10)": (lambda s, X=np.hstack([h1_a3[mi][:, None],
                                                    h1_m[:, None]]):
                             noise_padding(X, 10, s, 6)),
        "COUNT+SHAN_B+NOISE(10)": (lambda s, X=np.hstack(
            [LOGM, h1_a3[mi][:, None], h1_m[:, None]]):
            noise_padding(X, 10, s, 7)),
        "META+NOISE(8)": (lambda s, X=METAm: noise_padding(X, 8, s, 8)),
        "_pairs": DIM_MATCHED_MENTION,
        "_family": {"SPEC_B_MENTION": spec_m, "SPEC_B": arms_m["SPEC_B"],
                    "COUNT+SPEC_B": arms_m["COUNT+SPEC_B"]},
    }

    # Unmatched gated floor verdicts (D10/protocol §3: every family against
    # every floor) -- the dim-matched rows attach the equal-dimension reading.
    GATED_ALPHA = (
        ("SPEC_B_ALPHA", "COUNT"), ("SPEC_B_ALPHA", "META"),
        ("SPEC_B_ALPHA", "H1_alpha"),
        ("COUNT+SPEC_B_ALPHA", "COUNT"),
    )
    GATED_MENTION = (
        ("SPEC_B_MENTION", "COUNT"), ("SPEC_B_MENTION", "META"),
        ("SPEC_B_MENTION", "H1_mention"),
        ("SPEC_B", "SHAN_B"), ("COUNT+SPEC_B", "COUNT"),
    )

    def evaluate(arms, noise, yv, gated):
        res = run_arms({k: v for k, v in arms.items()}, yv, seeds,
                       "hgb", args.quiet)
        res.update(run_arms({k: v for k, v in noise.items()
                             if not k.startswith("_")}, yv, seeds,
                            "hgb", args.quiet))
        comps = {}
        for fam, fl in gated:
            c = paired(res[fam]["auc"], res[fl]["auc"])
            comps[f"{fam}_vs_{fl}"] = {"family": fam, "floor_arm": fl,
                                       "kind": "gated", **c}
        for floor_parts, fam in noise["_pairs"]:
            fl = "+".join(floor_parts) + \
                f"+NOISE({res[fam]['n_features'] - res['+'.join(floor_parts)]['n_features']})"
            c = paired(res[fam]["auc"], res[fl]["auc"])
            c["verdict"] = interpret_dim_matched(c["mean_diff"],
                                                 c["significant"])
            comps[f"{fam}_vs_{fl}"] = {"family": fam, "floor_arm": fl,
                                       "kind": "dim_matched", **c}
        return res, comps

    print("\n[alpha sample] collapsed encoding (headline, D11)")
    res_a3, comp_a3 = evaluate(arms_alpha(True), noise_alpha(True), y,
                               GATED_ALPHA)
    print("[alpha sample] raw 4-symbol encoding (sensitivity axis)")
    res_a4, comp_a4 = evaluate(arms_alpha(False), noise_alpha(False), y,
                               GATED_ALPHA)
    print("[mention sample]")
    res_m, comp_m = evaluate(arms_m, noise_m, ym, GATED_MENTION)

    # sigma_config across the published encoding axis (D3)
    for key in comp_a3:
        deltas = [comp_a3[key]["mean_diff"], comp_a4[key]["mean_diff"]]
        for c in (comp_a3[key], comp_a4[key]):
            c["sigma_config"] = sigma_config(deltas)
    for key in comp_m:
        comp_m[key]["sigma_config"] = 0.0
        comp_m[key]["sigma_config_note"] = ("mention block is "
            "encoding-invariant (D11 collapses post types only); single "
            "encoding evaluated")

    # ---- H2 directional gates (pre-registered) -----------------------------
    def directional(values, yy, direction="less"):
        b, h = values[yy == 1], values[yy == 0]
        p = float(mannwhitneyu(b, h, alternative=direction).pvalue)
        return {"median_bot": float(np.median(b)),
                "median_human": float(np.median(h)),
                "delta_median_bot_minus_human": float(np.median(b)
                                                      - np.median(h)),
                "direction_tested": f"bots {direction}",
                "p_one_sided_mannwhitney": p,
                "pass": bool(p < 0.05),
                "auc_of_order_as_is": float(roc_auc_score(yy, values))}

    h2 = {}
    for enc, sa in (("collapsed", spec_a3), ("raw4", spec_a4)):
        h2[f"alphabet_H2_{enc}"] = directional(sa[:, 3], y, "less")
    for enc, sm in (("collapsed", spec_m),):
        h2[f"mention_H0_{enc}"] = directional(sm[:, 0], ym, "less")
    h2_pass = bool(h2["alphabet_H2_collapsed"]["pass"]
                   and h2["mention_H0_collapsed"]["pass"])
    signs_consistent = bool(
        h2["alphabet_H2_collapsed"]["delta_median_bot_minus_human"] < 0
        and h2["mention_H0_collapsed"]["delta_median_bot_minus_human"] < 0)
    h2["H2_gate_pass"] = h2_pass
    h2["signs_consistent"] = signs_consistent
    h2["sigma_config_alphabet_H2"] = sigma_config(
        [h2["alphabet_H2_collapsed"]["delta_median_bot_minus_human"],
         h2["alphabet_H2_raw4"]["delta_median_bot_minus_human"]])
    h2["sigma_config_mention_H0"] = 0.0
    h2["fallback"] = ("H2 passed; no P3 fallback needed" if h2_pass else
                      "H2 FAILED: docs/03-PHASES.md P3 rule -- return to P1 "
                      "and re-examine P8; inconsistent signs would mean the "
                      "volume/length detector is firing")

    # ---- G4 shuffle null ----------------------------------------------------
    rng = np.random.default_rng(SHUFFLE_SEED)
    spec_shuffle = np.zeros_like(spec_a3)
    for j, tweets in enumerate(seq):
        types = np.array([t for t, _ in tweets], dtype=np.int64)
        perm = rng.permutation(len(types))
        spec_shuffle[j] = alphabet_spectrum(types[perm], collapse=True)
    shuffle_max_diff = float(np.abs(spec_shuffle - spec_a3).max())
    auc_unshuf = [eval_arm(spec_a3, y, s)["auc"] for s in seeds]
    auc_shuf = [eval_arm(spec_shuffle, y, s)["auc"] for s in seeds]
    shuffle_null = {
        "definition": "per-account marginal-preserving permutation of the "
                      "post-type sequence (default_rng(42)), full pipeline "
                      "re-run",
        "separating_world": "a world where this null separates would require "
                            "order-dependent features; none exist -- the "
                            "block is a histogram of symbols, and the "
                            "spectrum is permutation-invariant (P6), so "
                            "expected silence is exact identity (S4.1: the "
                            "null is uninformative for marginal spectra and "
                            "is run to prove no order leakage)",
        "max_abs_spec_diff": shuffle_max_diff,
        "auc_mean_unshuffled": float(np.mean(auc_unshuf)),
        "auc_mean_shuffled": float(np.mean(auc_shuf)),
        "auc_delta_shuffled_minus_unshuffled": float(np.mean(auc_shuf)
                                                     - np.mean(auc_unshuf)),
        "silent": bool(shuffle_max_diff == 0.0),
    }

    # ---- figures (G1b) ------------------------------------------------------
    FIGURES.mkdir(parents=True, exist_ok=True)
    hi_bot = int(np.argmax(np.where(y == 1, counts[idx], -1)))
    hi_hum = int(np.argmax(np.where(y == 0, counts[idx], -1)))
    examples = {"bot": {"uid": str(ev.user_ids[idx[hi_bot]]),
                        "row": hi_bot, "label": ""},
                "human": {"uid": str(ev.user_ids[idx[hi_hum]]),
                          "row": hi_hum, "label": ""}}
    alpha_curves(spec_a3, y, FIGURES / "p3h_spec_b_alpha.png",
                 "SPEC_B_ALPHA — Rényi spectrum of the D11-collapsed "
                 "post-type distribution (mean ± 1 SD)",
                 examples=examples, types_seq=seq, quiet=args.quiet)
    # mention example lists rendered as text panels
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6))
    for c, (name, colour, row_g) in enumerate(
            (("bot", BOT_C, hi_bot), ("human", HUM_C, hi_hum))):
        ax = axes[0, c]
        mc = mentions[row_g]
        top = ", ".join(f"@{t}×{n}" for t, n in
                        sorted(mc.items(), key=lambda kv: -kv[1])[:12])
        ax.axis("off")
        ax.text(0.0, 0.95, f"{name} {examples[name]['uid']} — "
                f"{len(mc)} distinct targets, {sum(mc.values())} mentions",
                fontsize=10, color=colour, weight="bold", va="top")
        ax.text(0.0, 0.75, top if top else "(no mentions)", fontsize=8.5,
                wrap=True, va="top")
    axes[1, 0].axis("off"); axes[1, 1].axis("off")
    xs = np.arange(6)
    axc = fig.add_subplot(2, 1, 2)
    for mask, nm, c in ((ym == 0, "human", HUM_C), (ym == 1, "bot", BOT_C)):
        m, s = spec_m[mask].mean(axis=0), spec_m[mask].std(axis=0)
        axc.plot(xs, m, "o-", color=c, lw=2, label=nm)
        axc.fill_between(xs, m - s, m + s, color=c, alpha=0.18)
    axc.set_xticks(xs); axc.set_xticklabels(ORDERS)
    axc.set_ylabel("bits"); axc.legend()
    axc.set_title("SPEC_B_MENTION — Rényi spectrum of the mention-target "
                  "distribution (mean ± 1 SD)", fontsize=10.5)
    fig.suptitle("WP-H — mention-target objects (G1b) and alpha-curves",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "p3h_spec_b_mention.png", dpi=130)
    plt.close(fig)
    if not args.quiet:
        print(f"  [G1] rendered -> {FIGURES / 'p3h_spec_b_mention.png'}")

    # ---- JSON ---------------------------------------------------------------
    report = {
        "phase": "P3h-behaviour (WP-H)",
        "seeds": seeds,
        "classifier": "HistGradientBoostingClassifier(max_iter=200, "
                      "early_stopping=False), StratifiedKFold(5)",
        "feature_config": {"min_events": 5, "n_kept": int(len(idx)),
                           "age_reference": "2015-01-01 (botsage recipe)"},
        "D11_collapse": {
            "rule": "quote -> original for spectra (plan WP-H task 1)",
            "share_before": share_before,
            "share_after": share_after,
            "raw4_kept_for": "sensitivity axis here; DNA encoding in P4"},
        "capture": capture,
        "dim_matched": {
            "definition": "plan §8 D2; salts by declaration order",
            "salts": {"alpha": [0, 1, 2, 3], "mention": [4, 5, 6, 7, 8]},
            "interpretation_rules": "pre-registered plan WP-E task 3 [rev1]"},
        "fidelity": {"kept_index_matches_temporal": True,
                     "count_block_max_abs_diff": fid_count},
        "alpha_sample": {
            "n": int(len(idx)),
            "majority_baseline": float(max(y.mean(), 1 - y.mean())),
            "arms_collapsed": res_a3, "arms_raw4": res_a4,
            "comparisons_collapsed": comp_a3,
            "comparisons_raw4": comp_a4},
        "mention_sample": {
            "n": int(n_mention),
            "majority_baseline": float(max(ym.mean(), 1 - ym.mean())),
            "exclusions_per_class": n_excl,
            "arms": res_m, "comparisons": comp_m},
        "H2_directional": h2,
        "G4_shuffle_null": shuffle_null,
        "examples": examples,
    }
    OUT_JSON.write_text(json.dumps(report, indent=1))

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 92)
    print("WP-H — BEHAVIOURAL FRONT SPEC_B")
    print("=" * 92)
    for label, res in (("alpha (collapsed)", res_a3), ("mention", res_m)):
        print(f"\n[{label} sample]")
        print(f"{'arm':<24}{'dim':>4}{'AUC':>10}{'±SD':>8}{'TPR@1%':>9}"
              f"{'TPRfold':>9}")
        for k, v in res.items():
            a = np.array(v["auc"]); t = np.array(v["tpr01"])
            tf = np.array(v["tpr01_foldmean"])
            print(f"{k:<24}{v['n_features']:>4}{a.mean():>10.4f}"
                  f"{a.std():>8.4f}{t.mean():>9.3f}{tf.mean():>9.3f}")
    print("\n[gated + dim-matched verdicts]")
    for key, c in list(comp_a3.items()) + list(comp_m.items()):
        extra = f"  {c['verdict']}" if c["kind"] == "dim_matched" else ""
        print(f"  {key:<58}{c['mean_diff']:>+8.4f} {c['wins']:>6} "
              f"p={c['p']:.4f} σcfg={c['sigma_config']:.4f}{extra}")
    print("\n[H2 directional]")
    for k in ("alphabet_H2_collapsed", "mention_H0_collapsed"):
        v = h2[k]
        print(f"  {k:<28} med b/h {v['median_bot']:.4f}/"
              f"{v['median_human']:.4f}  p={v['p_one_sided_mannwhitney']:.4g}"
              f"  {'PASS' if v['pass'] else 'FAIL'}")
    print(f"  H2 gate: {'PASS' if h2_pass else 'FAIL'} "
          f"(signs consistent: {signs_consistent})")
    print(f"\n[G4] shuffle null: max |spec diff| {shuffle_max_diff:.2e}, "
          f"AUC delta {shuffle_null['auc_delta_shuffled_minus_unshuffled']:+.2e}"
          f" — {'SILENT (expected)' if shuffle_null['silent'] else 'NOT SILENT'}")
    print(f"json -> {OUT_JSON}")
    print("=" * 92)


if __name__ == "__main__":
    main()
