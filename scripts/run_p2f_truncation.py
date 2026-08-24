#!/usr/bin/env python
"""WP-F -- truncation controls on the real corpus (plan review D1, D2).

Two questions the WP-A probe left open for real accounts:

    1. Equal windows.  How much of SPEC_T's edge survives when every account
       is observed for the same length of time -- the first ``K`` days of its
       OWN activity (origin = account's own first event, boundary convention
       identical to the probe's ``s[s <= W*MS_PER_DAY]``, plan §8 D5)? The
       censoring amendment (bitacora 06) bounds every P2 "shape" reading by
       the probe ceiling until this arm reports; this script reports it.
       K swept over {7, 14, 30, 90}; **headline K = 30, pre-registered v1.0**;
       the others are sensitivity and are reported in full (G3).

    2. The API cap.    How much of COUNT's edge is Twitter's timeline cap?
       Cap = mode of the human upper tail (printed, interior-checked);
       ``frac_humans_at_cap`` reported; COUNT-alone and COUNT+SPEC_T
       recomputed excluding capped humans (sensitivity rows).

Arm set at each K -- the six families the plan names plus the two composites
its own figure spec needs (clause deltas require COUNT+SPEC_T; the
burstiness verdict requires COUNT+BURST):

    COUNT, BURST, SHAN, SPEC_T, SHAPE, COUNT+SHAPE, COUNT+SPEC_T,
    COUNT+BURST

with the WP-E standard dim-matched controls attached wherever
dim(family) > dim(floor): SHAN+NOISE(10), BURST+NOISE(9), COUNT+NOISE(12),
COUNT+BURST+NOISE(9) (plan §8 D2; interpretation rules binding). All
evaluation runs through ``renyiext.evaluate``.

An unwindowed reference block re-evaluates the same eight arms at the
headline config and is gated elementwise against BOTH committed artefacts --
``p2_temporal.json`` (six shared arms) and ``p2b_decomposition.json``
(SHAPE, COUNT+SHAPE) -- so this producer is provably the same pipeline that
produced every number already published (tolerance 1e-4, expect 0.0).

sigma_config (§8 D3): population SD (ddof=0) of each comparison's per-K mean
delta across the four published K values -- here the K sweep IS the published
config sweep. Reported beside every floor verdict.

Datasaurus: G1 -- windowed rasters of one real bot and one real human beside
the statistic curves; G2 -- capped-vs-uncapped compared on the same kept
sample, same seeds; G3 -- K swept, headline pre-declared, everything
reported; G4 -- the probe cells are the null world this arm is read against,
drawn as a band on the equal-window figure.

Usage:
    python scripts/run_p2f_truncation.py [--quiet] [--seeds N] [--fast]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import FIGURES, RESULTS, DATA_PROCESSED
from renyiext.events import load_events_cached
from renyiext.features import temporal_blocks, temporal_blocks_windowed, MS_PER_DAY
from renyiext.evaluate import (eval_arm, run_arms, paired, noise_padding,
                               interpret_dim_matched, sigma_config)

warnings.filterwarnings("ignore", category=UserWarning)

CACHE = DATA_PROCESSED / "cresci_events_d9.npz"
OUT_JSON = RESULTS / "p2c_truncation.json"
P2_JSON = RESULTS / "p2_temporal.json"
P2B_JSON = RESULTS / "p2b_decomposition.json"
PROBE_JSON = RESULTS / "p2c_probe.json"
GATE_TOL = 1e-4

WINDOWS_D = (7, 14, 30, 90)
HEADLINE_K = 30                    # pre-registered v1.0
HEADLINE_FEAT = {"n_bins": 24, "hi": 400 * MS_PER_DAY, "min_events": 5}

BLOCK_DIMS = {"COUNT": 1, "BURST": 3, "SHAN": 2, "SPEC_T": 12}

# (floor blocks, family blocks) -- fixed order fixes the D2 salt (arm_index).
DIM_MATCHED = (
    (("SHAN",),           ("SPEC_T",)),         # clause (ii) reading
    (("BURST",),          ("SPEC_T",)),         # spectrum vs burstiness floor
    (("COUNT",),          ("COUNT+SPEC_T",)),   # clause (i) reading
    (("COUNT", "BURST"),  ("COUNT+SPEC_T",)),   # burstiness verdict
)


def materialize(blocks, parts):
    return np.hstack([blocks[p] for p in parts])


def shape_arm(spec: np.ndarray) -> np.ndarray:
    """Level removed within each half: index 2 of each half is alpha = 1."""
    out = spec.copy()
    out[:, :6] -= spec[:, :6][:, [2]]
    out[:, 6:] -= spec[:, 6:][:, [2]]
    return out


def build_arm_matrices(blocks):
    """The eight families, as matrices, from one block dict."""
    spec = blocks["SPEC_T"]
    return {
        "COUNT":              blocks["COUNT"],
        "BURST":              blocks["BURST"],
        "SHAN":               blocks["SHAN"],
        "SPEC_T":             spec,
        "SHAPE":              shape_arm(spec),
        "COUNT+SHAPE":        np.hstack([blocks["COUNT"], shape_arm(spec)]),
        "COUNT+SPEC_T":       np.hstack([blocks["COUNT"], spec]),
        "COUNT+BURST":        np.hstack([blocks["COUNT"], blocks["BURST"]]),
    }


def noise_specs(blocks):
    """D2 padded-floor arms + their G3 echo. k is formula-determined."""
    specs, echo = {}, {}
    fam_cache = build_arm_matrices(blocks)
    for i, (floor_parts, fam_parts) in enumerate(DIM_MATCHED):
        X_floor = materialize(blocks, floor_parts)
        k = int(fam_cache["+".join(fam_parts)].shape[1] - X_floor.shape[1])
        name = "+".join(floor_parts) + f"+NOISE({k})"
        specs[name] = (lambda s, Xf=X_floor, k=k, i=i:
                       noise_padding(Xf, k, s, i))
        echo[name] = {"floor_blocks": list(floor_parts),
                      "family_blocks": list(fam_parts),
                      "k": k, "arm_index": i,
                      "rng": "default_rng(seed*1000 + arm_index)",
                      "k_rule": "dim(family) - dim(floor)"}
    return specs, echo


COMPARISONS = (
    # key            family arm        floor arm        kind
    ("clause_i",     "COUNT+SPEC_T",   "COUNT",         "gated"),
    ("clause_ii",    "SPEC_T",         "SHAN",          "gated"),
    ("burstiness",   "COUNT+SPEC_T",   "COUNT+BURST",   "gated"),
    ("count_shape",  "COUNT+SHAPE",    "COUNT",         "gated"),
    ("clause_i_matched",    "COUNT+SPEC_T", "COUNT+NOISE(12)",       "matched"),
    ("clause_ii_matched",   "SPEC_T",       "SHAN+NOISE(10)",        "matched"),
    ("burstiness_matched",  "COUNT+SPEC_T", "COUNT+BURST+NOISE(9)",  "matched"),
    ("count_shape_matched", "COUNT+SHAPE",  "COUNT+NOISE(12)",       "matched"),
    ("shape_vs_burst_matched", "SHAPE",     "BURST+NOISE(9)",        "matched"),
)


def evaluate_window(blocks, y, seeds, quiet):
    arms = build_arm_matrices(blocks)
    res = run_arms(arms, y, seeds, "hgb", quiet)
    ns, echo = noise_specs(blocks)
    res.update(run_arms(ns, y, seeds, "hgb", quiet))
    comps = {}
    for key, fam, fl, kind in COMPARISONS:
        c = paired(res[fam]["auc"], res[fl]["auc"])
        if kind == "matched":
            c["verdict"] = interpret_dim_matched(c["mean_diff"],
                                                 c["significant"])
        comps[key] = {"family": fam, "floor_arm": fl, "kind": kind, **c}
    return res, comps, echo


def find_cap(ev):
    """API-cap identification: mode of the human upper tail (plan WP-F task 3).

    Upper tail = event counts >= the human median. Mode ties resolve to the
    smallest tied count (np.argmax over ascending unique values); the top of
    the modal table is reported either way so the choice is auditable.
    """
    counts = ev.counts()
    hum = ev.labels == 0
    hc = counts[hum]
    thr = float(np.median(hc))
    tail = hc[hc >= thr]
    vals, cnts = np.unique(tail, return_counts=True)
    order = np.argsort(-cnts, kind="stable")       # most frequent first
    top = [{"events_per_account": int(vals[i]), "n_humans": int(cnts[i])}
           for i in order[:5]]
    cap = int(vals[order[0]])
    n_tied = int((cnts == cnts[order[0]]).sum())
    interior = {"human_min": int(hc.min()), "human_max": int(hc.max()),
                "pass": bool(hc.min() < cap < hc.max())}
    return {
        "definition": "mode of human event counts within the upper tail "
                      "(counts >= human median); ties -> smallest count",
        "tail_threshold_median": thr,
        "top_modal_values": top,
        "mode_ties_at_top": n_tied,
        "cap_events": cap,
        "interior_check": interior,
        "frac_humans_at_cap": float((hc == cap).mean()),
        "frac_humans_ge_cap": float((hc >= cap).mean()),
        "frac_humans_ge_095cap": float((hc >= 0.95 * cap).mean()),
        "n_humans": int(hum.sum()),
        "bots_ge_cap": int((counts[~hum] >= cap).sum()),
        "n_bots": int((~hum).sum()),
    }


def fidelity_gate_unwindowed(ref_res):
    """Elementwise gate of the unwindowed reference block against both
    committed artefacts (G2). Expect exact equality; tolerance 1e-4."""
    worst, where = 0.0, ""
    sources = []
    if P2_JSON.exists():
        sources.append((P2_JSON.name,
                        json.loads(P2_JSON.read_text())["arms"],
                        ("COUNT", "BURST", "SHAN", "SPEC_T", "COUNT+BURST",
                         "COUNT+SPEC_T")))
    if P2B_JSON.exists():
        sources.append((P2B_JSON.name,
                        json.loads(P2B_JSON.read_text())["arms"],
                        ("SHAPE", "COUNT+SHAPE", "COUNT+BURST",
                         "COUNT+SPEC_T")))
    n_arrays = 0
    for src_name, src, arms in sources:
        for arm in arms:
            if arm not in ref_res or arm not in src:
                continue
            for metric in ("auc", "tpr01", "macro_f1", "accuracy"):
                if metric not in src[arm] or metric not in ref_res[arm]:
                    continue
                got = np.array(ref_res[arm][metric])
                want = np.array(src[arm][metric])
                n_arrays += 1
                d = float(np.abs(got - want).max())
                if d > worst:
                    worst, where = d, f"{src_name}:{arm}.{metric}"
    if worst > GATE_TOL:
        raise AssertionError(
            f"fidelity gate FAILED: max |diff| {worst:.3e} > {GATE_TOL:g} at "
            f"{where} -- pipeline drift; do not quote these numbers")
    print(f"  [gate] unwindowed reference vs committed artefacts: PASS "
          f"(max |diff| {worst:.2e}, {n_arrays} array comparisons)")
    return {"compared_arrays": n_arrays, "max_abs_diff": worst,
            "tol": GATE_TOL, "pass": bool(worst <= GATE_TOL)}


def pick_raster_accounts(ev, kept_index):
    """Deterministic examples (G1b): per class, the kept-at-headline account
    with the most corpus events; ties -> lowest index."""
    idx = np.asarray(kept_index)
    counts = ev.counts()[idx]
    out = {}
    for cls, name in ((1, "bot"), (0, "human")):
        mask = ev.labels[idx] == cls
        cand = np.where(mask)[0]
        best = cand[np.argmax(counts[cand])]     # argmax -> first max
        gid = int(idx[best])
        ts, _ = ev.events_of(gid)
        out[name] = {"user_id": str(ev.user_ids[gid]),
                     "global_index": gid,
                     "corpus_events": int(counts[best]),
                     "ts_ms": ts.tolist()}
    return out


def render_equal_window(windows, rasters, unwin_res, probe_band, quiet):
    FIGURES.mkdir(parents=True, exist_ok=True)
    ks = list(WINDOWS_D)

    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], hspace=0.42,
                          wspace=0.24)
    gs_left = gs[:, 0].subgridspec(2, 1, hspace=0.38)

    day = MS_PER_DAY
    ax_rasters = {}
    for row, (name, colour) in enumerate((("bot", "#e76f51"),
                                          ("human", "#2a9d8f"))):
        axr = fig.add_subplot(gs_left[row])
        ax_rasters[name] = axr
        info = rasters[name]
        rel = (np.array(info["ts_ms"]) - info["ts_ms"][0]) / day
        axr.plot(rel, np.arange(len(rel)), "|", ms=3.5, color=colour,
                 alpha=0.75)
        axr.axvline(HEADLINE_K, color="k", ls=":", lw=1.6)
        axr.set_xlim(-1, min(rel.max() * 1.02, 400))
        axr.set_ylabel("event #")
        n_in = int((rel <= HEADLINE_K).sum())
        axr.set_title(f"{name} {info['user_id']} — {info['corpus_events']} "
                      f"events, {n_in} inside K={HEADLINE_K}d",
                      loc="left", fontsize=9.5)
    ax_rasters["human"].set_xlabel("days since account's own first event")

    axd = fig.add_subplot(gs[0, 1])
    styles = {"clause_i": ("CS − COUNT", "#3b6ea5", "o"),
              "clause_ii": ("SPEC_T − SHAN", "#9b5de5", "s"),
              "burstiness": ("CS − CB", "#e76f51", "^"),
              "count_shape": ("C+SHAPE − C", "#1baf7a", "D")}
    for key, (lab, col, mk) in styles.items():
        vals = [windows[str(k)]["comparisons"][key]["mean_diff"] for k in ks]
        axd.plot(ks, vals, "-", marker=mk, color=col, lw=1.8, label=lab)
    mstyles = {"clause_i_matched": ("CS − C+N12", "#3b6ea5", "o"),
               "clause_ii_matched": ("SPEC_T − SHAN+N10", "#9b5de5", "s"),
               "burstiness_matched": ("CS − CB+N9", "#e76f51", "^")}
    for key, (lab, col, mk) in mstyles.items():
        vals = [windows[str(k)]["comparisons"][key]["mean_diff"] for k in ks]
        axd.plot(ks, vals, "--", marker=mk, mfc="none", color=col, lw=1.2,
                 alpha=0.85, label=lab)
    axd.axhline(0.02, color="crimson", ls="--", lw=1.6,
                label="registered floor 0.02")
    axd.axhline(0.0, color="grey", lw=0.8)
    axd.axvspan(HEADLINE_K - 2, HEADLINE_K + 2, color="gold", alpha=0.18)
    axd.set_xticks(ks)
    axd.set_xlabel("equal-window K (days)")
    axd.set_ylabel("AUC delta (seed-mean)")
    axd.set_title("Clause deltas vs K — solid: registered pairs; dashed: "
                  "dim-matched (D2)", fontsize=10)
    axd.legend(fontsize=7.6, ncol=2, loc="lower right")

    axa = fig.add_subplot(gs[1, 1])
    plotted = []
    for arm, lab, col in (("COUNT+SPEC_T", "COUNT+SPEC_T", "#3b6ea5"),
                          ("SPEC_T", "SPEC_T", "#9b5de5"),
                          ("COUNT", "COUNT", "#e76f51")):
        vals = [float(np.mean(windows[str(k)]["arms"][arm]["auc"])) for k in ks]
        plotted += vals
        axa.plot(ks, vals, "-o", ms=4, color=col, lw=1.8, label=arm)
        full_v = float(np.mean(unwin_res[arm]["auc"]))
        plotted.append(full_v)
        axa.plot([135], [full_v], "*", ms=11, color=col)
    if probe_band is not None:
        axa.axhspan(probe_band["lo"], probe_band["hi"], color="#e76f51",
                    alpha=0.13,
                    label=f"WP-A censoring ceiling [{probe_band['lo']:.3f}, "
                          f"{probe_band['hi']:.3f}]")
        axa.axhline(0.85, color="crimson", ls=":", lw=1.2)
    axa.set_xticks(list(ks) + [135])
    axa.set_xticklabels([str(k) for k in ks] + ["full"])
    lo = max(0.5, min(plotted) - 0.03)
    axa.set_ylim(lo, 1.005)
    axa.set_xlabel("equal-window K (days); ★ = unwindowed headline")
    axa.set_ylabel("AUC")
    axa.set_title("Separation vs K, read against the same-generator "
                  "truncation null (G4)", fontsize=10)
    axa.legend(fontsize=7.8, loc="center right")

    fig.suptitle("WP-F — equal-window truncation controls on Cresci-2015 "
                 f"(headline K = {HEADLINE_K} d, shaded)", fontsize=12,
                 fontweight="bold")
    fig.savefig(FIGURES / "p2f_equal_window.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    if not quiet:
        print(f"  [G1/G4] rendered -> {FIGURES / 'p2f_equal_window.png'}")


def render_api_cap(ev, cap_info, quiet):
    FIGURES.mkdir(parents=True, exist_ok=True)
    counts = ev.counts()
    hum = ev.labels == 0
    cap = cap_info["cap_events"]

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    bins = np.logspace(0, np.log10(max(counts.max(), 10)) + 0.05, 70)
    ax.hist(counts[hum], bins=bins, color="#2a9d8f", alpha=0.65,
            label=f"human (n={int(hum.sum())})")
    ax.hist(counts[~hum], bins=bins, color="#e76f51", alpha=0.55,
            label=f"bot (n={int((~hum).sum())})")
    ax.axvline(cap, color="black", ls="--", lw=2)
    ax.text(cap * 1.08, ax.get_ylim()[1] * 0.86,
            f"API cap = {cap}\n"
            f"{100*cap_info['frac_humans_at_cap']:.1f}% of humans at cap\n"
            f"{cap_info['bots_ge_cap']} bots ≥ cap",
            fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="lemonchiffon", alpha=0.85))
    ax.set_xscale("log")
    ax.set_xlabel("events per account (log scale)")
    ax.set_ylabel("accounts")
    ax.set_title("Events-per-account histogram with the collection cap — "
                 "both classes on one axis (G2)",
                 loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "p2f_api_cap.png", dpi=130)
    plt.close(fig)
    if not quiet:
        print(f"  [G1/G2] rendered -> {FIGURES / 'p2f_api_cap.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = list(range(42, 42 + args.seeds))

    print("loading events ...", flush=True)
    ev = load_events_cached(CACHE)
    print(" ", ev, flush=True)

    # ---- API cap (task 3) -------------------------------------------------
    cap_info = find_cap(ev)
    if not args.quiet:
        print(f"\n[cap] {cap_info['definition']}")
        print(f"  cap = {cap_info['cap_events']} events "
              f"(interior check pass={cap_info['interior_check']['pass']}); "
              f"humans at cap {100*cap_info['frac_humans_at_cap']:.2f}%, "
              f">= cap {100*cap_info['frac_humans_ge_cap']:.2f}%")

    assert cap_info["interior_check"]["pass"], \
        "cap not interior to the human count range -- definition broken"

    # ---- unwindowed reference + fidelity gate -----------------------------
    print("\n[reference] unwindowed headline blocks (also the cap-sensitivity "
          "sample)")
    bl_full = temporal_blocks(ev, **HEADLINE_FEAT)
    y_full = bl_full["y"]
    arms_full = build_arm_matrices(bl_full["blocks"])
    unwin_res = run_arms(arms_full, y_full, seeds, "hgb", args.quiet)
    fid = fidelity_gate_unwindowed(unwin_res)
    majority_full = float(max(y_full.mean(), 1 - y_full.mean()))

    # ---- cap sensitivity ----------------------------------------------------
    print("[cap sensitivity] excluding humans at >= cap")
    counts_all = ev.counts()
    idx_full = bl_full["index"]
    capped_rows = (ev.labels[idx_full] == 0) & (counts_all[idx_full]
                                                >= cap_info["cap_events"])
    keep_mask = ~capped_rows
    y_sens = y_full[keep_mask]

    def _series_incl(Xall):
        """Full kept sample -- the 'including capped' arm."""
        aucs = [eval_arm(Xall, y_full, s)["auc"] for s in seeds]
        return {"auc_mean": float(np.mean(aucs)),
                "auc_sd": float(np.std(aucs)), "auc_per_seed": aucs}

    def _series_excl(Xall):
        """Capped humans removed -- the 'excluding capped' arm."""
        X = Xall[keep_mask]
        aucs = [eval_arm(X, y_sens, s)["auc"] for s in seeds]
        return {"auc_mean": float(np.mean(aucs)),
                "auc_sd": float(np.std(aucs)), "auc_per_seed": aucs}

    X_c = bl_full["blocks"]["COUNT"]
    X_cs = np.hstack([bl_full["blocks"]["COUNT"], bl_full["blocks"]["SPEC_T"]])
    sens = {
        "sample": "headline-config kept sample (no window); humans with "
                  "corpus-wide events >= cap removed",
        "n_before": int(len(y_full)), "n_after": int(keep_mask.sum()),
        "humans_removed": int(capped_rows.sum()),
        "majority_baseline_before": majority_full,
        "majority_baseline_after": float(max(y_sens.mean(), 1 - y_sens.mean())),
        "COUNT_including_capped": _series_incl(X_c),
        "COUNT_excluding_capped": _series_excl(X_c),
        "COUNT+SPEC_T_including_capped": _series_incl(X_cs),
        "COUNT+SPEC_T_excluding_capped": _series_excl(X_cs),
        "exploratory": True,
        "note": "populations differ between including/excluding rows; paired "
                "stats deliberately omitted (descriptive sensitivity only)",
    }
    sens["delta_COUNT_excl_minus_incl"] = float(
        sens["COUNT_excluding_capped"]["auc_mean"]
        - sens["COUNT_including_capped"]["auc_mean"])
    sens["delta_CS_excl_minus_incl"] = float(
        sens["COUNT+SPEC_T_excluding_capped"]["auc_mean"]
        - sens["COUNT+SPEC_T_including_capped"]["auc_mean"])

    # Post-hoc wide sensitivity: the registered at-cap rule removes only the
    # accounts sitting exactly at/above the modal value, but the G1 render
    # shows the cap phenomenon is a SPIKE REGION (the pre-cap log-bin holds
    # ~20x more humans than the exact mode). This row excludes the whole
    # >= 0.95*cap region. Explicitly post-hoc; the registered row above stays
    # the headline answer to review D2.
    wide_mask = ~((ev.labels[idx_full] == 0)
                  & (counts_all[idx_full] >= 0.95 * cap_info["cap_events"]))
    y_wide = y_full[wide_mask]

    def _series_wide(Xall):
        X = Xall[wide_mask]
        aucs = [eval_arm(X, y_wide, s)["auc"] for s in seeds]
        return {"auc_mean": float(np.mean(aucs)),
                "auc_sd": float(np.std(aucs)), "auc_per_seed": aucs}

    wide_count = _series_wide(X_c)
    wide_cs = _series_wide(X_cs)
    sens["post_hoc_wide"] = {
        "rule": "humans with corpus-wide events >= 0.95 * cap removed",
        "threshold_events": float(0.95 * cap_info["cap_events"]),
        "humans_removed": int((~wide_mask).sum()),
        "n_after": int(wide_mask.sum()),
        "majority_baseline_after": float(max(y_wide.mean(), 1 - y_wide.mean())),
        "COUNT_excluding_wide": wide_count,
        "COUNT+SPEC_T_excluding_wide": wide_cs,
        "delta_COUNT_wide_minus_incl": float(
            wide_count["auc_mean"]
            - sens["COUNT_including_capped"]["auc_mean"]),
        "delta_CS_wide_minus_incl": float(
            wide_cs["auc_mean"]
            - sens["COUNT+SPEC_T_including_capped"]["auc_mean"]),
        "post_hoc": True,
    }
    del X_c, X_cs

    # ---- equal-window sweep (task 2) ---------------------------------------
    windows = {}
    echo = None
    bl_head = None
    for K in WINDOWS_D:
        print(f"\n[K={K}d] windowed blocks", flush=True)
        blw = temporal_blocks_windowed(ev, K, **HEADLINE_FEAT)
        yw = blw["y"]
        if K == HEADLINE_K:
            bl_head = blw
        if not args.quiet:
            print(f"  kept {blw['n_kept']} (bot {blw['n_kept_bot']}, human "
                  f"{blw['n_kept_human']}); excluded {blw['n_excluded']} "
                  f"(bot {blw['n_excluded_bot']}, human "
                  f"{blw['n_excluded_human']}); lost to window alone: bot "
                  f"{blw['n_lost_to_window_bot']}, human "
                  f"{blw['n_lost_to_window_human']}")
        res, comps, echo = evaluate_window(blw["blocks"], yw, seeds, args.quiet)
        windows[str(K)] = {
            "window_days": K,
            "n_kept": blw["n_kept"], "n_kept_bot": blw["n_kept_bot"],
            "n_kept_human": blw["n_kept_human"],
            "n_excluded": blw["n_excluded"],
            "n_excluded_bot": blw["n_excluded_bot"],
            "n_excluded_human": blw["n_excluded_human"],
            "n_lost_to_window_bot": blw["n_lost_to_window_bot"],
            "n_lost_to_window_human": blw["n_lost_to_window_human"],
            "majority_baseline": float(max(yw.mean(), 1 - yw.mean())),
            "arms": res, "comparisons": comps,
        }

    # sigma_config (D3): population SD of each delta across the published K
    # sweep -- here the K axis IS the config sweep.
    for comp_key, _, _, kind in COMPARISONS:
        per_k = [windows[str(k)]["comparisons"][comp_key]["mean_diff"]
                 for k in WINDOWS_D]
        s = sigma_config(per_k)
        for k in WINDOWS_D:
            windows[str(k)]["comparisons"][comp_key]["sigma_config"] = s

    # ---- probe band (G4) ----------------------------------------------------
    probe_band = None
    probe_note = "probe json absent -- band skipped"
    if PROBE_JSON.exists():
        pr = json.loads(PROBE_JSON.read_text())
        cells = {key: v["COUNT+SPEC_T"]["auc_mean"]
                 for key, v in pr["cells"].items()}
        near = [v for key, v in cells.items()
                if any(f"W={w}d" in key for w in (30, 90))]
        allv = list(cells.values())
        probe_band = {"lo": float(min(near)), "hi": float(max(near)),
                      "cells_used": [key for key in cells
                                     if any(f"W={w}d" in key
                                            for w in (30, 90))],
                      "all_cells_min": float(min(allv)),
                      "all_cells_max": float(max(allv))}
        probe_note = ("band = min/max COUNT+SPEC_T over the probe's W=30,90 "
                      "cells (three generators each); all-nine-cell range "
                      "[{:.4f}, {:.4f}]".format(probe_band["all_cells_min"],
                                                probe_band["all_cells_max"]))
    if not args.quiet:
        print(f"\n[G4] probe band: {probe_note}")

    # ---- rasters + figures --------------------------------------------------
    rasters = pick_raster_accounts(ev, bl_head["index"])
    render_equal_window(windows, rasters, unwin_res, probe_band, args.quiet)
    render_api_cap(ev, cap_info, args.quiet)

    report = {
        "phase": "P2f-truncation (WP-F)",
        "definition": "plan WP-F tasks 1-3; window origin = each account's "
                      "own first event; boundary <= W days (probe-identical, "
                      "§8 D5); headline K = 30 pre-registered v1.0",
        "seeds": seeds,
        "feature_config": {"n_bins": HEADLINE_FEAT["n_bins"],
                           "hi_days": HEADLINE_FEAT["hi"] / MS_PER_DAY,
                           "min_events": HEADLINE_FEAT["min_events"]},
        "classifier": "HistGradientBoostingClassifier(max_iter=200, "
                      "early_stopping=False), StratifiedKFold(5)",
        "dim_matched": {
            "definition": "plan §8 D2; interpretation rules pre-registered "
                          "in plan WP-E task 3 [rev1], binding",
            "knobs": echo,
            "sigma_config_definition":
                "population SD (ddof=0) of the per-K mean delta across the "
                "published K sweep (D3 adapted to this WP's sweep axis)"},
        "unwindowed_reference": {
            "purpose": "fidelity anchor to p2_temporal/p2b_decomposition and "
                       "'full' point on the figure",
            "n": int(len(y_full)),
            "majority_baseline": majority_full,
            "arms": unwin_res,
            "fidelity_gate": fid},
        "api_cap": {**cap_info, "sensitivity": sens},
        "probe_reference": {"file": str(PROBE_JSON.name),
                            "trigger_threshold": 0.85,
                            "band": probe_band, "note": probe_note},
        "raster_accounts": {k: {kk: vv for kk, vv in v.items() if kk != "ts_ms"}
                            for k, v in rasters.items()},
        "windows": windows,
    }
    OUT_JSON.write_text(json.dumps(report, indent=1))

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 96)
    print("WP-F — TRUNCATION CONTROLS ON CORPUS (equal windows + API cap)")
    print("=" * 96)
    print(f"{'K':>5} {'n':>6} {'maj':>6} | {'CS−C':>7} {'S−SHAN':>7} "
          f"{'CS−CB':>7} {'C+S−C':>7} | {'CS':>7} {'SPEC_T':>7} {'COUNT':>7}")
    for k in WINDOWS_D:
        w = windows[str(k)]
        c = w["comparisons"]
        a = lambda arm: float(np.mean(w["arms"][arm]["auc"]))
        print(f"{k:>5} {w['n_kept']:>6} {w['majority_baseline']:>6.3f} | "
              f"{c['clause_i']['mean_diff']:>+7.4f} "
              f"{c['clause_ii']['mean_diff']:>+7.4f} "
              f"{c['burstiness']['mean_diff']:>+7.4f} "
              f"{c['count_shape']['mean_diff']:>+7.4f} | "
              f"{a('COUNT+SPEC_T'):>7.4f} {a('SPEC_T'):>7.4f} "
              f"{a('COUNT'):>7.4f}")

    print(f"\n{'dim-matched verdicts at K (headline)':<52}"
          f"{'delta':>9}{'wins':>8}{'p':>9}{'σ_cfg':>9}  verdict")
    ch = windows[str(HEADLINE_K)]["comparisons"]
    for key, _, _, kind in COMPARISONS:
        if kind != "matched":
            continue
        c = ch[key]
        nm = f"{c['family']}  vs  {c['floor_arm']}"
        print(f"{nm:<52}{c['mean_diff']:>+9.4f}{c['wins']:>8}{c['p']:>9.4f}"
              f"{c['sigma_config']:>9.4f}  {c['verdict']}")

    print(f"\ncap = {cap_info['cap_events']} events "
          f"(mode of human upper tail; interior pass="
          f"{cap_info['interior_check']['pass']}); "
          f"humans at cap {100*cap_info['frac_humans_at_cap']:.2f}%")
    print(f"cap sensitivity: humans removed {sens['humans_removed']} "
          f"(n {sens['n_before']} -> {sens['n_after']}); "
          f"COUNT {sens['COUNT_including_capped']['auc_mean']:.4f} -> "
          f"{sens['COUNT_excluding_capped']['auc_mean']:.4f} "
          f"(Δ{sens['delta_COUNT_excl_minus_incl']:+.4f}); "
          f"CS {sens['COUNT+SPEC_T_including_capped']['auc_mean']:.4f} -> "
          f"{sens['COUNT+SPEC_T_excluding_capped']['auc_mean']:.4f} "
          f"(Δ{sens['delta_CS_excl_minus_incl']:+.4f})")
    pw = sens["post_hoc_wide"]
    print(f"cap sensitivity (post-hoc wide, >=0.95·cap): humans removed "
          f"{pw['humans_removed']} (n -> {pw['n_after']}); "
          f"COUNT -> {pw['COUNT_excluding_wide']['auc_mean']:.4f} "
          f"(Δ{pw['delta_COUNT_wide_minus_incl']:+.4f}); "
          f"CS -> {pw['COUNT+SPEC_T_excluding_wide']['auc_mean']:.4f} "
          f"(Δ{pw['delta_CS_wide_minus_incl']:+.4f})")
    print(f"\nunwindowed fidelity gate vs committed artefacts: "
          f"{'PASS' if fid['pass'] else 'FAIL'} "
          f"(max |diff| {fid['max_abs_diff']:.2e} over {fid['compared_arrays']} arrays)")
    print(f"json -> {OUT_JSON}")
    print("=" * 96)


if __name__ == "__main__":
    main()
