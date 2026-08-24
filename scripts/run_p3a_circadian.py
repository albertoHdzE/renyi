#!/usr/bin/env python
"""WP-G -- circadian adjudication (HANDOFF open item 1; review D5).

The anomaly (``results/p2_temporal.json``, ``partial_correlations``): every
circadian order's raw correlation with the bot label is NEGATIVE (bots post in
fewer distinct hours) but flips POSITIVE given count (+0.21..+0.27); the
inter-arrival half shows no such reversal. Per the pre-registered rule (plan
WP-G task 2) this script decides:

    kept_suppression_explained  stable matched-count difference agreeing with
                                the conditioned sign -> circadian orders KEPT,
                                mechanism cited;
    dropped_from_SPEC_B         reversal disappears under matching or flips
                                with the timezone offset -> circadian six
                                DROPPED from SPEC_B; P2's cd half keeps a
                                caveat;
    ambiguous_drop_exploratory  matched-class TV < 0.05 at EVERY offset ->
                                dropped, exploratory-only.

Mechanics: hour-of-day histograms (the objects, G1) bot vs human in UTC and
local +1 h / +2 h (CET/CEST both plausible for an Italian corpus -- the
offset is a knob, G3), overall and within count-caliper strata (plan §8 D4:
deciles of log1p(count) on the headline kept sample, n = 4,770; >= 20 per
class per stratum, adjacent deciles merged and said so, sizes enumerated --
G2). Per stratum and offset: class histograms, total-variation distance,
per-order class means and bot-minus-human deltas, raw and given-count partial
correlations. Matched strata remove the volume world (G4).

Operationalisation (recorded in the JSON): the ambiguity predicate is
evaluated first (it is the no-signal case); "agreeing with the conditioned
sign" = matched delta positive for all six orders (the conditioned
coefficients are all positive); "stable" = same sign across all three offsets
and >= half the strata positive per order/offset.

Fidelity gates tie this producer to the committed record: recomputed
circadian spectra at offset 0 must equal SPEC_T's cd half elementwise, and
the overall raw/given-count partials at offset 0 must reproduce
``p2_temporal.json``'s ``partial_correlations`` (tolerance 1e-9).

No classifier, no seeds -- descriptive statistics; two runs byte-identical
(S2.6).

Usage:
    python scripts/run_p3a_circadian.py [--quiet]
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
from renyiext.features import temporal_blocks, MS_PER_DAY
from renyiext.spectrum import (spectrum, counts_to_probabilities,
                               spectrum_labels, SPECTRUM_ALPHAS)
from renyiext.evaluate import partial_corr

warnings.filterwarnings("ignore", category=UserWarning)

CACHE = DATA_PROCESSED / "cresci_events_d9.npz"
P2_JSON = RESULTS / "p2_temporal.json"
OUT_JSON = RESULTS / "p3a_circadian.json"
FID_TOL = 1e-9

OFFSETS_H = (0, 1, 2)              # UTC, CET, CEST -- the G3 knob
HEADLINE = {"n_bins": 24, "hi": 400 * MS_PER_DAY, "min_events": 5}
MIN_PER_CLASS = 20                 # D4 stratum validity
TV_AMBIGUOUS = 0.05                # pre-registered ambiguity threshold
BOT_C, HUM_C = "#e76f51", "#2a9d8f"

ORDERS = spectrum_labels()         # H_0, H_0.5, H_1, H_2, H_4, H_inf


def hour_histogram(ts: np.ndarray, offset_h: int) -> np.ndarray:
    """24-bin hour-of-day counts, shifted by ``offset_h`` local hours."""
    hours = ((ts + offset_h * 3_600_000) % MS_PER_DAY) // 3_600_000
    return np.bincount(hours.astype(np.int64), minlength=24).astype(np.float64)


def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Total variation between two count vectors read as distributions."""
    pa = a / a.sum() if a.sum() else a
    pb = b / b.sum() if b.sum() else b
    return float(0.5 * np.abs(pa - pb).sum())


def build_strata(log_counts: np.ndarray, y: np.ndarray) -> list[dict]:
    """D4 count-caliper strata: deciles of log1p(count); adjacent deciles
    merge left-to-right while either class holds < 20 accounts."""
    edges = np.quantile(log_counts, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    merged, p_lo = [], None
    for i in range(10):
        lo = edges[i] if p_lo is None else p_lo
        hi = edges[i + 1]
        m = (log_counts >= lo) & (log_counts < hi)
        n_bot, n_hum = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if n_bot >= MIN_PER_CLASS and n_hum >= MIN_PER_CLASS:
            merged.append({"lo": float(lo), "hi": float(hi),
                           "n_bot": n_bot, "n_hum": n_hum})
            p_lo = None
        else:
            p_lo = lo
    if p_lo is not None:               # trailing remnant that never validated
        m = log_counts >= p_lo
        merged.append({"lo": float(p_lo), "hi": float("inf"),
                       "n_bot": int((y[m] == 1).sum()),
                       "n_hum": int((y[m] == 0).sum()),
                       "invalid": "fewer than 20 per class after merging; "
                                  "excluded from matched analyses"})
    return merged


def render(offset: int, overall: dict, rows: list[dict], fname: str,
           quiet: bool):
    """G1: the objects -- overall + per-stratum hour histograms, overlaid."""
    hours = np.arange(24)
    n_strata = len(rows)
    ncols = min(4, max(1, n_strata))
    nrows = 1 + int(np.ceil(n_strata / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 2.9 * nrows),
                             squeeze=False)
    ax = axes[0, 0]
    hb = np.array(overall["hist_bot"]); hh = np.array(overall["hist_human"])
    ax.bar(hours - 0.2, hb / hb.sum(), width=0.4, color=BOT_C, label="bot")
    ax.bar(hours + 0.2, hh / hh.sum(), width=0.4, color=HUM_C, label="human")
    ax.set_title(f"overall — TV {overall['tv']:.3f}", fontsize=10)
    ax.set_ylabel("share of events")
    ax.legend(fontsize=8)
    for k, r in enumerate(rows):
        a = axes[1 + k // ncols, k % ncols]
        b, h = r["_hist_bot"], r["_hist_human"]
        a.bar(hours - 0.2, b / b.sum(), width=0.4, color=BOT_C)
        a.bar(hours + 0.2, h / h.sum(), width=0.4, color=HUM_C)
        a.set_title(f"stratum {k} — TV {r['tv']:.3f} "
                    f"(b{r['n_bot']}/h{r['n_hum']})", fontsize=9)
    for a in axes[-1]:
        a.set_xlabel("hour of day" + (f" (+{offset} h local)" if offset else
                                      " (UTC)"), fontsize=9)
    for row in axes:
        for a in row:
            a.set_xticks([0, 6, 12, 18]); a.tick_params(labelsize=7)
    for k in range(n_strata + 1, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    title = (f"WP-G — hour-of-day, bot vs human, "
             f"{'UTC' if offset == 0 else f'local +{offset} h'} "
             f"(objects G1; strata = count deciles, D4)")
    fig.suptitle(title, fontsize = 12, fontweight = "bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIGURES / fname
    fig.savefig(path, dpi=130)
    plt.close(fig)
    if not quiet:
        print(f"  [G1] rendered -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("loading events ...", flush=True)
    ev = load_events_cached(CACHE)
    print(" ", ev, flush=True)

    print("[headline] blocks (n_bins=24, hi=400d, min_events=5)")
    bl = temporal_blocks(ev, **HEADLINE)
    y = bl["y"]
    idx = bl["index"]
    log_counts = bl["blocks"]["COUNT"][:, 0]
    spec_cd = bl["blocks"]["SPEC_T"][:, 6:12]

    hists, spectra = {}, {}
    for off in OFFSETS_H:
        hs, ss = [], []
        for i in idx:
            ts, _ = ev.events_of(int(i))
            h = hour_histogram(ts, off)
            hs.append(h)
            ss.append(spectrum(counts_to_probabilities(h), SPECTRUM_ALPHAS))
        hists[off] = np.array(hs)
        spectra[off] = np.array(ss)

    fid_spec = float(np.abs(spectra[0] - spec_cd).max())
    print(f"[gate] circadian spectra @UTC vs SPEC_T cd half: "
          f"max |diff| {fid_spec:.2e}")
    assert fid_spec < FID_TOL, "circadian recompute drifted from SPEC_T"

    bot = y == 1
    strata = build_strata(log_counts, y)
    n_invalid = sum(1 for s in strata if s.get("invalid"))
    if not args.quiet:
        print(f"[G2] strata: {len(strata)} ({n_invalid} invalid after merge)")
        for s in strata:
            tag = " INVALID" if s.get("invalid") else ""
            print(f"    [{s['lo']:+.3f},{s['hi']:+.3f}) "
                  f"bot {s['n_bot']:>4}  human {s['n_hum']:>4}{tag}")

    overall, strata_stats, aggregates = {}, {}, {}
    for off in OFFSETS_H:
        hb = hists[off][bot].sum(axis=0)
        hh = hists[off][~bot].sum(axis=0)
        raw = {o: float(np.corrcoef(spectra[off][:, k], y)[0, 1])
               for k, o in enumerate(ORDERS)}
        given = {o: partial_corr(spectra[off][:, k], y.astype(float),
                                 log_counts) for k, o in enumerate(ORDERS)}
        overall[str(off)] = {"hist_bot": hb.tolist(),
                             "hist_human": hh.tolist(),
                             "tv": tv_distance(hb, hh),
                             "raw_corr": raw, "given_count_partial": given}

        rows = []
        for s in strata:
            if s.get("invalid"):
                continue
            m = (log_counts >= s["lo"]) & (log_counts < s["hi"])
            mb, mh = m & bot, m & ~bot
            hist_b, hist_h = (hists[off][mb].sum(axis=0),
                              hists[off][mh].sum(axis=0))
            mean_b = spectra[off][mb].mean(axis=0)
            mean_h = spectra[off][mh].mean(axis=0)
            rows.append({
                "lo": s["lo"], "hi": s["hi"],
                "n_bot": int(mb.sum()), "n_hum": int(mh.sum()),
                "tv": tv_distance(hist_b, hist_h),
                "mean_H_bot": {o: float(mean_b[k])
                               for k, o in enumerate(ORDERS)},
                "mean_H_human": {o: float(mean_h[k])
                                 for k, o in enumerate(ORDERS)},
                "delta_bot_minus_human": {o: float(mean_b[k] - mean_h[k])
                                          for k, o in enumerate(ORDERS)},
                "raw_corr": {o: float(np.corrcoef(spectra[off][m, k],
                                                  y[m])[0, 1])
                             for k, o in enumerate(ORDERS)},
                "given_count_partial": {
                    o: partial_corr(spectra[off][m, k], y[m].astype(float),
                                    log_counts[m])
                    for k, o in enumerate(ORDERS)},
                "_hist_bot": hist_b, "_hist_human": hist_h,
            })
        strata_stats[str(off)] = rows
        aggregates[str(off)] = {
            "mean_stratum_tv": float(np.mean([r["tv"] for r in rows])),
            "delta_mean": {o: float(np.mean([r["delta_bot_minus_human"][o]
                                             for r in rows]))
                           for o in ORDERS},
            "frac_strata_delta_positive": {
                o: float(np.mean([r["delta_bot_minus_human"][o] > 0
                                  for r in rows])) for o in ORDERS},
            "partial_mean": {o: float(np.mean([r["given_count_partial"][o]
                                               for r in rows]))
                             for o in ORDERS},
            "partial_frac_positive": {
                o: float(np.mean([r["given_count_partial"][o] > 0
                                  for r in rows])) for o in ORDERS},
        }

    # ---- fidelity gate 2: overall partials @UTC vs p2_temporal.json --------
    fid_part, fid_where = 0.0, ""
    if P2_JSON.exists():
        pc = json.loads(P2_JSON.read_text())["partial_correlations"]
        for k, o in enumerate(ORDERS):
            for stat, key in (("raw_corr", "raw"),
                              ("given_count_partial", "given_count")):
                d = abs(overall["0"][stat][o] - pc[f"{o}_cd"][key])
                if d > fid_part:
                    fid_part, fid_where = d, f"{o}_cd.{key}"
        print(f"[gate] overall partials @UTC vs p2_temporal.json: "
              f"max |diff| {fid_part:.2e} ({fid_where or 'all zero'})")
        assert fid_part < FID_TOL, "overall partials drifted from the record"

    # ---- decision rule (pre-registered v1.0), operationalised --------------
    tvs = [aggregates[str(o)]["mean_stratum_tv"] for o in OFFSETS_H]
    deltas = {o: np.array([aggregates[str(o)]["delta_mean"][ord]
                           for ord in ORDERS]) for o in OFFSETS_H}
    frac_pos = {o: [aggregates[str(o)]["frac_strata_delta_positive"][ord]
                    for ord in ORDERS] for o in OFFSETS_H}

    ambiguous = all(t < TV_AMBIGUOUS for t in tvs)
    stable_positive = all(bool((d > 0).all()) for d in deltas.values())
    majority_strata = all(f >= 0.5 for fs in frac_pos.values() for f in fs)
    flips_with_offset = any(
        np.sign(deltas[a][k]) != np.sign(deltas[b][k])
        for k in range(len(ORDERS))
        for a, b in ((0, 1), (0, 2), (1, 2)))

    if ambiguous:
        branch = "ambiguous_drop_exploratory"
    elif stable_positive and majority_strata and not flips_with_offset:
        branch = "kept_suppression_explained"
    else:
        branch = "dropped_from_SPEC_B"

    # The offset knob is structurally inert: a constant hour shift is a
    # cyclic relabelling of the 24 bins, and TV, Renyi entropies and
    # correlations are all permutation-invariant (cf. property P6). Measure
    # the effect anyway (G3) so the inertness is proven, not asserted.
    def _flat(off):
        a = aggregates[str(off)]
        return np.array(
            [a["mean_stratum_tv"]]
            + list(a["delta_mean"].values())
            + list(a["partial_mean"].values())
            + list(overall[str(off)]["raw_corr"].values())
            + list(overall[str(off)]["given_count_partial"].values())
            + [overall[str(off)]["tv"]])

    offset_effect = float(max(np.abs(_flat(a) - _flat(b)).max()
                              for a, b in ((0, 1), (0, 2), (1, 2))))

    mechanisms = {
        "kept_suppression_explained":
            "Suppression: posting more spreads the hour-of-day distribution, "
            "so raw hour-entropy tracks volume; at matched count the class "
            "difference reverses sign, as the matched histograms show.",
        "dropped_from_SPEC_B":
            "The reversal does not survive count matching with a stable sign "
            "across offsets, so the conditioned positive coefficient is not a "
            "matched-count behavioural difference; the circadian six are "
            "dropped from SPEC_B.",
        "ambiguous_drop_exploratory":
            f"Matched-class hour distributions are near-identical (mean "
            f"stratum TV < {TV_AMBIGUOUS} at every offset); dropped from "
            f"SPEC_B, exploratory-only.",
    }
    spec_b = {
        "kept_suppression_explained":
            "SPEC_B remains alphabet(6) + mention-targets(6) per plan WP-H; "
            "the circadian six stay in SPEC_T without caveat.",
        "dropped_from_SPEC_B":
            "SPEC_B confirmed as alphabet(6) + mention-targets(6); the "
            "circadian six are dropped from behavioural fronts; P2's SPEC_T "
            "circadian half carries a caveat.",
        "ambiguous_drop_exploratory":
            "SPEC_B confirmed as alphabet(6) + mention-targets(6); circadian "
            "features exploratory-only.",
    }[branch]

    render(0, overall["0"], strata_stats["0"], "p3g_circadian_utc.png",
           args.quiet)
    render(1, overall["1"], strata_stats["1"], "p3g_circadian_local.png",
           args.quiet)
    render(2, overall["2"], strata_stats["2"],
           "p3g_circadian_local_plus2.png", args.quiet)

    def _strip(rows):
        return [{k: v for k, v in r.items() if not k.startswith("_")}
                for r in rows]

    report = {
        "phase": "P3a-circadian (WP-G)",
        "definition": "plan WP-G tasks 1-2; strata per §8 D4; anomaly under "
                      "adjudication = p2_temporal.json partial_correlations "
                      "cd orders (raw negative, given-count positive)",
        "randomness": "none (descriptive statistics; no classifier, no seeds)",
        "feature_config": {"n_bins": 24, "hi_days": 400, "min_events": 5,
                           "n_kept": int(len(y))},
        "offsets_hours": list(OFFSETS_H),
        "operationalisation": {
            "ambiguous_first": "TV < 0.05 at every offset is the no-signal "
                               "case and is evaluated before the other branches",
            "agreeing_with_conditioned_sign": "matched delta > 0 for all six "
                                              "orders (conditioned coefficients all positive)",
            "stable": "same sign across offsets and >= 0.5 strata positive",
        },
        "fidelity": {
            "spec_cd_recompute_max_abs_diff": fid_spec,
            "overall_partials_vs_p2_temporal_max_abs_diff": fid_part,
            "tol": FID_TOL, "pass": True},
        "overall": overall,
        "strata_sizes": [{k: v for k, v in s.items()} for s in strata],
        "strata": {o: _strip(strata_stats[o]) for o in aggregates},
        "aggregates": aggregates,
        "decision": {
            "branch": branch,
            "predicates": {
                "mean_stratum_tv_per_offset": dict(zip(map(str, OFFSETS_H),
                                                       tvs)),
                "ambiguous": ambiguous,
                "stable_positive": stable_positive,
                "majority_strata_positive": majority_strata,
                "flips_with_offset": flips_with_offset,
                "delta_mean_per_offset": {str(o): deltas[o].tolist()
                                          for o in OFFSETS_H},
                "frac_strata_delta_positive_per_offset": {
                    str(o): frac_pos[o] for o in OFFSETS_H},
            },
            "mechanism_sentence": mechanisms[branch],
            "offset_invariance": {
                "max_abs_effect_across_offsets": offset_effect,
                "note": "a constant hour offset is a cyclic relabelling of "
                        "the 24 bins; TV, Renyi entropies and correlations "
                        "are permutation-invariant (cf. P6), so 'flips with "
                        "timezone offset' is unreachable by construction. "
                        "The sweep is retained as the proof of inertness "
                        "(G3); the adjudication rests on the matching leg.",
            },
            "spec_b_dimension_decision": spec_b,
            "spec_t_caveat": ("circadian half of SPEC_T keeps a caveat" if
                              branch != "kept_suppression_explained" else
                              "no caveat required; mechanism cited"),
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=1))

    print("\n" + "=" * 88)
    print("WP-G — CIRCADIAN ADJUDICATION")
    print("=" * 88)
    print(f"{'offset':>7}{'TV overall':>11}{'TV strata':>10}"
          f"{'raw H_0':>9}{'given H_0':>10}{'raw H_inf':>10}"
          f"{'given H_inf':>12}")
    for o in OFFSETS_H:
        ov = overall[str(o)]
        print(f"{o:>7}{ov['tv']:>11.4f}"
              f"{aggregates[str(o)]['mean_stratum_tv']:>10.4f}"
              f"{ov['raw_corr']['H_0']:>9.4f}"
              f"{ov['given_count_partial']['H_0']:>10.4f}"
              f"{ov['raw_corr']['H_inf']:>10.4f}"
              f"{ov['given_count_partial']['H_inf']:>12.4f}")
    print(f"\nmatched delta (bot - human), mean over strata:")
    for o in OFFSETS_H:
        dm = aggregates[str(o)]["delta_mean"]
        print(f"  +{o} h: " + "  ".join(f"{ord} {dm[ord]:+.4f}"
                                         for ord in ORDERS))
    print(f"\nBRANCH: {branch}")
    print(f"  {mechanisms[branch]}")
    print(f"  SPEC_B: {spec_b}")
    print(f"json -> {OUT_JSON}")
    print("=" * 88)


if __name__ == "__main__":
    main()
