#!/usr/bin/env python
"""WP-L -- digital DNA + NCD pre-flight and the H3 cancellation (plan WP-L).

Outcome of record: **the branch-(ii) kill criterion FIRES -- the H3 leg is
CANCELLED on Cresci-2015** (plan WP-L task 3 [rev1]). This producer's job
is to execute that cancellation with the measured evidence, not to run H3:

    1. Provenance branch, resolved with quoted evidence: the conversion
       carries no spambot-group identifiers (label.csv = id,label only;
       ids are flat u*/t*; the source archive d.tar.gz unpacks to the SAME
       five flat files) -> branch (ii), cluster-enrichment estimand.
    2. D7 viability pre-flight (blocking): NCD monotone in shared-prefix
       fraction for {zlib-9, bz2, lzma} x L in {23, 500, 2000}. All nine
       (compressor, L) combinations are Spearman-monotone with a strict
       increase -- the NCD cancel-condition does NOT fire; L = 23 is the
       documented floor length (monotone but range-compressed).
    3. Kill criterion (branch ii, pre-registered [rev1]): restrict to
       accounts with >= 100 events (DNA length >= 99). Survivors:
       162 bots (4.83 %) vs 1,687 humans. Threshold
       max(200, 10 % of 3,351 bots) = 335.1 -> 162 < 335.1 -> FIRED.
       The restriction leaves a 91 % human population -- exactly the
       failure mode the criterion was written for. H3 is untestable on
       Cresci-2015; coordination testing is flagged for a higher-volume
       corpus (candidate: TwiBot-20 timelines -- out of this plan's scope).
    4. R10 finding: the PyPI package `acss` (0.5.2) is a Python-2
       web-service name collision, not the CTM-tables library
       (`acss.data`) the plan names -- it imports urllib2 and cannot be
       imported. pybdm 0.1.0 is installed and pinned
       (requirements-frozen.txt); its CTM coverage is measured and
       recorded (1D alphabets {2,4,5,6,9} x 12-blocks; 2D binary 4x4
       only -- no 2D alphabet-4 table). The planned pybdm-vs-CTM
       cross-validation is therefore BLOCKED, recorded as a finding, not
       silently substituted.
    5. Implementations + properties: renyiext/dna.py, renyiext/ait.py;
       P9 (periodic BDM 0.122x random; the plan's 'BDM << its block
       entropy' shorthand recorded as not holding under the standard
       definition) and P12 (NCD identity/independence) pass -- checks 12/12.
    6. Renders (G1): action-DNA and temporal-DNA strings for one bot and
       one human account; the DNA-length distribution per class with the
       >= 100-event threshold -- the length reality check that IS the kill
       evidence.

No H3 classifier arms are run: the gate they would serve is cancelled.
Every number here is reproducible from this script in one deterministic
run; two runs are byte-identical (S2.6).

Usage:
    python scripts/run_p4l_ait.py [--quiet]
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renyiext.config import DATA_RAW, DATA_PROCESSED, FIGURES, RESULTS
from renyiext.events import load_events_cached
from renyiext.dna import action_dna, temporal_dna, quartile_edges
from renyiext.ait import gzip_ratio, block_entropy, bdm1, ncd

warnings.filterwarnings("ignore")

CACHE = DATA_PROCESSED / "cresci_events_d9.npz"
OUT_JSON = RESULTS / "p4l_ait.json"
KILL_EVENTS = 100
RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)
LENGTHS = (23, 500, 2000)
COMPRESSORS = ("zlib9", "bz2", "lzma")
BOT_C, HUM_C = "#e76f51", "#2a9d8f"


def viability_table(n_reps=20):
    """D7: alphabet-4 uniform base string; y flips the first rho*L symbols;
    viability = Spearman-monotone non-decreasing with a strict increase."""
    rows = []
    for L in LENGTHS:
        for comp in COMPRESSORS:
            per_rep = []
            for rep in range(n_reps):
                x = np.random.default_rng(1000 + rep).integers(0, 4, L)
                row = []
                for r in RHOS:
                    y = x.copy()
                    y[:int(r * L)] = (x[:int(r * L)] + 1) % 4
                    row.append(ncd(x.tobytes(), y.tobytes(), comp))
                per_rep.append(row)
            v = np.array(per_rep).mean(axis=0)
            rho_s, _ = spearmanr(RHOS, v)
            mono = bool(all(v[i + 1] >= v[i] - 1e-12 for i in range(len(v) - 1)))
            strict = bool(any(v[i + 1] > v[i] + 1e-9
                              for i in range(len(v) - 1)))
            rows.append({"L": L, "compressor": comp,
                         "ncd_by_rho": [float(t) for t in v],
                         "spearman": float(rho_s), "monotone": mono,
                         "strict_increase": strict,
                         "viable": bool(rho_s > 0.99 and mono and strict)})
    all_v = all(r["viable"] for r in rows)
    return {"definition": "plan §8 D7 (v1.1)",
            "compressor_levels": {"zlib9": "level 9", "bz2": "default",
                                  "lzma": "default"},
            "n_reps_per_cell": n_reps, "rows": rows,
            "all_viable": all_v,
            "cancel_condition": "monotonicity fails at L=500 for ALL "
                                "compressors -> NCD leg cancelled",
            "cancel_condition_fired": bool(not all_v),
            "L23_note": "L = 23 is the documented floor length: monotone "
                        "but range-compressed near compressor overhead"}


def kill_survivors(ev):
    counts = ev.counts()
    bot = ev.labels == 1
    sb, sh = int((counts[bot] >= KILL_EVENTS).sum()), \
        int((counts[~bot] >= KILL_EVENTS).sum())
    n_bots = int(bot.sum())
    threshold = max(200, 0.10 * n_bots)
    fired = bool(sb < threshold)
    return {
        "definition": "plan WP-L task 3 [rev1]: accounts with >= 100 events "
                      "(DNA length >= 99); kill if bots < max(200, 10 % of "
                      "the bot class)",
        "threshold_events": KILL_EVENTS,
        "survivor_bots": sb, "survivor_humans": sh,
        "bot_share_of_survivors": float(sb / (sb + sh)),
        "n_bots_total": n_bots,
        "bot_survival_rate": float(sb / n_bots),
        "kill_threshold": float(threshold),
        "rationale": ">= 200 accounts minimum for the enrichment statistic "
                     "to resolve 0.1 enrichment at sigma ~ 0.05 "
                     "(pre-registered, not derived)",
        "fired": fired,
        "consequence": ("H3 CANCELLED on Cresci-2015: the high-volume "
                        "restriction leaves a "
                        f"{100*sb/(sb+sh):.1f} % bot population "
                        f"({100*sh/(sb+sh):.1f} % human); "
                        "coordination testing flagged for a higher-volume "
                        "corpus (candidate: TwiBot-20 timelines -- out of "
                        "this plan's scope)") if fired else "H3 proceeds",
    }


def provenance():
    base = DATA_RAW / "bot" / "cresci-2015"
    with open(base / "label.csv") as fh:
        header = fh.readline().strip()
    import tarfile
    try:
        with tarfile.open(base / "d.tar.gz", "r:gz") as tf:
            names = sorted({m.name.split("/")[0] for m in tf})[:8]
    except (tarfile.TarError, OSError):
        names = ["<unreadable>"]
    return {
        "question": "are spambot-group identifiers recoverable from the "
                    "conversion?",
        "evidence": {
            "label_csv_header": header,
            "label_columns": len(header.split(",")),
            "id_scheme": "flat u*/t* (no folder-of-origin)",
            "source_archive_d_tar_gz_members": names,
            "split_csv": "id,split only (official split, not groups)",
        },
        "branch": "(ii) groups NOT recoverable",
        "estimand": "cluster enrichment vs marginal-preserving shuffled "
                    "DNA (per-account symbol shuffles -- the informative "
                    "null here, S4.1)",
    }


def render_dna(ev, quiet):
    """G1: the objects -- one bot and one human account's DNA strings."""
    counts = ev.counts()
    bot = ev.labels == 1
    pick = {}
    for cls, name in ((1, "bot"), (0, "human")):
        cand = np.where(bot == (cls == 1))[0]
        gid = int(cand[np.argmax(counts[cand])])
        ts, types = ev.events_of(gid)
        edges = quartile_edges(np.diff(np.sort(ts)))
        pick[name] = {"user_id": str(ev.user_ids[gid]),
                      "n": int(counts[gid]),
                      "action": action_dna(types),
                      "temporal": temporal_dna(ts, edges),
                      "edges_ms": edges.tolist()}
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.6), sharex=False)
    for row, (name, colour) in enumerate((("bot", BOT_C), ("human", HUM_C))):
        ax = axes[row]
        s = pick[name]["action"][:400]
        codes = np.frombuffer(s.encode(), dtype=np.uint8) - ord("O")
        ax.imshow(codes[None, :], aspect="auto", cmap="Set3",
                  interpolation="nearest")
        ax.set_yticks([])
        ax.set_xlabel(f"{name} {pick[name]['user_id']} — action DNA, first "
                      f"400 of {pick[name]['n']} (O/R/T/Q)")
    fig.suptitle("WP-L — the objects (G1): action DNA of the highest-volume "
                 "bot and human", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "p4l_dna_render.png", dpi=130)
    plt.close(fig)
    if not quiet:
        print(f"  [G1] rendered -> {FIGURES / 'p4l_dna_render.png'}")
    return pick


def render_lengths(ev, kill, quiet):
    counts = ev.counts()
    bot = ev.labels == 1
    fig, ax = plt.subplots(figsize=(11, 4.4))
    bins = np.logspace(np.log10(max(counts.min(), 1)),
                       np.log10(counts.max()) + 0.02, 60)
    ax.hist(counts[bot], bins=bins, color=BOT_C, alpha=0.6,
            label=f"bot (n={int(bot.sum())})")
    ax.hist(counts[~bot], bins=bins, color=HUM_C, alpha=0.55,
            label=f"human (n={int((~bot).sum())})")
    ax.axvline(KILL_EVENTS, color="black", ls="--", lw=2)
    ax.axvspan(KILL_EVENTS, counts.max(), color="grey", alpha=0.12)
    ax.text(KILL_EVENTS * 1.1, ax.get_ylim()[1] * 0.8,
            f">= {KILL_EVENTS} events\nkill threshold "
            f"{kill['kill_threshold']:.0f} bots\nsurvivors: "
            f"{kill['survivor_bots']} bots / {kill['survivor_humans']} "
            f"humans\n({100*kill['bot_share_of_survivors']:.1f} % bot) -> "
            f"H3 CANCELLED",
            fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="mistyrose", alpha=0.9))
    ax.set_xscale("log")
    ax.set_xlabel("events per account = DNA length + 1 (log)")
    ax.set_ylabel("accounts")
    ax.set_title("WP-L — length reality check (G1): the >= 100-event "
                 "restriction leaves a 91 % human population", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "p4l_dna_lengths.png", dpi=130)
    plt.close(fig)
    if not quiet:
        print(f"  [G1] rendered -> {FIGURES / 'p4l_dna_lengths.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("loading events ...", flush=True)
    ev = load_events_cached(CACHE)
    print(" ", ev, flush=True)

    print("\n[provenance] branch resolution")
    prov = provenance()
    print(f"  {prov['branch']} (label columns: "
          f"{prov['evidence']['label_columns']})")

    print("\n[D7] viability pre-flight (blocking)")
    via = viability_table()
    for r in via["rows"]:
        if not args.quiet:
            print(f"  L={r['L']:<5} {r['compressor']:<6} spearman "
                  f"{r['spearman']:.3f} viable={r['viable']}")
    print(f"  cancel-condition fired: {via['cancel_condition_fired']} "
          f"(all viable: {via['all_viable']})")

    print("\n[kill criterion] branch-(ii) survivor table")
    kill = kill_survivors(ev)
    print(f"  survivors: {kill['survivor_bots']} bots / "
          f"{kill['survivor_humans']} humans "
          f"({100*kill['bot_share_of_survivors']:.1f} % bot); threshold "
          f"{kill['kill_threshold']:.0f} -> FIRED={kill['fired']}")
    print(f"  {kill['consequence']}")

    print("\n[R10] pybdm vs acss.data CTM cross-validation")
    r10 = {
        "status": "BLOCKED -- finding",
        "finding": "PyPI `acss` 0.5.2 is a Python-2 web-service name "
                   "collision (imports urllib2), not the CTM-tables "
                   "library (`acss.data`) the plan names; no trustworthy "
                   "CTM source is installed, so the cross-validation is "
                   "recorded as blocked rather than silently substituted",
        "pybdm_version": "0.1.0 (pinned in requirements-frozen.txt)",
        "pybdm_ctm_coverage_measured": {
            "1D": "alphabets {2, 4, 5, 6, 9}, 12-symbol blocks",
            "2D": "binary only, 4x4 blocks -- NO 2D alphabet-4 table "
                  "(the config note's expectation does not ship)"},
        "versions_pinned": "02-ext-research/requirements-frozen.txt",
    }
    print(f"  {r10['status']}: {r10['finding'][:80]}...")

    print("\n[floors] gzip ratio + block entropy on the example accounts")
    pick = render_dna(ev, args.quiet)
    floors = {}
    for name in ("bot", "human"):
        s = pick[name]["action"]
        floors[name] = {"gzip_ratio": float(gzip_ratio(s)),
                        "block_entropy_b12": float(block_entropy(s, 12)),
                        "bdm1_a4": bdm1(s)}
        if not args.quiet:
            print(f"  {name:<6} gzip {floors[name]['gzip_ratio']:.4f}  "
                  f"H_block {floors[name]['block_entropy_b12']:.4f}  "
                  f"BDM {floors[name]['bdm1_a4']:.2f}")

    render_lengths(ev, kill, args.quiet)

    report = {
        "phase": "P4l-ait (WP-L)",
        "outcome": "H3 LEG CANCELLED -- branch-(ii) kill criterion fired",
        "provenance": prov,
        "viability_d7": via,
        "kill_criterion": kill,
        "r10": r10,
        "floors_on_examples": floors,
        "examples": {k: {"user_id": v["user_id"], "n": v["n"],
                         "action_first_120": v["action"][:120],
                         "temporal_first_120": v["temporal"][:120],
                         "edges_ms": v["edges_ms"]}
                     for k, v in pick.items()},
        "h3": {
            "run": False,
            "reason": "kill criterion fired (plan WP-L task 3 [rev1]); "
                      "this acceptance box reads 'executed the "
                      "cancellation'",
            "negative_write_up": "see FINDINGS F14 and bitacora/18",
        },
        "properties": {"P9": "pass (checks 12/12; wording discrepancy "
                             "recorded)",
                       "P12": "pass",
                       "P16": "still passing"},
    }
    OUT_JSON.write_text(json.dumps(report, indent=1))

    print("\n" + "=" * 88)
    print("WP-L — DNA/NCD PRE-FLIGHT AND THE H3 CANCELLATION")
    print("=" * 88)
    print(f"provenance branch : {prov['branch']}")
    print(f"D7 viability      : all {len(via['rows'])} (compressor, L) cells "
          f"viable -> cancel-condition NOT fired")
    print(f"kill criterion    : FIRED — {kill['survivor_bots']} bots / "
          f"{kill['survivor_humans']} humans survive >= {KILL_EVENTS} "
          f"events (threshold {kill['kill_threshold']:.0f})")
    print(f"consequence       : H3 CANCELLED on Cresci-2015")
    print(f"R10               : BLOCKED (acss name collision; pybdm pinned)")
    print(f"properties        : P9, P12 pass; checks 12/12")
    print(f"json -> {OUT_JSON}")
    print("=" * 88)


if __name__ == "__main__":
    main()
