#!/usr/bin/env python
"""P0 -- data layer, and the datasaurus gate on decision D1.

D1 claims that Cresci-2015's tweet timestamps are recoverable from the snowflake
ids. Bitacora 00 sect. 4 supports that with a decoded count and a date range,
which is precisely what gate G2 says is *not* agreement. This script settles it.

    G1 RENDER      the objects, at full length, before any number:
                   inter-arrival distribution, hour-of-day, per-account rasters,
                   events-per-account, the whole corpus timeline
    G2 ELEMENTWISE decoded tweet time vs the account's own ``created_at`` --
                   an independent field that never passed through the decoder.
                   2.8M per-event constraints. Symmetric difference and where.
    G3 KNOBS       every threshold this script invents, printed and swept
    G4 MECHANISM   the counter null: what the decode looks like if the top 41
                   bits are a monotone counter rather than a clock

Usage:
    python scripts/run_p0_events.py [--quiet]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from renyiext.config import Config, FIGURES, RESULTS, DATA_PROCESSED, CRESCI_WINDOW
from renyiext import events as ev_mod
from renyiext.config import FIRST_SNOWFLAKE_ID  # noqa: F401
from renyiext.events import (
    load_cresci_events, account_age_violations, counter_null_timestamps,
    decode_snowflake_array, POST_TYPE,
)

MS_PER_DAY = 86_400_000.0


def _hours_utc(ts_ms: np.ndarray) -> np.ndarray:
    """Hour of day in UTC, as a float, from epoch milliseconds."""
    return ((ts_ms % MS_PER_DAY) / 3_600_000.0)


# ---------------------------------------------------------------------------
# GATE 1 -- render the objects
# ---------------------------------------------------------------------------

def gate1_render(ev, cfg, quiet=False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    counts = ev.counts()
    bot = ev.labels == 1
    ts = ev.ts_ms

    # --- the corpus timeline: the raw thing, at full length ---
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    ax = axes[0, 0]
    days = (ts - ts.min()) / MS_PER_DAY
    ax.hist(days, bins=400, color="#3b6ea5")
    ax.set_title("Corpus timeline — all 2.8M decoded tweets, full length")
    ax.set_xlabel(f"days since {datetime.fromtimestamp(ts.min()/1000, timezone.utc):%Y-%m-%d}")
    ax.set_ylabel("tweets")

    # --- hour of day: the circadian object. This is the G4 discriminator. ---
    ax = axes[0, 1]
    h_real = _hours_utc(ts)
    null_ts = counter_null_timestamps(
        np.array([int(x) for x in range(0)], dtype=np.int64)) if False else None
    ax.hist(h_real, bins=48, density=True, color="#3b6ea5", label="decoded")
    ax.axhline(1 / 24, color="crimson", ls="--", lw=2, label="uniform (counter null)")
    ax.set_title("Hour of day (UTC) — circadian structure or none")
    ax.set_xlabel("hour"); ax.set_ylabel("density"); ax.legend()

    # --- inter-arrival: the object the temporal front is built on ---
    ax = axes[1, 0]
    ia_bot, ia_hum = [], []
    for i in range(ev.n_users):
        d = ev.inter_arrival_ms(i)
        if len(d):
            (ia_bot if bot[i] else ia_hum).append(d)
    ia_bot = np.concatenate(ia_bot) if ia_bot else np.empty(0)
    ia_hum = np.concatenate(ia_hum) if ia_hum else np.empty(0)
    # log bins from 1 ms to 1 year; swept below in G3
    bins = np.logspace(0, np.log10(365 * MS_PER_DAY), 80)
    for arr, lab, c in ((ia_hum, "human", "#2a9d8f"), (ia_bot, "bot", "#e76f51")):
        a = arr[arr > 0]
        if len(a):
            ax.hist(a, bins=bins, density=True, histtype="step", lw=2, label=lab, color=c)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Inter-arrival time — full range, both classes")
    ax.set_xlabel("Δt (ms, log)"); ax.set_ylabel("density (log)"); ax.legend()

    # --- events per account: the R1 confound, drawn ---
    ax = axes[1, 1]
    b = np.logspace(0, np.log10(max(counts.max(), 2)), 60)
    ax.hist(counts[~bot], bins=b, histtype="step", lw=2, label="human", color="#2a9d8f")
    ax.hist(counts[bot], bins=b, histtype="step", lw=2, label="bot", color="#e76f51")
    ax.axvline(cfg.n_events, color="k", ls=":", lw=2, label=f"n_events={cfg.n_events}")
    ax.set_xscale("log")
    ax.set_title("Events per account — the volume confound (R1)")
    ax.set_xlabel("tweets (log)"); ax.set_ylabel("accounts"); ax.legend()

    # --- per-account rasters: THE OBJECT, not a statistic of it (G1b) ---
    rng = np.random.default_rng(cfg.seed)
    for row, (mask, name, colour) in enumerate(
            [(~bot, "human", "#2a9d8f"), (bot, "bot", "#e76f51")]):
        ax = axes[2, row]
        idx = np.where(mask & (counts > 50))[0]
        pick = rng.choice(idx, size=min(30, len(idx)), replace=False)
        for k, i in enumerate(sorted(pick, key=lambda j: counts[j])):
            t, _ = ev.events_of(i)
            ax.plot((t - ts.min()) / MS_PER_DAY, np.full(len(t), k), "|",
                    ms=3, color=colour, alpha=0.6)
        ax.set_title(f"Posting rasters — 30 {name} accounts (the object itself)")
        ax.set_xlabel("days"); ax.set_ylabel("account (sorted by volume)")

    fig.tight_layout()
    fig.savefig(FIGURES / "p0_g1_objects.png", dpi=130)
    plt.close(fig)

    if not quiet:
        print(f"[G1] rendered -> {FIGURES / 'p0_g1_objects.png'}")
    return {"ia_bot": ia_bot, "ia_hum": ia_hum, "counts": counts, "hours": h_real}


# ---------------------------------------------------------------------------
# GATE 4 -- the counter null, rendered beside the real decode
# ---------------------------------------------------------------------------

def gate4_counter_null(ev, cfg, quiet=False):
    """If the top 41 bits were a counter, hour-of-day would be uniform.

    The separating world, named before the test (G4): a global monotone counter
    advances with tweet *volume*, not with time, so decoded hour-of-day carries
    no diurnal cycle. Humans sleep. If the real decode shows a strong cycle and
    the null does not, the top bits are a clock.
    """
    rng = np.random.default_rng(cfg.seed)
    sample = rng.choice(ev.n_events, size=min(300_000, ev.n_events), replace=False)
    ts_real = ev.ts_ms[sample]
    ids = ((ts_real - ev_mod.TWITTER_EPOCH_MS) << 22).astype(np.int64)
    ts_null = counter_null_timestamps(ids)

    h_real, h_null = _hours_utc(ts_real), _hours_utc(ts_null)
    d_real, _ = np.histogram(h_real, bins=24, range=(0, 24), density=True)
    d_null, _ = np.histogram(h_null, bins=24, range=(0, 24), density=True)
    unif = np.full(24, 1 / 24)

    # Total variation distance from uniform: 0 = no circadian, 1 = maximal
    tv_real = float(0.5 * np.abs(d_real - unif).sum())
    tv_null = float(0.5 * np.abs(d_null - unif).sum())
    # peak/trough ratio -- the interpretable version
    ptr_real = float(d_real.max() / d_real.min())
    ptr_null = float(d_null.max() / d_null.min())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(np.arange(24), d_real, color="#3b6ea5", label="decoded")
    ax[0].axhline(1/24, color="crimson", ls="--", lw=2)
    ax[0].set_title(f"Decoded hour-of-day — TV from uniform {tv_real:.3f}, "
                    f"peak/trough {ptr_real:.2f}")
    ax[1].bar(np.arange(24), d_null, color="#999999", label="counter null")
    ax[1].axhline(1/24, color="crimson", ls="--", lw=2)
    ax[1].set_title(f"Counter null — TV {tv_null:.3f}, peak/trough {ptr_null:.2f}")
    for a in ax:
        a.set_xlabel("hour (UTC)"); a.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(FIGURES / "p0_g4_counter_null.png", dpi=130)
    plt.close(fig)

    return {"tv_real": tv_real, "tv_null": tv_null,
            "peak_trough_real": ptr_real, "peak_trough_null": ptr_null,
            "n_sampled": int(len(sample))}


# ---------------------------------------------------------------------------
# GATE 3 -- every knob this script invented
# ---------------------------------------------------------------------------

def gate3_knobs(ev, cfg, quiet=False):
    """Print every threshold, and sweep the ones that could hide a finding."""
    types, cnt = np.unique(ev.post_type, return_counts=True)
    inv = {v: k for k, v in POST_TYPE.items()}
    type_share = {inv[int(t)]: float(c / cnt.sum()) for t, c in zip(types, cnt)}

    # n_events is the one knob with teeth: it decides who is excluded.
    counts = ev.counts()
    bot = ev.labels == 1
    sweep = {}
    for n in (32, 64, 128, 256, 512):
        keep = counts >= n
        sweep[n] = {
            "kept_total": float(keep.mean()),
            "kept_bot": float(keep[bot].mean()),
            "kept_human": float(keep[~bot].mean()),
        }

    # D9's pre-snowflake exclusion is class-dependent and must be visible.
    dpu = ev.dropped_per_user
    presnow = {
        "total": int(dpu.sum()),
        "bot": int(dpu[bot].sum()), "human": int(dpu[~bot].sum()),
        "accounts_affected_bot": int((dpu[bot] > 0).sum()),
        "accounts_affected_human": int((dpu[~bot] > 0).sum()),
    }

    return {"post_type_share": type_share, "n_events_sweep": sweep,
            "presnowflake_dropped": presnow,
            "epoch_ms": ev_mod.TWITTER_EPOCH_MS, "shift_bits": 22,
            "first_snowflake_id": FIRST_SNOWFLAKE_ID,
            "note": "epoch, shift and the first-snowflake id are fixed "
                    "constants of the id format, not fitted parameters; they "
                    "have no bracket to be interior to"}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = Config()

    RESULTS.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("loading Cresci-2015 ...", flush=True)
    ev = load_cresci_events()
    if not args.quiet:
        print(" ", ev, flush=True)

    g1 = gate1_render(ev, cfg, args.quiet)
    g2 = account_age_violations(ev)
    g3 = gate3_knobs(ev, cfg, args.quiet)
    g4 = gate4_counter_null(ev, cfg, args.quiet)

    ts = ev.ts_ms
    lo = datetime.fromtimestamp(ts.min() / 1000, timezone.utc)
    hi = datetime.fromtimestamp(ts.max() / 1000, timezone.utc)
    win_lo = datetime.fromisoformat(CRESCI_WINDOW[0]).replace(tzinfo=timezone.utc)
    win_hi = datetime.fromisoformat(CRESCI_WINDOW[1]).replace(tzinfo=timezone.utc)

    counts = ev.counts()
    bot = ev.labels == 1
    keep = counts >= cfg.n_events

    # ---- P14/P15 properties ----
    p14 = bool(lo >= win_lo and hi <= win_hi)
    rng = np.random.default_rng(0)
    s = rng.choice(ev.n_users, size=min(200, ev.n_users), replace=False)
    p15 = all(np.all(np.diff(ev.events_of(i)[0]) >= 0) for i in s)

    # ---- G0 ----
    g0_pass = bool(
        p14 and p15
        and g2["frac_violations"] < 0.001
        and keep[bot].mean() >= 0.80 and keep[~bot].mean() >= 0.80
    )

    report = {
        "phase": "P0",
        "n_users": int(ev.n_users), "n_events": int(ev.n_events),
        "n_bot": int(bot.sum()), "n_human": int((~bot).sum()),
        "n_dropped_nonsnowflake": int(ev.n_dropped_nonsnowflake),
        "range": [lo.isoformat(), hi.isoformat()],
        "events_per_user_median": float(np.median(counts)),
        "P14_range_in_window": p14, "P15_sorted": bool(p15),
        "G2_elementwise": g2, "G3_knobs": g3, "G4_counter_null": g4,
        "kept_at_n_events": {"n_events": cfg.n_events,
                             "bot": float(keep[bot].mean()),
                             "human": float(keep[~bot].mean())},
        "G0_pass": g0_pass,
    }
    (RESULTS / "p0_events.json").write_text(json.dumps(report, indent=1))

    print("\n" + "=" * 68)
    print("P0 — DATA LAYER, DATASAURUS GATE ON D1")
    print("=" * 68)
    print(f"users {ev.n_users}  events {ev.n_events}  "
          f"bot {int(bot.sum())} / human {int((~bot).sum())}")
    print(f"range {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}   "
          f"median events/user {np.median(counts):.0f}")
    print(f"dropped as non-snowflake: {ev.n_dropped_nonsnowflake}")
    print(f"\n[G2 ELEMENTWISE]  decoded time vs the account's own created_at")
    print(f"  events checked        {g2['n_events_checked']:,}")
    print(f"  violations            {g2['n_violations']:,} "
          f"({100*g2['frac_violations']:.4f}%)")
    print(f"  accounts with any     {g2['n_users_with_any_violation']} "
          f"of {g2['n_users_checked']}")
    print(f"  worst backdate        {g2['worst_backdate_days']:.2f} days")
    print(f"\n[G4 MECHANISM]  circadian structure vs the counter null")
    print(f"  decoded    TV from uniform {g4['tv_real']:.4f}   "
          f"peak/trough {g4['peak_trough_real']:.2f}")
    print(f"  counter    TV from uniform {g4['tv_null']:.4f}   "
          f"peak/trough {g4['peak_trough_null']:.2f}")
    print(f"\n[G3 KNOBS]  post-type share: " +
          ", ".join(f"{k} {v:.3f}" for k, v in g3["post_type_share"].items()))
    print("  n_events sweep (fraction of accounts kept):")
    for n, d in g3["n_events_sweep"].items():
        print(f"    n={n:<4} total {d['kept_total']:.3f}   "
              f"bot {d['kept_bot']:.3f}   human {d['kept_human']:.3f}")
    print(f"\nP14 range in window: {p14}    P15 sorted: {p15}")
    print(f"\nGATE G0: {'PASS' if g0_pass else 'FAIL'}")
    print("=" * 68)


if __name__ == "__main__":
    main()
