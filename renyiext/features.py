"""Feature blocks for the temporal front (docs/02-PROTOCOL.md sect. 2).

Every block is named, built independently, and evaluated alone and in
combination. Ablation is by block, never by column.

The blocks here are those P2 needs:

    COUNT    1   log event count -- the incumbent. AUC 0.939 alone (bitacora 01)
    BURST    3   CV, burstiness B, memory M -- the standard burstiness floor
    SHAN     2   Shannon only: H_1 of inter-arrival and of hour-of-day
    SPEC_T  12   the full Renyi spectrum of both, 6 orders each

``SHAN`` is exactly the ``alpha = 1`` slice of ``SPEC_T``, by construction, so
``AUC(SPEC_T) - AUC(SHAN)`` is H1 clause (ii) with no other difference between
the two arms.

**Nothing here corrects for sample size.** Per decision D3' the bias is real
(P7: H_0 drifts 1.314 bits from n=8 to n=4096 on a *fixed* distribution) and is
handled by conditioning on ``COUNT`` downstream, not by correcting the
estimator. Any block used without ``COUNT`` beside it is being misread.
"""

from __future__ import annotations

import numpy as np

from .spectrum import (spectrum, counts_to_probabilities, log_bin_counts,
                       renyi_entropy, spectrum_labels, SPECTRUM_ALPHAS)

__all__ = ["burstiness", "memory_coefficient", "temporal_blocks",
           "temporal_blocks_ts", "temporal_blocks_windowed",
           "BLOCK_NAMES", "MS_PER_DAY"]

MS_PER_DAY = 86_400_000.0

BLOCK_NAMES = ("COUNT", "BURST", "SHAN", "SPEC_T")


def burstiness(dt: np.ndarray) -> float:
    """Goh & Barabasi burstiness ``B = (s - m) / (s + m)``.

    ``B = -1`` periodic, ``0`` Poisson, ``+1`` maximally bursty. This is the
    standard one-number summary of the very thing H1 claims the spectrum sees
    better, so it is a mandatory floor, not a nicety.
    """
    if dt.size < 2:
        return 0.0
    m, s = float(dt.mean()), float(dt.std())
    return (s - m) / (s + m) if (s + m) > 0 else 0.0


def memory_coefficient(dt: np.ndarray) -> float:
    """Lag-1 autocorrelation of consecutive inter-arrival times.

    Burstiness and memory are the two independent axes of Goh & Barabasi's
    decomposition; reporting only ``B`` would leave the ordering axis untested,
    and ordering is precisely what a marginal spectrum cannot see (S4.1).
    """
    if dt.size < 3:
        return 0.0
    a, b = dt[:-1], dt[1:]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def temporal_blocks_ts(ts_ms_list, labels=None, n_bins: int = 24,
                       lo: float = 1.0, hi: float | None = None,
                       min_events: int = 5, alphas=SPECTRUM_ALPHAS) -> dict:
    """Build every temporal block from raw per-account timestamp arrays.

    Generic core of :func:`temporal_blocks` (same block definitions, same
    defaults, same loop) so that synthetic accounts -- the S4.2 controls and
    the censoring probe -- go through the *identical* feature pipeline as the
    real corpus. ``ts_ms_list[k]`` is one account's decoded timestamps in
    milliseconds, any order; ``labels[k]`` is its class (1 = bot by
    convention). ``labels=None`` builds features only.

    ``hi`` fixes the top of the log grid. Leaving it ``None`` makes the grid
    *account-dependent*, which silently encodes each account's own maximum gap
    into the binning -- a per-unit parameter masquerading as a fixed one.
    Callers must pin ``hi`` to a constant across accounts (the P2 headline and
    sweep do; this is restated here because the probe inherits the same rule).

    Returns blocks, labels, the kept mask, and per-class exclusion counts.
    """
    ts_ms_list = [np.asarray(t, dtype=np.int64) for t in ts_ms_list]
    counts = np.array([len(t) for t in ts_ms_list], dtype=np.int64)
    keep = counts >= max(min_events, 2)
    idx = np.where(keep)[0]

    rows_count, rows_burst, rows_shan, rows_spec = [], [], [], []
    for k in idx:
        ts = np.sort(ts_ms_list[k])
        dt = np.diff(ts).astype(np.float64)

        h_ia = log_bin_counts(dt, n_bins=n_bins, lo=lo, hi=hi)
        p_ia = counts_to_probabilities(h_ia)

        hours = ((ts % MS_PER_DAY) / 3_600_000.0).astype(int)
        h_cd = np.bincount(hours, minlength=24).astype(np.float64)
        p_cd = counts_to_probabilities(h_cd)

        s_ia, s_cd = spectrum(p_ia, alphas), spectrum(p_cd, alphas)

        rows_count.append([np.log1p(len(ts))])
        rows_burst.append([burstiness(dt), memory_coefficient(dt),
                           float(dt.std() / dt.mean()) if dt.mean() > 0 else 0.0])
        rows_shan.append([renyi_entropy(p_ia, 1.0), renyi_entropy(p_cd, 1.0)])
        rows_spec.append(np.concatenate([s_ia, s_cd]))

    lab = spectrum_labels(alphas)
    out = {
        "blocks": {
            "COUNT": np.array(rows_count, dtype=np.float64),
            "BURST": np.array(rows_burst, dtype=np.float64),
            "SHAN": np.array(rows_shan, dtype=np.float64),
            "SPEC_T": np.array(rows_spec, dtype=np.float64),
        },
        "columns": {
            "COUNT": ["log1p_n"],
            "BURST": ["B", "M", "CV"],
            "SHAN": ["H1_ia", "H1_cd"],
            "SPEC_T": [f"{c}_ia" for c in lab] + [f"{c}_cd" for c in lab],
        },
        "index": idx,
        "n_kept": int(len(idx)),
        "n_excluded": int((~keep).sum()),
    }
    if labels is not None:
        labels = np.asarray(labels)
        bot = labels[idx] == 1
        excl = labels[~keep] == 1
        out["y"] = bot.astype(int)
        out["n_excluded_bot"] = int(excl.sum())
        out["n_excluded_human"] = int((~excl).sum())
        out["n_kept_bot"] = int(bot.sum())
        out["n_kept_human"] = int((~bot).sum())
    return out


def temporal_blocks(ev, n_bins: int = 24, lo: float = 1.0,
                    hi: float | None = None, min_events: int = 5,
                    alphas=SPECTRUM_ALPHAS) -> dict:
    """Build every temporal block for accounts with ``>= min_events`` events.

    Thin wrapper over :func:`temporal_blocks_ts`; see it for the contract.
    """
    ts_list = [ev.events_of(i)[0] for i in range(ev.n_users)]
    return temporal_blocks_ts(ts_list, labels=ev.labels, n_bins=n_bins,
                              lo=lo, hi=hi, min_events=min_events,
                              alphas=alphas)


def temporal_blocks_windowed(ev, window_days: float, n_bins: int = 24,
                             lo: float = 1.0, hi: float | None = None,
                             min_events: int = 5,
                             alphas=SPECTRUM_ALPHAS) -> dict:
    """Equal-window blocks (plan WP-F): truncate each account to its own
    first ``window_days`` of activity, then run the standard pipeline.

    Per account, keep only events with ``ts - ts[0] <= window_days * MS_PER_DAY``
    -- the same boundary convention as the WP-A probe's class-B truncation
    (``s[s <= W*MS_PER_DAY]``, plan §8 D5), so corpus rows and probe cells sit
    in a common coordinate and G4's comparison is like-for-like. The origin is
    each account's OWN first event, not a corpus-wide date: the confound under
    test is observation-window length, not calendar era. Everything downstream
    -- inter-arrival grid at the same pinned ``hi``, hour-of-day, count,
    burstiness, spectrum -- then runs unchanged in :func:`temporal_blocks_ts`.

    Exclusions are reported twice on purpose: ``n_excluded_*`` are the
    standard post-pipeline counts (fewer than ``min_events`` events IN the
    window); ``n_lost_to_window_*`` additionally isolates accounts that pass
    the global cutoff but fail only because of the window -- the population
    the equalisation actually sacrifices, per class.
    """
    W = float(window_days) * MS_PER_DAY
    ts_list = []
    for i in range(ev.n_users):
        ts, _ = ev.events_of(i)
        if len(ts):
            ts = ts[ts - ts[0] <= W]
        ts_list.append(ts)
    out = temporal_blocks_ts(ts_list, labels=ev.labels, n_bins=n_bins,
                             lo=lo, hi=hi, min_events=min_events,
                             alphas=alphas)
    out["window_days"] = float(window_days)
    passed_global = ev.counts() >= max(min_events, 2)
    kept = np.zeros(ev.n_users, dtype=bool)
    kept[out["index"]] = True
    lost = passed_global & ~kept
    bot = ev.labels == 1
    out["n_lost_to_window"] = int(lost.sum())
    out["n_lost_to_window_bot"] = int((lost & bot).sum())
    out["n_lost_to_window_human"] = int((lost & ~bot).sum())
    return out
