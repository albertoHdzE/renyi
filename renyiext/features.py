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


def temporal_blocks(ev, n_bins: int = 24, lo: float = 1.0,
                    hi: float | None = None, min_events: int = 5,
                    alphas=SPECTRUM_ALPHAS) -> dict:
    """Build every temporal block for accounts with ``>= min_events`` events.

    ``hi`` fixes the top of the log grid. Leaving it ``None`` makes the grid
    *account-dependent*, which silently encodes each account's own maximum gap
    into the binning -- a per-unit parameter masquerading as a fixed one. The
    P2 sweep therefore pins ``hi`` to a constant across accounts.

    Returns blocks, labels, the kept mask, and the per-class exclusion counts,
    which are reported with every downstream result (D3').
    """
    keep = ev.counts() >= max(min_events, 2)
    idx = np.where(keep)[0]

    rows_count, rows_burst, rows_shan, rows_spec = [], [], [], []
    for i in idx:
        ts, _ = ev.events_of(i)
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
    bot = ev.labels[idx] == 1
    excl = ev.labels[~keep] == 1

    return {
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
        "y": bot.astype(int),
        "index": idx,
        "n_excluded_bot": int(excl.sum()),
        "n_excluded_human": int((~excl).sum()),
        "n_kept_bot": int(bot.sum()),
        "n_kept_human": int((~bot).sum()),
    }
