"""Tail statistics for the inter-arrival front (plan WP-J; §8 D8).

Two per-account statistics aimed at the tail mechanism H1 names:

    TAIL   Hill tail-index estimate alpha_hat of the inter-arrival
           distribution (§8 D8): sort gaps ascending, k = max(3, n//4)
           top gaps, H = mean(ln gap - ln gap_(j)) over the top k,
           alpha_hat = 1/H clipped to [0.3, 20]. k is returned alongside.
    SURV   empirical survival P(gap > t) at the fixed corpus-wide lags
           {1 h, 1 d, 7 d} -- three proportions, nothing fitted.

Bias stance (house rule, D8): plug-in, no bias correction. For an exact
Pareto(alpha) sample the top-k spacing sum is unbiased for 1/alpha, so the
upward bias in alpha_hat is the Jensen effect of the reciprocal
(E[1/H] > 1/E[H] = alpha), largest at small k; P17 measures the direction
and magnitude at n = 500 and the docstring states it. Zero-length gaps
(simultaneous posts -- real and kept everywhere else in this repository)
carry no tail information and are excluded from TAIL only; SURV uses all
gaps (P(0 > t) = 0 is a true statement about the distribution).
"""

from __future__ import annotations

import numpy as np

__all__ = ["hill_alpha", "survival_rates", "SURV_LAGS_MS", "tail_features"]

SURV_LAGS_MS = (3_600_000.0, 86_400_000.0, 604_800_000.0)   # 1 h, 1 d, 7 d
_ALPHA_MIN, _ALPHA_MAX = 0.3, 20.0


def hill_alpha(dt: np.ndarray, k: int | None = None):
    """§8 D8 Hill estimate. Returns ``(alpha_hat, k_used, n_positive)``.

    ``alpha_hat`` is NaN when fewer than two positive gaps exist (the tail
    is undefined); the clip to [0.3, 20] bounds degenerate cases (e.g. all
    top gaps equal -> H = 0).
    """
    dt = np.asarray(dt, dtype=np.float64)
    dt = dt[dt > 0]
    n = dt.size
    if n < 2:
        return float("nan"), 0, int(n)
    k_used = max(3, n // 4) if k is None else k
    k_used = min(k_used, n - 1)
    dt = np.sort(dt)
    j = n - k_used
    h = float(np.mean(np.log(dt[j:]) - np.log(dt[j])))
    if h <= 0:
        return float(_ALPHA_MAX), int(k_used), int(n)
    a = 1.0 / h
    return float(min(max(a, _ALPHA_MIN), _ALPHA_MAX)), int(k_used), int(n)


def survival_rates(dt: np.ndarray, lags=SURV_LAGS_MS) -> np.ndarray:
    """Empirical P(gap > t) at the fixed lags; all gaps count."""
    dt = np.asarray(dt, dtype=np.float64)
    if dt.size == 0:
        return np.full(len(lags), np.nan)
    return np.array([float((dt > t).mean()) for t in lags])


def tail_features(ts: np.ndarray):
    """``(alpha, k, n_positive, surv3)`` from one account's timestamps."""
    ts = np.sort(np.asarray(ts, dtype=np.int64))
    dt = np.diff(ts).astype(np.float64) if ts.size > 1 else np.empty(0)
    a, k, npos = hill_alpha(dt)
    return a, k, npos, survival_rates(dt)
