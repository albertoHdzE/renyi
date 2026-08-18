"""Synthetic accounts with known generators -- the S4.2 positive control.

Standing rule S4.2: any estimator applied to real accounts is applied unchanged,
in the same run, to accounts whose generating process is known. If it cannot
separate those, the result on real data is uninterpretable and is not reported.

Three generators, spanning the regimes H1's mechanism is about:

* **periodic**  -- a cron job. Inter-arrival times concentrate on one value.
* **poisson**   -- a randomised scheduler. Exponential inter-arrivals, CV = 1.
* **bursty**    -- human activity. Heavy-tailed (Pareto) inter-arrivals, CV >> 1;
  Barabasi (2005), *The origin of bursts and heavy tails in human dynamics*.

**They are generated with identical event counts on purpose.** That is what makes
this the decisive control for R1: if the spectrum separates periodic from Poisson
at matched ``n``, it is reading distributional shape; if it cannot, it is reading
volume, and P0 measured volume at AUC 0.939 on the real corpus.
"""

from __future__ import annotations

import numpy as np

__all__ = ["periodic_account", "poisson_account", "bursty_account",
           "GENERATORS", "generate_control_set"]

_MS_PER_HOUR = 3_600_000.0


def periodic_account(n: int, rng, period_ms: float = 6 * _MS_PER_HOUR,
                     jitter: float = 0.02) -> np.ndarray:
    """A cron job: fixed period with a little multiplicative jitter."""
    d = period_ms * (1.0 + jitter * rng.standard_normal(n - 1))
    return np.maximum(d, 1.0)


def poisson_account(n: int, rng, rate_per_hour: float = 1 / 6) -> np.ndarray:
    """A randomised scheduler: exponential inter-arrivals, CV = 1."""
    scale = _MS_PER_HOUR / rate_per_hour
    return np.maximum(rng.exponential(scale, size=n - 1), 1.0)


def bursty_account(n: int, rng, tail: float = 1.2,
                   scale_ms: float = 60_000.0) -> np.ndarray:
    """Human activity: Pareto inter-arrivals, heavy-tailed, CV >> 1.

    ``tail = 1.2`` puts the exponent in the range reported for human
    communication (Barabasi 2005 finds ~1 to ~1.5 across media).
    """
    return np.maximum(scale_ms * (rng.pareto(tail, size=n - 1) + 1.0), 1.0)


GENERATORS = {
    "periodic": periodic_account,
    "poisson": poisson_account,
    "bursty": bursty_account,
}


def generate_control_set(n_per_class: int = 100, n_events: int = 200,
                         rng=None, jitter: float = 0.5, tail: float = 1.2
                         ) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """``(inter_arrival_series, generator_index, names)`` at **matched n**.

    Every synthetic account has exactly ``n_events`` events, so event count
    carries zero information and any separation is distributional shape.

    ``jitter`` and ``tail`` set the difficulty. The defaults are deliberately
    **hard**: at ``jitter=0.02`` the three generators separate at accuracy 1.000
    and so does Shannon alone, which is a control that licenses nothing.
    """
    rng = np.random.default_rng(rng)
    names = list(GENERATORS)
    kw = {"periodic": {"jitter": jitter}, "poisson": {}, "bursty": {"tail": tail}}
    series, y = [], []
    for gi, name in enumerate(names):
        for _ in range(n_per_class):
            series.append(GENERATORS[name](n_events, rng, **kw[name]))
            y.append(gi)
    return series, np.array(y), names
