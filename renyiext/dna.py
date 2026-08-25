"""Digital DNA encodings (phase P4, plan WP-L; decision D5).

Two per-account strings over the four-symbol action alphabet
{original, reply, retweet, quote} -- the DNA keeps all four symbols
(including quote); the CTM alphabet-4 tables bind here, unlike the
spectra where D11 collapses quote into original.

    action_dna     symbols in time order of the account's tweets
    temporal_dna   each inter-arrival gap binned into 4 quartiles; the
                   edges are computed FROM A REFERENCE SUBSET supplied by
                   the caller -- for any train/test protocol the edges
                   must come from the TRAIN side only (leakage rule)

Symbol alphabet: ``O`` original, ``R`` reply, ``T`` retweet, ``Q`` quote
(temporal: ``1``..``4``, shortest to longest gap). Strings are Python str
over that fixed alphabet; ``ait`` handles byte encoding for the
compressors and integer arrays for BDM.
"""

from __future__ import annotations

import numpy as np

from .events import POST_TYPE

__all__ = ["ACTION_SYMBOLS", "TEMPORAL_SYMBOLS", "action_dna",
           "temporal_dna", "quartile_edges"]

ACTION_SYMBOLS = "ORTQ"          # index = POST_TYPE value
TEMPORAL_SYMBOLS = "1234"


def action_dna(types) -> str:
    """Per-account action DNA: one symbol per tweet, time order (D5)."""
    t = np.asarray(types)
    return "".join(ACTION_SYMBOLS[int(v)] for v in t)


def quartile_edges(dt_reference) -> np.ndarray:
    """Quartile edges of the inter-arrival distribution of a REFERENCE
    subset (train side under any split protocol -- leakage rule)."""
    dt = np.asarray(dt_reference, dtype=np.float64)
    dt = dt[dt > 0]
    return np.quantile(dt, [0.25, 0.5, 0.75])


def temporal_dna(ts, edges) -> str:
    """Per-account temporal DNA: gaps binned by the supplied (train-side)
    quartile edges into symbols 1..4 (shortest to longest)."""
    ts = np.sort(np.asarray(ts, dtype=np.int64))
    if ts.size < 2:
        return ""
    dt = np.diff(ts).astype(np.float64)
    idx = np.searchsorted(edges, dt, side="right")
    return "".join(TEMPORAL_SYMBOLS[i] for i in idx)
