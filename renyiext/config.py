"""Paths and hyperparameters for 02-ext-research.

Single source of hyperparameters, following the convention set by
``dtwre.config``, ``disinfo.config`` and ``botsage.config``. Anything the design
documents do not fix is listed in :data:`INFERRED_PARAMETERS`.

Design documents live in ``02-ext-research/docs/``; this module must not
contradict them. Where it does, the document is right and this is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

__all__ = ["Config", "ROOT", "PROJECT_DIR", "DATA_RAW", "DATA_PROCESSED",
           "RESULTS", "FIGURES", "BITACORA", "TWITTER_EPOCH_MS",
           "CRESCI_WINDOW", "INFERRED_PARAMETERS"]

ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT / "02-ext-research"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed" / "ext"
RESULTS = PROJECT_DIR / "results"
FIGURES = RESULTS / "figures"
BITACORA = PROJECT_DIR / "bitacora"

# Twitter snowflake epoch, 2010-11-04T01:42:54.657Z, in milliseconds.
# Decision D1: timestamp_ms = (tweet_id >> 22) + TWITTER_EPOCH_MS
TWITTER_EPOCH_MS = 1_288_834_974_657

# First snowflake tweet id. Ids below this are the pre-2010 *sequential*
# scheme and carry NO timestamp -- decoding them yields the epoch instant
# itself, not a posting time. Decision D9; found by the P0 G1 render, which
# showed 63,830 Cresci-2015 tweets piled on a single millisecond.
# The corpus confirms the boundary: its largest sub-threshold id is
# 29,700,661,919 and its smallest snowflake is 292,906,606,796,800 -- a gap of
# four orders of magnitude with nothing in it.
FIRST_SNOWFLAKE_ID = 29_700_859_247

# Cresci-2015 was collected in 2015 but its tweets predate that. The window
# below is the *acceptance* range for property P14, not a measurement --
# it is deliberately wider than the observed range so the check can fail.
CRESCI_WINDOW = ("2010-11-01", "2015-12-31")


@dataclass
class Config:
    """Hyperparameters for one experiment."""

    # ---- entropy (docs/01-METHODS.md sect. 1) ----
    # Base 2 throughout (decision D2): CTM values are in bits, and mixing
    # bases inside one feature vector is a silent unit error.
    log_base: float = 2.0
    alpha_grid: tuple = (0.0, 0.5, 1.0, 2.0, 4.0, float("inf"))

    # ---- finite-sample bias control (decision D3, gate G1) ----
    # Fixed-n subsampling rather than bias correction: no standard correction
    # exists for general alpha. n identical across accounts makes the bias a
    # constant offset per alpha that cannot differ by class.
    n_events: int = 128            # sensitivity at 64 and 256
    n_bootstrap: int = 100

    # ---- behavioural alphabet (decision D5) ----
    # Four symbols, forced by tooling: acss.data and pybdm ship CTM tables for
    # alphabets 2, 4, 5, 6, 9. There is no alphabet-3 table.
    dna_alphabet: tuple = ("original", "reply", "retweet", "quote")
    dna_temporal_bins: int = 4     # quartiles of inter-arrival, fitted on train

    # ---- statistics (docs/02-PROTOCOL.md sect. 5) ----
    n_seeds: int = 10
    seed: int = 42
    # Differences below this are not claimed regardless of p: it is the
    # measured seed-to-seed sigma in this repository.
    effect_size_floor: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)


INFERRED_PARAMETERS = [
    ("n_events", "events per account for fixed-n subsampling",
     "guess at kick-off; gate G0 may force it down. Whatever it becomes is "
     "reported with every result, with per-class exclusion counts"),
    ("n_bootstrap", "subsample draws per account", "100; not tuned"),
    ("dna_temporal_bins", "quantisation of inter-arrival for temporal DNA",
     "4, to match the action alphabet and the available CTM tables"),
    ("alpha_grid", "which orders enter the spectrum",
     "endpoints 0, 1, inf are the named measures; 0.5, 2, 4 sample between "
     "them. Not tuned -- tuning alpha is the thing this design avoids"),
]
