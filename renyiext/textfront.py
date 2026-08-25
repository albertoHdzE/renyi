"""Text-front features (phase P3b, plan WP-I).

SPEC_X: word- and character-frequency spectra on **raw uncleaned text**
(decision D4). Word tokens are ``\\w+`` over Unicode with **no lowercasing**
-- casing is signal (D4); ``Sì`` and ``sì`` are different tokens. Character
frequencies run over every raw character of every kept tweet, counted
per tweet and summed (no separator characters are injected).

H₂ on the word-frequency distribution is the collision entropy
``-log2 sum p_i^2`` -- the same quadratic functional Yule's K normalises, so
if H₂ separates the classes the PHASES-P3 note applies: the result is
Yule's-K-flavoured vocabulary concentration, reported as such.

URL handling: the registered default keeps raw text untouched (D4). The
with/without-URL variant is a census/sensitivity item
(:func:`strip_urls`), never a silent substitution.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .spectrum import spectrum, SPECTRUM_ALPHAS

__all__ = ["WORD_RE", "URL_RE", "word_tokens", "strip_urls",
           "freq_spectrum", "account_text_features"]

WORD_RE = re.compile(r"\w+")          # Unicode-aware; no lowercasing (D4)
URL_RE = re.compile(r"https?://\S+")


def word_tokens(text: str | None) -> list[str]:
    """``\\w+`` tokens of the raw text, case-preserved."""
    return WORD_RE.findall(text or "")


def strip_urls(text: str | None) -> str:
    """Census/sensitivity variant: blank out ``http(s)://...`` spans."""
    return URL_RE.sub(" ", text or "")


def freq_spectrum(counts: dict | Counter | np.ndarray,
                  alphas=SPECTRUM_ALPHAS) -> np.ndarray:
    """6-vector Rényi spectrum of a frequency distribution.

    Dict/Counter keys are iterated in sorted order for byte-stability (the
    spectrum is order-invariant, P6; the float summation order is not).
    An empty count vector yields the all-zero spectrum (defined).
    """
    if isinstance(counts, (dict, Counter)):
        vec = np.array([counts[k] for k in sorted(counts)], dtype=np.float64)
    else:
        vec = np.asarray(counts, dtype=np.float64)
    from .spectrum import counts_to_probabilities
    return spectrum(counts_to_probabilities(vec), alphas)


def account_text_features(texts: list[str], strip: bool = False,
                          alphas=SPECTRUM_ALPHAS) -> tuple[np.ndarray, np.ndarray, int]:
    """One account's ``(word_spectrum, char_spectrum, n_tokens)``.

    ``texts`` are the account's raw tweet texts in time order. Character
    counts are summed per tweet (no separators); word tokens come from the
    raw text unless ``strip`` blanks URL spans first (census variant only).
    """
    wc: Counter = Counter()
    cc: Counter = Counter()
    n_tokens = 0
    for t in texts:
        body = strip_urls(t) if strip else (t or "")
        toks = word_tokens(body)
        n_tokens += len(toks)
        wc.update(toks)
        cc.update(t or "")
    return (freq_spectrum(wc, alphas), freq_spectrum(cc, alphas), n_tokens)
