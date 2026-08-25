"""Behavioural-alphabet and mention-target features (phase P3a, plan WP-H).

Two blocks, each a 6-vector Rényi spectrum (base 2, plug-in, no bias
correction -- the D3' stance shared with ``features.py``):

    SPEC_B_ALPHA     spectrum of the per-account post-type distribution
    SPEC_B_MENTION   spectrum of the per-account mention-target distribution

Post-type collapse (decision D11, pre-registered v1.0): for *spectra* the
four-symbol action alphabet is collapsed ``quote -> original`` to
{original, reply, retweet}. The 11.6 % quote share implied by the
trailing-t.co proxy is not credible for 2011-13 Twitter, and the CTM
alphabet-4 constraint (D5) binds only P4's DNA strings, not spectra. The
4-symbol encoding stays available (:func:`post_type_counts` with
``collapse=False``) as the published sensitivity axis; the raw 4-symbol form
remains the DNA encoding in P4.

Mention-target specification (pre-registered v1.0): all ``@\\w+`` tokens in
raw text; the leading token is excluded on reply-classified tweets (the
replied-to party is not audience); self-mentions are excluded (token matches
the account's own username, case-insensitively); an account with no
remaining mention is excluded from the mention block and counted per class.
Retweets keep their ``RT @a`` target -- only the reply-leading token is
dropped, per the registered rule.

The spectrum is order-invariant (P6), so a marginal-preserving shuffle of a
post-type sequence must leave these features bit-identical;
:func:`run_p3h_behaviour` runs that null end-to-end and states the
separating world in its JSON (S4.1 logic).
"""

from __future__ import annotations

import re

import numpy as np

from .spectrum import spectrum, SPECTRUM_ALPHAS
from .events import POST_TYPE

__all__ = ["ALPHABET3", "MENTION_RE", "collapse_post_types",
           "post_type_counts", "alphabet_spectrum", "extract_mentions",
           "mention_spectrum"]

# D11 collapsed alphabet for spectra; D5's four-symbol form stays with DNA.
ALPHABET3 = ("original", "reply", "retweet")
MENTION_RE = re.compile(r"@\w+")


def collapse_post_types(types: np.ndarray) -> np.ndarray:
    """Map quote (3) -> original (0); other symbols unchanged."""
    t = np.asarray(types)
    out = t.copy()
    out[t == POST_TYPE["quote"]] = POST_TYPE["original"]
    return out


def post_type_counts(types: np.ndarray, collapse: bool = True) -> np.ndarray:
    """Counts over the (collapsed) post-type alphabet, fixed symbol order."""
    t = collapse_post_types(types) if collapse else np.asarray(types)
    n = len(ALPHABET3) if collapse else 4
    return np.bincount(t, minlength=n)[:n].astype(np.float64)


def alphabet_spectrum(types: np.ndarray, alphas=SPECTRUM_ALPHAS,
                      collapse: bool = True) -> np.ndarray:
    """SPEC_B_ALPHA: 6-vector spectrum of the post-type distribution."""
    from .spectrum import counts_to_probabilities
    return spectrum(counts_to_probabilities(post_type_counts(types, collapse)),
                    alphas)


def extract_mentions(text: str | None, post_type: int,
                     own_username: str | None) -> tuple[list[str], dict]:
    """Registered mention rules, with per-rule capture accounting.

    Returns ``(kept_targets, info)`` where ``info`` counts, for this tweet:
    tokens found, whether the reply-leading token was dropped, and how many
    self-mentions were dropped. Kept targets keep their text case; the
    self-mention match is case-insensitive (usernames are on Twitter).
    """
    text = text or ""
    tokens = [m.group(0) for m in MENTION_RE.finditer(text)]
    info = {"n_tokens": len(tokens), "leading_dropped": False,
            "n_self_dropped": 0}
    if not tokens:
        return [], info
    if post_type == POST_TYPE["reply"]:
        tokens = tokens[1:]                    # replied-to party != audience
        info["leading_dropped"] = True
    own = (own_username or "").lower()
    kept = []
    for tok in tokens:
        if own and tok[1:].lower() == own:
            info["n_self_dropped"] += 1
            continue
        kept.append(tok)
    return kept, info


def mention_spectrum(counts: dict | np.ndarray, alphas=SPECTRUM_ALPHAS
                     ) -> np.ndarray:
    """SPEC_B_MENTION: 6-vector spectrum of the mention-target distribution.

    ``counts`` maps target -> occurrences (dict; iterated in sorted key
    order for byte-stability -- the spectrum is order-invariant, P6, but the
    float summation order is not) or is a count vector in that order.
    """
    from .spectrum import counts_to_probabilities
    if isinstance(counts, dict):
        vec = np.array([counts[k] for k in sorted(counts)], dtype=np.float64)
    else:
        vec = np.asarray(counts, dtype=np.float64)
    return spectrum(counts_to_probabilities(vec), alphas)
