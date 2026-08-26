"""Language-ID front for amendment LS-1 (bitacora 24): is the char-spectrum
inversion (F16) mediated by corpus-language composition?

Additive module (plan §9.7): nothing existing imports it; closed producers
are untouched. Tooling decision recorded in bitacora 24 §3: ``langid``
1.1.6 — verified installable on this machine's py3.13 BEFORE registration;
deterministic forced-choice naive Bayes over byte n-grams with the model
shipped in the distribution (no runtime download). ``lingua-py`` was NOT
resolvable from this machine's package index and is recorded as such
(the R10 lesson: check PyPI before registering a dependency).

Semantics frozen by the registration:

- restriction set :data:`LANGS` (16 codes), installed globally at import;
- account-level unit: texts joined with ``"\\n"`` in loader order,
  truncated to the first :data:`ACCOUNT_BUDGET_DEFAULT` chars, ONE
  classify call -> ``(lang, confidence)`` (bitacora 24 §3);
- per-tweet secondary: first :data:`TWEET_HEAD_CHARS` chars of EVERY
  tweet — execution measured langid at ~0.05 ms per short call, so no
  subsample knob exists to invent;
- forced-choice argmax among the set: no unknown class, no confidence
  threshold; the confidence distribution is recorded descriptively.
"""
from __future__ import annotations

from collections import Counter
from importlib.metadata import version as _pkg_version

import langid

__all__ = ["LANGS", "ACCOUNT_BUDGET_DEFAULT", "TWEET_HEAD_CHARS",
           "WHEEL_SHA256", "classify_account", "tweet_en_stats",
           "assert_deterministic", "tool_echo"]

LANGS = ("en", "it", "es", "pt", "fr", "de", "ja", "ko", "zh", "ru",
         "ar", "tr", "id", "tl", "nl", "pl")
ACCOUNT_BUDGET_DEFAULT = 20_000
TWEET_HEAD_CHARS = 200

# sha256 of the langid 1.1.6 sdist exactly as fetched from PyPI on
# 2026-08-25 via `pip download --no-deps langid==1.1.6`; config-echo
# provenance so the tool identity is pinned beside its outputs.
WHEEL_SHA256 = ("044bcae1912dab85c33d8e98f2811b8f4ff1213e5e9a9e95101"
                "37b84da2cb293")

langid.set_languages(list(LANGS))


def classify_account(texts, budget: int = ACCOUNT_BUDGET_DEFAULT):
    """One account -> ``(lang, confidence)`` on the concatenated timeline.

    Concatenation order is the loader's time-sorted order; the join is
    ``"\\n"`` so adjacent tweets cannot fuse into one n-gram stream.
    Empty input returns ``("", nan)`` (the token-positive population of
    the registration never hits this branch; kept for honesty).
    """
    concat = "\n".join(t for t in texts if t)[:budget]
    if not concat:
        return "", float("nan")
    lang, conf = langid.classify(concat)
    return lang, float(conf)


def tweet_en_stats(texts, first_chars: int = TWEET_HEAD_CHARS) -> dict:
    """Per-tweet secondary (descriptive, never gating): classify the first
    ``first_chars`` of every tweet and return the account's EN share plus
    the full label counter."""
    labels = []
    for t in texts:
        if not t:
            continue
        lab, _conf = langid.classify(t[:first_chars])
        labels.append(lab)
    c = Counter(labels)
    n = sum(c.values())
    return {"en_share": (c.get("en", 0) / n if n else float("nan")),
            "n_tweets": n,
            "labels": dict(c)}


def assert_deterministic() -> None:
    """Inline property check (repo style: when touching a tool, check a
    property): same fixture twice — including after re-installing the
    language set — must classify identically."""
    fixture = ("just had the best coffee of my life, good morning everyone "
               "buongiorno a tutti, oggi che bella giornata al mare")
    a = langid.classify(fixture)
    langid.set_languages(list(LANGS))
    b = langid.classify(fixture)
    assert a == b, f"langid nondeterministic on fixture: {a} vs {b}"


def tool_echo() -> dict:
    """Config echo: the exact tool identity, frozen into every artefact."""
    return {
        "library": "langid",
        "version": _pkg_version("langid"),
        "wheel_sha256_sdist": WHEEL_SHA256,
        "restriction_set": list(LANGS),
        "account_join": "\\n",
        "account_budget_chars_default": ACCOUNT_BUDGET_DEFAULT,
        "tweet_head_chars": TWEET_HEAD_CHARS,
        "semantics": "forced-choice argmax among restriction set; "
                     "no unknown class, no confidence threshold",
        "registered_in": "bitacora 24 sect. 3",
    }
