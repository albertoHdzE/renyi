# Bitácora 14 — WP-I, the text front: the spectrum finally beats its Shannon slice, Holm-surviving

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** exploratory under H2 (passed, bitacora 13); Holm-corrected within
the SPEC_X family. Property checks 9/9. Two runs byte-identical (S2.6).
**Artefacts:** `renyiext/textfront.py` (new),
`renyiext/events.py::load_cresci_text_side` (additive loader),
`scripts/run_p3i_textfront.py` (new), `results/p3i_textfront.json`,
`figures/p3i_spec_x_objects.png`, `figures/p3i_length.png`.
**Seeds:** 42–51; fidelity vs the temporal producer (kept index + COUNT) at
0.0. Every number in this artefact is EXPLORATORY.

---

## 1. The headline: SPEC_X_CHAR 0.9889, and the spectrum beats its Shannon slice at matched dimensions

| arm (n = 4,770) | AUC ± SD | TPR@1% pooled/fold |
|---|---|---|
| COUNT | 0.9400 ± 0.0011 | 0.141 / 0.141 |
| TOKENS | 0.9498 ± 0.0009 | 0.280 / 0.260 |
| META-lite | 0.9972 ± 0.0003 | 0.976 / 0.973 |
| SHAN_CHAR (H_1 alone) | 0.6751 ± 0.0051 | 0.164 / 0.157 |
| SPEC_X_WORD | 0.9741 ± 0.0009 | 0.598 / 0.579 |
| **SPEC_X_CHAR** | **0.9889 ± 0.0004** | 0.837 / 0.821 |
| **SPEC_X (12)** | **0.9940 ± 0.0003** | 0.915 / 0.904 |
| COUNT+SPEC_X | 0.9948 ± 0.0003 | 0.934 / 0.917 |

This is the first front where the full order-set decisively beats its own
Shannon slice **at matched dimensions**: SPEC_X_CHAR vs SHAN_CHAR+NOISE(5) =
**+0.3332**; SPEC_X vs SHAN_X+NOISE(10) = **+0.0537**; SPEC_X_WORD vs
SHAN_WORD+NOISE(5) = **+0.0466**; COUNT+SPEC_X vs COUNT+SHAN_X+NOISE(10) =
**+0.0318** — all `supports_clause`, 10/10, and **all 20 family comparisons
survive Holm** (raw p = 0.0020 each, adjusted 0.0391). The char block's
signal lives in the tail orders the Shannon slice throws away — the
α-curve render shows near-overlapping means at H_1 but separation growing
toward H_4/H_inf, carried multivariately.

## 2. Against the floors: volume and length beaten; META still not beaten

vs COUNT +0.0540 (SPEC_X), vs TOKENS +0.0446 (TOKENS+SPEC_X) — the text
front beats both volume covariates, and the ≥ 512-token sensitivity
(631 bots / 1,817 humans survive) makes the word-vs-COUNT edge *larger*
(+0.0636; σ_cfg across the split 0.0147). URL stripping moves SPEC_X_WORD by
only −0.0077 (census: 27.02 % of tweets carry a URL; bot 24.25 % / human
27.22 %) — the front is not a URL artefact.

vs META the WP-E semantics fire again: SPEC_X_WORD vs META+NOISE(2)
**−0.0233 → `confounded_dimensionality`**; SPEC_X vs META+NOISE(8)
**−0.0036 → `confounded_dimensionality`** (though the gap is now ~0.003 —
SPEC_X at 0.9940 is nearly at the incumbent's 0.9975 ceiling). Downgrade
executed in HANDOFF/FINDINGS: not claimable beyond the incumbent on
Cresci-15; the transfer question stays with WP-N.

## 3. Length control (R2 amended) — another suppression, different shape

Raw correlations of word orders are strongly negative (H_0_word −0.801) and
collapse to ~0 given tokens (+0.042) — word diversity tracks volume.
Character orders behave oppositely: positive given tokens (H_0_char +0.351).
The two blocks are not redundant: word spectra encode *diversity vs length*,
char spectra encode *character usage given length*. All 12 orders' raw /
given-tokens / given-count partials are in the JSON (acceptance box).

## 4. Census and objects (G1–G3)

Token regex `\w+` Unicode, NO lowercasing — only the registered rule was
tried; the with/without-URL variant was counted, not adopted. Zero-token
exclusions: 0/0 (every kept account has text). The objects render: the
example bot's longest tweet is a copied news snippet whose top tokens all
appear exactly once (the copied-text signature: uniform singletons); the
example human's is conversational with repeated function words. Length
render: bot median **272** tokens vs human **11,478** — the confound the
design controls.

Yule's K note (PHASES P3): H₂ on word frequency is the collision entropy
−log₂ Σp² — Yule's K's functional. H_2_word given tokens is +0.132 (raw
−0.274): the vocabulary-concentration signal is length-confounded raw and
survives conditioning weakly; the word front's separation is carried mostly
by H_0/H_0.5 (vocabulary size), not H_2.

## 5. Decisions taken under the ambiguity protocol

1. **Loader promoted to the data layer** (`events.load_cresci_text_side`,
   generalised from the WP-H producer's script-local loader with an
   elementwise alignment assertion per account). The closed WP-H producer is
   left untouched — no re-run of a closed phase (§9.7); its local copy stays
   with a provenance note in the new function's docstring.
2. **Holm family membership** = all paired comparisons in this artefact
   (13 gated + 7 dim-matched = 20), documented in the JSON; step-down,
   α = 0.05.
3. **σ_config axis**: the ≥ 512-token sensitivity split is this WP's only
   config axis; σ_cfg recorded where both sides exist
   (SPEC_X_WORD-vs-COUNT: 0.0147), absent otherwise and marked so.

## 6. What failed and was not fixed

Two crashes before any output: a naive-datetime `timestamp()` (local-time
age reference) and a missing TOKENS+SHAN_X floor arm. Fixed; nothing
outstanding.

## 7. Multiple comparisons counted

20 paired comparisons (13 gated + 7 matched), Holm-corrected as a single
family; 12 × 3 partial correlations; 1 URL-stripped sensitivity arm; 1
≥512-token sensitivity with 4 arms. All exploratory; no hypothesis-gate
verdict changes hands here.
