# Bitácora 24 — Amendment LS-1 (pre-registered BEFORE any language computation): is the char-spectrum inversion a corpus-language effect?

**Date:** 2026-08-25
**Branch:** `main`
**Nature:** post-plan follow-up diagnostic under plan §9.3, registered before
any language identification has been run or any language-tagged view of
either corpus has been seen. It adjudicates the suspect F16 named
("hypothesis, not conclusion"): **H_LS**. It is NOT a charter-hypothesis
amendment: H4-T's failure stands regardless of what LS-1 finds.
**Provenance status:** PRELIMINARY — inherited unchanged from bitacora 23
§1 (Kaggle copy; sha256 chain and re-derivation obligation on the
authorized copy apply to every artefact of this amendment too).

---

## 1. What is being explained

The registered facts LS-1 must speak to (all committed; JSON paths in
EVIDENCE-INDEX "Amendment H4-T"):

- SPEC_X_CHAR transfers at **0.4840** (CI95 [0.4719, 0.4925]) from a
  within-corpus **0.9898** — below chance (`families.SPEC_X_CHAR|hgb`).
- Target-local char orders are near-chance ON TwiBot-20 alone:
  H₁ **0.5110**, H∞ **0.5358** (probe recorded in bitacora 23 §7.2).
- The G4 null band is max |Δ| **0.0134**
  (`sanity_null_g4.max_abs_delta`); equal-cap window shifts ≤ **0.0126**
  (`equal_cap_block.sigma_config_across_configs_hgb`) — so neither draw
  noise nor the observed-window mismatch explains the inversion.

**H_LS**: the inversion is mediated by corpus-language composition —
Cresci-2015's bot/human classes differ in language mix, TwiBot-20's do not
(English-dominant in both classes), so a source-fitted character-usage
boundary partially encodes an English-ness axis whose class direction does
not survive the crossing.

## 2. Scope and non-goals

- Primary object: the SPEC_X_CHAR inversion specifically (the only
  below-chance transfer measured).
- LS-1 does not reopen H4-T, does not touch any floor or threshold of any
  closed phase (plan §9.3), and cannot upgrade H4-T's verdict whatever its
  outcome. Its own failure mode is pre-committed as a result (§7).
- Nothing new is invented for the spectra: features stay WP-I definitions
  via `renyiext.textfront.account_text_features`; the estimator stays D6.

## 3. Tooling (verified BEFORE registration — the R10 lesson)

- **langid == 1.1.6**: installed and probed today on this machine
  (py3.13 venv, deterministic forced-choice NB over byte n-grams, model
  shipped in the wheel — no runtime download). Added to
  `requirements-frozen.txt` at execution time with this entry.
- `lingua-py` is **NOT resolvable** from this machine's package index
  (checked twice, `pip index`/`pip download`); recorded like WP-L's R10,
  not worked around.
- Frozen restriction set (16 codes, major Twitter languages of these
  corpora's era; restriction is known to help short-text accuracy vs
  all-97): `{en, it, es, pt, fr, de, ja, ko, zh, ru, ar, tr, id, tl, nl,
  pl}`. Semantics: forced-choice argmax among the set — no unknown class,
  no confidence threshold to tune; per-account confidence distribution
  recorded descriptively.
- **Account-level unit (primary)**: concatenate the account's texts in
  loader order (time-sorted), truncate to the first **20,000 chars**,
  classify once → EN-account iff label `'en'`. The budget is an invented
  parameter, so it is swept: a **10,000-char** variant is reported beside
  the primary (σ_config-style) and must not change any verdict (§7).
- **Per-tweet secondary (descriptive only)**: first 200 chars per tweet →
  per-account EN-tweet share; rendered, never gating.
- Determinism asserted inline: same string twice ⇒ identical label; wheel
  sha256 + import version echoed into the JSON config block.

## 4. Populations

H4-T §3 populations verbatim, then intersected with the LID rule:

- token-positive source n = 4,770 (bot 2,846);
- token-positive target n = 11,746 (bot 6,561)
  (`composition.token_positive_*`);
- restriction exclusions counted per class per side and landed in the
  JSON beside the H4-T exclusion counts;
- the EN∩EN restricted population's balance is reported next to every
  restricted reading (composition caveat discipline, bitacora 17).

## 5. Arms

- **A · Census (descriptive).** Language-label distribution per corpus ×
  class at account grain; per-class histograms of per-tweet EN share;
  EN-share table per class per corpus. Figures render the distributions
  themselves (account/tweet grain), not summary bars alone
  (`p6nls_language_census.png`, `p6nls_tweet_share.png`).
- **B · PRIMARY intervention.** Re-run `d6_family` byte-identical to
  H4-T/WP-K/N (R = 20, `default_rng(42)`; B = 1000, `default_rng(1042)`
  shared resamples; HGB rs=42; LR secondary) for SPEC_X_CHAR on
  (i) the full token-positive population [reference: already measured] and
  (ii) the EN∩EN population. Context arms on (ii): META_raw[naive],
  SHAN_CHAR+NOISE(5) (salt 104, the arm's declared salt), SPEC_X_WORD.
  Sensitivities on the primary arm only: equal-cap config (one row) and
  the 10k concat budget (§3).
- **B2 · Directionality placebo (context, ungated).** EN-restricted
  SOURCE → FULL target, SPEC_X_CHAR only: if repair survives here, the
  mismatch is target-side; if not, source-side purity was doing it.
- **C · Local restoration (exploratory).** Within-TB20 5-fold AUC of the
  char spectra on EN∩EN vs full target (`renyiext.evaluate.run_arms`,
  producer seed convention `range(42, 52)`): does the locally absent
  fingerprint reappear once language-matched?
- **D · Nulls (calibration).** G4 permutation-half null (`default_rng(77)`)
  through BOTH pipelines (full and restricted), SPEC_X_CHAR + META arms,
  expected silent in both. Named in advance: a null separates here only if
  partition noise alone exceeded the bar — its silence bounds how large a
  Δ_inv the noise floor can fake.

## 6. Verdict rules (pre-committed)

Primary quantity (hgb headline):
`Δ_inv = transfer_mean(full) − transfer_mean(EN∩EN)`.

- Bar **0.05** — the charter threshold inherited verbatim: the programme
  already refuses to call smaller degradations effects (H4-T §5), and a
  mediation claim deserves no softer ruler than the claim it mediates.
- **SUPPORTED** iff Δ_inv > 0.05 AND ≥ 16/20 paired draws individually
  > 0.05 AND repair holds: restricted `transfer_mean > 0.5` with CI95
  lower bound > 0.5 (inversion actually flips back above chance — movement
  without sign repair is not mechanism).
- P1 composition corroboration (share points): e(C15,bots) − e(C15,hum)
  > **0.20** AND min(e(TB20,bots), e(TB20,hum)) − e(C15,hum) > **0.20**.
  The bar is declared a-priori as a large-effect margin on the share
  scale, and the census is reported in full regardless, so the exact value
  cannot hide anything (insensitive anywhere in [0.15, 0.30]).
- Combined vocabulary (partial success must not read as whole success):
  **SUPPORTED** (primary + P1) · **PARTIAL-MECHANISM** (primary yes, P1
  no — mediation real but not through measured composition) ·
  **PARTIAL-EFFECT** (primary no, P1 yes — shift real, innocent of the
  inversion) · **REFUTED** (both no).
- Exploratory family (Holm within it): C-arm restricted-vs-full local
  separability (SPEC_X_CHAR + H₁/H∞ slices), B2 row, budget/equal-cap
  sensitivity rows. The gated set is exactly {Δ_inv conjuncts, P1(i),
  P1(ii)} — four numbers, declared before execution.
- Either way the outcome lands in FINDINGS as F17 with the same weight a
  pass would carry (charter H0).

## 7. Execution protocol

- Script `scripts/run_p6n_languageshift.py --quiet`; artefact
  `results/p6n_ls1.json`; figures `p6nls_*.png`. OMP pinned in-script.
- Two-run byte-equality excluding the declared volatile block (wall-time
  fields), same mechanism as H4-T.
- `renyiext.checks.run_all()` must stay 12/12: changes are additive (LID
  wrapper module + loader reuse); closed producers untouched (§9.7).
- No language statistic exists at commit time of this registration; the
  diff boundary "registered vs seen" is this commit.

*Execution results append as §8+ after the runs.*

---

## 8. Execution results (2026-08-25; run twice, byte-identical; checks 12/12 after the additive `renyiext.lid` module)

**Gates first.** Fidelity vs the committed H4-T artefact: max |diff|
**0.0** over all eight full-reference D6 arms × seven fields each
(incl. both CI bounds) — the machinery provably did not drift
(`fidelity_gate_vs_h4t`). The full-pipeline G4 null reproduces its
committed values exactly (`sanity_null_g4.committed_fidelity_max_abs_diff`
= 0.0). Two-run byte-equality PASS (no volatile fields in the JSON).
Artefact: `results/p6n_ls1.json`; figures `p6nls_*.png`.

Execution resolutions (sub-decisions, recorded per §0 convention): the
per-tweet secondary ran on **every** tweet — measured langid cost
(~0.05 ms/short call) made the subsample knob the registration left
unspecified unnecessary, so none was invented; the account concat joins
with `"\n"` (frozen in `lid.py` and the config echo); the
restricted-pipeline null sizes its own boot stream; `c_local_restoration`
reuses `evaluate.paired`, whose `clears_floor` key carries the 0.02 AUC
floor semantics (not the P1 share bar).

### 8.1 Arm A — the shift is real and large: P1 PASS

| EN share (account grain) | Cresci-2015 | TwiBot-20 |
|---|---|---|
| bot | **0.5418** | **0.9308** |
| human | **0.0567** (93.1 % Italian) | **0.7556** |

Predicates: gap(i) = **+0.4852**, margin(ii) = **+0.6990** — both ≫ the
0.20 bar (`p1_composition`). Per-tweet means agree (0.591 / 0.176 /
0.858 / 0.711). The census figure renders the distributions: Cresci
humans are an Italian corpus; TwiBot-20 is English-dominant in BOTH
classes, exactly the registered pattern.

### 8.2 Arm B — the primary: language-matching does NOT repair the inversion

| quantity (headline config) | hgb | lr |
|---|---|---|
| Δ_INV (full − EN∩EN) | **−0.0028** | +0.0360 |
| draws > 0.05 | **0/20** | 0/20 |
| restricted transfer (CI95 lo) | 0.4867 (0.4754) | 0.4955 (0.4840) |

`verdict_primary` = **FAIL_bar_or_draws_or_repair** on both heads: the
inversion persists inside the language-matched subpopulations, still
below chance. Sensitivities: budget-10k restriction moves Δ_INV by
**0.0000**; equal-cap source config by **0.0270** — both far under the
bar, so the failure is not a config artefact. B2 placebo (EN-source →
FULL-target): transfer 0.4800 hgb / 0.5023 lr — source-side purity alone
does nothing either.

**Composition caveat, carried beside the verdict (it does not flip the
registered rules):** the EN∩EN source holds only **109 humans among
1,651 accounts** (`composition.restricted_source`; exclusions
`source_non_en_human` 1,815) — a repair that needed many English source
humans was never testable with real power. The B2 placebo and the C-arm
corroborate the negative from designs that do not share this limitation.

### 8.3 Arm C — the locally absent fingerprint does not come back

Within-TwiBot-20 local AUC (10 seeds, hgb): SPEC_X_CHAR 0.6381 ± 0.0026
full → **0.5969 ± 0.0034** restricted (diff **−0.0412**, 0/10 seeds,
holm p 0.0059); H₁ −0.0190 (0/10); H∞ +0.0146 (10/10, subfloor).
Language matching makes the multivariate char separability WORSE, not
better (`c_local_restoration`).

### 8.4 Arm D — nulls

Full pipeline max |Δ| **0.0134** (byte-identical to the committed band);
restricted pipeline **0.0148** — same order. Neither pipeline's noise
floor approaches any quantity under adjudication.

### 8.5 Combined verdict: PARTIAL-EFFECT

The corpus-language shift is real and enormous, and it is **innocent of
the inversion**: F16's named suspect is REFUTED as its mediator, as
pre-committed by the vocabulary for "primary no, P1 yes". The
programme-level reading sharpens: the char fingerprint's transfer
failure is **corpus-lineage-bound, not language-tag-bound** — the
transferred boundary is bimodal-confident on both classes (mass at both
score extremes, classes mixed at both; `figures/p6nls_inversion.png`)
and language matching moves nothing. Whatever the char spectrum keyed on
in Cresci-2015, it is a property of how that corpus was collected, not
of what language its accounts tweeted in.

Consequences (FINDINGS F17): any future transfer claim leaning on text
features must sit a same-lineage exam; F16's "language profile matched
to the question" recommendation is sharpened to "language matching is
insufficient — the variable the data cannot supply is collection
lineage". The authorized-copy re-derivation obligation (§1) covers this
artefact too.
