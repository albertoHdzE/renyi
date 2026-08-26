# Bitácora 23 — Amendment H4-T (pre-registered BEFORE outcome analysis): the text front sits the transfer exam on TwiBot-20

**Date:** 2026-08-25
**Branch:** `main`
**Nature:** post-plan amendment under plan §9.3, registered before any
target-side feature or model computation. Structure-only facts it relies on
are those of bitacora 22 (counts, types, overlap — no outcomes exist yet).
**Provenance status:** PRELIMINARY — see §1.

---

## 1. Provenance decision (project owner, 2026-08-25)

The `train/dev/test.json` copy (~342 MB, sha256 below) comes from the
third-party Kaggle re-upload (`marvinvanbo/twibot-20`), not from the
authors. The owner's decision, recorded verbatim in intent: *"the full
dataset is subject to authorization … we can run over this downloaded copy
and take partial views till we get the full one"* — permission e-mail in
preparation. Consequently:

- every artefact of this amendment carries `provenance.preliminary = true`
  and figure titles carry "(Kaggle copy — preliminary)";
- chain of custody: sha256 of the three files recorded in the JSON config
  echo;
- **re-derivation obligation**: when the authorized copy arrives, every
  number is regenerated and compared elementwise; identical ⇒ preliminary
  labels lift; any difference ⇒ a finding about the copy.

```
dev.json   1bb0897a7b49b1c2ded50723615d7442c6423024f014ab17b99bba68c7c0eb26
test.json  e33ac24a132711c0bc5a781c1f3aecc6d4390895c14b824fc96d59dcbb6150ea
train.json 18bef76a8b848eef5d1a9845bfae7a525c779bd3cf67051ea71e9d5fedfe3df8
```

## 2. Scope

**H4-T**: transferring Cresci-2015 → TwiBot-20, the AUC degradation of the
TEXT spectrum families is smaller than metadata's by more than **0.05**
(charter threshold inherited):
`Δ_META − Δ_SPEC_X > 0.05`. The temporal families and SPEC_B_ALPHA are not
scoped into H4-T: uncomputable on this target by benchmark construction /
definition-dependent respectively (bitacora 22 §2) — recorded, not silently
dropped.

Populations: **source** = event-cache kept accounts (`min_events = 5`,
n = 4,770 — the population every committed front uses); **target** = all
11,826 labelled raw-release users. Overlap between the two id sets: 0
(bitacora 22).

## 3. Features and arms (definitions frozen upstream; nothing new invented)

Target-side features reuse the committed WP-I definitions verbatim via
`renyiext.textfront.account_text_features` (`\w+` Unicode tokens, NO
lowercasing, char counts summed per tweet; orders α ∈ {0, .5, 1, 2, 4, ∞},
base 2):

| arm | dim | source | target |
|---|---|---|---|
| SPEC_X_WORD / SPEC_X_CHAR / SPEC_X | 6/6/12 | text spectra over kept events' texts | text spectra over the user's ≤200 stored texts |
| META_raw | 4 | followers, following, tweet_count, age — `load_cresci_text_side` recipe | same four fields from `profile`, RAW scale |
| VOL_PROFILE | 1 | log1p(tweet_count) | log1p(statuses_count) |
| TOKENS | 1 | log1p(n_tokens) | log1p(n_tokens) |
| SHAN_WORD / SHAN_CHAR / SHAN_X | 1/1/2 | H₁ slices (WP-I convention) | same |

COUNT (decoded-event volume) has no target counterpart and is **replaced,
namedly**, by VOL_PROFILE/TOKENS on this target — stated because protocol
§4 forbids silent substitution.

Exclusions: accounts with **zero tokens** are excluded from all
text-bearing arms and TOKENS, counted per class per side; META_raw and
VOL_PROFILE run on complete profiles for everyone. All exclusion counts
land in the JSON.

## 4. Controls (every one inherited, pre-committed)

1. **D6 estimator, byte-identical machinery** to WP-K/N: R = 20 draws
   (`default_rng(42)`), fit 80 % source → within on held-out 20 % →
   transfer on the FULL labelled target; Δ_r = within_r − transfer_r;
   mean ± SD; paired bootstrap over target users (B = 1000,
   `default_rng(1042)`), same resamples across arms; HGB
   `random_state=42`; LR secondary reported.
2. **Window control for the ≤200-text cap (the WP-A lesson)**: headline
   config = source FULL timeline vs target ≤200; **equal-cap sensitivity
   arm** = source capped to its **200 most recent** texts (sequences are
   time-sorted, so the cap is exact). `sigma_config` across these two
   configs is reported beside every floor verdict (D3). Per-class
   text-count distributions rendered for both corpora (G1).
3. **Dim-matched arms** (D2, WP-E binding semantics via
   `interpret_dim_matched`): SPEC_X_WORD vs SHAN_WORD+NOISE(5),
   SPEC_X_CHAR vs SHAN_CHAR+NOISE(5), SPEC_X vs SHAN_X+NOISE(10);
   vs-metadata: WORD/CHAR vs META+NOISE(2), SPEC_X vs META+NOISE(8).
4. **R8 transforms**: raw-scale alignment is now literal — naive =
   (raw_target − μ_src)/σ_src; recal = StandardScaler fitted on the
   target's labelled columns. Both variants × both heads reported; HGB
   primary.
5. **Age reference fixed symmetric**: 2020-01-01T00:00Z applied to BOTH
   corpora (retires the mismatch caveat bitacora 19 recorded).
6. **G4 sanity null**: within-Cresci exchangeable halves, same machinery
   (permutation halves per WP-K; the id-hash deviation stands as ledgered
   in bitacora 19 §5.4). Expected silent; max |Δ| calibrates reading.

## 5. Verdict rules (pre-committed)

- Primary: Δ_META − Δ_SPEC_X > 0.05 on the headline config, with the
  paired-bootstrap CI on the transfer term and the per-draw paired count
  reported; σ_config (headline vs equal-cap) beside it.
- Floor comparisons use `interpret_dim_matched` semantics verbatim
  (confounded_dimensionality / real_but_subfloor_not_claimable /
  supports_clause) — binding on FINDINGS/HANDOFF.
- Exploratory set = all SPEC_X-vs-floor comparisons, Holm-corrected within
  that family (WP-I convention); the primary gap is single,
  pre-registered, uncorrected.
- Failure is a result: an honest negative lands in FINDINGS F16 with the
  same weight as a pass (charter H0).

## 6. Multiple-comparisons census (declared in advance)

Arms: 6 base + noise-padded variants (≈10) × 2 transform variants × 2
heads for the D6 runs; 1 primary gap; 1 G4 null pair; 2 window configs.
Holm family pre-declared: {WORD vs SHAN_WORD+n, CHAR vs SHAN_CHAR+n,
X vs SHAN_X+n, WORD vs META+n, CHAR vs META+n, X vs META+n} = 6
comparisons per config per head at most.

---

*Execution results will be appended to this entry as §7+ after the runs;
this registration section is committed first so the diff boundary between
"registered" and "seen" is auditable.*

---

## 7. Execution results (run twice, byte-identical; checks 12/12 after the additive loader)

### 7.1 The primary verdict: H4-T FAILS — decisively, and in an informative direction

| quantity (hgb, token-positive population, headline config) | value |
|---|---|
| Δ_META_raw[naive] | **+0.3502 ± —** (within 0.9976 → transfer AUC 0.6474) |
| Δ_SPEC_X_WORD / CHAR / X | +0.4502 / **+0.5058** / +0.5292 |
| **primary gap Δ_META − Δ_SPEC_X** | WORD −0.1000 · CHAR −0.1556 · X −0.1790 |
| draws with gap > 0.05 | **0/20** for all three families |
| LR head | same ordering (META lr transfer 0.7398 vs CHAR 0.5315) |

The text spectra degrade MORE than metadata — by 0.10–0.18 AUC, an order
of magnitude beyond the G4 null band (max |Δ| 0.0134). The charter's
inequality holds nowhere near its bar.

### 7.2 The mechanism finding: the fingerprint does not fade — it INVERTS

SPEC_X_CHAR transfers at **AUC 0.4840 (CI95 [0.4719, 0.4925])** — below
chance: source-fitted character-usage structure anti-ranks TwiBot-20's
classes. Target-only probes show why this is genuine shift, not plumbing:
char H₁ = 0.5110 and H∞ = 0.5358 measured ON TwiBot-20 alone (vs 0.9898
within-Cresci) — the locally near-perfect fingerprint is ~chance there,
and the multivariate residue flips sign. Named suspect for follow-up
(hypothesis, not conclusion): corpus-language shift — Cresci-2015's human
class vs TwiBot-20's English-dominant population loads character
distributions differently from bots, so a source-fitted human/bot char
boundary lands on the wrong side. No NaNs anywhere; label alignment
re-verified against the independent tensor counts (5,237/6,589).

### 7.3 Robustness of the negative

- **Window control (equal-cap)**: capping the SOURCE side at its 200 most
  recent texts moves deltas ≤ 0.0126 (σ_config block) — the verdict does
  not depend on the observed-window mismatch.
- **Floor table**: all six Holm-corrected family-vs-floor comparisons per
  head read `confounded_dimensionality` (0–1/20 draws); the Shannon-word
  floor itself collapses (+0.41) — even "how many distinct words" does not
  survive this pair of corpora.
- **R8 at raw scale**: naive vs recal META agree on AUC (0.6474 vs 0.6498)
  while recal repairs accuracy (0.6596 vs 0.4989 — calibration again);
  bitacora 22 §3c's prediction confirmed empirically.
- **Context rows**: full-population runs agree with token-positive ones
  (META Δ +0.3546 vs +0.3502).
- Within-sides match committed artefacts: SPEC_X_CHAR within 0.9898 ≈ p3i's
  0.9889; META within 0.9976 ≈ p3h's 0.9972.

### 7.4 Consequences

FINDINGS F16 records: on the only modality-bearing obtainable target, the
text front fails the transfer exam outright — it is LESS transferable than
the metadata it was meant to beat. With WP-K (era shift within corpus:
nothing degrades) and WP-N-scoped (metadata cross-corpus collapse
measured), the programme's transfer picture is now: shape statistics are
era-robust within a corpus and language-lineage-bound across corpora.
H4-as-chartered remains closed pending the authorized copy (whose
re-derivation obligation stands — these numbers must reproduce on it);
no further family remains that could sit any version of the exam on this
target pair.
