# Bitácora 19 — WP-N, Protocol C scoped: the metadata side of H4 measured (+0.3143); the Δ_ours side is untestable on the obtainable target

**Date:** 2026-08-25
**Branch:** `main`
**Gate:** G6 = H4 — **not evaluable as chartered**: the scoping audit
(§1) shows no spectral/temporal/textual family is computable on the only
obtainable TwiBot-20 artefact. Everything that CAN run did run: D6
degradation for META-aligned and its floors, the mandatory R8
with/without-alignment control (both heads), dim-matched arms, the G4
sanity null, majority baselines and TPR@1%FPR beside everything.
Property checks **12/12** after the run. Two runs byte-identical (S2.6).
**Artefacts:** `scripts/run_p6n_transfer.py` (new),
`results/p6n_transfer.json`, `figures/p6n_degradation.png`,
`figures/p6n_alignment.png`, `figures/p6n_score_distributions.png`.

---

## 1. The scoping decision (ambiguity protocol §0 step 4, recorded before any transfer number was read)

WP-N task 1 says "every family available from the done deps". On contact
with the target side, "available" collapses:

| family | Cresci source | TwiBot-20 target | blocker |
|---|---|---|---|
| META_aligned(4) | ✓ | ✓ | — |
| VOL_PROFILE(1) | ✓ | ✓ | — |
| COUNT(decoded), BURST, SHAN, SPEC_T, SHAPE, TAIL, SURV | ✓ | ✗ | needs event timestamps |
| SPEC_B_ALPHA | ✓ | ✗ | needs post types / mention tokens |
| SPEC_X_WORD / CHAR | ✓ | ✗ | needs raw tweet text |

Evidence, elementwise and dated **2025-08-25**:

1. `tweets_tensor.pt`: shape (229,580 × 768), float32,
   **non-zero row fraction = 1.000000**, row norms 9.29 / 10.93 / 11.18 /
   11.40 / 13.62 (pctls 0/25/50/75/100) — dense pooled embeddings; no
   per-tweet structure, no timestamps, no text. Measured in a torch
   session (`torch.load(...); (t != 0).any(1).mean()`,
   `np.percentile(np.linalg.norm(t, axis=1), ...)`); the runner
   re-measures live when torch is present and falls back to these
   constants otherwise. This confirms bitacora 07 §3.2 by re-derivation.
2. Raw TwiBot-20 is **gated by its authors**
   (github.com/bunsenfeng/twibot-20 README, quoted in the JSON): sample
   only is public; full set requires emailing the maintainer. The HF
   mirror we fetched from holds exactly the five BotRGCN files (listing
   checked 2025-08-25).
3. The open TwiBot-22 `user.json` (Zenodo) was inspected as a candidate
   amended target: profiles only — no tweets field, no timestamps. No
   modality there either.

Decision taken (most conservative pre-registration-preserving option):
**execute the maximal honest exam on the commensurable families and
record H4 as UNTESTABLE_PENDING_DATA** — not passed, not failed, not
re-scoped into something else. The charter's H4 inequality keeps its
form; this run measures its metadata side and pins what would complete
the other side. A target-swap amendment was considered and rejected:
no open artefact carries the modalities, so a swap would have weakened
the claim's data basis to fit the tooling — the opposite of
conservative.

## 2. What ran (D6 machinery identical to WP-K's committed producer)

R = 20 draws from `default_rng(42)` (fit 80 % of Cresci labelled → within
on held-out 20 % → transfer on ALL 11,826 labelled TB20 users);
paired bootstrap over target users, B = 1000, `default_rng(1042)`, same
resamples across families; HGB `random_state=42`; LR secondary reported
for both transform variants. Arms: `META_aligned(4)`, `VOL_PROFILE(1)`
(the statuses/tweet_count profile column — the volume anchor WP-B named),
`VOL_PROFILE+NOISE(3)` dim-matched per D2 (fixed noise draws, salts 101/102
printed).

| arm \| head | within | transfer | Δ | CI95 transfer | tpr@1% |
|---|---|---|---|---|---|
| META_aligned[naive] \| hgb | 0.9974 | 0.6831 | **+0.3143 ± 0.0075** | [0.6764, 0.6938] | 0.0137 |
| META_aligned[recal] \| hgb | 0.9974 | 0.6639 | +0.3335 | [0.6512, 0.6711] | 0.0189 |
| META_aligned[naive] \| lr | 0.9589 | 0.6810 | +0.2779 | [0.6715, 0.6916] | 0.0047 |
| META_aligned[recal] \| lr | 0.9589 | 0.6667 | +0.2922 | [0.6559, 0.6764] | 0.0093 |
| VOL_PROFILE[naive] \| hgb | 0.9390 | 0.5624 | +0.3767 | [0.5512, 0.5677] | 0.0062 |
| VOL_PROFILE[recal] \| hgb | 0.9390 | 0.6179 | +0.3211 | [0.6086, 0.6301] | 0.0220 |
| VOL_PROFILE[naive/recal] \| lr | 0.9393 | 0.6180 | +0.3213 | [0.6071, 0.6288] | 0.0059 |
| VOL+NOISE(3)[naive] \| hgb | 0.9355 | 0.5399 | +0.3957 | [0.5240, 0.5448] | 0.0102 |
| VOL+NOISE(3)[recal] \| hgb | 0.9355 | 0.6151 | +0.3205 | [0.6037, 0.6260] | 0.0187 |

Majority baselines beside everything: train 0.6321, test 0.5572.
Composition shift quantified: −7.4983 pp bot share (0.6321 → 0.5572);
label-source heterogeneity unrecoverable from the tensors (recorded);
era statement in the JSON (2011-13 fake-follower/social-spam vs 2020
diverse-domain populations — feature-family robustness, not bot identity).

**Reading.** The metadata half of H4's inequality is enormous: Cresci's
near-ceiling META (within 0.9974) degrades to 0.68 on TwiBot-20 — while
WP-K showed the same family transfers *within* Cresci at Δ +0.0005. The
collapse is cross-corpus, not cross-era-within-corpus. Had any of our
families been computable target-side, the > 0.05 bar would have been
trivially cleared by Δ_META alone — which sharpens, but cannot settle,
H4. The untestability is now the binding constraint, measured.

## 3. R8 alignment control — verdict: effect present under BOTH variants

Naive (source stats onto shipped z-scores; charter-faithful) vs
target-marginal-recalibrated (StandardScaler on the labelled target
columns; same definition as the WP-B diagnostic): Δ_META agrees to
|gap| 0.0192 (hgb) / 0.0142 (lr), both variants > 0.02 above floor →
**not a transform artefact**. Open item 7 discharged: LR diagnostics are
reported on both variants. Two mechanism notes recorded in the JSON:
(a) single-feature LR AUCs coincide across variants exactly (positive-
slope affine maps preserve a monotone score's ranking — VOL lr rows are
byte-equal at 0.6180, expected, not a bug); (b) under naive the collapsed
column scales (sd ≈ σ_subset/σ_src ~ 1e-3–1e-4) partially break HGB's
binning on the single volume feature (0.5624 vs recal 0.6179 ≈ lr 0.6180)
— trees lose the ranking that LR retains.

Also recorded: the corpus-reconstruction route (z·σ_corpus + μ_corpus,
then source stats) is provably a near-no-op because the shipped columns
ARE corpus-z-scored (μ |.| < 0.01, σ 1.000–1.002 verified in WP-B) — it
would reproduce naive to float drift; documented rather than run.

## 4. G4 sanity null — silent, as required

Pseudo-transfer within Cresci (exchangeable halves, same machinery):
Δ hgb +0.0016, lr −0.0088; max |Δ| 0.0088 ≪ every real delta above.
The estimator is sound; the deltas measure shift, not plumbing.

## 5. What failed and was not fixed (all caught before any number entered a document)

1. First-draft "recal" variant reconstructed target raw via corpus
   marginals — provably a no-op given the verified z-scoring; caught by
   comparing the two draft runs (identical numbers where difference was
   expected), redefined to the WP-B diagnostic definition before the
   results were read into any document.
2. The null reused the full-target bootstrap index matrix (IndexError at
   the null stage of draft 1); fixed with its own correctly-bounded
   stream.
3. My config echo initially asserted HGB numbers "coincide across
   variants by construction" — false once recal stopped being a no-op
   (target units vs source thresholds differ); corrected to the accurate
   statement: within-variant affine invariance yes, across-variant
   differences meaningful (they ARE the R8 measurement).
4. Plan said the null splits "by account id hash"; `cresci_meta_v1.npz`
   carries no ids, so WP-K's permutation-halves machinery is used
   instead — conservative-equivalent (exchangeable halves), deviation
   recorded here per §0.

## 6. Multiple comparisons counted

12 D6 runs (6 arms × 2 heads); 2 R8 pairings; 2 null heads; zero
hypothesis tests fired (H4 not evaluable — nothing to compare against a
p-value without both sides of the gap). No exploratory claims made.

## 7. Consequences for the programme

Charter success criteria revisited in FINDINGS F15: H1 supported on
clause (i) only (bitacora 11); H2 directional gates passed with the
vs-META confound downgraded (bitacora 13); H3 cancelled by kill rule
(bitacora 18); **H4 untestable-as-chartered pending data acquisition** —
the honest terminal state of P6 on the obtainable artefacts. The path to
sitting the exam exists and is written into the JSON: gated raw TwiBot-20
(timestamps/text/post-types) or a pre-registered amendment naming a
modality-bearing target. P7/P8 remain governed by docs/03-PHASES.md;
the write-up phase inherits a programme whose headline is a *negative
and a measurement*, which the charter's H0 explicitly honours.
