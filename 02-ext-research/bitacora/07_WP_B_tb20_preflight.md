# Bitácora 07 — WP-B, TwiBot-20 preflight and the H4 framing

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** none (preflight). Contains the **framing decision** that plan WP-B
task 2 requires to be recorded before any transfer run exists.
**Artefacts:** `scripts/run_p6b_tb20_preflight.py`,
`data/processed/ext/twibot20_preflight_v1.npz` (torch→numpy conversion),
`results/p6b_tb20_preflight.json`, `results/figures/p6b_tb20_volume.png`,
EVIDENCE-INDEX rows.
**Seeds:** 42–51, StratifiedKFold(5), HGB(max_iter=200, early_stopping=False).
Two runs byte-identical.

---

## 1. Framing decision (pre-transfer, binding for WP-N)

Rule (pre-registered): AUC(statuses alone) ≥ 0.85 ⇒ H4′ fires; else H4 stands
as chartered.

| volume candidate | AUC alone |
|---|---|
| **statuses (the named volume column)** | **0.6073 ± 0.0022** |
| followers | 0.7414 ± 0.0019 |
| active_days | 0.6186 ± 0.0027 |
| friends | 0.5417 ± 0.0031 |
| screen_name_length | 0.5298 ± 0.0024 |

**Decision: H4′ does NOT fire — H4 runs exactly as chartered.** Even the
strongest profile scalar (followers, 0.7414) sits far below the threshold, and
far below Cresci-2015's count-alone 0.939. Metadata magnitude is *weak* on
TwiBot-20's labelled population, bot-majority at 0.5572 (vs Cresci 0.6321).

## 2. What the numbers say

- **Volume direction matches across corpora**: bot share falls monotonically-
  -ish with statuses decile, 0.637 → 0.406 (JSON `retention_on_status_deciles`)
  — bots are the *quieter* class on both corpora, so the transfer comparison
  keeps a consistent sign on its incumbent.
- **The z-score claim is true where it was made**: full-corpus column means
  |μ| < 0.01, sds 1.000–1.002 → verified. The **labelled subset drifts hard**
  (subset sd up to 3.76 on followers): the annotated 11,826 users are a
  heavy-tailed, unrepresentative slice of the 229,580. Reported separately;
  the first version of this script tested the claim on the subset and wrongly
  printed False — fixed to test what the claim actually asserts.
- **R8 made measurable — and refuted for our classifier**: strict D8 transfer
  (source scaler onto z-scored target) collapses target columns to
  sd 0.0002–0.004, yet 4-field META scores **0.7864 naive vs 0.7859
  marginal-recalibrated**. HistGradientBoosting splits are invariant to
  per-feature monotone rescaling, so the feared "H4 passes because the scaler
  broke" mechanism does not exist for the primary head. It WOULD exist for a
  scale-sensitive head (LR); WP-N must keep any LR diagnostics on the
  recalibrated variant or report both.

## 3. Deviations from the plan text (recorded per §0)

1. Plan task 3 said the alignment dry-run fits "no classifier yet"; two
   diagnostic univariate-model AUCs were added anyway (naive vs recalibrated)
   because the offsets table alone could not distinguish "scale artefact"
   from "no artefact" — datasaurus G4 prefers measuring over assuming. The
   addition is conservative (it guards against a false-positive H4).
2. Retention curve uses statuses deciles, not tweet-count cutoffs: per-user
   tweet counts are unrecoverable from the BotRGCN tensor artefact
   (`tweets_tensor.pt` is dense pooled embeddings, all rows non-zero — text
   availability likewise uninformative by construction). Recorded in JSON.
3. Account age uses a fixed reference date 2020-01-01 on both sides (printed);
   Cresci ages land ~2,500 d, making the era gap visible rather than hidden.

## 4. What failed and was not fixed

Nothing failed. The notable negative: **volume is not the threat on TwiBot-20**
that it is on Cresci-2015 (0.61 vs 0.94). If H4 passes, it will not be because
META collapsed under its own weight on the target — the interpretation must
rest on the degradation comparison itself.

## 5. Multiple comparisons counted

Five property AUCs (all reported); two R8 diagnostics; one branch evaluation
against a pre-named column and threshold. No hypothesis test performed.
