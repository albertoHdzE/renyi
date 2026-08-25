# Bitácora 20 — Corrections to the WP-N record: the volume ordering is variant-dependent, the transfer AUC hides a calibration collapse, one figure sign bug

**Date:** 2026-08-25
**Branch:** `main`
**Status:** corrections to bitacora 19, which stands unedited (append-only).
Every quoted number below was re-derived elementwise from
`results/p6n_transfer.json` before entering this entry. No published number
changed; three documents' *reading* of unchanged numbers was wrong.

---

## 1. Defect 1 — "volume degrades even more" was variant-dependent, and we quoted the broken arm

Bitacora 19 §2's prose sourced volume's degradation to hgb/naive (+0.3767;
dim-matched +0.3957). Paired per-draw over the 20 D6 draws (every arm
re-instantiates `default_rng(42)`, so draw r shares its partition across
arms — the pairing is by construction):

| head | variant | Δ_META | Δ_VOL | VOL−META | draws VOL>META |
|---|---|---|---|---|---|
| hgb | naive | +0.3143 | +0.3767 | +0.0624 | 20/20 |
| hgb | recal | +0.3335 | +0.3211 | **−0.0124** | **3/20** |
| lr | naive | +0.2779 | +0.3213 | +0.0434 | 20/20 |
| lr | recal | +0.2922 | +0.3213 | +0.0291 | 20/20 |

Dim-matched (VOL+NOISE(3) vs META): hgb naive +0.0814 (20/20); hgb recal
−0.0130 (**2/20**). By the R8 rule stated in bitacora 19 itself — an effect
that exists only under one transform variant is that variant's artefact —
the hgb volume-vs-meta ordering FAILS it, for exactly the mechanism §3(b)
recorded: naive collapses the volume column's sd to ~1e-3–1e-4, breaking
HGB binning (transfer 0.5624 vs recal 0.6179 ≈ lr 0.6180). The comparison
is claimed **only on LR, both variants, 20/20 draws** (one external review
pass said "19/20" for lr/recal; our elementwise recount from the stored
per-draw arrays is 20/20 — the artefact-backed count governs).

The claim is now artefact-backed, not prose: the runner computes this table
into `alignment_r8.volume_vs_meta_variant_dependence`. Propagation repaired:
HANDOFF state, FINDINGS F15, EVIDENCE-INDEX (⚠ rows + LR-sourced rows),
notebook 05 §5.1.

## 2. Defect 2 — transfer AUC 0.6831 sits on total calibration collapse

`META_aligned[naive]|hgb` on target: accuracy **0.5585 vs majority 0.5572**
(+0.0013), macro-F1 **0.3612**, TPR@1%FPR **0.0137**; the G1 score render
shows both class piles at predict_proba ≈ 1. So "within 0.9974 → 0.6831"
is a decay to a score pile with residual ranking only — not to a mediocre-
but-working classifier. This STRENGTHENS F15 and is now said in F15,
EVIDENCE-INDEX (accuracy/macro-F1/majority row per the botsage
beside-everything rule), HANDOFF, and notebook 05.

## 3. Defect 3 — figure sign bug and a clipped title

`run_p6n_transfer.py` drew the bootstrap CI as `w − within_mean`, i.e. −Δ,
mirroring the interval onto the wrong side of zero in
`p6n_degradation.png`; fixed to `within_mean − w[1] … within_mean − w[0]`.
`p6n_score_distributions.png`'s suptitle clipped; re-laid out. No published
number is affected; figures regenerated. Post-fix runs byte-identical, and
the JSON diff against the pre-fix version is exactly 2 date-string leaves +
the additive variant-dependence block — zero family-value changes.

## 4. Minor corrections

1. **Date typo**: the audit constants carried `2025-08-25`; corrected to
   2026-08-25 in the runner and regenerated JSON. Bitacora 19 §1 contains
   the same typo twice; it stands unedited, corrected here by reference.
2. **Provenance precision**: the tensor audit WAS measured interactively in
   a torch session (commands in bitacora 19 §1), but the ext venv ships no
   torch, so the committed RUN transcribes those constants through the
   documented fallback (`scoping.measured.fallback`) rather than
   re-deriving them; the runner re-measures live wherever torch exists.
   Any summary saying simply "measured myself" without that qualifier
   overstates the committed chain.
3. **HF mirror completeness**: the listing holds 9 files; the four never
   fetched (`cat_properties_tensor.pt`, `des_tensor.pt`, `edge_type.pt`,
   `label_new.json`) are profile/categorical tensors and label metadata —
   none temporal or textual. Recorded so the door-audit is exhaustive, not
   sampled.
4. **Checks caveat**: "checks 12/12" is `renyiext/checks.py`, the standing
   estimator suite. WP-N adds no estimator, so the suite passing is true
   but carries no weight as a gate on the transfer code; the transfer's
   gates are the byte-identity runs, the R8 control and the G4 null.

## 5. What failed and was not fixed

Nothing new failed. The three defects were caught by external review of
committed prose/figures whose numbers were themselves correct — the failure
was reading our own figure (the degradation plot already drew the reversal)
and quoting arms without their variant labels. Both habits are now ledgered
as the thing to distrust.

## 6. Multiple comparisons counted

Descriptive paired counts only (8 arm-pairs × 20 draws); no hypothesis test,
no threshold crossed, no claim beyond what the tables above state.
