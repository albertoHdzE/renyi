# Bitácora 15 — correction to the WP-I record: what the headline buried

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** n/a (correction entry; supersedes the *framing* of bitacora 14 §1
and FINDINGS F11, not their numbers — every number stands as measured in
`results/p3i_textfront.json`, byte-identical runs unchanged).

Trigger: session review asked whether the WP-I summary reads as a
datasaurus-style overclaim. Audit verdict: the artefacts were gate-clean
(G1 objects rendered; G2 elementwise vs the committed record; G3 census
printed; numbers byte-identical twice), but the **prose framing failed the
reporting protocol** in four ways. Recorded here, in place, per S2.

---

## 1. The partial success read as the whole one (material)

Bitacora 14 §1 and FINDINGS F11 headline "the spectrum beats its Shannon
slice at matched dimensions (char +0.3332)" and do not state, in prose, the
comparison that bounds it:

    TOKENS+SPEC_X  vs  TOKENS+SHAN_X+NOISE(10)  =  +0.0166
    -> real_but_subfloor_not_claimable at the registered 0.02 floor

The value was in the JSON and the EVIDENCE-INDEX row, but a reader of the
headline would never learn that **once token count is its own feature, the
spectrum's residual edge over its Shannon slice is subfloor**. The honest
split: the +0.3332 is real, and it is largely *length-mediated* — the
character-frequency orders encode how much an account posts, which TOKENS
already supplies. The tail-order mechanism render stands; its classifier-
level residual, given length, does not clear the floor.

## 2. Holm "survival" presented as confirmation strength (structural)

Every raw p in the family is 0.001953 — the **floor** of the Wilcoxon
signed-rank test at 10 seeds (2⁻¹⁰ × 2; it cannot go lower). "All 20
comparisons survive Holm (adjusted p = 0.0391)" therefore carries exactly
the information "all 20 comparisons won 10/10 seeds and the family size is
≤ 25": at family size 26 the same unanimous results would **all fail** Holm
(26 × 0.001953 = 0.0508). The adjustment is arithmetically correct and its
membership was documented; it is not independent evidence of anything, and
future fronts with larger exploratory families will hit this bound.

## 3. Cross-sample "edge grows" (descriptive, reworded)

"SPEC_X_WORD vs COUNT grows to +0.0636 on the ≥ 512-token subsample"
compares deltas across **different populations** — on that subsample COUNT
itself drops to 0.881. The like-for-like statement: on the high-token
subsample the margin is +0.0636 (σ_cfg 0.0147 across the split); whether
the edge is "larger" is not a controlled comparison and is not claimed.

## 4. Mechanism and "inert" wording beyond measurement (reworded)

- "The char signal lives in the tail orders (H_4/H_inf)": an eyeball
  reading of the mean±SD α-render. No order-ablation was run. The render
  *suggests* tail-order separation; which orders carry the classifier-level
  information is unmeasured.
- "URL stripping inert": a single descriptive delta (−0.0077) with no test.
  Reworded: "small and untested; the front is not obviously a URL artefact."

## 5. What was right and stands

The numbers themselves, the elementwise gates, the byte-identity, the
objects renders, the length-control partials (all 12 orders × 3), the META
confound downgrade, and the exploratory/Holm bookkeeping — all as recorded.
The failure was emphasis, not arithmetic; this entry repairs the emphasis
where the next reader will hit it (F11 update below, HANDOFF, plan board).

## 6. Multiple comparisons counted

No new comparisons. Two structural facts derived from existing artefacts
(p-floor 0.001953; Holm family bound 25).
