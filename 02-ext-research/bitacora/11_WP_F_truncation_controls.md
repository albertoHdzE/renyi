# Bitácora 11 — WP-F, equal windows on the real corpus: clause (i) survives, clause (ii) does not

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** G1 (rasters of real accounts beside the statistic curves) +
G2 (capped-vs-uncapped on the same sample; unwindowed reference gated
elementwise against both committed artefacts) + G3 (K swept, headline
pre-declared, cap knob printed) + G4 (probe cells drawn as the null band).
**Fidelity gate: PASS at max |diff| 0.0 over 36 arrays.** Property checks 9/9.
**Artefacts:** `renyiext/features.py::temporal_blocks_windowed` (new),
`scripts/run_p2f_truncation.py` (new), `results/p2c_truncation.json`,
`figures/p2f_equal_window.png`, `figures/p2f_api_cap.png`.
**Seeds:** 42–51; two consecutive runs byte-identical (S2.6).

---

## 1. What survived equal windows

Clause (i) — COUNT+SPEC_T over COUNT — is **window-robust and grows with K**:

| K (days) | n kept (bot/human) | majority | CS − COUNT | dim-matched CS − C+N12 |
|---|---|---|---|---|
| 7 | 1,743 (567/1,176) | 0.675 hum | −0.0004 (4/10, p = 0.77) | +0.0130 |
| 14 | 2,747 (1,425/1,322) | 0.519 | +0.0122 (10/10, 0.002) | +0.0210 |
| **30** | **3,634 (2,152/1,482)** | **0.592** | **+0.0423 (10/10, 0.002)** | **+0.0488 → supports_clause** |
| 90 | 4,339 (2,660/1,679) | 0.613 | +0.0681 (10/10, 0.002) | +0.0778 |

σ_cfg across the K sweep: 0.0266 (unmatched) / 0.0255 (matched) — larger than
the 0.02 floor itself, so the honest form of the claim is conditional: clause
(i) clears at every K ≥ 30 with the delta increasing in K, and is ~zero at
K = 7, where only 1,743 accounts survive windowing at all (the sample flips
human-majority, 0.675). The headline K = 30 was pre-registered (v1.0); K = 7
reads as the tiny-n regime, not as a refutation.

The **level-removed spectrum is the strongest survivor**: SHAPE vs
BURST+NOISE(9) = **+0.0234 at K = 30** (10/10, 0.002; supports_clause), and
positive at every K (+0.0601 / +0.0213 / +0.0234 / +0.0245). COUNT+SHAPE −
COUNT = +0.0371 at K = 30 (matched +0.0437, supports_clause). Removing the
level — the count-correlated mass — leaves signal that is robust to both
window equalisation and dimension matching. This is the first result in the
programme that clears every registered control simultaneously.

## 2. What shrank: clause (ii), and the downgrade is EXECUTED

Clause (ii) — SPEC_T over SHAN — **collapses under equal windows**:
+0.0381 unwindowed → +0.0048 at K = 30; at no K does it reach the 0.02 floor
(max +0.0084 at K = 90; σ_cfg 0.0049). The pre-registered dim-matched
semantic then fires at the headline: **SPEC_T vs SHAN+NOISE(10) = −0.0003
(6/10, p = 0.92) → `confounded_dimensionality`** (plan WP-E task 3, rule 1).

**Execution of the failure semantic** (not a note): clause (ii)'s support for
H1-as-amended is **downgraded** in HANDOFF and in FINDINGS F8 — the
Shannon-floor edge measured at P2 is not separable from observation-window
truncation; bots and humans differ in how much of their timeline the corpus
retains, and that difference, not tail-resolution, carries the clause-(ii)
edge. H1-as-amended required both clauses; its support is therefore downgraded
to clause-(i)-only. The registered unwindowed gate verdicts stand as computed
(no floor moved, no history edited); this is the interpretation the controls
attach to them, exactly as pre-registered.

The burstiness verdict behaves consistently: +0.0120 at K = 30 (fails;
matched +0.0066, `real_but_subfloor_not_claimable`). At K = 90 the unmatched
delta reaches +0.0204 — a sensitivity-row observation, not a verdict change.

## 3. Relation to the probe ceiling

The unwindowed corpus CS (0.9764) sits just **below** the probe's W = 30/90
band [0.9783, 1.0000]; equal-window corpus CS drops to 0.9146 (K = 30) and
0.9435 (K = 90) — well below the same-generator truncation world. Reading:
truncation alone separates synthetic same-generator pairs at ≈ 1 through this
pipeline, but real accounts under equal windows are *harder* than that null —
so the corpus's separation was never purely censoring. What the equal-window
control removes is specifically the Shannon-floor component (clause ii) and
the burstiness margin; what it leaves is a count-anchored spectrum edge
(clause i) and a level-removed shape edge (SHAPE arms). Amendment 06's
ceiling annotation on SPEC-family "shape" readings is thereby partially
discharged: the surviving edges are those measured *under* the ceiling, not
at it.

## 4. The API cap is not COUNT's secret (review D2)

Cap = mode of the human upper tail = **3215 events** (8 humans; interior
check passes; the modal table shows the spike region 3203–3215). The G1/G2
render shows the real phenomenon: a spike *region* just below the cap
(~375 humans in the top pre-cap log-bin; 19.6 % of humans ≥ 0.95·cap), while
only 3 bots reach it. Sensitivities on the headline kept sample (n = 4,770):

| exclusion | humans removed | COUNT | Δ | COUNT+SPEC_T | Δ |
|---|---|---|---|---|---|
| none | 0 | 0.9400 | – | 0.9764 | – |
| ≥ cap (registered) | 131 | 0.9353 | −0.0047 | 0.9750 | −0.0014 |
| ≥ 0.95·cap (post-hoc) | 383 | 0.9267 | −0.0133 | 0.9729 | −0.0035 |

Even removing the entire cap-edge region moves COUNT by 0.013 — the volume
edge is a property of bots' low counts, not of truncated humans. D2 is
answered empirically: no artefact to control away.

## 5. What failed and was not fixed

Two producer bugs were written, caught, and fixed **before any number entered
any document** (both found by impossible-number inspection, the datasaurus
loop working as designed):

1. The cap-sensitivity "including capped" rows were computed on the
   already-masked matrix — the incl/excl AUCs came out bit-identical, which
   is impossible for different samples. Fixed by separating the series
   functions; the corrected deltas (−0.0047 / −0.0014) replaced the
   meaningless 0.0000.
2. The post-hoc wide row's CS delta subtracted COUNT's baseline
   (printed +0.0329 where incl−excl bound the truth near −0.0035). Fixed;
   corrected value −0.0035.

Nothing outstanding. The first (discarded) run of the buggy producer never
reached a document; its JSON was deleted and regenerated.

## 6. Decisions taken under the ambiguity protocol

1. **Arm set.** Task 2 names six families; the figure spec requires clause
   deltas, which need COUNT+SPEC_T, and the burstiness verdict continuity
   needs COUNT+BURST. Both composites added to the evaluated set (eight arms
   + four D2 noise floors). Recorded here; most-conservative reading was to
   add arms, never to drop a named one.
2. **sigma_config axis.** D3 says "the full published config sweep"; this
   WP's published sweep axis is K, so each comparison's σ_cfg is the
   population SD (ddof = 0) of its per-K mean delta across K ∈ {7, 14, 30,
   90}. Stated in the JSON config echo.
3. **Post-hoc wide cap row.** The registered cap definition (mode of the
   human upper tail) is reported as pre-registered; the ≥ 0.95·cap exclusion
   is added, labelled `post_hoc: true`, because the G1 render shows the
   at-cap rule understates the spike region. The render forced the extra
   knob; the registered row remains the headline answer to D2.
4. **Unwindowed reference block** doubles as the fidelity anchor: 36 arrays
   elementwise-gated against `p2_temporal.json` and `p2b_decomposition.json`
   at max |diff| 0.0, tying this new producer to every published number.

## 7. Multiple comparisons counted

Eight arms + four noise floors per K (12 × 10 seeds × 4 K) plus eight
unwindowed arms plus four sensitivity series; nine named comparisons per K
(four gated, five matched) with the pre-registered interpretation rules
applied to the matched ones; one post-hoc cap sensitivity (labelled). Two
hypothesis-level consequences, both pre-registered semantics: clause (ii)
downgraded; clause (i) and the SHAPE arms recorded as surviving. No floor or
threshold moved.
