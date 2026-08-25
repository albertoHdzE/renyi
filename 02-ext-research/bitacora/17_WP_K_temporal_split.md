# Bitácora 17 — WP-K, the era shift: nothing SPEC degrades; two floor statistics improve; one structural caveat

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** G2 disjointness asserted in code and JSON (0 intersecting ids);
G3 boundary sensitivity at three dates; G4 shuffled-assignment null run
(max |Δ| 0.0158). Two runs byte-identical (S2.6). Checks 10/10.
**Artefacts:** `scripts/run_p3k_timesplit.py` (new),
`results/p3k_timesplit.json`, `figures/p3k_era_split.png`,
`figures/p3k_degradation.png`.

---

## 1. The registered result

Split at 2012-07-01T00:00Z (pre-registered): **train 2,763 (bot 0.5074) /
test 2,007 (bot 0.7195)**; 2,689 accounts (56.37 %) span the boundary.
D6 per family (R = 20 draws + B = 1000 shared bootstrap resamples):

| family | within | transfer | Δ | transfer CI95 |
|---|---|---|---|---|
| TAIL | 0.7804 | 0.9074 | **−0.1271** | [0.8869, 0.9192] |
| BURST | 0.8653 | 0.9237 | **−0.0584** | [0.9105, 0.9420] |
| SPEC_B_ALPHA | 0.9355 | 0.9735 | −0.0380 | [0.9620, 0.9832] |
| SHAPE | 0.9404 | 0.9634 | −0.0230 | [0.9509, 0.9714] |
| SPEC_T | 0.9516 | 0.9695 | −0.0179 | [0.9643, 0.9804] |
| TAIL+SURV | 0.9514 | 0.9539 | −0.0025 | [0.9218, 0.9463] |
| SPEC_X | 0.9917 | 0.9930 | −0.0013 | [0.9910, 0.9966] |
| META | 0.9955 | 0.9950 | **+0.0005** | [0.9912, 0.9989] |
| SURV | 0.9507 | 0.9477 | +0.0030 | [0.9284, 0.9503] |
| SHAN | 0.9201 | 0.9079 | +0.0122 | [0.9073, 0.9362] |
| COUNT | 0.9237 | 0.9020 | **+0.0217** | [0.8746, 0.9167] |

## 2. The preview guard: nothing SPEC crosses it — and what fires is inverted

Δ_META − Δ_family > 0.05 fires for **META vs TAIL (+0.1276)** and
**META vs BURST (+0.0589)** only. Every SPEC family stays under
(max SPEC_B_ALPHA +0.0385; SPEC_T +0.0184; SPEC_X +0.0018). And the fired
pairs are **inverted-direction**: they fire because TAIL/BURST *improve*
under the shift (negative Δ), not because META collapses — META transfers
at ceiling (+0.0005). So there is **no H4-preview of the feared kind**
(the spectrum degrading faster than metadata); if anything the era shift
favours the shape/tail statistics and mildly hurts volume (COUNT +0.0217).
All of it labelled **preview**, never H4, per the plan.

## 3. The structural caveat that leads every reading (G2 honesty)

The two sides are **not in a common coordinate**: the test era is
bot-majority (0.7195) while train is balanced (0.5074) — the classes have
different era profiles. Δ therefore mixes era shift with class-composition
shift; AUC is not invariant to that rebalancing. The G4 null (random halves,
balanced by construction, max |Δ| = 0.0158) calibrates *sampling* noise
only. Consequently the negative deltas (TAIL/BURST/SPEC improvements)
should be read as "shift + composition", and the near-zero deltas
(META ±0.001, SPEC_X −0.0013) as the genuinely flat ones. Recorded here,
in FINDINGS F13, and in the JSON.

Second-order facts: R8 age-guard immaterial (META_no_age −0.0010 vs META
+0.0005 — age carries no extra era artefact); boundary sensitivity is
strong (2012-01-01: 1,213/3,557, train bot 0.2597; 2013-01-01: 3,425/1,345,
test bot 0.8781 — the registered date sits mid-gradient).

## 4. Decisions taken under the ambiguity protocol

1. **No scaler fitted**: HGB is scale-invariant and no LR head runs here;
   the leakage rule (scaler fitted train-side) is satisfied vacuously and
   stated in the JSON rather than obeyed with a no-op transform whose float
   rounding could perturb trees.
2. **Same D6 draw stream across families** (default_rng(42) re-instantiated
   per family): identical 80 % partitions make the within terms paired;
   bootstrap resamples shared via default_rng(1042).
3. **Preview pairs reported for ALL non-META/COUNT families**, not just
   SPEC — with the inversion made explicit, because suppressing TAIL/BURST
   would have hidden that the guard's letter fires for reasons opposite to
   its motivation.

## 5. What failed and was not fixed

One dead import line (syntax) removed before the first run. Nothing else.
Nothing outstanding.

## 6. Multiple comparisons counted

14 families × (20 draws + full-train fit + 1000 bootstrap resamples);
11 preview pairs evaluated against 0.05; 1 G4 null over 14 families;
3 boundary dates. All preview-labelled; no H4 claim made or implied.
