# 07 — Findings of `02-ext-research`

The `DISCREPANCIES`-analogue for this project: one split-interpretation section
per finding, newest last. Maintained per plan WP-M. Every number carries its
`results/*.json` source (see `../EVIDENCE-INDEX.md`). Append-only.

---

## F1 — The corpus has timestamps after all, and half of them were lying

**Finding (P0).** Tweet IDs are snowflakes; `(id >> 22) + epoch` recovers
posting time (decision D1). The first render then caught that pre-snowflake
sequential ids decode to a few ms past the epoch — **63,830 tweets on one
millisecond** — fixed by thresholding at the first snowflake id, not a bit width
(decision D9). The elementwise created_at check passed at 0 violations *before
and after* the fix; only the render saw it.

**Status:** closed. Source: `bitacora/01`, `results/p0_events.json`.

---

## F2 — Volume is the dominant signal in Cresci-2015, and the fixed-n design died of it

**Finding (P0).** Bots post a median of 23 tweets vs humans' 834; event count
alone scores AUC 0.939; D3's fixed-n subsampling retains ≥ 80 % of both classes
only at n ≤ 12. G0 failed; H1 was amended (bitacora 02) to test shape against
volume directly.

**Status:** closed as a gate failure that reshaped the design.
Source: `bitacora/01/02`, `results/p0_events.json`.

---

## F3 — H1 passed its clauses; the mechanism did not survive its own render

**Finding (P2).** SPEC_T beat count (+0.0367) and Shannon (+0.0380), 10/10
seeds, p = 0.0020 — but failed the burstiness floor (+0.019 < 0.02) and the
α-curves are near-parallel offsets: tail-resolution is not demonstrated.

**Status:** verdict stands; **reading superseded by F4.**
Source: `results/p2_temporal.json`, `bitacora/04`.

---

## F4 — Censoring alone can produce ~all of it (WP-A probe)

**Finding (WP-A, 2026-08-24).** Same generator, same rate, shorter observation
window: AUC(B vs A) = **0.9224–1.0000** across all nine generator × window
cells through this exact pipeline (trigger ≥ 0.85, fired at 1.0000). Any claim
that SPEC_T's edge over count measures *behavioural shape* is bounded by a
censoring artefact family whose ceiling here is ≈ 1.

**What this does NOT show:** that P2's +0.0367 *is* censoring. Real accounts
fade gradually rather than cutting off; the equal-window arms (plan WP-F) test
the real corpus directly.

**Status:** open → WP-F adjudicates; every SPEC-family number carries the
ceiling annotation until then. Sources: `results/p2c_probe.json`,
`bitacora/05`, amendment `bitacora/06`.

---

## F5 — TwiBot-20 inverts the volume story, and the scale trap is empty

**Finding (WP-B, 2026-08-24).** On TwiBot-20's labelled population (bot-majority
0.5572 — unlike every other corpus here), profile magnitude is *weak*: statuses
alone AUC 0.6073, best scalar followers 0.7414 — versus Cresci-2015 where count
alone scores 0.939. The pre-registered branch therefore did **not** fire: H4
runs as chartered, no incremental-over-volume amendment.

Two secondary facts worth keeping:

1. **The labelled slice is unrepresentative of its own corpus** — z-scoring was
   verified at corpus level (means 0, sds 1), but the annotated 11,826 users
   drift hard (subset sd up to 3.76). Transfer conclusions apply to the
   annotated population, which is the one anyone would deploy on.
2. **The feared standardisation trap is empty for trees**: strict source-scaler
   transfer collapses target columns to sd ≈ 10⁻³ yet META's AUC moves 0.0005
   (0.7864 vs 0.7859 recalibrated) — HGB rescaling invariance. The trap is real
   only for scale-sensitive heads; WP-N keeps LR diagnostics on recalibrated
   features.

**Status:** closed for framing purposes; feeds WP-N's caveat set.
Sources: `results/p6b_tb20_preflight.json`, `bitacora/07`.

---

## F6 — The estimator was silently losing mass; the check caught the fix's own leak

**Finding (WP-D, 2026-08-24).** `log_bin_counts` dropped all intervals outside
`[lo, hi]` — 73 bot vs 17 human intervals at hi = 400 d (0.0393 % / 0.0007 %).
The first fix (overflow cell only) still leaked below `lo`; property P16's
first draft failed on it and forced the symmetric underflow cell. Full P2
re-run: every verdict unchanged (clauses +0.0364/+0.0381 clear at p = 0.002;
burstiness floor +0.0170 still fails; max arm shift 0.0003). Probe re-run:
trigger still fired at 1.0000.

**Status:** closed. Sources: `results/p2_temporal.json` (post-fix),
`results/p2b_decomposition.json`, `figures/p2d_overflow_mass.png`,
`bitacora/09`.
