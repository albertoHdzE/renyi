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
