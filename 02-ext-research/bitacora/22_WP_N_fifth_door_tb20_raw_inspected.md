# Bitácora 22 — A fifth door materialises: the full labelled TwiBot-20 raw release inspected — text-bearing, timestamp-FREE; the amendment's shape changes

**Date:** 2026-08-25
**Branch:** `main`
**Status:** feasibility/provenance only — no features built, no model run,
no outcome analysis. Counts and structure verified elementwise;
`train/dev/test.json` (~342 MB total, owner-supplied from the Kaggle
re-upload `marvinvanbo/twibot-20`) sit outside the repo pending the
provenance decision.

---

## 1. What the files are — verified, including where my own heuristic failed

`ID/profile/tweet/neighbor/domain/label` objects; splits train/dev/test
(support absent). Totals:

| quantity | value | cross-check |
|---|---|---|
| labelled users | **11,826** | equals `label.pt` count exactly |
| human / bot | **5,237 / 6,589** | equals the tensor audit's counts exactly |
| overlap vs Cresci-2015 labelled ids | **0** (`|tb20 ∩ c15| = 0`) | clean cross-group target |
| tweet entries (exhaustive, all files) | **1,999,788**, every one a plain string | no per-tweet object exists |
| users with `profile.created_at` | 11,826/11,826 | account age at RAW scale |
| users with NO tweet field | **80** | future exclusion census |

Recorded against myself: I argued the Kaggle zip (114 MB) could not be the
full release. Wrong — JSON compresses ~3×; the elementwise count against
`label.pt` governs over size heuristics. This is the second time this
programme caught its own summary failing a datasaurus gate (bitacora 19 §5
has the first).

## 2. The modality re-audit under the REAL raw schema

| family | computable target-side? | why |
|---|---|---|
| META_aligned(4), RAW scale | **YES — improved** | followers/friends/statuses/created_at present un-z-scored; true commensurability replaces the affine reconstruction |
| VOL_PROFILE(1) | YES | raw statuses_count |
| SPEC_X_WORD / SPEC_X_CHAR | **YES** | up to ~200 raw texts per user (WP-I's front) |
| SPEC_B_ALPHA (structured post types) | NO | no typed post objects; only "RT @"-prefix heuristics — a NEW feature definition, not D11's collapse; out unless separately pre-registered |
| SPEC_T, SHAPE, BURST, SHAN, TAIL, SURV | **NO — permanently, for this corpus as distributed** | zero timestamps in 1,999,788 entries; `profile.created_at` is ONE point per account, not an event stream |

**This corrects HANDOFF item 8's premise.** Emailing the authors yields
exactly these fields (the GitHub-documented release schema is what this
copy matches) — acquisition unlocks the TEXT front, not the temporal half.
H4-as-chartered over the full family list is unsatisfiable on TwiBot-20
*by construction of the benchmark*, not merely by our artefacts.

## 3. What the exam can now legitimately be

An amendment (NOT yet registered) could run: fit Cresci-2015 → test
TwiBot-20 on {META_aligned(4) raw, VOL_PROFILE, SPEC_X_WORD, SPEC_X_CHAR,
SPEC_X} under D6 + R8 + dim-matched arms + G4 pseudo-null. Why it is
informative and not a consolation prize: locally, SPEC_X could NOT beat
META (confounded near ceiling, −0.0036, bitacora 14); cross-corpus META
collapses (+0.3143, bitacora 19). Transfer is precisely where the two
fronts separate — the question WP-N was written to ask.

Design questions the amendment must pre-register before any outcome is
seen: (a) the target's ≤200-recent-tweets cap is a WINDOW — the WP-A
lesson says shape claims need an equal-window/caliper control on both
sides, or at minimum the length distribution rendered per class;
(b) disposition of the 80 no-tweet users (exclusion counted per class);
(c) LR diagnostics stay on both transform variants (raw-scale target makes
naive-vs-recal nearly moot — record rather than assume);
(d) provenance stance.

## 4. Provenance — an owner-level decision, documented not buried

The copy matches the official release exactly (schema, counts, id space),
but it is a third-party re-upload whose card claims MIT over data the
authors gate "due to privacy issues". Two clean paths: (a) email the
maintainer and use the identical files obtained from the source (HANDOFF
item 8's route — nothing about §2–3 changes); (b) the owner accepts the
re-upload with the licence caveat recorded here and in FINDINGS. This
entry deliberately analyses structure only; outcome analysis waits for
that decision plus the §3 registrations.

## 5. What failed and was not fixed

My own size-based dismissal of the Kaggle item (§1). Nothing else.

## 6. Multiple comparisons counted

Structure only: 3 file loads, one exhaustive type scan, one id-set
intersection, one field-presence census. No model, no test, no claim.
