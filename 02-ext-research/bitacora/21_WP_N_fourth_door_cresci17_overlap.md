# Bitácora 21 — WP-N follow-up: the fourth door checked — no TwiBot-20 anywhere in the conversion archive, and the cresci-2017 amended-target option DIES on the overlap check

**Date:** 2026-08-25
**Branch:** `main`
**Status:** feasibility/provenance only — no features built, no model run,
no outcome data seen. This is the check that decides whether an amendment
may even be drafted (plan §9.3: before seeing affected data).
**Artefact:** `data/raw/bot/_tmp_other.zip` (807,252,443 bytes,
Other-Dataset-TwiBot22-Format.zip from Drive id `1flNklDJG8wrv4oj5JHA9WAtV6JeGi1dW`,
the archive `get_bot_data.sh` extracts cresci-2015 from and deletes;
kept at `data/raw/bot/_tmp_other.zip` pending the target decision).

---

## 1. Door four, listed with my own eyes

46 entries; seven corpora under `Other-Dataset-TwiBot22-Format/`:
botometer-feedback-2019, **cresci-2015**, **cresci-2017**, cresci-rtbust-2019,
cresci-stock-2018, gilani-2017, midterm-2018. **No twibot-20.** The
UNTESTABLE_PENDING_DATA verdict stands on all four doors now:
HF BotRGCN mirror (9 files, none temporal/textual) · author-gated GitHub raw ·
TwiBot-22 Zenodo user.json (profiles-only) · this archive.

## 2. What cresci-2017 in fact offers (all verified here)

- `label.csv`: header `id,label`; **14,368 unique labelled users, 10,894 bot
  / 3,474 human** (majority 0.7582). CRLF line endings confirmed
  (`b"\r\n"` present) — the same silent-truncation genus as LIAR's dialect
  trap; any future parser strips `\r` explicitly.
- `edge.csv`: schema `source_id,relation,target_id`; first rows are
  `u678033,post,t593932392663912449` — real post relations to snowflake
  tweet ids. Decode of that id under D1: **2015-05-01T00:18:11Z** — wholly
  post-snowflake-epoch, so D9 discards nothing.
- `node.json` / `split.csv`: the same four-file schema
  `renyiext/events.py` already parses for cresci-2015.

So yes — modality-bearing. The option died on provenance, not modality:

## 3. The overlap check FIRES — the amendment is dead as a clean cross-corpus claim

Elementwise against `data/raw/bot/cresci-2015/label.csv`
(5,301 labelled users):

| quantity | value |
|---|---|
| \|c17 ∩ c15\| | **3,351 = 63.21 % of c15's labelled users, 23.32 % of c17's** |
| where they sit in c17 | **3,351/3,351 inside c17's bot class** (30.76 % of it) |
| label agreement on shared ids | **3,351/3,351** |

The entire Cresci-2015 bot population IS cresci-2017's fake-follower
lineage, relabelled `bot` with perfect agreement. A Cresci-2015 →
cresci-2017 transfer would compute both sides of H4's inequality on a
target that contains the source's whole positive class by construction —
partly within-sample, exactly the failure mode the pre-registration was
written to avoid. **The amended-target route via cresci-2017 is closed**
for any claim phrased "cross-corpus".

## 4. Residual options (recorded, not decided)

1. **Contamination-excluded variant**: target = c17 minus the 3,351 shared
   accounts → 11,017 users (7,543 bot / 3,474 human), none of them in the
   source. Genuinely unseen sample — but the reviewer's weaker-test caveat
   stands and must headline any such amendment: same group, same
   repository, adjacent eras, same annotation lineage —
   *cross-corpus-but-not-cross-group*. It bounds what the result may
   claim (robustness within one annotation programme), not what H4
   chartered (transfer beyond the annotation lineage).
2. Raw TwiBot-20 remains the only route to H4 as written (HANDOFF item 8).

Decision deferred to the project owner; nothing is re-registered until
one is chosen.

## 5. What failed and was not fixed

Nothing. One count differs from the external review pass by one
(10,894 bots here vs 10,893 quoted); the elementwise recount governs.

## 6. Multiple comparisons counted

One listing; one overlap decomposition (overall + per-class + label
agreement); one decode spot-check. No hypothesis test.
