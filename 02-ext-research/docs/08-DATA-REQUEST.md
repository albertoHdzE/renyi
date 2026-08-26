# 08 — DATA REQUEST — what the authorized TwiBot-20 copy must carry

**Status:** the authorization e-mail has been sent (2026-08-25); this
document is the field-level spec behind it, so the reply can be checked
mechanically on arrival.
**Provenance state:** every TwiBot-20 number in FINDINGS F15–F17 is
PRELIMINARY (third-party Kaggle re-upload `marvinvanbo/twibot-20`,
sha256 chain in `results/p6n_transfer_text.json` / `p6n_ls1.json`; owner
decision and re-derivation obligation in `bitacora/23` §1 and
`bitacora/24` §1). The authorized copy triggers a re-derivation, not a
redesign — §6.

## 1. What the current copy already supports

The Kaggle copy ships per user: `ID`, `label`, `domain`, `profile`
(followers/friends/statuses counts, `created_at`, plus decorative
fields), and `tweet` = a list of ≤ 200 raw text strings, `neighbor`
(empty in the inspected copy). From this our pipeline already ran:
META_raw, VOL_PROFILE, TOKENS, SPEC_X_WORD/CHAR/X + Shannon slices
(H4-T), and the language census (LS-1). SPEC_B_ALPHA is text-derivable
(post types classified from text per D11, mentions via the registered
regex) — definition-dependent, recorded as such.

## 2. The single decisive gap: per-event time

The `tweet` field carries **no tweet IDs and no timestamps**, so every
temporal family is uncomputable on this benchmark *as distributed*
(bitacora 22) — this is a property of the copy, not of our code:

| family | needs | status on Kaggle copy |
|---|---|---|
| SPEC_T / BURST / SHAPE | per-event UTC ms | **MISSING** |
| TAIL / SURV (Hill, survival lags) | inter-event times | **MISSING** |
| circadian H_cd (WP-G design) | hour-of-day | **MISSING** |
| era split (WP-K design on target side) | event dates | **MISSING** |
| everything in §1 | — | present |

**Snowflake IDs are sufficient** — no explicit timestamps required: the
top 41 bits of Twitter's snowflake format encode UTC milliseconds, our
decoder is built, gated and validated (decision D1, `renyiext.events`).
The one hard requirement is that IDs be the **original numeric tweet
IDs**, not re-indexed or hashed surrogates.

## 3. Field-by-field ask

1. **Per-tweet numeric IDs** (snowflakes) — or explicit creation
   timestamps per tweet; unlocks every temporal family above.
2. **Raw tweet text**, untruncated and unnormalised (current copy is
   already usable; confirm the authorized copy matches it byte-for-byte
   or document the difference).
3. **Post-type / interaction annotations** (retweet/reply/quote
   provenance) if the release carries them — otherwise our
   text-classified D11 collapse stands.
4. **Profile fields as distributed** (incl. `created_at`) — present in
   the copy; confirm unchanged.
5. **The follow/neighbor network**, if the full distribution ships one
   (charter P5 / SPEC_N is TwiBot-20-only and has never been runnable).
6. **The original train/dev/test split layout with user IDs**, so the
   re-derivation is elementwise-comparable to the preliminary artefacts.
7. **Release version / date / changelog**, and any label-provenance
   documentation.

## 4. The coverage question (headline-validity, not a nicety)

Is the ≤ 200-tweets-per-user list the complete timeline the release
ships, or a 200-item sample of longer histories? Our own censoring probe
(WP-A) measured that **window truncation alone separates the classes at
AUC 0.92+** through this pipeline, so observed-window length is a
headline-validity variable: every transfer claim is stated net of it
(equal-cap controls, bitacora 23 §4). The answer determines which
window controls the authorized-copy re-runs must carry.

## 5. Nice-to-have (only if cheap for the maintainers)

- per-tweet language or locale annotations (would let LS-1's census be
  cross-checked against an independent tag);
- bot-label provenance per class (human-verified vs heuristics).

## 6. Re-derivation protocol on arrival (pre-committed)

1. Record sha256 of every received file beside the Kaggle chain.
2. Regenerate `p6n_transfer_text.json` and `p6n_ls1.json` unchanged.
3. Compare elementwise: **identical ⇒ PRELIMINARY labels lift**;
   **any difference ⇒ a finding about the copy** (FINDINGS, same weight
   as a pass — charter H0).
4. If per-event time arrives, the WP-N runner extends to every family
   and H4-as-chartered becomes computable as pre-registered (HANDOFF
   "WP-N as planned" preserves the exact protocol).

## 7. Ready-to-paste e-mail paragraph

> We are an academic replication group studying which behavioural and
> text features of social accounts transfer across corpora (Cresci-2015
> → TwiBot-20). We currently work from the public labelled TwiBot-20
> release and would like to request access to the authorized full
> distribution. Specifically, beyond the profile/text fields the public
> copy already carries, our pipelines are blocked on one decisive field:
> **per-tweet numeric IDs (snowflakes) or creation timestamps**, which
> determine event timing (posting rhythms, burstiness, circadian
> patterns). We would also gratefully receive: untruncated raw tweet
> texts; post-type/interaction annotations if available; the
> follow/neighbor network if shipped; the original train/dev/test split
> with user IDs; and the release version/date. We record sha256
> provenance for every artefact, cite the release formally, and will
> share our replication code and findings.
