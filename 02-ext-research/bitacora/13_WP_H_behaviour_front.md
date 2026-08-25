# Bitácora 13 — WP-H, the behavioural front: mechanism confirmed, incumbent unbeatable, volume beaten

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** G3 = H2 (directional) — **PASS**, both predictions, p ≈ 0, signs
consistent. Property checks 9/9. Two runs byte-identical (S2.6).
**Artefacts:** `renyiext/behaviour.py` (new), `scripts/run_p3h_behaviour.py`
(new), `results/p3h_behaviour.json`, `figures/p3h_spec_b_alpha.png`,
`figures/p3h_spec_b_mention.png`.
**Seeds:** 42–51; fidelity vs the temporal producer (kept index + COUNT
block) at max |diff| 0.0.

---

## 1. H2 passes — the mechanism is real (gate G3)

Both pre-registered directional predictions hold with one-sided
Mann-Whitney p underflowing to 0.0:

| prediction | median bot | median human | AUC of the order, as-is |
|---|---|---|---|
| bots lower H₂ on the alphabet (collapsed) | **0.0000** | 1.1969 | 0.0450 |
| bots lower H₀ on mention-targets | **1.0000** | 7.2384 | 0.0398 |

Over half the bots post a single post-type exclusively (H₂ = 0) and mention
a single target; humans mix. Signs consistent, both p < 0.05 ⇒ **H2 PASS**;
the docs/03-PHASES.md P3 fallback (return to P1/P8′) is not invoked. The
raw-4-symbol sensitivity agrees (medians 0.0497 / 1.4528; σ_cfg of the
alphabet delta across encodings 0.1031).

The objects render the mechanism: the example bot's sequence is a wall of
`orig` with two lone `reply`/`retweet` bursts; its mention list is news
media (@msnbc×254, @YahooNoticias×93, …) — promotion, not conversation. The
example human mixes all three types continuously and mentions people
(@mirenzo61×352, …).

## 2. The floors split the verdict — downgrade EXECUTED for the incumbent claim

| comparison (paired, 10 seeds) | Δ | verdict |
|---|---|---|
| SPEC_B_ALPHA vs COUNT | **+0.0211**, 10/10, p = 0.0020, σ_cfg 0.0024 | **clears the volume floor** |
| SPEC_B_MENTION vs COUNT | **+0.0414**, 10/10, p = 0.0020 | **clears** |
| COUNT+SPEC_B vs COUNT | **+0.0539**, 10/10, p = 0.0020 | **clears** |
| SPEC_B_ALPHA vs META (−0.0361); matched vs META+NOISE(2) (**−0.0362**) | 0/10 | **`confounded_dimensionality` — DOWNGRADED** |
| SPEC_B_MENTION vs META (−0.0401); matched (**−0.0397**) | 0/10 | **`confounded_dimensionality` — DOWNGRADED** |
| SPEC_B vs META+NOISE(8) (−0.0294); COUNT+SPEC_B_ALPHA vs COUNT+META+NOISE(2) (−0.0200) | 0/10 | **`confounded_dimensionality`** |
| vs Shannon slices (unmatched +0.0060/+0.0029/+0.0037; matched +0.0086/+0.0062/+0.0034/+0.0030) | all 10/10 | **`real_but_subfloor_not_claimable`** at the 0.02 floor |

**Execution of the failure semantics** (plan WP-E task 3, rule 1): every
behavioural claim against the metadata incumbent is recorded as
**confounded** — META-lite alone scores **0.9972** (alpha sample) / 0.9946
(mention sample) on Cresci-2015, and at matched dimensions the SPEC_B blocks
add nothing to it (they subtract). Downgrade recorded in HANDOFF and
FINDINGS F10: on this corpus the behavioural front is NOT claimable as
information beyond metadata; its transfer value is exactly what WP-N tests
(WP-B already showed TB20's META is a different beast at 0.79).

The positive reading survives intact against the floor the amendment made
mandatory: the front **beats volume** everywhere, and the mechanism the
orders name (determinism, target narrowness) is the one H2 predicted.

## 3. Class-dependent exclusion is itself a finding (G2)

1,490 bots (44.4 %) vs 11 humans (0.6 %) have zero valid mention targets and
are excluded from the mention block; the mention sample (n = 3,269) is
therefore human-majority (0.5852) unlike the corpus. Every mention-block
number carries this selection; the exclusion counts are in the JSON per
class.

## 4. Registered rules, capture rates, collapse shares (G3)

Regex `@\w+`; only the registered rules were tried — no other tokenisation
was attempted (census). 2,763,184 kept-sample tweets: 63.49 % carry ≥ 1
token; 818,229 reply-leading tokens dropped; 39,194 self-mentions dropped;
499,856 mentions kept over 196,676 distinct targets. D11 collapse: quote
share **11.64 %** before (the plan's 11.6 %, confirmed) → original
**44.60 %** after; reply 29.61 %, retweet 25.79 % unchanged.

## 5. G4 shuffle null: exactly silent, as predicted

Per-account marginal-preserving permutation of post-type sequences
(default_rng(42)), full pipeline re-run: max |spec diff| **0.0**; AUC delta
**+0.00e+00**. Separating world stated in the JSON (S4.1): a separating
shuffle would require order-dependent features; the block is a symbol
histogram and the spectrum is permutation-invariant (P6) — the null proves
no order leakage end-to-end.

## 6. Decisions taken under the ambiguity protocol

1. **Unmatched gated verdicts added** beside the dim-matched rows (D10/
   protocol §3 require every family against every floor; the first complete
   run recorded only matched pairs — fixed before any document quoted a
   number).
2. **sigma_config axis** = the published D11 encoding axis {collapsed,
   raw 4-symbol} for alpha rows (population SD, ddof = 0); mention rows are
   encoding-invariant by construction (σ exactly 0, noted per row).
3. **META-lite recipe** = botsage's read-only fields (followers, following,
   tweet_count, age vs 2015-01-01) per the plan's "as used by botsage";
   `listed_count` stays dropped (WP-B alignment).

## 7. What failed and was not fixed

Four producer crashes before any output existed (a NameError in the
text-alignment loop — fixed by rebuilding the loader's exact stable-sort
ordering with an elementwise assertion against the cache; a missing
COUNT+META floor arm; a 2-D array passed to the sequence renderer; a variable
collision `nm`), plus the §6.1 completeness fix. All fixed; the committed
producer runs clean twice, byte-identical. Nothing outstanding.

## 8. Multiple comparisons counted

Alpha sample: 7 arms × 2 encodings; mention sample: 14 arms; 9 gated + 9
matched comparisons per encoding; 3 H2 directional tests (pre-registered
directions); 1 shuffle null. Headline claims: H2 PASS; vs-COUNT clears
(3 comparisons); vs-META confounded (4, downgraded); vs-Shannon subfloor
(5, not claimable). Census: no unregistered regex, rule, or encoding was
tried beyond the two declared.
