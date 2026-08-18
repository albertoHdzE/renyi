# Bitácora 01 — P0, data layer and the gate on D1

**Date:** 2026-08-18
**Branch:** `main`
**Gate G0:** **FAIL** — on the `n_events` criterion, not on the decode.
**Artefacts:** `renyiext/{config,events}.py`, `scripts/run_p0_events.py`,
`results/p0_events.json`, `results/figures/p0_g1_objects.png`,
`results/figures/p0_g4_counter_null.png`
**Seed:** 42 (`Config.seed`); the run is deterministic and was executed twice with
identical output.

---

## 1. D1 is confirmed — but not by the evidence bitácora 00 offered

Bitácora 00 §4 supported the snowflake decode with "200,000 decoded, 0 failures,
range 2010-11-04 → 2013-06-03". That is a count and a range, which G2 says is not
agreement, and it was labelled NOT YET GATED. Correctly so — it was hiding a bug.

**G1 RENDER found what the counts could not.** The corpus timeline showed a spike of
~64,000 tweets on day 0, and hour-of-day showed an artefactual peak at hour 1. Both sat
exactly at the snowflake epoch, 2010-11-04T01:42:54Z.

**Cause.** Tweet ids issued before 2010-11-04 are *sequential*, not snowflakes, and carry
no time information — but they are large numbers (~3 × 10¹⁰), so they survived the naive
`id ≥ 2²²` guard and decoded to a few milliseconds past the epoch. The id distribution
confirms it with a clean four-order-of-magnitude gap and nothing in it:

```
largest sub-threshold id     29,700,661,919   -> 2010-11-04 01:43:01
smallest snowflake       292,906,606,796,800  -> 2010-11-04 21:06:49
```

**63,830 tweets (2.26%) were on a single millisecond.** Decision **D9**: the threshold is
the first snowflake id (29,700,859,247), not a bit width. The exclusion is carried
per account (`dropped_per_user`) because it is class-dependent — older accounts lose more.

**G2 could not have caught this**, and it is worth saying why: those fake timestamps are
2010, and the accounts they belong to were created 2007–2009, so every one of them
satisfies `ts ≥ created_at`. The elementwise check passed at 0 violations *both before and
after* the bug was fixed. **G1 caught what G2 missed.** That is the argument for rendering
first, made concrete on the first day of the project.

### After the fix

| Gate | Evidence | Result |
|---|---|---|
| **G1** | timeline, hour-of-day, inter-arrival, counts, per-account rasters | both artefacts gone; circadian curve clean |
| **G2** | decoded time vs the account's own `created_at`, an independent field that never passed through the decoder | **0 violations / 2,763,927** events, 0 of 5,101 accounts |
| **G3** | epoch, shift, first-snowflake id are format constants, not fitted; post-type shares and the `n_events` sweep printed | no knob interior-to-bracket issue |
| **G4** | counter null — if the top 41 bits were a counter, hour-of-day is uniform | decoded TV **0.2248**, peak/trough **12.82**; null TV **0.0002**, peak/trough **1.00** |

The circadian shape is independently plausible: trough 03–05 UTC, peak 20–22 UTC, which
for a largely Italian corpus (CET) is 04–06 and 21–23 local. The world named in advance
as separating did separate. **D1 stands.**

**Corpus after D9:** 5,301 users (3,351 bot / 1,950 human), **2,763,927** events,
2010-11-04 → 2013-06-06.

## 2. G0 FAILS, and the reason changes the design

### 2.1 A datasaurus of my own

Bitácora 00 §4 reported "~533 per user". That is the **mean** over a bimodal
distribution. The median is **38**. The per-class picture:

| quantile | bot | human |
|---|---|---|
| 25 | 16 | 237 |
| **50** | **23** | **834** |
| 75 | 38 | 2,551 |
| 90 | 63 | 3,193 |
| mean | 55 | 1,322 |

196 bots have **zero** decodable events; 4 humans do. The human mass near 3,200 is the
Twitter API's per-user timeline cap, not behaviour.

### 2.2 Volume is the dominant signal in this corpus

**AUC of event count alone = 0.939.** One feature, already present in `META` as
`tweet_count`, and bots are the *low*-volume class — the opposite of the usual assumption.

This is R1 realised, and it is worse than anticipated. It is not that an uncorrected
spectrum *might* encode volume; it is that volume is worth 0.939 here, so anything
correlated with it will look excellent.

### 2.3 D3's control is infeasible as specified

Fixed-n subsampling was designed to make the bias a constant offset per α. On this corpus
it cannot be applied at any useful n:

| n | kept, bot | kept, human |
|---|---|---|
| 8 | 0.829 | 0.981 |
| **12** | **0.808** | **0.973** |
| 16 | 0.751 | 0.969 |
| 32 | 0.314 | 0.941 |
| 64 | 0.097 | 0.900 |
| 128 | 0.043 | 0.833 |

The G0 criterion (≥80% retained in **both** classes) is met only at **n ≤ 12**. A Rényi
spectrum on 12 events is not a measurement: `H₀ ≤ log₂ 12 = 3.58` bits and every order is
bias-dominated. Raising n to where the spectrum is meaningful (≥64) discards 90% of the
bot class — and discards it *non-randomly*, keeping exactly the atypical high-volume bots.

**The two requirements are in direct conflict on Cresci-2015.** This is a property of the
corpus, not a bug, and no amount of tuning dissolves it.

## 3. What the render suggests anyway

Two observations, recorded as **exploratory** and not tested:

1. The inter-arrival distributions separate mainly in the **tail**, Δt ≳ 10⁸ ms (~1 day),
   where bots carry more mass. Tail is where α ≠ 1 has leverage, so H1's mechanism is at
   least visible in the object. Not a result — no null, no seeds, and confounded with
   volume via the observation window.
2. The bot rasters concentrate in days 600–900 while human rasters span the full window.
   Bot accounts appear to be *younger and shorter-lived* in this corpus, which is a third
   route by which volume, window length and class are entangled.

Both are exactly the "per-unit feature that is really sample size, window length or age"
failure that deconv-lab rule 13 names. Neither is quoted further until controlled.

## 4. Decision required — this amends a pre-registered protocol

Per standing rule S2, protocol changes are new numbered entries and never silent edits,
and per the charter the hypotheses were pre-registered so they cannot be retrofitted.
**I am not amending them unilaterally.** The options, with my recommendation:

**(a) Reframe H1 as incremental over volume.** Keep variable n; report the spectrum's
incremental AUC **over event count**, and require it to clear the same 0.02 floor.
*This is arguably the question we meant to ask all along* — "does shape add anything
beyond magnitude" — and it is the only option that uses the whole corpus.
Cost: no longer a clean fixed-n comparison; needs count as a mandatory covariate in
every model, and a partial-correlation control.

**(b) Volume-matched subpopulation.** Caliper-match bots to humans on log event count,
run the fixed-n design at n = 64 inside the matched set. Clean, but ~325 bots survive,
which is underpowered for a 0.02 effect.

**(c) Move the primary front to TwiBot-20** and treat Cresci-2015 as the transfer
*target* rather than the source, inverting protocol C. Requires measuring TwiBot-20's
volume confound first; unknown today.

**Recommendation: (a) as primary, (b) as the robustness check**, with (c) decided after
TwiBot-20's count distribution is measured. Under (a) the pre-registered H1 threshold is
unchanged; what changes is the baseline it is measured against — Shannon-alone **and**
event-count-alone, both.

## 5. Open items carried forward

1. **The amendment in §4 needs assent before P1 proceeds.** Blocking.
2. Post-type shares are original 0.330, reply 0.296, retweet 0.258, quote 0.116. The
   `quote` rule (trailing t.co link) is a proxy and 11.6% is higher than plausible for
   2011–13 quote tweets; it is probably catching link-sharing originals. Needs a better
   rule or a 3-symbol collapse — noting D5's constraint that there is no alphabet-3 CTM
   table, so the collapse would be to alphabet 4 with one symbol unused, or to 2.
3. TwiBot-20's volume confound is unmeasured and decides option (c).
4. `n_events = 128` in `Config` is now known to be wrong for this corpus and is left in
   place only until §4 is decided.

## 6. Multiple comparisons counted

Tried and recorded: `n_events` ∈ {2, 4, 8, 10, 12, 16, 20, 24, 32, 64, 128, 256, 512}
(13 values, all reported); inter-arrival log-binning at 80 bins over 1 ms – 1 year (one
choice, not swept — flagged for P1); hour-of-day at 24 and 48 bins (both rendered).
No model was fitted and no hypothesis was tested in this phase.
