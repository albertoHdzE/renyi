# HANDOFF

**Updated:** 2026-08-18
**Branch:** `main`
**State:** P0 run. Gate **G0 FAILED**. P1 is **blocked** on a decision.

## Read first

1. [bitacora/01_p0_data_layer.md](bitacora/01_p0_data_layer.md) — what P0 found
2. [bitacora/00_kickoff.md](bitacora/00_kickoff.md) — why the project exists
3. [docs/00-CHARTER.md](docs/00-CHARTER.md) — the four pre-registered hypotheses

## Confirmed state

**D1 (snowflake decode) is confirmed** and properly gated: G1 rendered, G2 gives 0
violations across 2,763,927 elementwise constraints against an independent field, G4
separates the decode from its counter null (circadian TV 0.2248 vs 0.0002).

**D9 added:** pre-snowflake ids carry no timestamp. 63,830 tweets (2.26%) discarded;
found by the G1 render, invisible to G2. Corpus is 5,301 users / 2,763,927 events.

**G0 failed** on retention. Bots post a median of 23 tweets, humans 834. Event count
alone scores **AUC 0.939**. Fixed-n subsampling (D3) retains ≥80% of both classes only
at n ≤ 12, where a Rényi spectrum is bias-dominated.

`renyiext/` has `config.py` and `events.py`. No spectrum estimator yet.

## Single next action — needs a decision, not code

The D3 amendment in [bitacora/01](bitacora/01_p0_data_layer.md) §4. Three options;
recommendation is **(a) reframe H1 as incremental over event count**, with (b)
volume-matched subpopulation as the robustness check. This touches a pre-registered
protocol and is deliberately not taken unilaterally.

Once decided: P1, the spectrum estimator and properties P1–P8.

## Open items

1. **D3 amendment.** Blocking P1.
2. `quote` post-type rule (trailing t.co link) gives an implausible 11.6% share; likely
   catching link-sharing originals. Needs a better rule.
3. TwiBot-20's volume confound unmeasured; decides option (c).
4. `Config.n_events = 128` is known wrong for this corpus; left until item 1 is decided.
