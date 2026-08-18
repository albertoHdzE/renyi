# HANDOFF

**Updated:** 2026-08-18
**Branch:** `main`
**State:** P0 done (G0 fail, corpus property). H1 amended. P1 done (G1 pass, 8/8).
Next: **P2, the temporal front**.

## Read first

1. [bitacora/03_p1_spectrum.md](bitacora/03_p1_spectrum.md) §4 — the standing warning on H1
2. [bitacora/02_h1_amendment.md](bitacora/02_h1_amendment.md) — H1 as now in force
3. [bitacora/01_p0_data_layer.md](bitacora/01_p0_data_layer.md) — the corpus

## Confirmed state

**D1 confirmed, D9 added.** Snowflake decode gated: 0 violations across 2,763,927
elementwise constraints; counter null separates (circadian TV 0.2248 vs 0.0002).
63,830 pre-snowflake tweets discarded. Corpus 5,301 users / 2,763,927 events.

**H1 amended** (option a): variable n, spectrum must beat BOTH Shannon-alone AND
event-count-alone by >0.02. Count alone scores **AUC 0.939** on Cresci-2015.

**P1 gate passed 8/8.** Estimator in `renyiext/spectrum.py`, base 2, plug-in, no bias
correction by design (D3'). Two defects caught and fixed: catastrophic cancellation near
alpha=1 (now a series expansion, err 7.55e-07 -> 1.04e-09), and a one-sided tolerance in
the P4 check itself.

## The warning that should shape P2

On the synthetic positive control -- engineered to have exactly H1's mechanism -- the
spectrum's gain over Shannon alone is **within +-0.014 across the whole difficulty
sweep**, inside the 0.02 floor and negative at both ends. See
[bitacora/03](bitacora/03_p1_spectrum.md) §4. This is the DTWRE alpha-flatness pattern
on a second substrate. Not falsified; three testable explanations are listed.

## Single next action

**P2 — temporal front.** First job is item 1 of bitacora 03 §5: sweep the log-binning
grid (`n_bins`, `lo`, `hi`) as a classifier parameter, since it may be destroying the
tail before the spectrum sees it. Render the per-class alpha-curve with error bands
before quoting any AUC.

## Open items

1. Log-binning grid unswept for the classifier. Blocking P2's interpretation.
2. `quote` post-type rule gives an implausible 11.6% share.
3. TwiBot-20 volume confound unmeasured.
4. `Config.n_events` now used only by the (unrun) robustness arm.
