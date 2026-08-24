# Bitácora 05 — WP-A, the censoring probe

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** none (this is a control, not a family claim). Its output is an
**amendment trigger**, evaluated below.
**Artefacts:** `scripts/run_p2c_probe.py`, `results/p2c_probe.json`,
`results/figures/p2c_probe_grid.png`, `results/figures/p2c_probe_rasters.png`,
rows in `EVIDENCE-INDEX.md`.
**Seeds:** classifiers refit on seeds 42–51; accounts drawn once per
(generator, window) from `default_rng(1_000_003 · (gen_index+1))` — the same
underlying processes feed every window, so class differences are truncation and
nothing else. Two runs byte-identical.

---

## 1. What ran

Plan WP-A (v1.1), Definition D5: for each generator g ∈ {periodic(jitter 0.5),
poisson, bursty(tail 1.2)} — each from its **own renewal process**, no separate
Poisson — 120 + 120 accounts; class A observed 900 days, class B the *same*
process truncated to its first W ∈ {30, 90, 400} days. Identical feature
pipeline (`temporal_blocks_ts`, the generic core that `temporal_blocks`
delegates to — refactor regression-checked byte-identical on a full P2 rerun
before the probe was written). Binary AUC(B vs A), HGB, 5-fold CV, 10 seeds.

One calibration, printed (G3): bursty's scale is set so its median Δt equals
the other generators' 6 h (`scale = 6 h / (2^{1/1.2} − 1)`); tail exponent
untouched. Uncalibrated, that generator emits ~7×10⁵ events/account over 900 d
— a compute artefact, not behaviour.

## 2. Result — the trigger fires at the ceiling

| cell | SPEC_T | COUNT+SPEC_T | SHAPE |
|---|---|---|---|
| periodic \| 30 d | 1.0000 | **1.0000** | 1.0000 |
| periodic \| 90 d | 1.0000 | **1.0000** | 1.0000 |
| periodic \| 400 d | 0.9347 | **1.0000** | 0.9400 |
| poisson \| 30 d | 1.0000 | **1.0000** | 1.0000 |
| poisson \| 90 d | 1.0000 | **1.0000** | 1.0000 |
| poisson \| 400 d | 0.9402 | **1.0000** | 0.9582 |
| bursty \| 30 d | 0.9972 | **0.9969** | 0.9887 |
| bursty \| 90 d | 0.9739 | **0.9781** | 0.9765 |
| bursty \| 400 d | 0.9302 | **0.9737** | 0.9224 |

**Trigger (≥ 0.85 on COUNT+SPEC_T): FIRED — worst cell 1.0000.**
Minimum over all 27 cell-metric readings: **0.9224**.

The separating world named in advance (G4) — identical generator, identical
rate, only the window differs — separates essentially perfectly. Truncation
alone pushes this pipeline to AUC ≈ 1.

## 3. What the picture says (G1)

The rasters show the mechanism plainly: class B's events stop at the vertical
line; its inter-arrival histogram therefore has *no mass above W*, while class
A's does. On a shared grid pinned at hi = 400 d, B occupies fewer high bins →
lower H₀, thinner upper tail, different level — and because we subtracted H₁
per half for SHAPE, support/occupancy differences survive level removal.
This is exactly the failure mode bitacora 01 §3.2 named ("bot rasters
concentrate in days 600–900 while humans span the full window") now measured as
a ceiling rather than observed as a suspicion.

## 4. Interpretation limits of the probe itself (stated before anyone else says them)

1. It is a ceiling construction: births at t = 0 with a hard cutoff. Real
   accounts fade gradually; the artefact on real data need not reach 1.00.
2. The pipeline ran pre-WP-D (`overflow_cell: false` in the JSON): intervals
   above 400 d are dropped, which *removes* distinguishing mass from class A.
   The post-WP-D re-run (obliged by plan WP-A task 4) can only raise this
   ceiling.
3. min_events = 5 excludes heavily-truncated accounts; kept-B medians are in
   the table (28 at worst).
4. This does not prove P2's +0.0367 IS censoring. It proves this pipeline
   family cannot distinguish "behavioural shape" from "window shape" without
   further controls — which is what WP-F's equal-window arms are for, and why
   every SPEC-family claim must now be stated net of this ceiling.

## 5. What failed and was not fixed

Nothing failed operationally (determinism clean, all cells reported). The
finding itself is the negative: the amendment trigger fired at its maximum.

## 6. Multiple comparisons counted

Nine cells × three metrics = 27 readings, all reported above and in the JSON;
one trigger evaluation (pre-registered metric and threshold); one generator
calibration constant (recorded §1). No hypothesis test was performed; no floor
was evaluated.

## 7. Consequences (binding per plan §7.1)

The censoring amendment fires. From WP-F onward: SPEC-family claims are stated
net of the probe ceiling; WP-N/H4 framing cites this entry; HANDOFF carries the
downgrade of the "shape" reading of P2 pending WP-F's equal-window result.
