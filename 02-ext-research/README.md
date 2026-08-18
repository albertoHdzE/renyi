# 02-ext-research — Distributional shape and algorithmic structure in bot detection

**Status:** P0 run, gate G0 FAILED, P1 blocked on a decision. See [HANDOFF.md](HANDOFF.md).
**Started:** 2026-08-18.
**Relation to `01-info-propagation/`:** extension in subject, **separate** in scope, code and
claims. The three replications in `01-info-propagation/` reproduce published work. This
project asks a new question and must stand on its own evidence.

---

## The question in one paragraph

Bot detection today rests on account metadata (follower counts, account age, tweet
volume) and on text embeddings. Both are *era-specific* and *purchasable*: follower
counts can be bought, and a 2015 metadata profile does not describe a 2020 bot. We ask
whether two families of measure that describe the **shape** of an account's behaviour,
rather than its magnitude, are (i) discriminative and (ii) more stable across dataset
and era shift:

1. the **Rényi spectrum** `{H_α}` of an account's behavioural distributions — a
   tail-resolved characterisation of distributional shape, of which Shannon entropy is
   the single point `α = 1`;
2. **algorithmic-information measures** (BDM, NCD, and BDM 2.0's reuse gain) over
   behavioural sequences, which detect *shared generators* across accounts — the
   signature of a fleet driven by one script.

The primary claim under test is **generalisation, not peak accuracy** (see
[docs/00-CHARTER.md](docs/00-CHARTER.md) §3).

## Documents — read in this order

| File | Purpose |
|---|---|
| [docs/00-CHARTER.md](docs/00-CHARTER.md) | Objectives, pre-registered hypotheses, scope, non-goals, success criteria |
| [docs/01-METHODS.md](docs/01-METHODS.md) | Mathematical specification of every estimator, with the properties each must satisfy |
| [docs/02-PROTOCOL.md](docs/02-PROTOCOL.md) | Data, encodings, splits, baselines, statistics, reporting rules |
| [docs/03-PHASES.md](docs/03-PHASES.md) | Phase plan with deliverables and go/no-go gates |
| [docs/04-DECISIONS.md](docs/04-DECISIONS.md) | Running log of decisions and their justification (the `DISCREPANCIES.md` analogue) |
| [docs/05-RISKS.md](docs/05-RISKS.md) | Threats to validity, and the control for each |
| [docs/06-STANDING-RULES.md](docs/06-STANDING-RULES.md) | Datasaurus gate, bitácora, didactic notebooks, nulls and positive controls |
| [bitacora/](bitacora/) | Append-only record of record. Start at `00_kickoff.md` |

**This directory is the source of truth.** Where a phase result contradicts a document,
update the document in the same commit as the result.

## Layout (as it will be)

```
02-ext-research/
  README.md            this file — status board
  HANDOFF.md           current state for session resumption
  docs/                the seven documents above — design, frozen at kick-off
  bitacora/            NN_name.md — history, append-only, never silently edited
  .venv/               environment (per repo convention: beside the work it serves)
  notebooks/           narrative notebooks; build artefacts of scripts/build_*.py
  results/             tables, figures, JSON — gitignored
```

The reusable package will live at the repository root as `renyiext/`, following the
`dtwre/`, `disinfo/`, `botsage/` convention. Paper-specific artefacts stay here.

## Phase board

| Phase | Name | Gate | Status |
|---|---|---|---|
| P0 | Data layer and event reconstruction | 2.83M timestamps decoded, sanity checks pass | **FAIL** — see [bitacora/01](bitacora/01_p0_data_layer.md) |
| P1 | Rényi spectrum estimator | property checks + finite-sample bias control | **blocked** on the D3 amendment |
| P2 | Temporal front | H1 | not started |
| P3 | Behavioural and text fronts | incremental AUC over P2 | not started |
| P4 | Digital DNA, BDM 1.0, NCD | H3, beats gzip and block entropy | not started |
| P5 | Network front (TwiBot-20) | permutation + configuration-model controls pass | not started |
| P6 | **Cross-dataset generalisation** | **H4 — the primary claim** | not started |
| P7 | BDM 2.0, Tiers 1–2 | reuse gain beats NCD | not started |
| P8 | Conditional CTM (optional) | only if P7 passes | not started |
| P9 | Write-up and artefacts | notebook reproduces every figure | not started |

## Standing rules

From this repository:

- No test suite. Correctness is asserted inline against **properties the mathematics
  must satisfy** (`checks.py`). When touching an estimator, check a property.
- Every batch script ships `--quiet`.
- Every reported comparison carries **≥10 seeds, paired Wilcoxon, and the majority
  baseline** printed beside it.
- Entropy estimators state their **base, α, and bias correction** in the code.
- Scalers and vocabularies are fitted on **training folds only**.

From the `CausalBool` / `deconv-lab` programmes — see
[docs/06-STANDING-RULES.md](docs/06-STANDING-RULES.md):

- **Datasaurus gate.** No number enters any document until its object has been rendered
  and compared elementwise. Enforced by the `datasaurus` skill
  (`.claude/skills/datasaurus/`). Three of the eleven risks here are datasaurus-shaped
  and would each produce a *positive* result.
- **Bitácora.** Append-only numbered record; protocol changes are new entries, never
  silent edits. Negatives carry the same weight as positives.
- **Didactic notebooks.** Every phase that produces a result produces a notebook that
  teaches the object before the result, executable end to end from a clean checkout.
- **Marginal-preserving nulls and synthetic positive controls** on every estimator, in
  the same run as the real data.
