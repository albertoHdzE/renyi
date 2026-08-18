# 02 — Experimental protocol

Data, splits, baselines, statistics and reporting rules. Fixed before any experiment.
Deviations are recorded in [04-DECISIONS.md](04-DECISIONS.md) with a justification, in
the same commit as the result they affect.

---

## 1. Datasets

| Dataset | Role | n labelled | Balance | What we can use |
|---|---|---|---|---|
| **Cresci-2015** | primary | 5,301 | 63.2% bot | metadata, 2.83M tweets + reconstructed timestamps, text; graph unusable |
| **TwiBot-20** | transfer target | ~11k | — | graph, 5 (different) properties, BERT embeddings, labels |
| **TwiBot-22** | baseline sanity only | 1,000,000 | 13.99% bot | labels + split only (open release has no `edge.csv`, no tweets) |

Known corpus facts that must appear in every report:

- Cresci-2015 is **63.2% bot / 36.8% human**, not the "almost 50/50" its source paper
  claims. The majority baseline is **0.6321**.
- TwiBot-22's majority baseline is **0.8601**, and its official split is not stratified
  (train 7.80% bot, val 27.96%, test 29.44%). Numbers from 5-fold CV and from the
  official split describe tasks with class balances differing by a factor of two.
- TwiBot-20's five properties are **not** Cresci's five (screen-name length replaces
  favourites) and arrive already z-scored. This matters for H4 and is handled in §5.

## 2. Feature families

Each is a named block, evaluated alone and in combination. Ablation is by family, never
by individual column.

| Family | Dim | Contents |
|---|---|---|
| `META` | 5 | the source paper's metadata counts — **the incumbent to beat** |
| `SPEC_T` | 12 | Rényi spectrum on inter-arrival (6) and circadian (6) |
| `SPEC_B` | 12 | spectrum on post-type (6) and mention-target (6) |
| `SPEC_X` | 12 | spectrum on word- and character-frequency |
| `SPEC_N` | 6 | spectrum on ego-degree distribution (TwiBot-20 only) |
| `AIT` | ~6 | BDM 1.0 on action-DNA and temporal-DNA; NCD cohesion; gzip ratio |
| `AIT2` | ~4 | BDM 2.0 Tier 1–2 reuse gain (P7) |
| `TEXT` | 768 | BERT embeddings — imported from `botsage`, unchanged, for reference only |

## 3. Mandatory floors

No claim about any family is made until it has been compared against **all** of these on
the same split, with the same seeds:

1. **Majority class** — printed beside every accuracy, always.
2. **`META`** (5 metadata features) — the incumbent.
3. **Shannon only** — the α = 1 slice of whatever spectrum is being claimed. This is
   the floor that makes H1 meaningful.
4. **gzip compression ratio** of the DNA string — one scalar. The brutal free baseline
   for anything complexity-flavoured.
5. **Block Shannon entropy** of the DNA string at the same block size as BDM. The floor
   that makes any BDM claim meaningful, given that BDM 1.0 provably converges to it.
6. **Coefficient of variation** and **Fano factor** of inter-arrival times — the
   standard burstiness statistics, and the floor for the temporal front.
7. **Event count alone** — added 2026-08-18 by
   [../bitacora/02_h1_amendment.md](../bitacora/02_h1_amendment.md). On Cresci-2015 this
   scores **AUC 0.939** by itself. It is the hardest floor in the project and no
   spectrum number is quoted without it beside.

A family that fails to beat its floors is reported as failing. It is not rescued by
tuning.

## 4. Protocols — three distinct tasks, never mixed

| Protocol | Definition | Purpose |
|---|---|---|
| **A — 5-fold CV** | stratified, within dataset | comparability with `botsage` results |
| **B — official split** | the corpus's own train/test | distribution-shift stress within a corpus |
| **C — cross-dataset** | fit on Cresci-2015, test on TwiBot-20 | **H4, the primary claim** |

Protocol A and Protocol B numbers are **never** placed in the same column without a
label. The 9.2-point gap between them on this pipeline
(`docs/DISCREPANCIES_BOTSAGE.md` §5) is larger than the entire spread between competing
published methods, and conflating them is the specific error the source paper made.

### Protocol C mechanics

The feature blocks must be **commensurable** across corpora, which `META` is not:
TwiBot-20 ships different properties, already z-scored. Handling:

- `META` is mapped to the four overlapping fields (followers, following, statuses,
  account age); the non-overlapping fifth is dropped from **both** sides. The drop is
  reported.
- Spectra are corpus-independent by construction — a 6-vector of entropies in bits,
  computed at the same fixed `n_events`. This is the point of the hypothesis and must
  not be quietly undermined by per-corpus rescaling.
- Standardisation for Protocol C is fitted on **Cresci-2015 only** and applied to
  TwiBot-20. Refitting on the target is target leakage and would invalidate H4.

## 5. Statistics

- **Seeds:** ≥10 for every reported comparison. Seeds control fold assignment, subsample
  draws, and classifier initialisation.
- **Test:** paired Wilcoxon signed-rank across seeds, α = 0.05, two-sided. Report the
  mean difference, its SD, the win count, and p — the format already used by
  `scripts/significance.py`.
- **Multiple comparisons:** the four hypotheses are pre-registered, so no correction is
  applied to them. Every *exploratory* comparison beyond H1–H4 is labelled exploratory
  and Holm-corrected within its family. This distinction is recorded per result.
- **Effect size floor:** differences below **0.02 AUC** are not claimed as real
  regardless of p, because that is the measured seed-to-seed σ in this repository.

## 6. Metrics

Reported together, every time:

| Metric | Why |
|---|---|
| AUC + SD | rank quality, threshold-free |
| **TPR @ FPR = 1%** | the deployment regime; suspending a real user is the cost of a false positive |
| Macro-F1 | class-imbalance-aware |
| Accuracy **+ majority baseline** | comparability with the literature, with its floor visible |
| n excluded by `n_events` cutoff, per class | the fixed-n subsampling bias (§2 of [01-METHODS.md](01-METHODS.md)) |

## 7. Reproducibility rules

- `renyiext/config.py` is the single source of hyperparameters, carrying published
  comparison values, and listing anything inferred in `INFERRED_PARAMETERS` — the
  convention set by `dtwre.config`.
- Every script takes `--quiet` (summaries and final table only) and `--seeds`.
- Long runs go to background with output to a log; poll with `tail`, never block on
  accumulated output.
- Notebooks are **build artefacts** of `scripts/build_*.py`. Edit the builder.
- Caches key on the fields that actually change the result, following
  `disinfo.experiments.FeatureCache`. Subsample draws and CTM lookups are the expensive
  stages here and must be cached.
- Results land in `02-ext-research/results/` (gitignored); the package lives at the repo
  root as `renyiext/`.

## 8. What gets written down when a phase completes

Each phase appends to [04-DECISIONS.md](04-DECISIONS.md):

1. the gate, and whether it passed;
2. the measured number with error bars and seed count;
3. every decision made during the phase that a reader could not infer from the code;
4. anything that failed and was *not* fixed, with the reason.

Item 4 is the one that is usually omitted and is usually the most useful. The three
`DISCREPANCIES` documents in this repository exist because that habit paid off.
