# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A research repository ("Renyi tests", per `README.md`) for work on Rényi entropy and
algorithmic information theory. **As of the initial commit it contains no source code** —
only a reference library of PDFs and an empty directory scaffold for a first research
project. Do not assume a build system, test suite, or package layout exists; check before
referencing one.

The only tracked files are `README.md` and `.gitignore`. Everything else listed below is
present on disk but untracked.

## Layout

- `RenyiPapers/` (8), `FirstCollection/` (14), `Causation/` (10) — the reference library
  (PDF only). These were deduplicated on 2026-07-30: every paper now exists in exactly one
  directory, verified by SHA-256 and by filename. `RenyiPapers/` holds the Rényi/AIT core
  (items 011–017) that previously also sat in `FirstCollection/`; `Causation/` holds the
  network-and-causality material. Keep this invariant — check hashes before adding a PDF
  that may already be filed elsewhere.

  Numeric filename prefixes (`001.`–`017.`) are a single sequence spanning `FirstCollection/`
  and `RenyiPapers/`, so `FirstCollection/` now has gaps at 011–017 by design.
- `01-info-propagation/` — the first research project, numbered-prefix style implying
  further `NN-topic/` projects will be added as siblings. It holds three sub-projects
  (`entropia-paper/`, `desinformation-paper/`, `bot-detection-paper/`), each currently
  containing only a `paper/` directory with the source PDF being worked from.

The `NN-topic/<name>-paper/paper/` nesting is the established convention: the `paper/`
subdirectory holds source material, leaving room alongside it for code, data, and
write-ups. Follow it when adding work.

## Reference library topics

Useful when locating background for a task: Rényi's own *On Measures of Entropy and
Information* and Crooks' treatment of the same; Campbell's coding theorem for Rényi
entropy; Bercher on escort distributions; Grünwald & Vitányi on algorithmic information
theory and MDL; Li & Vitányi's Kolmogorov complexity textbook and Chaitin's AIT; a large
body of Zenil's work on algorithmic probability, the coding theorem method, BDM, and
causal deconvolution; and network/causality material (Milo's motifs, graph complexity
from an algorithmic-information perspective).

`Causation/` retains three *distinct versions* of the Zenil causal-deconvolution work,
which are deliberately not deduplicated because they are different documents:
`nature zenil deconvolution.pdf` is the 9-page published article ("Causal deconvolution by
algorithmic generative models"), while `ACausalDeconNetGeneMecha-NaturePreprint.pdf` (36pp)
is a longer preprint under a different title. Cite the published version.

`01-info-propagation/` targets dynamic time-weighted Rényi entropy with graph neural
networks for opinion propagation, plus GNN-based disinformation and bot detection.

## The `dtwre` package

`dtwre/` at the repository root is a from-scratch replication of Tong et al.,
*Entropy* 2025, 27, 516 (the `01-info-propagation/entropia-paper/` paper), which has
no official code release. Read `docs/DISCREPANCIES.md` before changing anything in
`entropy.py`, `pipeline.py` or `embeddings.py` — it records four non-obvious
decisions that the results depend on, each found by a failure rather than by reading
the paper:

1. Eq. 1 is renormalised per neighbourhood; the literal global form diverges at α=1.
2. Eq. 4's `t - t_k` is in time-step indices; seconds underflow the weight to zero.
3. Message-passing and supervision edges must be disjoint (`mp_frac`), or test AUC
   falls to 0.36 — below chance.
4. Node2Vec needs `lr=0.005`; the word2vec default of 0.025 under Adam produces
   embeddings weaker than Adamic-Adar.

Layering is strict: `entropy`/`embeddings` → `features` → `models` → `pipeline` →
`experiments`/`plots`. No scientific logic lives in the notebook; it imports from
here so the same code is testable outside Jupyter.

`config.Config` is the single source of hyperparameters, and carries the published
results (`PAPER_TABLE1`, `PAPER_TABLE2`, …) so any run can be compared against the
paper. Values the paper does not state are listed in `config.INFERRED_PARAMETERS` —
keep that list accurate when adding parameters.

## The `disinfo` package

`disinfo/` at the repository root replicates Lakzaei, Haghir Chehreghani &
Bagheri, *Artificial Intelligence Review* 2024, 57:52 (the
`01-info-propagation/desinformation-paper/` paper). Read
`docs/DISCREPANCIES_SURVEY.md` before changing `layers.py`, `survey_data.py` or
`data.py`.

**This one is a survey**, which changes what the code is for. It has no
experiments, no code release, and all six figures are diagrams. The replication
therefore has three layers, and they are independent — do not collapse them:

1. `taxonomy.py` + `plots.py` reproduce Figs. 1–6 by *rendering encoded
   structures*, not by tracing pictures. Fig. 3's computation tree is unrolled
   from the adjacency on purpose, so it checks the Eq. 1 semantics.
2. `survey_data.py` transcribes Tables 1–4 and states Sect. 5.3.2's and Sect. 7's
   prose as predicates. `verify_claims()` returns 5/6 supported; the failure
   (Sect. 7's "multiclass accuracy below 50%", contradicted by 35 of 36
   accuracies in the survey's own tables) is a real finding, not a bug — do not
   "fix" it.
3. `layers.py` implements Eqs. 2–10 **as printed**, with the literature-standard
   form available as an option wherever the survey and its cited source disagree
   (they do, in Eqs. 2, 4 and 8). Both forms must stay.

Four things that are load-bearing and non-obvious:

1. LIAR needs `csv.QUOTE_NONE`; the default dialect silently drops 47 of 12,836
   rows and raises nothing.
2. Similarity/attribute graphs give **node** classification over one graph;
   propagation/heterogeneous give **graph** classification over one graph per
   item. That split is why all three semi-supervised rows in Table 1 are
   similarity-graph methods.
3. `attribute_graph` must cap group size. Hu et al.'s rule read literally is a
   union of cliques, and LIAR's `party` field would emit ~12M edges.
4. TF-IDF vocabulary and the scaler are fitted on **training rows only**
   (`fit_on=train_idx`). Fitting on all rows leaks and is the common silent error
   here.

There is no test suite; `checks.py` asserts six properties the layers must
satisfy (permutation equivariance, GCN regularity, attention normalisation, GIN
injectivity, finite output on isolated nodes, GATv2 dynamic attention). Preserve
that style — when touching a layer, check a property. Note the LSTM aggregator is
deliberately excluded from the equivariance check: Eq. 9 genuinely is
order-dependent.

`experiments.FeatureCache` keys stage 1 on the fields that actually change it, so
architecture and graph-type sweeps extract features once. A full
`--experiments` run is ~1.8 h; the cache saves roughly 8× on the sweeps.

## The `botsage` package

`botsage/` at the repository root replicates Deshmukh, *Bot Detection in Social
Media using GraphSage and BERT* (SJSU MS Project 1465, 2025) — the
`01-info-propagation/bot-detection-paper/` paper. Read
`docs/DISCREPANCIES_BOTSAGE.md` before changing `sage.py`, `data.py` or
`pipeline.py`.

Unlike the other two, **everything paper-specific lives in the paper folder**
(`.venv`, `replication.ipynb`, `results/`, `results/figures/`); the root holds
only the reusable package. Follow that split for future papers.

The replication succeeds (97.89% vs the paper's 98.68% on Cresci-15, ordering
preserved). Five things are load-bearing and non-obvious:

1. **The GraphSAGE layer is never trained** (Sect. 3.5). An untrained
   `SAGEConv(5,128)` is affine in `[x_v ‖ mean N(v)]`, so the embedding has
   **rank 10, not 128**. Do not "fix" this — reproducing it is the point, and
   `TrainedSAGE` exists for the comparison.
2. **The gap between the 128-d embedding and its 10 inputs is a regularisation
   artefact**, not information: it decays from +0.033 to +0.0004 as `svm_C`
   grows. Any claim about the two being equivalent must be made at large `C`;
   at the default `C=1` they differ by ~1 point for basis-geometry reasons.
3. **The binding constraint is the linear kernel**, not the untrained layer.
   Untrained+MLP scores 0.9775; trained+MLP 0.9739. Training the layer buys
   nothing; dropping linearity buys +4.2 points.
4. **Protocol dominates on both datasets.** Cresci-15 official split 0.886 vs
   5-fold CV 0.978 (the split is not random — test membership is predictable at
   AUC 0.79). TwiBot-22's majority baseline is 0.8601 and *all* of Table 5 sits
   below it. Always report `majority_baseline` next to accuracy.
5. **Never materialise every tweet vector.** `user_text_embeddings` accumulates
   per-user sums batch by batch; the naive version peaks at 12 GB on Cresci-15
   and would need ~250 GB on TwiBot-22.

`checks.py` asserts seven properties (PyG closed form, affineness, rank bound,
linear-SVM equivalence, permutation equivariance, isolated-node behaviour, seed
dependence). Preserve that style. Note `sage_embeddings` is seed-dependent by
construction, so `Config.sage_seed` must be threaded through anything comparable.

Text embeddings are cached under `data/processed/bot/` as
`{dataset}_{model}_mean_labelled.npy` — **labelled rows only** (5,301 of 1.29M
for Cresci-15; the full matrix is 3.7 GB of mostly zeros). `pipeline.expand_text`
accepts either shape.

## Commands

The virtualenv lives beside the paper it serves, not at the root:

```bash
P=01-info-propagation/entropia-paper/.venv/bin/python

bash scripts/get_data.sh                    # fetch CollegeMsg + CED cascades
$P scripts/run_all.py --quiet --weibo       # all tables/figures -> results/ (~15 min)
$P scripts/build_notebook.py                # regenerate replication.ipynb
```

```bash
D=01-info-propagation/desinformation-paper/.venv/bin/python

bash scripts/get_disinfo_data.sh            # LIAR, Twitter15/16, PHEME (+ reuses CED)
$D scripts/run_disinfo.py --quiet           # figs + meta-analysis -> results/disinfo (~30 s)
$D scripts/run_disinfo.py --quiet --experiments   # + live GNN runs (~1.8 h)
$D scripts/build_disinfo_notebook.py        # regenerate replication.ipynb
```

```bash
B=01-info-propagation/bot-detection-paper/.venv/bin/python

bash scripts/get_bot_data.sh                # cresci-2015, twibot-20/22 (~2.5 GB)
$B scripts/prepare_bot_embeddings.py --model distilbert-base-uncased   # ~28 min
$B scripts/prepare_bot_embeddings.py --model bert-base-uncased         # ~50 min
$B scripts/run_botsage.py --quiet           # checks + figures + baselines (~5 s)
$B scripts/run_botsage.py --quiet --experiments
$B scripts/build_botsage_notebook.py        # regenerate replication.ipynb
```

All three `replication.ipynb` files are **build artefacts** of their builder
script — edit the builder, not the notebook, or changes are lost on the next
regeneration. Each notebook is pinned to its own kernel (`dtwre-entropia`,
`disinfo-venv`, `botsage-venv`); register it with `ipykernel install` before
executing.

Long runs belong in the background with output to a log; `run_all.py` takes `--quiet`
(summaries only) and `--seeds`. Node2Vec dominates runtime, so reuse
`experiments.EmbeddingCache` across sweeps — α, λ, negative-ratio and window never
change the message-passing graph, so recomputing the embedding is pure waste.

There is no test suite. Correctness is instead asserted inline against properties the
maths must satisfy (uniform → `log n` for every α, exact Shannon at α=1, monotonicity
in α, random-walk steps being real edges). Preserve that style: when touching an
estimator, check a property, not just that the code runs.

## Tooling

`.gitignore` is GitHub's standard Python template plus `data/raw/`, `results/`,
`.venv/`. There is no dependency manifest; deps are installed ad hoc (`torch numpy
scipy networkx scikit-learn matplotlib pandas tqdm jupyter ipykernel`). DGL and
PyTorch Geometric are deliberately **not** used — GraphSAGE and Node2Vec are
implemented directly in PyTorch to keep the install trivial and the notebook
readable, even though the paper used DGL.

Note the template ignores `lib/` and `share/python-wheels/` — avoid those names for
source directories.

## Domain-code expectations

Per the user's global instructions, information-theoretic metrics, statistical
computations, numerical precision, and verification algorithms are not subject to the
usual brevity rules: read full context, reason explicitly, and verify before committing.
Entropy estimators in particular need their base, α parameter, and bias-correction
choices stated in the code rather than left implicit.

Batch and evaluation scripts should ship a `--quiet` flag that suppresses per-item
progress and prints only summaries and the final table.
