# Overview — what three replications actually established

**Purpose.** This folder is a self-contained briefing on the findings that came out of
replicating three papers on bot detection, disinformation detection and opinion
propagation. It exists so that the argument can be handed to someone who has not read
the code, and so that the follow-on project (`02-ext-research/`) has a citable,
numerically-grounded starting point rather than a recollection.

**Status.** Documentation only. No new experiments are run here. Every number in this
folder is copied from a result file produced by the three replications, and every one is
traceable through [EVIDENCE-INDEX.md](EVIDENCE-INDEX.md) to the file it came from and the
command that regenerates it.

**Audience.** Two, deliberately:

1. a reader who wants to know *why* the published numbers in this area should be read
   with care — start at [01-FINDINGS.md](01-FINDINGS.md);
2. a reader who is going to build on it — start at
   [05-WHY-A-NEW-PROTOCOL.md](05-WHY-A-NEW-PROTOCOL.md).

---

## The short version

Three papers were replicated from scratch, with no code release available for any of
them. All three replications succeeded in the narrow sense — the pipelines run, and the
headline numbers come out close to published. What the replications also produced was a
set of measurements the original papers did not report, and those measurements change how
the published results should be read.

**Five findings, each a measured number in this repository:**

| # | Finding | The number |
|---|---|---|
| **F1** | The text branch is nearly free-standing, and the graph branch is nearly free | BERT alone **0.9730**; five metadata numbers + an MLP **0.9775**; the full 896-dim pipeline **0.9779** |
| **F2** | The "GraphSage embedding" is 128 columns of rank 10, and is never trained | rank **10**, σ₁₀/σ₁₁ = 6.8 × 10⁶; training the layer is worth **−0.0036** |
| **F3** | Rényi's α has no measurable effect on the task it was introduced for | AUC spread **0.0098** across α ∈ [0.2, 5]; seed-to-seed σ **0.0179** |
| **F4** | The paper's own novelty is not significant against its own ablation | DTWRE vs static Rényi: **+0.0035**, 6/10 wins, **p = 0.625** |
| **F5** | Protocol and baseline choices dominate method choices | 5-fold CV vs official split: **9.2 points**. TwiBot-22: **8 of 8** published rows sit below the majority baseline |

None of these is a claim that the papers are wrong. F1–F5 are claims that **the published
comparisons do not isolate the thing they say they isolate**, and that the headroom the
field believes it is working in does not exist on these benchmarks.

## Documents

| File | What it covers |
|---|---|
| [01-FINDINGS.md](01-FINDINGS.md) | The five findings in full, with the tables behind each |
| [02-THE-TEXT-BRANCH.md](02-THE-TEXT-BRANCH.md) | F1 and F2 — what the embeddings and the graph layer actually contribute |
| [03-THE-ALPHA-QUESTION.md](03-THE-ALPHA-QUESTION.md) | F3 and F4 — the Rényi α evidence, and why the paper's own numbers already showed it |
| [04-PROTOCOL-AND-FLOORS.md](04-PROTOCOL-AND-FLOORS.md) | F5 — protocol dominance, majority baselines, and the degenerate graph |
| [05-WHY-A-NEW-PROTOCOL.md](05-WHY-A-NEW-PROTOCOL.md) | How F1–F5 produced the design of `02-ext-research/` |
| [EVIDENCE-INDEX.md](EVIDENCE-INDEX.md) | Every number → its file → the command that regenerates it |

## The papers

| Short name | Paper | Replication package |
|---|---|---|
| **bot-detection** | Deshmukh, *Bot Detection in Social Media using GraphSage and BERT*, SJSU MS Project 1465, 2025 | `botsage/` |
| **entropia** | Tong, Xu, Zhang & Xu, *Public Opinion Propagation Prediction Model Based on Dynamic Time-Weighted Rényi Entropy and Graph Neural Network*, **Entropy** 2025, 27, 516 | `dtwre/` |
| **desinformation** | Lakzaei, Haghir Chehreghani & Bagheri, *Disinformation detection using graph neural networks: a survey*, **Artificial Intelligence Review** 2024, 57:52 | `disinfo/` |

Each has a companion discrepancies document written during replication —
`docs/DISCREPANCIES_BOTSAGE.md`, `docs/DISCREPANCIES.md`, `docs/DISCREPANCIES_SURVEY.md`.
Those record the implementation decisions; **this folder records the findings.**

## Important: the evidence files are not in git

`.gitignore` line 237 ignores `results/` at every level, so every CSV and JSON cited here
is a **local build artefact**, not a committed file. That is deliberate — they are
regenerable — but it means a fresh clone has none of them.

[EVIDENCE-INDEX.md](EVIDENCE-INDEX.md) gives the exact command to regenerate each one.
The three notebooks (`*/replication.ipynb`) *are* committed and carry the narrative
version of most of these tables.

## Reading rules used throughout

Inherited from the replications and applied to every number quoted here:

- **A majority baseline is printed beside every accuracy.** Several published tables in
  this area do not clear theirs.
- **An effect smaller than the measured seed-to-seed σ is not an effect.** That σ is
  **0.0179** for the entropia pipeline and **0.0063** (fold σ) for bot-detection.
- **A protocol label is attached to every number.** 5-fold CV, official split, and
  cross-dataset are three different tasks and the gap between them here is larger than
  the gap between competing published methods.
