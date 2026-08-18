# 06 — Standing rules

Three practices adopted from the `CausalBool` and `deconv-lab` programmes, plus the
rules they carry with them. These are **process** rules; [02-PROTOCOL.md](02-PROTOCOL.md)
covers the experimental design. Where they conflict, these win, because they exist to
catch the case where the design was followed and the answer is still wrong.

Adopted 2026-08-18 at kick-off. Provenance is stated per rule so a reader can go to the
source.

---

## S1 — The datasaurus gate

**Source.** `deconv-lab/.claude/skills/datasaurus/SKILL.md`, ported verbatim to
`.claude/skills/datasaurus/SKILL.md` in this repository. Invocable as the `datasaurus`
skill.

**The law.**

> No claim of agreement or difference enters any document until the objects it is about
> have been rendered in a common coordinate and compared elementwise.

**When it fires.** Before writing *any* number into a document, bitácora, commit message
or handoff. Before any claim that two things agree, differ, match, are exact, sound,
validated or void.

**The four gates**, passed in the artifact, not in your head:

| Gate | Requirement |
|---|---|
| **G1 RENDER** | Draw the object at full length, in natural coordinates, **wider than the feature**. Save to `results/figures/`. Let the picture choose the statistic. Justify and **sweep** every window, threshold, bin and tolerance you invented. A plot of your *statistic* is not a render of your *object*. |
| **G2 ELEMENTWISE** | For any claim of equality, exactness, sufficiency or round-trip: compare the **elements in a common coordinate**, and report the **symmetric difference and where it lives**. Counts and ratios are not agreement. |
| **G3 KNOBS** | Print every fitted parameter and its spread across units. Check it is **interior to its bracket**. Verify the objective is monotone in the knob. Re-derive any inherited statistic from stored primitives — if its warrant is a docstring, it has no warrant. |
| **G4 MECHANISM** | What would this number be under a process known to contain **nothing**? Only the richest control licenses a claim. Check **effective N**. Check whether "independent" constraints are algebraically the same quantity. |

**LOOK BEFORE YOU TEST** (deconv-lab rule 11). Every experiment producing a scalar
summary of a relationship must **first** produce, and commit to `results/figures/`:
the scatter, the binned conditional mean, and the correlation structure among its own
predictors.

**Why this is not boilerplate here.** Three of the eleven risks in
[05-RISKS.md](05-RISKS.md) are datasaurus-shaped by construction, and all three would
produce a *positive* result that survives review:

- **R1** — a Rényi spectrum that encodes posting volume rather than distributional
  shape. G4's "check effective N" and deconv-lab rule 13 ("check every per-unit feature
  against sample size before calling it a property of the unit") are the same control as
  property **P8**, arrived at independently. That convergence is the reason to trust it.
- **R3** — graph BDM encoding node ordering or edge density rather than structure.
- **R6** — BDM 1.0 rediscovering block Shannon entropy and being reported as a
  complexity result.

Every α-curve, every DNA string, every adjacency matrix gets rendered before its number
is quoted. An α-curve figure is a render of a *statistic*; the inter-arrival series it
came from is the object, and both must be drawn (G1b).

## S2 — Bitácora

**Source.** `CausalBool/imp-prices/bitacora/`, `deconv-lab/bitacora/`.

An **append-only, numbered record of record** at `02-ext-research/bitacora/NN_name.md`.

**Rules.**

1. **Append-only.** Protocol changes are **new numbered entries, never silent edits** to
   an earlier one. A superseded entry stays where it is and the new one says what it
   supersedes.
2. **One entry per phase or gate**, written when the phase closes — not batched later.
3. **Every entry states:** date, branch, the gate, pass or fail, the measured number with
   error bars and seed count, what was decided, and **what failed and was not fixed**.
4. **Negatives carry the same weight as positives** and are not re-run in search of a
   better number.
5. **Multiple comparisons are counted**, including the ones that failed. Every threshold,
   encoding and parameter tried is recorded.
6. **Determinism.** Seeds pinned and recorded; two runs of any experiment must agree to
   the digit.

**Relation to the other documents.** `docs/` holds the *design*, frozen at kick-off and
amended only by dated entries. `bitacora/` holds the *history*, append-only.
[04-DECISIONS.md](04-DECISIONS.md) is the index of decisions; the bitácora is the
narrative that produced them. When they disagree, the bitácora is what happened.

**`HANDOFF.md`** at the project root carries current state for session resumption:
branch, SHA, one-line confirmed state, single next action.

## S3 — Didactic, fully replicable, executable notebooks

**Source.** This repository's own convention (`CLAUDE.md`), extended.

Every phase that produces a result also produces a notebook that **teaches** it. The
standard set by `core-concepts.ipynb` and `one-training-step.ipynb`:

1. **Executable end to end from a clean checkout**, with no hidden state. Restart kernel,
   run all, every number reproduces.
2. **Build artefacts** of `scripts/build_*.py`. Edit the builder; changes to the notebook
   are lost on regeneration.
3. **Pinned to a named kernel**, registered with `ipykernel install`.
4. **Didactic, not a log.** It explains the object before the result: what the Rényi
   spectrum *is* on a distribution you can see, what a digital-DNA string *looks like*
   for one real account, what BDM's block decomposition *does* to it — before any AUC
   appears.
5. **Every figure in the notebook is a figure in `results/figures/`**, produced by the
   same code path. No notebook-only plots.
6. **Teaching notebooks are synthetic and self-contained.** No dependency on `data/` or
   `results/`, so they run for a reader who has no corpus access. Their constants may be
   *tuned* to make a phenomenon visible — and where they are, the notebook says so and
   says what breaks if they change.

**Planned set.**

| Notebook | Teaches | Phase |
|---|---|---|
| `renyi-spectrum.ipynb` | Rényi spectrum on distributions you can see; why α resolves the tail; the finite-sample bias, demonstrated | P1 |
| `digital-dna.ipynb` | behavioural encoding, BDM block decomposition, NCD between two real accounts | P4 |
| `bdm2-reuse.ipynb` | BDM 1.0 vs 2.0 on a cyclic-shift construction where reuse is visible by eye | P7 |
| `replication.ipynb` | the full result; the only one that depends on the corpus | P9 |

## S4 — Inherited experimental rules

From `CausalBool/imp-prices/PROTOCOL_causal_timeseries.md` §1, adopted where they apply.

**S4.1 — Every positive needs a null**, and the null is not "random". It is the
**marginal-preserving** null appropriate to the claim: degree-preserving rewiring for
structural claims (already required by [01-METHODS.md](01-METHODS.md) §5), and
**time-shuffling that preserves the inter-arrival marginal while destroying order** for
temporal claims. A result that does not survive its null is reported `NEGATIVE`.

*Applies with force to H1.* A spectrum computed on shuffled inter-arrival times has the
identical marginal and therefore the identical spectrum — so the shuffle null is **not**
informative for H1 and its silence would prove nothing (G4: "before running a null, name
a plausible world in which it separates"). The correct null for the *ordering* component
is applied to the DNA strings and to burstiness, not to the marginal spectrum. Recorded
because getting this backwards is an easy and invisible error.

**S4.2 — Every method needs a positive control.** Any estimator applied to real accounts
is applied unchanged, in the same run, to a **synthetic account whose generator is
known**: a periodic poster (period p), a Poisson poster (rate λ), and a
heavy-tailed/bursty poster (power-law inter-arrivals). If the estimator cannot separate
those three, the result on real data is uninterpretable and is not reported.

This is cheap, and it is the single best defence against R1: a periodic and a Poisson
account with **identical event counts** must be separated by the spectrum, or the
spectrum is measuring volume.

**S4.3 — Negatives are results.** Entered in the bitácora with the same weight as a
positive. Already the charter's H0.

**S4.4 — Re-derive the statistic.** Before building on any per-account score from an
earlier phase, recompute it from stored primitives and check the definition matches the
prose.

---

## Enforcement

- S1 is a **skill** and fires automatically on the trigger vocabulary.
- S2 is enforced at phase close: a phase is not complete without its bitácora entry.
  Gate G9 checks this.
- S3 is enforced at G9.
- S4.2 (positive control) is added to `checks.py` and runs with every experiment, not as
  a separate step.
