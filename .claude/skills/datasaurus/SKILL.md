---
name: datasaurus
description: MANDATORY GATE before any claim that two things agree, differ, match, are exact, are sound, are validated, are confirmed, or are void — and before quoting ANY number in a report, bitácora, commit message or handoff. Fires on model-vs-data comparison, reconstruction or round-trip checks, generator validation, fitted parameters and calibrations, surrogate/null tests, and every summary statistic. Triggers on: exact, identical, matches, reproduces, "the machinery is sound", validated, round trip, sufficiency, gap, deficit, excess, ratio, mean, correlation, effect size, z-score, p-value, null, surrogate, baseline, band, calibration, fit, bisection, confirmed, "no effect", "closes", "stands". If you are about to write a number into a document, this skill applies.
---

# Datasaurus

> **Provenance.** Ported verbatim from the `deconv-lab` programme, 2026-08-18, as a
> standing rule of `02-ext-research/`. The four gates, the law and the tells are
> general and apply here unchanged. The **ledger** below is deconv-lab's — its rows
> are that programme's own casualties, kept because concrete examples transfer better
> than abstractions. As this project accumulates its own, they are appended to
> `02-ext-research/bitacora/`, not here.
>
> Three failure modes in this project are known in advance to be datasaurus-shaped
> and are named in `02-ext-research/docs/05-RISKS.md`: R1 (Rényi spectra encoding
> posting volume rather than shape), R3 (graph BDM encoding node ordering or edge
> density), R6 (BDM 1.0 rediscovering block entropy). Each has a gate that must be
> passed in the artifact.

Thirteen datasets share a mean, a variance and a correlation. One is a dinosaur.
Every summary says *nothing here*.

This is not a caution. In this programme it is **the** failure mode: it has killed
a result in at least six separate campaigns, it survives internal review every
time, and on 2026-07-29 it defeated the previous version of this skill — which
was read at session start and then not applied at the moment of the claim.

**So this is not a checklist to read. It is four gates to pass, at the moment you
make a claim, in writing, in the artifact.**

---

## The law

> **No claim of agreement or difference enters any document until the objects it
> is about have been rendered in a common coordinate and compared elementwise.**

A summary is a projection. A claim about objects that cites only summaries is a
claim about the projection.

---

## GATE 1 — RENDER

Before the number exists.

- Draw the object at **full length**, in its natural coordinates. Save to `figures/`.
- Draw it **wider than the feature**. A baseline measured inside a feature is a
  point on a slope. *(±15 bars made a 400-bar shoulder look like a floor; the "45×
  baseline deficit" was that window, b40 amendment.)*
- Let the picture choose the statistic; do not bring one. If the object is bursty,
  the mean is the **wrong** statistic, not a risky one. *(P28's mean co-pivot was
  55-of-60-names bursts, b32.)*
- Justify every window, threshold, bin and tolerance **out loud**, and sweep it.
  An invented parameter becomes a finding you cannot see. *(`τ = ±2 bars`, b35.)*

**G1b · A PLOT OF YOUR STATISTIC IS NOT A RENDER OF YOUR OBJECT (form K).** Ask of
every figure: *could someone identify the raw thing from this?* Error CDFs, effect-size
curves and scatters of derived quantities all pass "a figure exists" and fail G1.
Draw the price path, the duration distribution, the trajectory — the thing itself.
*(b48/b49: every figure was of a statistic; the objects were drawn only in b50.)*

## GATE 2 — COMPARE THE OBJECTS, NOT THEIR SUMMARIES

This is the gate that failed on 2026-07-29 and it is the one to distrust yourself on.

- **Two matched counts and a matched ratio are not agreement.** For any claim of
  equality, exactness, sufficiency, reconstruction, round trip or "the machinery is
  sound": compare the **elements** — the index sets, the positions, the members —
  and report the **symmetric difference** and *where it lives*.
- **Put both objects in a common coordinate first.** An unaligned comparison
  measures the offset, not the discrepancy. *(187,394 "discrepancies" that were
  162; a shift search over ±3 bars for an offset of hundreds — both on the same
  day, b42 §3.)*
- **Overlay them, rendered identically, in the same units.** If they sit on
  different scales the picture is worthless and so is the eye's verdict.
- Say explicitly **what is discarded** by the representation and prove it cannot
  affect the claim.
- Matching a model's summaries to reality's licenses **nothing**. Compare by
  picture, panel beside panel. *(b38 matched three numbers and declared two clean
  gaps; the panels showed four differences, two of them bugs in my generator, b39.)*

## GATE 3 — LOOK AT YOUR OWN KNOBS

Every fitted quantity is a claim too.

- **Print the fitted value and its distribution across units.** Check it is
  **interior to its bracket**. *(Median fitted scale = the ceiling of its own
  bracket, 15/25 names pinned — while the count-match error read 0.5%. It voided a
  campaign's closing result, b41.)*
- Before bisecting, **verify the objective is strictly monotone in the knob**. A
  saturating objective makes bisection return the bracket, not a fit.
- A knob its target cannot feel is a free parameter with a good disguise.
- Re-derive any inherited statistic from stored primitives, **and the object it is
  computed over**. If its warrant is a docstring, it has no warrant. *(`fold_gains`,
  four bitácoras, b21; DC-pivots-are-the-oracle, b26.)*

## GATE 4 — READ IT AGAINST A MECHANISM

- **What would this number be under a process known to contain nothing?**
  *(`drift` +1.92%/leg was an edge until GBM returned +1.76.)*
- Only the **richest** control licenses a positive claim; beating a poorer one
  means you rediscovered a known law. `gbm` → `shuffle` → `block`, three deep.
- Before running a null, name a plausible world in which it **separates**. If you
  cannot, its silence is not evidence.
- Check the **effective sample size** by decomposing per unit and per day. If 15
  observations carry the statistic, the SD over 8,677 is wrong in a direction you
  cannot bound.
- Check whether your "independent" constraints are **algebraically the same
  quantity**. *(lag-0 IS the overdispersion of S(t); "four marginals" were two.)*

---

## The ledger — what the summary said, and what the object was

| claimed | the summary that hid it | the object |
|---|---|---|
| out-of-sample gain | `fold_gains` +0.417 | difference from a future-containing constant; +0.028 |
| per-stock memory | `lag1_MI` | ≈ 1/N, an estimator bias |
| "2000–2010" | window label | scores 2008-07..2010-06 |
| amplitude comb ρ ≈ 2.5 | exact, universal, replicated | 97% a censoring identity; causal ρ = 1.041 |
| co-pivoting indeterminate | mean co-pivot | bursts, 55 of 60 names; residue 5σ |
| model 8× short on the spike | matched pivot counts | bisection pinned at its ceiling; spike actually 1.08× |
| "the machinery is sound, 80/80" | two counts and a ratio | 82% of pivot indices displaced (before alignment) |

**Every one of these survived internal review.** Five were mine.

---

## The tells that you are already in it

- You can quote the number but have not drawn the shape.
- You are **relieved** by a null result rather than informed by it.
- Your evidence chain contains a link whose warrant is a docstring.
- Your first confirmation arrives right after you changed how the data is assembled.
- You matched three summaries and declared the model adequate.
- You have never looked at the value of a parameter you fitted.
- You are about to write "exact", "identical" or "sound" having compared counts.

---

## Reporting protocol

State the **picture** first and the number second. For any equality claim, give
the **symmetric difference and where it lives**, never a percentage alone. Label
post-hoc analyses post-hoc. Record what you withdrew and why, **in place**, where
the next reader will hit it. Split a pre-committed interpretation so a partial
success cannot read as the whole one.

**And when the claim is constructive — a model, a reconstruction, a program —
the standard is exact equality of the objects, run and seen, not held-out
agreement of their statistics.** That is the whole point of this programme.
