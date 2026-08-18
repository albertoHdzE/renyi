# Bitácora 02 — Amendment to H1 and D3

**Date:** 2026-08-18
**Branch:** `main`
**Supersedes:** the H1 statement in `docs/00-CHARTER.md` §3 and decision D3 in
`docs/04-DECISIONS.md`, both as written at kick-off (bitácora 00).
**Status:** decided. Option (a) of [01_p0_data_layer.md](01_p0_data_layer.md) §4.

Per standing rule S2 this is a new numbered entry, not an edit to bitácora 00 or 01.
Those entries stand as written; this one says what replaced them.

---

## 1. What forced the amendment

P0 measured, on Cresci-2015:

- bots post a median of **23** tweets, humans **834**; 196 bots have zero decodable events
- **event count alone scores AUC 0.939**
- D3's fixed-n subsampling retains ≥80% of both classes only at **n ≤ 12**, where
  `H₀ ≤ log₂ 12 = 3.58` bits and every order is bias-dominated
- at n ≥ 64 it retains **9.7%** of bots, and retains them non-randomly

The original H1 compared the spectrum against Shannon-alone at fixed n. That design is
unexecutable here. The conflict is a property of the corpus, not of the estimator.

## 2. The amendment

**H1 (superseded).** `AUC(H_α vector) − AUC(H₁ alone) > 0.02` at fixed n = 128.

**H1 (in force from 2026-08-18).** Variable n, per account. The spectrum must clear
**both** floors:

```
(i)   AUC(SPEC_T ∪ {count}) − AUC(count alone)  > 0.02
(ii)  AUC(SPEC_T)           − AUC(H₁ alone)     > 0.02
```

each by paired Wilcoxon over ≥10 seeds at p < 0.05. The 0.02 threshold is unchanged and
is still the measured seed-to-seed σ.

**Why this is the better hypothesis, not merely the feasible one.** The charter's own
motivation (§1c) is that metadata *magnitude* is brittle and *shape* should transfer.
Event count is magnitude. Requiring the spectrum to beat it directly is the sharper form
of the question the project was always asking. The original fixed-n design tested shape
against Shannon *at controlled volume*; the amended design tests shape against volume
itself, which is a stronger claim.

**Why it is also more dangerous.** At AUC 0.939, count is a brutal floor, and any feature
correlated with it will look excellent while adding nothing. The amendment therefore
buys a better question at the cost of needing a harder control.

## 3. Consequent changes

**D3 (superseded).** Fixed-n subsampling with `n_events = 128`.

**D3′ (in force).** Variable n. `Config.n_events` is retained only for the robustness
arm. Three controls replace what fixed-n was providing:

1. **Count is a mandatory covariate** in every model and in every reported table. No
   spectrum number is quoted without the count-alone baseline beside it.
2. **Partial correlation.** Report `ρ(H_α, label | count)` alongside `ρ(H_α, label)` for
   every α. If the partial collapses, the feature is volume.
3. **Positive control at matched n** (S4.2), which is what actually rescues the design:
   synthetic periodic, Poisson and heavy-tailed accounts generated with **identical event
   counts**. If the spectrum cannot separate those three at equal n, it is measuring
   volume and nothing else — and this is checkable without touching the corpus.

**Property P8 (superseded).** `|ρ(H_α, event count)| < 0.1` after subsampling.

**P8′ (in force).** Under variable n that bound cannot and should not hold — count is now
an explicit covariate rather than something to be nulled out. P8′ is instead:

> the spectrum separates the three synthetic generators at **matched n** (S4.2), and
> `ρ(H_α, label | count)` is non-zero for at least one α.

The first half is the real gate: it is a property of the estimator, testable in
isolation, and it fails loudly if the estimator is a volume proxy.

**Retained unchanged.** Option (b), the volume-matched fixed-n arm, is *not* run as a
primary result but stays available as a robustness check if H1′ passes and a reviewer
asks whether the effect survives at controlled volume.

## 4. What this does not change

H2, H3, H4 are untouched. H4 in particular — the primary claim — is unaffected, since
cross-dataset degradation is measured per feature family and `count` is simply one more
family whose degradation is reported.

The kill criteria in `docs/03-PHASES.md` are unchanged, except that G1's hard stop now
refers to P8′ rather than P8.
