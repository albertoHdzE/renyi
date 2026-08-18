# 01 — Methods

Mathematical specification of every estimator, with the property each must satisfy.
Per this repository's convention there is no test suite: correctness is asserted against
properties the mathematics must satisfy. Every property below becomes a function in
`renyiext/checks.py`.

Per `CLAUDE.md`, entropy estimators must state their **base**, **α**, and
**bias-correction** choice in the code, not leave them implicit. All logarithms in this
project are **base 2** (bits), which differs from `dtwre/entropy.py` (nats) — a
deliberate change, because §5 mixes entropies with CTM values, which are in bits.

---

## 1. Rényi entropy and the spectrum

For a probability vector `p` on a finite support and order `α > 0`, `α ≠ 1`:

```
H_α(p) = 1/(1-α) · log₂ ( Σᵢ pᵢ^α )
```

with the limiting cases

```
H₀(p)  = log₂ |{i : pᵢ > 0}|          support richness
H₁(p)  = −Σ pᵢ log₂ pᵢ                 Shannon (the α → 1 limit)
H₂(p)  = −log₂ Σ pᵢ²                   collision entropy
H_∞(p) = −log₂ maxᵢ pᵢ                 min-entropy
```

**The spectrum** is the vector `S(p) = [H₀, H_½, H₁, H₂, H₄, H_∞]` ∈ ℝ⁶.

### Why the spectrum rather than a tuned α

Selecting a single α that maximises class separation is one-dimensional feature
selection with a multiple-comparisons hazard, and it discards information. The vector is
strictly more informative, requires no tuning, and is interpretable: **the shape of the
α-curve is the signature**, not its argmax.

The spectrum also has standing as a complexity-science object. `τ(q) = (q−1)·H_q` is the
mass-exponent function; its Legendre transform is the multifractal singularity spectrum
`f(α)`. Computing `{H_α}` over a range of q *is* multifractal analysis of the underlying
measure — an established methodology (Grassberger–Procaccia; Halsey et al.) rather than
an ad-hoc parameter sweep.

Two endpoints recover known measures, which is a useful sanity anchor:
- `H₂` on a word-frequency distribution is, up to normalisation, **Yule's K** (1944),
  the classical stylometric repeat-rate index: `K = 10⁴(Σᵢ i²Vᵢ − N)/N²` and
  `Σᵢ i²Vᵢ/N² = Σ p²= 2^{−H₂}`.
- `H₀` is log support size, i.e. type count — the oldest lexical-richness measure there
  is.

The novelty is therefore in the **joint use** of the spectrum and in the fronts it is
applied to, not in any single α. This must be stated plainly in the write-up.

### Properties to check

| # | Property | Assertion |
|---|---|---|
| P1 | Uniform limit | `H_α(uniform on n) = log₂ n` for every α, to 1e-10 |
| P2 | Shannon at α=1 | `H_{1±1e-9}` matches `−Σp log₂ p` to 1e-8 |
| P3 | Monotonicity | `H_α` is non-increasing in α for every p |
| P4 | Bounds | `H_∞ ≤ H₂ ≤ H₁ ≤ H₀ ≤ log₂ n` |
| P5 | Degenerate | point mass ⇒ `H_α = 0` for every α |
| P6 | Permutation invariance | `H_α(p) = H_α(σp)` for any permutation σ |

## 2. Finite-sample bias — the central estimation problem

**This is the most dangerous part of the project and it gets its own section.**

Plug-in Rényi estimators are biased, and *the bias depends on both n and α*. `H₀` is the
worst case: observed support size systematically underestimates true support and grows
monotonically with sample size. Bots and humans differ in tweet volume. Therefore an
uncorrected spectrum computed on all of an account's events would **partly measure
activity volume**, and a "result" could be nothing but the volume difference re-expressed.

**Control (mandatory, applies to every front).** Fixed-n subsampling with bootstrap:

1. Fix `n_events` per account (default **n = 128**; sensitivity at 64 and 256).
2. Accounts with fewer than `n_events` are excluded, and the exclusion count is reported
   per class — this is itself a bias and must be visible.
3. Draw `B = 100` subsamples of size `n_events` without replacement.
4. Report the **mean spectrum** over the B draws, and carry the bootstrap SD as an
   uncertainty estimate.

Because n is identical for every account, the bias is a constant offset per α and cannot
differ between classes. This trades statistical efficiency for validity, which is the
correct trade here.

**Reported alongside, never instead:** the uncorrected full-sample spectrum, so the size
of the volume confound is measurable rather than assumed away.

| # | Property | Assertion |
|---|---|---|
| P7 | Bias direction | plug-in `H₀` on subsamples of a known distribution increases with n |
| P8 | Bias neutrality | after fixed-n subsampling, `H_α` is uncorrelated with an account's total event count (\|ρ\| < 0.1) |

## 3. The four fronts and their distributions

Each front supplies one or more distributions per account; each yields one 6-vector.

| Front | Distribution | Support | Source |
|---|---|---|---|
| **T — temporal** | inter-arrival times Δt between consecutive posts, log-binned | ~20 bins | snowflake-decoded IDs (§6) |
| **T2 — circadian** | hour-of-day of posting | 24 | same |
| **B — behavioural** | post type (original/reply/retweet/quote) | 4 | edge/text structure |
| **B2 — interaction** | mention-target identity | variable | `@` extraction |
| **X — text** | word-frequency; character-frequency | variable; ~30 | raw text, **uncleaned** |
| **N — network** | ego-network degree distribution | variable | `edge_index` |

**Front X requires bypassing `botsage`'s cleaning.** `botsage/text.py` lowercases and
strips URLs, mentions, hashtags, non-ASCII and all non-alphabetic characters — which
removes most of the orthographic irregularity the spectrum would measure. Front X uses
raw text. Recorded as decision D4 in [04-DECISIONS.md](04-DECISIONS.md).

**Front N is blocked on Cresci-2015** and runs on TwiBot-20 only: 8,550 of 6,994,858
Cresci edges join two users that have features, and 64% of labelled users have an
all-zero neighbour mean (`docs/DISCREPANCIES_BOTSAGE.md` §3).

## 4. Digital DNA and algorithmic-information measures

### 4.1 Encoding

Following Cresci et al., *Social Fingerprinting* (IEEE TDSC 2017), each account's
timeline is encoded as a string over a small behavioural alphabet, in chronological
order.

**Alphabet size is 4**, not 3 or 5: `{original, reply, retweet, quote}`. This is forced
by tooling — the CTM lookup tables (`acss.data`, `pybdm`) cover alphabets
**2, 4, 5, 6, 9**; there is no alphabet-3 table. Decision D5.

A second encoding, **temporal DNA**, quantises inter-arrival times into 4 bins by
training-set quartiles, giving a string over the same alphabet size that describes
*rhythm* rather than *action type*.

### 4.2 BDM 1.0

```
BDM1(X) = Σᵢ ( K̂(xᵢ) + log₂ mᵢ )
```

over distinct blocks `xᵢ` with multiplicities `mᵢ`, `K̂` from the CTM tables. Block
length 12 (the table maximum), non-overlapping, with the trailing partial block
discarded and the discard length recorded.

**Known limitation, stated up front.** For objects much longer than the block size the
`log₂ mᵢ` term dominates and BDM 1.0 converges to the block Shannon entropy of the block
distribution. Sakabe et al. (arXiv:2606.23471) §2.4 state this explicitly as the
motivation for BDM 2.0. BDM 1.0 is therefore included as a **baseline to be beaten**,
not as the method — and `block entropy` is a mandatory floor for it (§6 of
[00-CHARTER.md](00-CHARTER.md)).

### 4.3 NCD — the coordination statistic available today

```
NCD(x,y) = ( C(xy) − min(C(x),C(y)) ) / max(C(x),C(y))
```

with `C` a real compressor (`zlib` at level 9; `bz2` as a robustness check). NCD is a
normalised estimate of algorithmic *mutual* information — the quantity that measures
**shared generators**, which is what coordination means. This is the operative measure
for H3 and it needs no new implementation.

**Group cohesion** for a candidate group G: the mean pairwise NCD within G, compared
against a null of size-matched random account sets.

### 4.4 BDM 2.0 — reuse gain

Sakabe, Abrahão, Hernández-Orozco, Gudwin & Zenil (arXiv:2606.23471, 22 Jun 2026):

```
BDM2(X) = min_P min_{∅≠S⊆P} [ Σ_{pᵢ∈S} K̂(pᵢ)
                            + Σ_{pⱼ∈P∖S} min_{pᵢ∈S} min{ K̂(pⱼ|pᵢ), K̂(xⱼ|xᵢ) } ]
        + Σᵢ log₂ mᵢ
```

The aggregation is no longer pure multiplicity counting: blocks that share algorithmic
information are described conditionally. `BDM2 ≤ BDM1 + O(1)` (their Theorem 1), strict
when the reuse gain exceeds representation overhead; the gain is bounded by algorithmic
mutual information (their Theorem 2). Exact selection of `S` is **NP-hard**.

**Implementation tiers.** The paper is explicit (§2.4) that `K̂` "can be instantiated by
different approximations of Kolmogorov complexity and not necessarily by CTM", and Eq. 1
accepts observation-space conditionals `K̂(xⱼ|xᵢ)`. This licenses a hybrid:

| Tier | Unconditional `K̂(x)` | Conditional `K̂(y\|x)` | Cost | Phase |
|---|---|---|---|---|
| **1 — hybrid** | CTM tables | `C(xy) − C(x)` (compression) | days | P7 |
| **2 — transformation library** | CTM tables | `log₂\|T\| + ε` if some `t ∈ T` maps `x→y`, else Tier 1 | days | P7 |
| **3 — conditional CTM** | CTM tables | partial enumeration of the machine space with `x` on the input tape | months | P8, optional |

`T = {identity, reverse, cyclic shift, symbol permutation, complement}`. **Cyclic shift
is the load-bearing member**: a fleet running one script with staggered start times
produces behavioural strings related by exactly that transformation. Tier 2 therefore
encodes the domain hypothesis directly into the estimator, which is the main reason to
prefer it over Tier 1.

Selection of `S` is greedy (largest marginal reuse gain first), with the greedy/exact
gap measured on small synthetic instances where exhaustive search is tractable.

| # | Property | Assertion |
|---|---|---|
| P9 | BDM 1.0 vs entropy | on a long periodic string, BDM 1.0 ≪ its own block-entropy value |
| P10 | BDM 2.0 bound | `BDM2(X) ≤ BDM1(X) + C_rep` on every test object (their Theorem 1) |
| P11 | Reuse detection | for `X` built from a block and its cyclic shifts, Tier-2 `BDM2 < BDM1` |
| P12 | NCD identity | `NCD(x,x) ≈ 0`; `NCD` of two independent random strings ≈ 1 |
| P13 | Graph BDM instability | BDM of a fixed graph varies under node relabelling — measured, not assumed (§5) |

## 5. Graph BDM and the permutation problem

BDM over an adjacency matrix is **not permutation-invariant**: relabelling nodes changes
the block decomposition and changes the value. BDM 2.0 does not address this. This
repository's existing `checks.py` files assert permutation *equivariance* as a
correctness property of every layer, so a naive graph-BDM feature would fail the house
standard.

**Control.** (i) canonical ordering by degree, ties broken by BFS from the highest-degree
node; (ii) report the SD of BDM over `K = 50` random relabellings as part of the feature
vector. If that SD exceeds the between-class effect, the feature is noise and is dropped
— that is the decision rule, fixed in advance.

**Second control — density.** On sparse adjacency matrices most blocks are all-zero, so
BDM is dominated by the multiplicity of the zero block and becomes a proxy for edge
density. Every graph-BDM number is therefore reported as a z-score against a
**degree-preserving configuration-model null** (double-edge-swap rewiring, 100
replicates). Raw graph BDM is never reported alone.

## 6. Timestamp reconstruction

Cresci-2015 tweet nodes in the TwiBot-22-schema conversion carry only `id` and `text` —
`botsage/text.py` documents that the corpus has no timestamps. It does: Twitter
snowflake IDs encode milliseconds in the top 41 bits.

```
timestamp_ms = (int(tweet_id) >> 22) + 1288834974657
```

Verified on a 200,000-tweet random sample: **200,000/200,000 decoded, 0 failures, 0
pre-2010 artefacts**, range 2010-11-04 → 2013-06-03, median 2012-11-28 — exactly the
Cresci-2015 collection window. Corpus totals: **5,301 users, 2,827,757 tweets** (~533
per user).

| # | Property | Assertion |
|---|---|---|
| P14 | Snowflake sanity | all decoded timestamps lie in [2010-11-01, 2013-07-01]; monotone in ID |
| P15 | Ordering | decoded order agrees with numeric ID order |

This unblocks the temporal front, which is the strongest hypothesis (H1), on the primary
dataset. Recorded as decision D1.

## 7. Classifier and readout

Held fixed so that the feature family is the only thing varying:

- **Primary:** gradient-boosted trees (`HistGradientBoostingClassifier`), which handle
  heterogeneous feature scales and are insensitive to monotone transforms — relevant
  because spectra and counts live on different scales.
- **Secondary:** L2 logistic regression on standardised features, for interpretability
  of per-α coefficients.
- **Not used:** linear SVM. `docs/DISCREPANCIES_BOTSAGE.md` §4 established the linear
  kernel as the binding constraint in the source paper, costing 4.2 points.

Scalers are fitted **on training folds only** (`disinfo` lesson: fitting on all rows
leaks and is the common silent error).
