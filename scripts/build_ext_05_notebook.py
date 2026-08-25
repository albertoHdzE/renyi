#!/usr/bin/env python
"""Build `02-ext-research/notebooks/05-from-papers-to-proposal.ipynb`.

Notebook 6 of the didactic series -- a **team tour**: the three base papers in
one picture each, the proposal that grew out of auditing them, the traps the
audit caught, and the final exam — sat scoped, blocked at the data wall. Built for a
live session: every
section is a runnable demonstration on synthetic data, every measured number is
a hard-coded constant with its artefact cited beside it.

Runs on a clean checkout with no `data/` and no `results/`.

    02-ext-research/.venv/bin/python scripts/build_ext_05_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ext_notebook import HOW_TO_RUN, PREAMBLE, Notebook, license_block  # noqa: E402

nb = Notebook("05-from-papers-to-proposal.ipynb")

# ===========================================================================
nb.md(r"""
# 05 — From the papers to our proposal: a team tour

**Sixth of the series**, built for a live team session. Two halves:

1. **The three base papers**, one picture each — what they did, what we took
   from them, and what our audit found when we measured their claims on their
   own data.
2. **Our proposal** — shape vs volume — told as a story with runnable demos:
   the heartbeat idea, the trap that almost sank it, the honesty machine, and
   the final exam, as designed and as far as it could be sat.

Everything runs on **synthetic data computed in front of you**; the handful of
real measured numbers are hard-coded constants, each labelled with the artefact
it comes from (`02-ext-research/EVIDENCE-INDEX.md` maps this programme's
numbers; the three inherited paper-audit numbers in §1 carry their own pointers
to `01-info-propagation/` artefacts inline).

> **Companions.** Notebook 00 covers the three papers in more depth; 01–03 cover
> the replication findings; 04 covers the programme's status as of its midpoint.
> This notebook assumes none of them and points back where depth lives.
""")

# This series' constants are indexed in 02-ext-research/EVIDENCE-INDEX.md, not
# the 01-info-propagation index the shared header names -- repoint it locally
# rather than touching the shared plumbing five published notebooks build from.
nb.md(HOW_TO_RUN.replace(
    "`01-info-propagation/overview/EVIDENCE-INDEX.md`",
    "`02-ext-research/EVIDENCE-INDEX.md`").replace("%(nn)s", "05"))

nb.code(PREAMBLE)

# ===========================================================================
nb.md(r"""
## §0 — Three numbering systems, so nobody gets lost

This project numbers things three ways. They are *not* the same numbers, and
mixing them up is the single most common confusion (it confused us too):

| system | what it counts | where it stands today |
|---|---|---|
| **Charter phases P0–P9** (`docs/03-PHASES.md`) | research phases | P0–P5 **closed** — note P4 (digital DNA / BDM / NCD) closed by its pre-registered kill criterion, honestly. **P6 = the primary claim (H4): executed scoped on 2026-08-25 — UNTESTABLE_PENDING_DATA** (§5). P7/P8 are stretch, outside the current plan; P9 is the write-up |
| **Work packages WP-A…WP-N** (`PLAN-02-ext-research.md`) | execution units of the plan | **all 14 done** (WP-N scoped; WP-M ledger pass ran with it). Bitacora 19 has the outcome |
| **Notebooks 00…05** | teaching artefacts | 00–04 published; **05 = this one** |

So: "notebook 04" is *not* "phase P4". Notebook 04 teaches the programme's
midpoint; phase P4 (Zenil's BDM on behaviour strings) was executed and closed by
its own kill rule — §6 of this notebook tells that story.
""")

# ===========================================================================
nb.md(r"""
## §1 — The three base papers, one picture each

We build on three published works. Everything this project does is either an
audit of their claims on their own data, or an extension they never ran.
""")

nb.md(r"""
### 1.1 Tong et al., *Entropy* 2025 — the Rényi dials

Their paper predicts opinion propagation using a **time-weighted Rényi entropy**
of interaction sequences, fed to a graph neural network. What we borrowed is the
*instrument*: not one entropy number but a **family of dials**.

The dials are easiest to see on a playlist. Two listeners, four songs, 100 plays
each — same size, different character:
""")

nb.code(r"""
counts_specialist = np.array([90, 6, 3, 1])    # one song dominates
counts_eclectic   = np.array([28, 26, 24, 22]) # spread evenly

def dials(counts):
    p = counts / counts.sum()
    h0   = np.log2((p > 0).sum())              # variety: how many songs exist
    h1   = -(p[p > 0] * np.log2(p[p > 0])).sum()  # evenness (Shannon)
    h2   = -np.log2((p ** 2).sum())            # collision: how little one dominates
    hinf = -np.log2(p.max())                   # the favourite's complement
    return h0, h1, h2, hinf

names = ["H0 variety", "H1 evenness", "H2 collision", "Hinf non-dominance"]
tab = pd.DataFrame(
    {n: dials(c) for n, c in (("specialist", counts_specialist),
                              ("eclectic", counts_eclectic))},
    index=names).round(3)
display(tab)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
for a, (lbl, c) in zip(ax, (("specialist", counts_specialist),
                            ("eclectic", counts_eclectic))):
    a.bar([f"s{i}" for i in range(4)], c, color=SERIES[:4])
    a.set_title(f"{lbl}: {c.sum()} plays")
fig.suptitle("Same size, different character — the dials summarise that "
             "difference in 4 numbers", y=1.04, fontsize=11, fontweight="bold")
plt.show()
""")

nb.md(r"""
Read the table: the **specialist** has low variety (H₀ = 2 — they only really
play two songs), low evenness, near-zero collision dial (one song dominates),
and near-zero non-dominance. The **eclectic** listener maxes every dial.
**Same 100 plays — the dials see the character, not the volume.**

That dial family, applied to *gaps between posts* and *kinds of actions*, is
our SPEC_T / SPEC_B / SPEC_X feature families. Tong et al. fed theirs to a
graph network for opinion propagation; we point them at bot behaviour.
""")

nb.md(r"""
### 1.2 Deshmukh, SJSU 2025 — GraphSAGE + BERT for bot detection

This paper owns our **task and our corpora**: bot detection on Cresci-2015
(training) and TwiBot-20 (target). Its recipe: (a) learn a per-account vector
from the **follower graph** with GraphSAGE — each node's embedding = a
learned function of its own features and the *mean of its neighbours'*; (b)
learn a text vector with **BERT**; (c) concatenate and classify.

One mean-aggregation step, by hand, on a 6-node graph:
""")

nb.code(r"""
# 6 accounts; edges = "follows". Features: [followers_z, post_rate_z]
X = np.array([[1., 2.], [0.9, 1.8], [1.1, 2.2],   # a small clique of
              [-1., -1.5], [-0.8, -1.2],           #   human-looking nodes
              [0.2, 0.1]])                          # one borderline node
adj = {0: [1, 2], 1: [0, 2], 2: [0, 1],           # clique 0-1-2
       3: [4],    4: [3, 5], 5: [4]}               # chain 3-4-5

W = np.array([[0.5, -0.3], [0.2, 0.6],      # rows 0-1: the account's own features
              [0.4, 0.1], [-0.2, 0.3]])     # rows 2-3: the neighbours' mean
def sage_step(X, adj, W):
    out = np.zeros_like(X)
    for v in range(len(X)):
        neigh = adj[v]
        mean_neigh = X[neigh].mean(axis=0)
        out[v] = np.tanh(X[v] @ W[:2] + mean_neigh @ W[2:])  # ReLU in the paper
    return out

H1_ = sage_step(X, adj, W)
print("features X (6 x 2):");  print(X)
print("\nafter one GraphSAGE-style mean-aggregation H1 (6 x 2):");  print(
    H1_.round(3))
print("\nnode 0 aggregated from its clique; node 5 from node 4 only --")
print("the graph's shape, not just each account's own features, moved the vectors.")
""")

nb.md(r"""
Two things our audit of this paper found (notebooks 01 and 03 have the full
story): its **GraphSAGE layer was never trained** — the "learned embedding" was
a fixed random projection, rank 10, not 128 — and its reported accuracy
depended more on the **train/test protocol** than on any model (a 9-point gap
between two equally defensible splits). Same task, same data — the lesson was:
*audit the pipeline before believing the score*.
""")

nb.md(r"""
### 1.3 Lakzaei et al., *Artificial Intelligence Review* 2024 — the map

A survey of disinformation detection with graph learning. What we took from it
is **discipline**, not a model: its taxonomy (content vs behaviour vs graph
signals; centralised vs propagation-based) is the checklist we use to say which
kind of signal a feature family actually exploits — and its layer catalogue
(GCN/GAT/GIN/GraphSAGE) is implemented and property-tested in `disinfo/`.

### 1.4 Papers vs our audit, side by side

| paper | what it claims / implies | what our audit measured |
|---|---|---|
| Tong et al. | Rényi entropy of behaviour carries signal beyond simple counts | the *dials* work (`renyiext/checks.py`, 12/12 properties) — but on real accounts the α-grid came out **level-dominated** (bitacora 04), and the tail orders carry the signal only in the text front (WP-I) |
| Deshmukh | GraphSAGE + BERT detect bots at ~97–99 % | the embedding was an **untrained random projection** (rank 10); the score was **protocol-dominated** (9.2 points between splits); volume/metadata confounds everywhere (notebooks 01, 03) |
| Lakzaei et al. | graph learning is the state of the art for disinformation | the survey's own tables contradict one of its prose claims (35 of 36 accuracies above its stated bound) — `disinfo/checks.py` encodes that audit |

**The proposal that follows:** if learned pipelines score high but hide *what*
they see, test whether **simple, plottable distributional statistics** of
behaviour carry the signal — and whether they survive the one test the papers
never ran: **transfer to a different corpus** (H4).
""")

# ===========================================================================
nb.md(r"""
## §2 — Our proposal: shape vs volume

The heartbeat idea. Three accounts, **same number of posts, same average rate**
— only the rhythm differs.
""")

nb.code(r"""
rng = np.random.default_rng(SEED)
H = 24 * 90                                    # 90 days of observation, in hours

gaps_human  = rng.exponential(43, 50)          # random: sometimes soon, sometimes late
gaps_robot  = np.full(50, 43.2)                # metronome: exactly every 43.2 h
gaps_bursty = (rng.pareto(1.2, 50) + 1) * 8    # long silence, then a burst
while gaps_human.sum() > H:                    # keep every account inside the
    gaps_human = gaps_human[:-1]               #   same 90-day observation window
while gaps_bursty.sum() > H:
    gaps_bursty = gaps_bursty[:-1]

fig, axes = plt.subplots(3, 1, figsize=(11, 4.6), sharex=True)
for ax, (nm, g, c) in zip(axes, (("human", gaps_human, C["aqua"]),
                                 ("metronome bot", gaps_robot, C["blue"]),
                                 ("bursty bot", gaps_bursty, C["orange"]))):
    t = np.concatenate([[0], np.cumsum(g)])
    ax.plot(t, np.arange(len(t)), "|", ms=6, color=c)
    ax.set_ylabel("post #")
    ax.set_title(f"{nm}  ({len(g)} posts, mean gap {g.mean():.1f} h)",
                 loc="left", fontsize=10)
axes[-1].set_xlabel("hours since start of observation")
fig.suptitle("Same size, same rate — different heartbeat", y=1.02,
             fontsize=12, fontweight="bold")
fig.tight_layout()
plt.show()
""")

nb.md(r"""
A volume detector sees three identical accounts (50 posts each). Now put the
**dials on the heartbeat**: bin each account's gaps into a histogram and read
the four dials off it.
""")

nb.code(r"""
def gap_dials(gaps, bins):
    h = np.histogram(gaps, bins=bins)[0].astype(float)
    return dials(h)

bins = np.linspace(0, 120, 9)          # gap histogram: 8 bins of 15 h
rows = {}
for nm, g in (("human", gaps_human), ("metronome", gaps_robot),
              ("bursty", gaps_bursty)):
    rows[nm] = gap_dials(g, bins)
display(pd.DataFrame(rows, index=names).round(3))
""")

nb.md(r"""
The **metronome** collapses to a single histogram bin, so its collision and
non-dominance dials hit **zero**: a perfectly predictable heartbeat. The human
spreads across bins. The bursty account spreads *unevenly* — visible in H₀/H₁
but with a fatter low tail.

That is the entire feature idea. SPEC_T applies these dials to gap histograms
(and to hour-of-day histograms); SPEC_B to action alphabets; SPEC_X to word and
character frequencies. Nothing more mysterious than a playlist summary.
""")

nb.md(r"""
### 2.1 The trap: watching-window censoring

Now the danger. Watch the **same bursty account** for 30 days instead of 90.
The account did not change — *our measurement did*. Long silences past the
cutoff are deleted, so every "long gap" statistic moves:
""")

nb.code(r"""
cut = np.cumsum(gaps_bursty) <= 30 * 24
g_short = gaps_bursty[cut]

fig, ax = plt.subplots(figsize=(11, 3.4))
for g, nm, c, ls in ((gaps_human, "human (90d)", C["aqua"], "-"),
                     (gaps_bursty, "bursty (90d)", C["orange"], "-"),
                     (g_short, "SAME bursty, watched 30d", C["red"], "--")):
    h = np.histogram(g, bins=bins)[0]
    ax.plot(np.arange(8), h / h.sum(), ls, marker="o", color=c, label=nm)
ax.set_xticks(np.arange(8))
ax.set_xticklabels([f"{int(b)}-{int(bins[i+1])}h" for i, b in enumerate(bins[:-1])],
                   rotation=30, fontsize=8)
ax.set_ylabel("share of gaps")
ax.set_title("Censoring deletes the long-gap mass — the dashed line is the "
             "SAME account as the orange one", loc="left")
ax.legend(fontsize=9)
plt.show()

p_h, p_f = (gaps_human > 24).mean(), (gaps_bursty > 24).mean()
p_s = (g_short > 24).mean()
print(f"P(gap > 24h):  human {p_h:.2f} | bursty full {p_f:.2f} | "
      f"bursty truncated {p_s:.2f}")
print(f"human-vs-bursty gap:      {abs(p_h - p_f):.2f}")
print(f"window-vs-window gap:     {abs(p_f - p_s):.2f}   <-- same account, "
      f"two windows")
""")

nb.md(r"""
**This is the single most important finding of the programme's first half.**
WP-A ran this trap through the *real* feature pipeline on synthetic accounts
whose only difference was the observation window: the detector separated them
at **AUC 1.00** in the worst cell (every one of the nine cells ≥ 0.978 —
`results/p2c_probe.json`, EVIDENCE-INDEX §WP-A). A difference in *measurement*
had become a difference in *apparent character*.

Every shape claim since carries that warning label until equal-window controls
are run — which is exactly what WP-F then did (§4).
""")

# ===========================================================================
nb.md(r"""
## §3 — The honesty machine: information, or just more columns?

Every comparison in this project is family (say, 12 spectrum features) vs floor
(say, 1 volume feature). But 12 features beat 1 feature partly by *being 12*.
WP-E's control: **pad the floor with random-noise columns until dimensions
match.** If the family still wins, the win is information. If it collapses to
the padded floor, the "win" was dimensionality.

Toy version, runnable:
""")

nb.code(r"""
n = 400
y_toy = (rng.random(n) < 0.5).astype(int)             # balanced labels
x_signal = rng.normal(0, 1, n) + y_toy * 1.2          # a genuinely useful dial
x_weak   = rng.normal(0, 1, n) + y_toy * 0.25         # a weak floor dial
noise    = rng.normal(0, 1, (n, 11))                  # pure-noise dials (11: see below)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def auc5(X):
    return cross_val_score(HistGradientBoostingClassifier(max_iter=60),
                           X, y_toy, cv=5, scoring="roc_auc").mean()

floor   = x_weak[:, None]
family  = np.hstack([x_weak[:, None], x_signal[:, None],
                     noise[:, :10]])                                # 12 dims
floor_p = np.hstack([x_weak[:, None], noise])                       # 12 dims too:
                                                                    # 1 real dial +
                                                                    # k = 12-1 noise
a_floor, a_family, a_padded = auc5(floor), auc5(family), auc5(floor_p)
print(f"floor (1 dial):            {a_floor:.4f}")
print(f"family (12 dials):         {a_family:.4f}   vs floor: {a_family - a_floor:+.4f}")
print(f"padded floor (12 dims):    {a_padded:.4f}   vs family: {a_family - a_padded:+.4f}")
print("\nthe family's edge over the PADDED floor is the honest, "
      "dimension-matched edge")
""")

nb.md(r"""
On the real data this machine produced the programme's most sobering numbers:

- **The metadata ceiling.** Cresci-2015's four profile fields alone score
  **AUC 0.9972** (`results/p3h_behaviour.json`, EVIDENCE-INDEX §WP-H) — and
  every behavioural family, matched to the same dimensionality, sits *below*
  that. Locally, metadata wins. (WP-B showed the target corpus is different:
  volume there scores only 0.6073 — which is why the transfer exam matters.)
- **The length trap.** WP-I's character spectrum beat its Shannon slice by
  +0.3332 at matched dimensions — but once *message length* was given its own
  feature, the residual was **+0.0166, below the 0.02 floor**: mostly
  length-mediated (`bitacora/15` — our own correction entry).
""")

# ===========================================================================
nb.md(r"""
## §4 — What survived, what collapsed: the equal-window story

WP-F re-ran the headline comparison with **equal observation windows** (every
account truncated to its own first K days). The registered clauses:

- clause (i): spectrum+volume vs volume — does rhythm beat *counting*?
- clause (ii): spectrum vs Shannon — does the *dial family* beat its own
  simplest dial?

Hard-coded from `results/p2c_truncation.json` (EVIDENCE-INDEX §WP-F):
""")

nb.code(r"""
Ks = [7, 14, 30, 90]
clause_i   = [-0.0004, 0.0122, 0.0423, 0.0681]   # CS - COUNT, seed-mean
clause_ii  = [ 0.0036, -0.0049, 0.0048, 0.0084]  # SPEC_T - SHAN
clause_i_m = [ 0.0130, 0.0210, 0.0488, 0.0778]   # vs COUNT+NOISE(12)

fig, ax = plt.subplots(figsize=(10.5, 4))
ax.plot(Ks, clause_i, "o-", color=C["blue"], label="clause (i): CS - COUNT")
ax.plot(Ks, clause_i_m, "o--", color=C["blue"], alpha=0.6,
        label="clause (i) dim-matched")
ax.plot(Ks, clause_ii, "s-", color=C["orange"],
        label="clause (ii): SPEC_T - SHAN")
ax.axhline(0.02, color=C["red"], ls="--", lw=1.5, label="claim bar 0.02")
ax.axhline(0.0, color=C["grey"], lw=0.8)
ax.set_xlabel("equal observation window K (days)");  ax.set_ylabel("AUC delta")
ax.set_title("Rhythm-beats-counting survives equal windows and grows;\n"
             "the Shannon-slice win collapses to zero", loc="left")
ax.legend(fontsize=9)
plt.show()
""")

nb.md(r"""
Reading: **clause (i) survives everything** — at 30-day windows the
volume-anchored rhythm edge is +0.0423 (10/10 seeds, p = 0.002; matched
+0.0488) and it *grows* with window. **Clause (ii) collapses** (+0.0048 at
30 days; dim-matched −0.0003 → formally downgraded): the part of the early
win that came from bots and humans being *watched for different lengths of
time* is gone once windows are equal.

And the referee-friendly winner of §WP-J: the margin over the burstiness floor
is carried by the simplest statistic on the table — **P(gap > 1 day)** — whose
dim-matched margin (+0.0555) beats the full spectrum's (+0.0361) at every
window, while the fitted "tail index" alone is *worse than volume* (−0.12).
Fancy lost to simple; the pre-registered rule said record that, and we did.
""")

# ===========================================================================
nb.md(r"""
## §5 — The final exam: H4, transfer to a different corpus

**H4 (charter):** transferring Cresci-2015 → TwiBot-20, the AUC degradation of
our features must be smaller than metadata's by more than **0.05**:
`Δ_metadata − Δ_ours > 0.05` over ≥ 10 seeds (`docs/00-CHARTER.md` §3). That
number is the bar because it is larger than the seed-to-seed noise we actually
measured — the config-sensitivity σ of the registered floor verdicts spans
0.0006–0.0094 (bitacora 10) — and larger than any single artefact we have
measured; it asks for a *robustness gap*, not a lucky flip.

The mechanics, on a toy shift you can see:
""")

nb.code(r"""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

n = 600
y_src = (rng.random(n) < 0.5).astype(int)
sig   = rng.normal(0, 1, n) + y_src * 1.2
X_src = np.column_stack([sig, rng.normal(0, 1, n)])       # dim 2 = nuisance
y_tgt = y_src.copy()
sig_t  = rng.normal(0, 1, n) + y_tgt * 1.2
X_tgt  = np.column_stack([sig_t, rng.normal(0, 3, n)])    # nuisance scale x3

# D6-style: 10 draws of 80% train -> within on held-out, transfer on full target
w, t = [], []
for r in range(10):
    idx = np.random.default_rng(r).permutation(n)
    m = int(0.8 * n)
    tr, ho = idx[:m], idx[m:]
    mod = HistGradientBoostingClassifier(max_iter=60, random_state=r).fit(
        X_src[tr], y_src[tr])
    w.append(roc_auc_score(y_src[ho], mod.predict_proba(X_src[ho])[:, 1]))
    t.append(roc_auc_score(y_tgt, mod.predict_proba(X_tgt)[:, 1]))
delta = np.mean(w) - np.mean(t)
print(f"within  (source, held-out): {np.mean(w):.4f}")
print(f"transfer (target era/scale): {np.mean(t):.4f}")
print(f"degradation delta = {delta:+.4f}   (H4 compares THIS across families)")

# sanity null: random halves of the SAME population must show delta ~ 0
w2, t2 = [], []
for r in range(10):
    idx = np.random.default_rng(100 + r).permutation(n)
    m = int(0.8 * n)
    tr, te = idx[:m], idx[m:]
    mod = HistGradientBoostingClassifier(max_iter=60, random_state=r).fit(
        X_src[tr], y_src[tr])
    w2.append(roc_auc_score(y_src[te], mod.predict_proba(X_src[te])[:, 1]))
    t2.append(roc_auc_score(y_src[te], mod.predict_proba(X_src[te])[:, 1]))
print(f"pseudo-transfer null (same population): "
      f"{np.mean(w2) - np.mean(t2):+.4f}  -- ~0 by construction")
""")

nb.md(r"""
Two design lessons already paid for, carried into the exam:

- **Composition** (WP-K): the two sides of any split must be compared with
  their class balances reported — an AUC delta between a 51 %-bot and a 72 %-bot
  population is not a pure shift measurement.
- **R8 alignment** (WP-B): TwiBot-20's metadata arrives z-scored with
  *different* fields; META transfers only with an explicit alignment, reported
  with and without.

And the honest context: on the *target* corpus, metadata is no cheat code —
volume scores 0.6073 and the best single field 0.7414 (`results/p6b_tb20_
preflight.json`, EVIDENCE-INDEX §WP-B) — so the exam genuinely separates
"robust features" from "corpus luck".

### 5.1 The exam as sat: scoped, and blocked at the data wall

WP-N ran on 2026-08-25 — and hit a finding bigger than any score. Four
doors were checked for a modality-bearing TwiBot-20; none has one. The HF
mirror's tweet file is **dense pooled BERT embeddings** (229,580 × 768,
every row non-zero — no timestamps, no post types, no text); raw TwiBot-20
is gated by its authors; TwiBot-22's open `user.json` is profiles-only;
the conversion archive holds seven corpora but no twibot-20.
`results/p6n_transfer.json`, `scoping` block, has the audit elementwise.

So the exam was sat on the one commensurable arm pair, with every registered
control (`bitacora/19`, corrections in `bitacora/20`):

| quantity (fit Cresci → test TwiBot-20, D6, hgb) | value |
|---|---|
| **Δ_META** (within 0.9974 → transfer AUC 0.6831, CI [0.6764, 0.6938]) | **+0.3143 ± 0.0075** |
| Δ_META under marginal recalibration | +0.3335 |
| calibration on target: accuracy / majority / macro-F1 / TPR@1%FPR | 0.5585 / 0.5572 / 0.3612 / 0.0137 |
| R8 verdict (naive vs recalibrated, both heads) | effect under both — not an artefact |
| G4 sanity null (pseudo-transfer, same population) | max abs Δ 0.0088 |

Two readings matter more than any single number:

- **Total calibration collapse.** The transferred model puts nearly every
  target user at predict_proba ≈ 1 (accuracy beats majority by 0.0013;
  macro-F1 0.36). What survives is *ranking* only — AUC 0.68 — so "0.9974 →
  0.6831" is a decay to a score pile, not to a mediocre-but-working model.
- **Volume vs metadata is variant-dependent** (`bitacora/20`). On LR,
  volume degrades more than META in 20/20 draws under both alignment
  variants. On HGB the naive arm says the opposite (+0.0624) — but that
  arm's collapsed column scale breaks tree binning by our own §3(b)
  mechanism, and under recalibration the ordering reverses (−0.0124,
  3/20 draws). The honest claim is LR-sourced; the figure shows both.

And the headline stands: Cresci's near-ceiling metadata collapses by
**0.31** across corpora (it transfers *within* Cresci at Δ +0.0005, WP-K) —
had any of our families been computable target-side, H4's > 0.05 bar would
have been cleared by Δ_META alone. Instead the claim's status is recorded
honestly: **UNTESTABLE_PENDING_DATA** — the Δ_ours side has no data to
compute. The path back is written down: obtain raw TwiBot-20 from its
authors, or pre-register an amended target before analysing it.
""")

# ===========================================================================
nb.md(r"""
## §6 — Two post-mortems worth teaching

### 6.1 Tong's twist we did not use yet: time-weighting

Tong et al.'s entropy is **time-weighted**: recent events count more than old
ones. On the dials, that is a *weighted* histogram. One runnable intuition:
""")

nb.code(r"""
# A listener: heavy on song 0 early, heavy on song 1 late (100 plays)
seq = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
def weighted_dials(seq, lam=0.98):
    n = len(seq)
    w = lam ** (n - 1 - np.arange(n))          # newest play has weight 1
    p = np.array([w[seq == k].sum() for k in range(4)])
    p = p / p.sum()
    return p
p_plain = np.bincount(seq.astype(int), minlength=4) / len(seq)
p_rec   = weighted_dials(seq)

fig, ax = plt.subplots(figsize=(8.5, 3.2))
x4 = np.arange(4)
ax.bar(x4 - 0.2, p_plain, 0.4, color=C["grey"], label="plain shares")
ax.bar(x4 + 0.2, p_rec, 0.4, color=C["blue"], label="recency-weighted (lambda=0.98)")
ax.set_xticks(x4); ax.set_xticklabels(["song0", "song1", "song2", "song3"])
ax.set_ylabel("probability mass"); ax.legend(fontsize=9)
ax.set_title("Same listener, two summaries: the weighted one says 'who are they NOW'",
             loc="left")
plt.show()
print(f"H1 plain {-(p_plain[p_plain>0]*np.log2(p_plain[p_plain>0])).sum():.3f}"
      f"  -> weighted {-(p_rec[p_rec>0]*np.log2(p_rec[p_rec>0])).sum():.3f}"
      "   (the weighted listener looks far more decided)")
""")

nb.md(r"""
Our registered features use *unweighted* dials (the plan froze that choice);
the weighted form is Tong's contribution and an obvious sensitivity arm for
WP-N's write-up — one knob, one sentence in the limitations.
""")

nb.md(r"""
### 6.2 The BDM post-mortem: killed by a rule, before it could lie

The plan's phase P4 (charter H3) was Zenil's **BDM/NCD**: encode each account's
behaviour as a string (the "digital DNA", e.g. `ORTQRTT…`), then ask whether
bot strings are *more compressible* — algorithmic fingerprints of coordination.

WP-L executed the whole pre-flight honestly:

- **Viability**: NCD behaves monotonically in shared-prefix fraction for all
  three compressors at all three tested lengths — the tool works
  (`results/p4l_ait.json`, D7 table).
- **The population fails, not the tool**: BDM needs strings of ≥ ~100 symbols,
  i.e. accounts with ≥ 100 posts. Cresci-2015 has **162 such bots out of
  3,351 (4.8 %)**; the pre-registered rule demanded ≥ max(200, 335). The
  survivor pool would be **91 % human** — any "coordination" result would be
  uninterpretable. **H3 was cancelled by its own rule**, survivor table
  published (`bitacora/18`).

**What resurrection on TwiBot-20 requires** (flagged, out of current scope):
(1) measure per-account tweet counts in that release — WP-B showed the
packaged tensors do not carry them; (2) verify ≥ max(200, 10 % of bots)
accounts clear ~100 tweets; (3) resolve the CTM-table source (the PyPI `acss`
package is an unrelated Python-2 project — recorded as finding, `bitacora/18`);
(4) only then re-run the enrichment estimand. Until those hold, *no BDM number
on this question is trustworthy* — that is the lesson, and it generalises.
""")

# ===========================================================================
nb.md(r"""
## §7 — Where the programme stands

| work package | one-line outcome |
|---|---|
| WP-A censoring probe | the trap is real: window truncation alone reaches AUC 1.00 — all shape claims labelled |
| WP-B TwiBot-20 preflight | the target corpus is winnable: volume weak there (0.61); H4 runs as chartered |
| WP-C evidence repair | every quoted number now has a producer script + artefact |
| WP-D estimator defects | mass-conserving histograms; the check caught the fix's own leak |
| WP-E evaluation layer | the honesty machine (per-fold TPR, noise-padded floors, config sigma) |
| WP-F equal windows | clause (i) survives and grows; clause (ii) collapsed → downgraded |
| WP-G circadian | sign reversal explained (volume suppression) — dials kept |
| WP-H behaviour front | mechanism real (bot = one action type, one target) but under the metadata ceiling locally |
| WP-I text front | char spectrum 0.9889 — beats its Shannon slice at last; largely length-mediated |
| WP-J tail statistics | **the winning statistic is P(gap > 1 day)**; mechanism narrative updated |
| WP-K era shift | nothing SPEC degrades 2011→2013; composition caveat recorded |
| WP-L DNA/BDM | H3 cancelled by kill criterion (162/335 bots) — negative result, published |
| **WP-N transfer (scoped)** | **the exam hit the data wall: target artefact has no timestamps/text — H4 UNTESTABLE_PENDING_DATA; metadata side measured (+0.3143)** |

Every number above is traceable: `02-ext-research/EVIDENCE-INDEX.md` maps each
to its `results/*.json` and the command that regenerates it. The narrative of
each work package — including what failed — is in `02-ext-research/bitacora/`.
""")

nb.md(license_block(
    licensed=[
        "Using the dial-family intuition (H₀/H₁/H₂/H∞) to explain every SPEC_* feature family in this repository",
        "Reusing the censoring, dimension-matching and transfer demonstrations to teach why each control exists",
        "Citing the measured constants in team discussions **with their artefact**, as done here",
    ],
    not_licensed=[
        "Quoting these constants without their artefact (every one lives in `02-ext-research/EVIDENCE-INDEX.md`)",
        "Reading the toy demonstrations as measurements — they are illustrations; the measurements live in `results/*.json`",
        "Treating H3 (BDM coordination) as 'failed' — it was **cancelled by a pre-registered population rule** before any result existed; retrying it needs a heavier corpus and the four steps in §6.2",
        "Calling H4 'failed' or 'passed' — it was **never sat**: the target artefact lacks the modalities (§5.1). What exists is the measured metadata side and a written path to run the exam",
        "Presenting §5's toy transfer as the real experiment — the real one is `scripts/run_p6n_transfer.py` → `results/p6n_transfer.json`",
    ],
))

nb.write()
