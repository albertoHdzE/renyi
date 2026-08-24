# Plan — remediation and completion of `02-ext-research`

**Version:** 1.1 · **Created:** 2026-08-24 · **Revised:** 2026-08-24 (v1.1 — see §10)
**Origin:** adversarial review of the project conducted 2026-08-24 (findings
**A1–D5**, summarised in Appendix R), then peer-reviewed; the review's five
amendments are incorporated and marked **[rev1]** where they changed structure.
**Authoritative companions:** `CLAUDE.md`, `02-ext-research/docs/00…06`,
`02-ext-research/bitacora/`, `02-ext-research/HANDOFF.md`.
**Scope:** repairs the evidence-chain and estimator defects found in the review,
runs the censoring probe and the TwiBot-20 preflight **first** (they can
invalidate or re-frame everything downstream) **[rev1]**, then drives P3
(behavioural/text fronts), P4 (digital DNA / NCD), the temporal-split
generalisation axis, and **executes P6 / H4 itself** — the primary claim —
under whichever framing the preflight selects **[rev1]**. P7–P9 remain governed
by `docs/03-PHASES.md` and are outside this plan except where touched
explicitly (WP-M).

---

## 0. How to use this plan (read first, every session)

You are an agent resuming work. Do this, in order:

1. Read `CLAUDE.md` (repo root), this file, `02-ext-research/HANDOFF.md`, and the
   **latest two** bitácora entries.
2. Load the `datasaurus` skill (`.claude/skills/datasaurus/SKILL.md`). It fires on
   every number you ever write down. This plan restates its four gates as
   **G1 RENDER, G2 ELEMENTWISE, G3 KNOBS, G4 MECHANISM**; where a work package
   says "datasaurus:", those gates are acceptance criteria, not advice.
3. Pick the **lowest-letter work package whose status is `pending`** and whose
   dependencies are `done`. Never start two packages in one session unless they
   are marked independent.
4. Claim it: set its Status cell to `in_progress (<date>, <your session id>)`
   **in this file, in the same commit as your first code change**.
5. Execute the tasks exactly. Where this plan states a **Decision rule**, that
   rule is binding — it was pre-registered (v1.0 on 2026-08-24, v1.1 revisions
   likewise dated) before the affected data was seen, precisely so you would not
   have to improvise. Do not relitigate it; if you believe it is wrong, stop,
   record the argument in a new bitácora entry, and mark the WP
   `blocked (decision challenged)`.
6. Tick every box in the WP's **Acceptance** section. A box you cannot tick is a
   blocked WP, not a finished one.
7. On completion: update the Status board, append a bitácora entry —
   **the next free number**, titled `NN_WP<letter>_<slug>.md` — update
   `HANDOFF.md`, add rows to `02-ext-research/EVIDENCE-INDEX.md` for every number
   you produced (create the file per WP-C if it does not exist yet), commit once
   with message prefixed `WP-<letter>:`. Set Status to `done (<date>)`.

### Ambiguity protocol (binding)

When something is not covered by this plan or by a Decision rule:

1. Follow the WP's Decision rules;
2. else `02-ext-research/docs/` (charter > protocol > methods > phases);
3. else `docs/06-STANDING-RULES.md`;
4. else take the **most conservative pre-registration-preserving option**
   (the one that makes a positive result *harder*, not easier), record it as a
   dated sub-item in the WP's bitácora entry, and add it to §10 (changelog) of
   this plan in the same commit.

Never resolve ambiguity by silently editing an earlier bitácora entry or by
changing a floor/threshold. Floors change only via a numbered amendment entry
(bitácora 02 is the precedent).

### Vocabulary

| Word | Meaning here |
|---|---|
| `done` | every Acceptance box ticked, artefacts committed, docs updated |
| `blocked` | cannot proceed; reason recorded in Status cell and bitácora |
| `cancelled` | a kill criterion fired (§7); negative result written up instead |
| floor | a baseline a family must beat before any claim (protocol §3 + D10) |

---

## 1. One-page context

`02-ext-research/` asks whether the **Rényi spectrum** of an account's
behavioural distributions and **algorithmic-information measures** over its
behavioural strings detect bots beyond activity volume, and whether they
generalise across datasets/eras better than purchasable metadata (H4 = primary).

State at plan creation: P0 failed its retention gate and forced the H1 amendment
(bitácora 02); P1 passed 8/8 property checks; P2 supported H1's two clauses
(+0.0367 over count, +0.0380 over Shannon; 10/10 seeds, p = 0.0020) but did **not**
clear the burstiness floor (+0.019 < 0.02) and its α-curves are near-parallel, so
the claimed tail-resolution mechanism is undemonstrated. The review found: four
quoted P2 qualification numbers with no committed code/artefact behind them (A1),
one factual mischaracterisation in bitacora 04 **that also propagated into
notebook 04 §6.1** (A3, **[rev1]**), a silently substituted floor statistic (B1),
an interval-truncation defect in `log_bin_counts` (C1) — reviewer-measured at
63 of 182,085 bot intervals (0.035 %) and 13 of 2,576,329 human intervals above
hi = 400 d, i.e. 2.21 % of bots vs 0.68 % of humans losing ≥ 1 interval: genuine,
class-dependent in the direction that flatters the result, and tiny
(**[rev1]** magnitudes to be confirmed by WP-D's render, not assumed) — a
dimensionality confound in clause-(ii)-style comparisons (C2), and two
uncontrolled upstream confounds — observation-window truncation and the Twitter
API timeline cap (D1, D2) — that sit under every "shape" claim made so far.

Why the order of §5 is what it is **[rev1]**: WP-A (pure synthetic, no corpus,
no cache, no dependencies) is the cheapest item in the plan and the only one
that can invalidate the P2 headline; WP-B is descriptive statistics on data
already on disk and decides whether H4 or H4′ is the claim P3/P4 features must
serve. Everything else follows.

---

## 2. Standing rules condensed (full text in sources)

| # | Rule | Source |
|---|---|---|
| S1 | Datasaurus gates G1–G4 before any number enters any document | `docs/06-STANDING-RULES.md` §S1 |
| S2 | Bitácora append-only; one entry per WP; negatives recorded; multiple comparisons counted; two runs agree to the digit | ibid. §S2 |
| S3 | Notebooks are build artefacts of `scripts/build_ext_*_notebook.py`; never edit `.ipynb` | ibid. §S3, `CLAUDE.md` |
| S4 | Positive controls at matched n; marginal-preserving nulls where informative; re-derive inherited statistics | ibid. §S4 |
| P-rules | Entropy code states base (2), α, bias stance in docstrings; scalers/vocab fitted on training folds only; every batch script ships `--quiet` and `--seeds` | `CLAUDE.md` |
| F | Every reported comparison: ≥10 seeds, paired Wilcoxon, majority baseline printed beside accuracy, TPR@1%FPR beside AUC, protocol labelled | `docs/02-PROTOCOL.md` §§5–6 |
| D10 | BURST (B, M, CV) is a first-class floor on every front; the level-removed SHAPE arm is a standard arm | `bitacora/04` §8 |
| NEW | **Every bitácora table that quotes a measurement carries the `results/*.json` path it came from.** Numbers living only in prose are forbidden. | this plan (review A1/A2) |
| NEW | **Notebook constants are derived from `results/*.json` at build time wherever an artefact exists**; inherited prose claims in builders are audited against the corrected record (review A3's propagation path) | this plan [rev1] |
| NEW | `OMP_NUM_THREADS=1` (and OPENBLAS/MKL/NUMEXPR/VECLIB) pinned inside every new script before numpy import — 133× penalty measured (`bitacora/04` §6) | this plan |

---

## 3. Environment and canonical commands

```bash
E=02-ext-research/.venv/bin/python        # the only env for this plan

# property checks (must pass before and after every estimator change):
$E -c "import sys; sys.path.insert(0,'.'); from renyiext.checks import run_all; run_all()"

# event cache (VALID through WP-K: binning/feature changes do NOT invalidate it;
# it would only be invalidated by changing the DECODE — see events.py docstring:
# that requires renaming to cresci_events_d10.npz)
#   data/processed/ext/cresci_events_d9.npz

# rebuild + execute a notebook after changing its builder:
$E scripts/build_ext_04_notebook.py
$E -m jupyter nbconvert --to notebook --execute --inplace \
     02-ext-research/notebooks/04-the-new-programme.ipynb    # ≈17 s for all five

# corpora for WP-B / WP-N (downloaded by the botsage step; verify before fetching):
ls data/raw/bot/            # expect cresci-2015/, twibot-20/, twibot-22/
```

Seeds are always `range(42, 42+n_seeds)`, default `n_seeds=10`. Classifier:
`HistGradientBoostingClassifier(max_iter=200, early_stopping=False,
random_state=seed)`; LR secondary `make_pipeline(StandardScaler(),
LogisticRegression(max_iter=5000))`. CV: `StratifiedKFold(5, shuffle=True,
random_state=seed)`.

---

## 4. Global acceptance standards (apply to EVERY work package)

A WP is `done` only if **all** of the following hold, in addition to its own boxes:

- **G-correctness** — every new mathematical estimator gains a property check in
  `renyiext/checks.py` (next free numbers **P16, P17, …**; P1–P15 are taken),
  and `run_all()` passes before and after the change.
- **G-integration** — the producing script runs end-to-end from its declared
  inputs (cache, `data/raw/bot/`, or pure synthetic) with `--quiet`; its JSON
  artefact is **byte-identical across two consecutive runs** (determinism, S2.6);
  any downstream notebook builder regenerates and executes without error.
- **G-datasaurus** — the WP's datasaurus items (listed per WP) have their
  artefacts in `02-ext-research/results/figures/`, and no number entered any
  document before its gate artefact existed.
- **G-ledger** — bitácora entry written (with "what failed and was not fixed"),
  `HANDOFF.md` updated, `EVIDENCE-INDEX.md` rows added, one commit
  `WP-<letter>:`. Bitácora numbers are assigned sequentially **at completion
  time**, never pre-assigned to a WP letter.
- **G-no-regression** — previously passing property checks still pass; previous
  headline numbers that change are listed old→new in the bitácora entry with the
  reason (an unexplained change to a committed number is a bug until proven
  otherwise).

Gate-mapping cheat-sheet: G1 ⇒ figure exists showing the **object** (not only the
statistic); G2 ⇒ elementwise comparison in a common coordinate, symmetric
difference reported; G3 ⇒ every invented constant printed + swept + interior to
bracket; G4 ⇒ null/counterfactual with a named separating world.

---

## 5. Work packages

Dependency graph in §6; effort: S ≤ ½ day, M ≈ 1 day, L ≥ 2 days.
Letters encode priority: execute in alphabetical order among ready WPs.

### Status board

| WP | Name | Depends | Effort | Status |
|---|---|---|---|---|
| A | Censoring probe — can truncation alone fake P2? | – | S | `done (2026-08-24) — trigger FIRED, bitacora 05/06` |
| B | TwiBot-20 preflight — volume landscape, H4 vs H4′ | – | S | `done (2026-08-24) — H4 as chartered, bitacora 07` |
| C | Evidence-chain repair + notebook-04 audit | – | M | `done (2026-08-24) — all five quoted numbers reproduced exactly, fidelity gate 0.0; bitacora 08` |
| D | Estimator defects: overflow cell, P8′ grid, hygiene | C | M | `pending` |
| E | Evaluation layer: per-fold TPR + dim-matched floors | D | M | `pending` |
| F | Truncation controls on corpus: equal-window + API cap | A, D, E | M | `pending` |
| G | Circadian adjudication | D | M | `pending` |
| H | Behavioural front SPEC_B (P3a) | E, G | M | `pending` |
| I | Text front SPEC_X (P3b) | E | M | `pending` |
| J | Tail-statistic arms (TAIL, SURV) | E | M | `pending` |
| K | Within-corpus temporal-split axis | D, E | M | `pending` |
| L | Digital DNA + NCD: pre-flight and P4 core (H3) | D, E | L | `pending` |
| M | Ledger, findings document, hand-off quality (recurring) | – | S | `pending` |
| N | **Protocol C execution — H4, the primary claim** | B, E, F, G, H, I | L | `pending` |

---

### WP-A — Censoring probe (review D1; plan v1.0 WP-D part 2) **[rev1: moved first]**
· S · depends: none (pure synthetic)

**Goal.** Measure, before anything else is built, how much classifier-separable
signal observation-window truncation alone produces through the exact P2 feature
pipeline. This is the only result in the plan that can invalidate the P2
headline; it is first because it costs almost nothing.

**Tasks.**

1. Script `scripts/run_p2c_probe.py` (no cache, no `data/`): per Definition D5 —
   for each generator g ∈ {periodic(jitter 0.5), poisson, bursty(tail 1.2)}:
   draw 120 accounts/class from g's **own** renewal process on a 900-day
   timeline; class A observed in full; class B truncated to its first
   W ∈ {30, 90, 400} days; identical feature pipeline; binary AUC(B vs A),
   10 seeds, standard classifier. Three metrics per cell: `SPEC_T` alone,
   `COUNT+SPEC_T` (the P2-headline analogue — **this cell owns the trigger**),
   and `SHAPE` (level removed).
2. Output `results/p2c_probe.json`; figures: `figures/p2c_probe_grid.png`
   (3×3 AUC heat/bar grid with the 0.85 trigger line drawn) and one G1 render of
   example class-A/class-B rasters for the worst cell.
3. **Amendment trigger (pre-registered v1.0, unchanged):** if any
   `COUNT+SPEC_T` cell reaches **AUC ≥ 0.85**, append an amendment bitácora
   entry stating that SPEC_T's edge may be substantially censoring; WP-F/H/N
   framing must cite it, and the "shape" reading of P2 is formally downgraded.
   Not a kill — a reinterpretation with teeth.
4. Pipeline-fidelity note recorded in the JSON `config_echo`: the probe runs on
   the pipeline as it exists when executed. **Re-run obligation:** after WP-D
   changes `log_bin_counts`, re-run this script and confirm the trigger verdict
   is unchanged (max cell delta ≤ 0.01); record both runs in EVIDENCE-INDEX.

**Datasaurus:** G4 — the probe *is* the mechanism null (named world: identical
generator, identical rate, only the window differs); G1 — rasters are the
objects, drawn beside the statistic; G3 — W swept and printed, all nine cells
reported regardless of outcome.

**Acceptance.**

- [ ] `run_p2c_probe.py --quiet` runs with no `data/` and no cache; JSON
      byte-identical twice; 3 generators × 3 windows × 3 metrics × 10 seeds.
- [ ] Trigger evaluated and stated explicitly (fired / not fired), with the
      owning cell's number.
- [ ] Both figures exist; worst-cell rasters rendered.
- [ ] Amendment entry written iff triggered; HANDOFF updated either way;
      EVIDENCE-INDEX rows added.

---

### WP-B — TwiBot-20 preflight (HANDOFF open item 3; v1.0 WP-K) **[rev1: moved second]**
· S · depends: none (data already on disk)

**Goal.** Measure TwiBot-20's volume landscape before any front is built, and
pre-register H4's exact estimator under both outcomes. Descriptive statistics
only; no model fitting beyond the single volume-AUC.

**Tasks.**

1. From `data/raw/bot/twibot-20/`: per-class distributions of statuses_count,
   followers, account age; **AUC(volume alone) on TwiBot-20** (10 seeds, CV);
   retention curve at candidate tweet-count cutoffs; tweet-text availability
   per labelled user. Output `results/p6_preflight_tb20.json`; renders:
   per-class count histograms for Cresci-2015 and TwiBot-20 overlaid on one
   common x-axis (log scale) — the G2 common coordinate.
2. **Pre-registered branch (fires here, before P3/P4 are built):**
   - If AUC(volume) on TwiBot-20 **≥ 0.85**: H4 executes in amended form
     **H4′** (mirror of H1′, bitacora-02 precedent): degradation measured
     incrementally over each corpus's own volume feature — i.e. every family
     runs as `family ∪ {count}` on both sides, deltas compared on those arms.
   - Else original **H4** stands as written in the charter.
   Either way the choice is recorded as an amendment bitácora entry in this WP,
   **before any P6 transfer number exists**.
3. Schema alignment dry-run per D8/R8: the four overlapping META fields mapped
   (followers, following, statuses, age), fifth dropped from both sides,
   z-score offset quantified (source scaler applied to the pre-z-scored target)
   — numbers in the JSON, no classifier yet.

**Datasaurus:** G1 — histograms; G2 — both corpora on one axis; G3 — cutoffs
swept and printed; G4 — the volume-AUC is itself the mechanism number that
selects the H4 framing.

**Acceptance.**

- [ ] Volume AUC + retention + availability + histograms present; byte-stable.
- [ ] Branch decision recorded as amendment prior to any transfer run.
- [ ] Alignment offsets tabulated; dropped-field reported.
- [ ] HANDOFF item 3 closed; EVIDENCE-INDEX rows; bitácora appended.

---

### WP-C — Evidence-chain repair + notebook audit (review A1, A2, A3 **[rev1]**) · M ·
depends: none

**Goal.** Every P2 qualification number becomes reproducible from a committed
script and JSON; the record's factual error is corrected without violating
append-only history; **and its propagation into notebook 04 is found and fixed**
(the didactic series currently prints the falsifiable "declines monotonically"
claim).

**Tasks.**

1. Create `scripts/run_p2b_decomposition.py`: loads the event cache, builds
   blocks at the headline config (`n_bins=24, hi=400d, min_events=5`),
   evaluates the four missing arms over seeds 42–51 with the standard classifier:
   `SPEC_T_MINUS_H0`, `SHAPE` (per-order H_α − H₁(ia) for the ia six and
   − H₁(cd) for the cd six), `COUNT+SHAPE`, `COUNT+BURST+SPEC_T`. Paired
   Wilcoxon for: SHAPE vs COUNT, SPEC_T vs SPEC_T_MINUS_H0,
   COUNT+BURST+SPEC_T vs COUNT+BURST. Writes `results/p2b_decomposition.json`
   (schema conventions of `p2_temporal.json`).
2. Expected values (from bitacora 04 §4 — the point is to confirm or refute):
   SHAPE ≈ 0.9596, COUNT+SHAPE ≈ 0.9673, Δ(COUNT+SHAPE−COUNT) ≈ +0.0273,
   Δ(SPEC_T−MINUS_H0) ≈ +0.0104, Δ(CB+S vs CB) ≈ +0.0192. Any |Δ| > 0.005 is a
   **finding**: record old→new in the bitácora and correct `HANDOFF.md`; tune
   nothing.
3. **Notebook-04 audit [rev1].** (a) Replace the hard-coded DECOMP constants
   (`build_ext_04_notebook.py` ~lines 915–940) with values loaded at build time
   from `results/p2b_decomposition.json` when present, else emitted behind a
   loud `STALE — regenerate results` banner. (b) Grep every
   `scripts/build_ext_*.py` for inherited prose claims that a JSON can falsify —
   known instance: notebook 04 §6.1 prints "DECLINES MONOTONICALLY in n_bins
   (0.0581 at 8 → 0.0248 at 48)", contradicted by `p2_temporal.json`
   (peaks 0.0626 @ 12 bins) — and rewrite such passages from the artefact.
   (c) Regenerate + execute all five notebooks.
4. Append a bitácora entry: (a) corrects bitacora 04 §2's monotonicity claim
   **by correction entry, not edit** ("peaks at n_bins=12 (0.0626) then declines
   to 0.0248 at 48"); (b) records the p2b confirmation/refutation table;
   (c) notes bitacora 01 §2.1's quantile table was independently recomputed from
   `cresci_events_d9.npz` on 2026-08-24 and matched elementwise (bot
   16/23/38/63 mean 55; human 237/834/2550/3193 mean 1322; 196 bot / 4 human
   zero-event); (d) lists every builder prose claim changed.
5. Create `02-ext-research/EVIDENCE-INDEX.md`, mirroring
   `01-info-propagation/overview/EVIDENCE-INDEX.md`: one row per quoted number →
   source JSON + regenerating command. Seed it with every number currently
   quoted in `HANDOFF.md`, README phase board, `docs/04-DECISIONS.md` P0/P2
   entries, and notebook 04. Maintained by every WP thereafter.

**Do NOT** edit bitacora 00–04 or any committed `.ipynb`.

**Datasaurus:** G2 — quoted-vs-measured settled elementwise (symmetric
difference listed), never by eyeballing means.

**Acceptance.**

- [ ] `run_p2b_decomposition.py --quiet` runs from cache; JSON byte-identical
      twice; four arms × 10 seeds.
- [ ] Quoted-vs-measured table in the bitácora; every |Δ| ≤ 0.005 **or**
      flagged as finding with HANDOFF corrected.
- [ ] Notebook-04 DECOMP constants JSON-derived; the monotonicity passage
      rewritten from the artefact; `grep -ri "monoton" scripts/build_ext_*.py`
      output clean or justified; all five notebooks regenerate + execute.
- [ ] `EVIDENCE-INDEX.md` covers all numbers in HANDOFF/README/DECISIONS/notebooks.
- [ ] Correction bitácora appended; `git diff` proves no edits to 00–04.

---

### WP-D — Estimator defects (review C1, C4, C5) · M · depends: WP-C

**Goal.** Fix the interval-truncation defect, the P8′ grid inconsistency, and
package hygiene, without invalidating the event cache.

**Tasks.**

1. `renyiext/spectrum.py::log_bin_counts`: append an explicit **overflow cell**
   so the return becomes `[zero_cell] + h + [overflow_cell]` (length `n_bins+2`,
   mirroring the zero-cell convention). Values above `hi` land in the overflow
   cell — mass conserved, resolution untouched. Docstring states base-2
   neutrality, both sentinel-cell conventions, and that corpus-wide `hi` pinning
   is mandatory outside sensitivity renders. Callers pick the change up via the
   signature.
2. New property **P16 (mass conservation)** in `renyiext/checks.py`: for random
   positive samples spanning decades around `hi`,
   `log_bin_counts(x, n_bins, lo, hi).sum() == x.size` exactly and the overflow
   cell equals `(x > hi).sum()`. Add to `CHECKS`.
3. `checks.py::check_spectrum_separates_generators_at_matched_n` (P8′): pin the
   grid to the corpus-wide default (`hi = 400·MS_PER_DAY`) instead of the
   account-dependent default banned by `features.py:71-75`. Report the gain as
   it lands; tune nothing.
4. Hygiene: delete the dead stub at `scripts/run_p0_events.py:74-76`;
   `renyiext/__init__.py` exports the real module list;
   `config.INFERRED_PARAMETERS`: stale `n_events` rationale replaced with the
   D3′ wording. Record the CRESCI_WINDOW-vs-METHODS P14 widening (review B2) in
   this WP's bitácora so the ledger owns what the code comment disclosed.
5. Re-run P2 end-to-end (`run_p2_temporal.py --quiet`) so `p2_temporal.json`,
   sweep, and `figures/p2_g1_spectrum.png` reflect the overflow cell; list every
   changed number old→new. Render per-class mass above `hi=400d`
   (`figures/p2d_overflow_mass.png`) — expected magnitude per the reviewer's
   measurement: 0.035 % of bot / 0.0005 % of human intervals; **confirm, don't
   copy**. **Decision rule (pre-registered v1.0):** if clause (i) or (ii) falls
   below floor after the fix, that is the reported outcome — floors untouched,
   H1 status honestly re-stated in HANDOFF (§7 trigger 3 fires).
6. Re-run the WP-A probe (its own acceptance item) and confirm the trigger
   verdict is unchanged.

**Datasaurus:** G3 — overflow cell printed, interior-checked; G1 — mass render;
G2 — P16 is an elementwise identity; G-no-regression on P1–P8′.

**Acceptance.**

- [ ] P1–P16 pass; `run_all()` output archived in the bitácora.
- [ ] `p2_temporal.json` regenerated; changed numbers tabulated old→new;
      floors untouched regardless of outcome.
- [ ] Overflow-mass figure exists; per-class shares measured (and compared to
      the reviewer's 0.035 % / 0.0005 %).
- [ ] Probe re-run verdict unchanged (≤ 0.01 cell delta).
- [ ] Hygiene greps clean; CRESCI_WINDOW item ledgered.

---

### WP-E — Evaluation layer: per-fold TPR + dimensionality-matched floors
(review C2, C3, D4) · M · depends: WP-D

**Goal.** One shared, correctly-calibrated evaluation module; a standard control
that separates "family information" from "extra dimensions" — **with failure
semantics named in advance**, per the project's own standard **[rev1]**.

**Tasks.**

1. Create `renyiext/evaluate.py` (layering: features → models → **evaluate**):
   generalise from `run_p2_temporal.py` — `tpr_at_fpr`, `eval_arm`, `run_arms`,
   `paired`, `partial_corr`, plus `noise_padding(X_floor, k, seed)` and
   `dim_matched_arm(...)` per Definition D2. `tpr_at_fpr` additionally computed
   **per-fold** (`tpr01_foldmean`), pooled kept for continuity.
2. Refactor `run_p2_temporal.py` onto it. **Regression gate:** AUC/macro-F1/
   accuracy arrays unchanged to 4 decimals; only the new TPR field may differ.
   Other drift = bug; find before committing.
3. Standard arm extension (all future fronts): wherever
   `dim(family) > dim(floor)`, add `<FLOOR>+NOISE(k=d_family−d_floor)` per D2.
   **Interpretation rules (pre-registered here, [rev1] — replacing v1.0's
   softer wording):**
   - If `Δ(family − floor−noise) ≤ 0`: the family measured **dimensionality,
     not information**; any clause resting on it is recorded as confounded, and
     the corresponding hypothesis clause's support is **downgraded in HANDOFF
     and FINDINGS** — a finding, not an uncertainty note.
   - If `0 < Δ < 0.02`: the family's edge is real-but-subfloor under matched
     dimensions; the clause is recorded as **not claimable** at the registered
     floor.
   - Only `Δ ≥ 0.02` (+ significance) supports the clause.
   Apply retroactively as a report: clause (ii) as `SPEC_T vs SHAN+NOISE(10)`;
   burstiness verdicts as `COUNT+SPEC_T vs COUNT+BURST+NOISE(9)`. P2's gate
   verdict stands as gated; these rows attach the matched-dimension reading to
   it, with the downgrade executed if it fires.
4. Uncertainty reporting (review D4): every floor verdict reports
   `sigma_config` (population SD of the delta across the published sweep)
   beside seed SD. The 0.02 floor itself is untouched (amendments only).

**Datasaurus:** G3 — noise knob (k, RNG salt) printed in JSON config echo;
G2 — regression gate elementwise on JSON arrays.

**Acceptance.**

- [ ] `evaluate.py` exists; regression gate holds to 1e-4 (max abs diff stated).
- [ ] `tpr01_foldmean` present for every arm.
- [ ] Retroactive dim-matched rows computed; whichever interpretation rule fires
      is **executed** (downgrade recorded) — box cannot be ticked with "noted
      only".
- [ ] `sigma_config` beside every touched floor verdict.

---

### WP-F — Truncation controls on corpus: equal-window + API cap
(review D1, D2; v1.0 WP-D parts 1 & 3) · M · depends: WP-A, WP-D, WP-E

**Goal.** Attach the probe's answer to the real corpus: how much of SPEC_T's
edge survives when observation windows are equalised, and how much of COUNT's
edge is the API cap.

**Tasks.**

1. `renyiext/features.py::temporal_blocks_windowed(ev, window_days, ...)`: keep
   per account only events within `window_days` of the account's **own first
   event**, then the standard block pipeline (dim-matched arms per WP-E).
   Exclusions per class reported.
2. Run the full arm set (COUNT/BURST/SHAN/SPEC_T/SHAPE/COUNT+SHAPE, dim-matched)
   at `window_days` ∈ {7, 14, **30**, 90} — **headline K = 30 pre-registered
   v1.0**; others are sensitivity. Output extends
   `results/p2c_truncation.json`; figure `figures/p2f_equal_window.png`
   (clause deltas vs K, floors as lines, probe ceiling from WP-A drawn as a band
   for comparison).
3. **API-cap render and sensitivity.** Events-per-account histogram with the cap
   annotated (`figures/p2f_api_cap.png`); cap defined as the mode of the human
   upper tail (printed, interior-checked); `frac_humans_at_cap` reported;
   COUNT-alone and COUNT+SPEC_T recomputed excluding capped humans
   (sensitivity rows in the same JSON).
4. Bitácora: split interpretation — §what-survived / §what-shrank /
§relation-to-probe-ceiling — and consequences for reading P2.

**Datasaurus:** G1 — windowed rasters for example accounts beside the statistic
curves; G2 — capped vs uncapped compared on the same sample; G3 — K swept,
headline pre-declared, all values reported; G4 — WP-A's probe is the null this
arm is read against.

**Acceptance.**

- [ ] `p2c_truncation.json` contains 4 K values × full dim-matched arm set +
      cap sensitivity; byte-identical twice.
- [ ] Both figures exist; probe ceiling drawn on the equal-window figure.
- [ ] Split interpretation written; HANDOFF/EVIDENCE-INDEX/bitácora updated.

---

### WP-G — Circadian adjudication (HANDOFF open item 1; review D5) · M ·
depends: WP-D · **blocks WP-H**

**Goal.** Explain or retire the circadian sign reversal before hour-of-day
features enter any downstream front.

**Tasks.**

1. `scripts/run_p3a_circadian.py`: bot vs human hour-of-day histograms side by
   side **in UTC and local offsets +1 h and +2 h** (CET/CEST both plausible for
   an Italian corpus — the offset is a knob, G3), overall and within
   **count-caliper strata** (Definition D4). Per-order raw and given-count
   partial correlations per stratum; sign-reversal survival test under matching.
   Output `results/p3a_circadian.json`; figures
   `figures/p3g_circadian_{utc,local}.png`.
2. **Decision rule (pre-registered v1.0):**
   - Stable matched-count difference agreeing with the conditioned coefficient
     sign ⇒ circadian orders **kept**, mechanism stated in one sentence
     (suppression effect explained), cited.
   - Reversal disappears under matching or flips with timezone offset ⇒ the
     circadian six are **dropped from SPEC_B** (which becomes 6-vector
     behavioural alphabet + 6-vector mention-targets); P2's SPEC_T circadian
     half keeps a caveat.
   - Ambiguous (matched-class TV < 0.05 at every offset) ⇒ drop from SPEC_B,
     mark exploratory-only.
   Whichever branch fires is recorded with its figures; no third path.
3. Close HANDOFF open item 1.

**Datasaurus:** G1 — histograms are the objects; G2 — strata enumerated with
sizes; G3 — offset swept; G4 — matched-count strata remove the volume world.

**Acceptance.**

- [ ] Renders for UTC and both offsets, overall + per-stratum.
- [ ] Branch named and executed per the decision rule.
- [ ] SPEC_B dimension decision recorded; HANDOFF open item closed.

---

### WP-H — Behavioural front SPEC_B (phase P3a) · M · depends: WP-E, WP-G ·
gate **G3 = H2** (directional)

**Goal.** Spectrum on the behavioural alphabet and on mention-targets, tested
against the amended floor set.

**Tasks.**

1. **Post-type collapse (decision D11, pre-registered v1.0):** for *spectra*,
   map `quote → original` (alphabet {original, reply, retweet}) — the 11.6 %
   quote share from the trailing-t.co proxy is not credible for 2011–13, and
   the CTM constraint (D5) binds only P4's DNA strings, not spectra. The
   4-symbol encoding remains for DNA. Share-after-collapse reported. Cache stays
   valid (feature-level collapse; decode untouched).
2. Create `renyiext/behaviour.py`: per-account post-type vector → probabilities
   → 6-vector spectrum. Mention-target specification (pre-registered v1.0): all
   `@\w+` tokens; leading token excluded on `reply`-classified tweets
   (replied-to party ≠ audience); self-mentions excluded; empty ⇒ account
   excluded from the mention block, exclusion counted per class. Blocks:
   `SPEC_B_ALPHA` (6), `SPEC_B_MENTION` (6).
3. Arms/floors: majority, COUNT, META-lite (followers/following/statuses/age,
   read-only fields as used by `botsage`), SHAN slices, COUNT+SHAN incumbent
   composite, dim-matched noise arms (BURST is temporal-specific; documented
   substitution). Gates: H2 directional — bots higher collision probability
   (lower H₂) on the alphabet, lower H₀ on mention-targets; both p < 0.05,
   consistent signs; else `docs/03-PHASES.md` P3 decision rule (return to
   P1/P8′ reasoning; inconsistency = the volume/length detector firing).
4. Output `results/p3h_behaviour.json`; figures: per-class α-curves for both
   blocks with the objects rendered above (example post-type sequence, example
   mention list — G1b).

**Datasaurus:** G1 — sequences rendered; G2 — exclusions elementwise per class;
G3 — regex rules printed with capture rates; G4 — marginal-preserving shuffle
null on post-type sequences run to prove expected silence (spectrum is
order-invariant; separating world named first, S4.1 logic, stated in JSON).

**Acceptance.**

- [ ] JSON complete (arms, floors, dim-matched, exclusions); byte-stable.
- [ ] H2 directional verdicts with wins/p/Δ/sigma_config; majority baseline and
      TPR@1%FPR beside everything.
- [ ] Multiple-comparisons census (every regex/rule tried, kept or not).
- [ ] HANDOFF/EVIDENCE-INDEX/bitácora updated.

---

### WP-I — Text front SPEC_X (phase P3b) · M · depends: WP-E · exploratory
under H2 (Holm-corrected), per charter

**Goal.** Word- and character-frequency spectra on **raw uncleaned text** (D4),
length confound controlled the amended way.

**Tasks.**

1. Create `renyiext/textfront.py`: raw text of every kept event; word tokens =
   `\w+` over Unicode, **no lowercasing** (casing is signal per D4); char
   frequencies over raw characters; per-account distributions → spectra.
   **Length control (R2, amended form):** token count is a covariate exactly
   like event count — `ρ(H_α, label | log tokens)` for every order; length
   distributions rendered per class; sensitivity arm restricted to accounts
   with ≥ 512 tokens. No fixed-n subsampling (D3′ supersedes it).
2. Arms: `SPEC_X_WORD` (6), `SPEC_X_CHAR` (6), composites with COUNT/TOKENS;
   floors = Shannon slices + dim-matched noise + META-lite. Exploratory label
   on every number; Holm within the SPEC_X family.
3. Output `results/p3i_textfront.json`; figures: per-class α-curves + length
   renders. Bitácora notes H₂-on-word-frequency ≈ Yule's K if it works
   (PHASES P3 note).

**Datasaurus:** G1 — one bot and one human account's raw text with frequency
histograms; G2 — tokenizer capture rates per class; G3 — token regex printed,
with/without-URL variants counted in the census.

**Acceptance.**

- [ ] JSON complete; partial correlations given tokens for all 12 orders.
- [ ] Holm applied and documented.
- [ ] HANDOFF/EVIDENCE-INDEX/bitácora updated.

---

### WP-J — Tail-statistic arms (method upgrade) · M · depends: WP-E ·
independent of WP-H/I/K/L

**Goal.** Give the referee-facing claim a statistic aimed at the mechanism H1
names, since the α-grid came out level-dominated.

**Tasks.**

1. `renyiext/tailstats.py`: **TAIL** = Hill estimator per Definition D8
   (α̂ clipped to [0.3, 20], k recorded); **SURV** = empirical survival
   P(Δt > t) at t ∈ {1 h, 1 d, 7 d} (fixed corpus-wide lags, printed, G3).
2. Property **P17**: synthetic Pareto(ν), n = 500, median α̂ within 15 % of ν
   for ν ∈ {1.2, 1.5, 2.0}; small-n bias direction documented in the docstring
   (bias stance rule).
3. Arms: TAIL, SURV, TAIL+SURV vs BURST and COUNT floors, dim-matched (SURV is
   naturally 3-d vs BURST's 3-d; TAIL 1-d vs COUNT 1-d). Added to
   `results/p2c_truncation.json` so WP-A's probe covers them too.
4. If TAIL+SURV beats SPEC_T's dim-matched margin, HANDOFF's mechanism narrative
   updates: separability is tail-magnitude, not multifractal shape — recorded
   either way.

**Acceptance.**

- [ ] P17 passes; bias stance in docstring.
- [ ] Probe + corpus rows present; verdict vs SPEC_T stated numerically.
- [ ] HANDOFF/EVIDENCE-INDEX/bitácora updated.

---

### WP-K — Within-corpus temporal-split generalisation axis · M · depends:
WP-D, WP-E · feeds H4 evidence

**Goal.** A second generalisation axis with fixed labels and shifted era,
enabled uniquely by the snowflake decode; previews H4 without schema mismatch.

**Tasks.**

1. **Split (pre-registered v1.0):** account → `train` if its first decoded
   event < 2012-07-01T00:00Z, else `test`. Report sizes, per-side class balance,
   fraction of accounts spanning the boundary, era-gap histogram (G1).
2. Per family (META-lite, COUNT, BURST, SHAN, SPEC_T, SHAPE, plus WP-H/I/J
   families if done): within = 5-fold CV on train only; transfer = single fit on
   train, scored on test; Δ_f per Definition D6. Scalers fitted train-side only.
3. Output `results/p3k_timesplit.json`; figure: degradation bars per family with
   bootstrap CIs and floors annotated.
4. Interpretation guard (R8 logic): META's Δ reported with and without
   era-mismatched fields. If `Δ_META − Δ_SPEC-family > 0.05` here, record as
   direct H4-preview evidence, labelled **preview**, not H4.

**Datasaurus:** G1 — era histograms; G2 — disjointness asserted elementwise on
id sets; G3 — boundary sensitivity at 2012-01-01 and 2013-01-01 in the JSON;
G4 — within-era shuffle of assignment drives Δ → 0 (run; expected silent; say why).

**Acceptance.**

- [ ] Disjointness assertion in code and JSON.
- [ ] Δ per family with CI; preview verdict separately labelled.
- [ ] HANDOFF/EVIDENCE-INDEX/bitácora updated.

---

### WP-L — Digital DNA + NCD: pre-flight and P4 core (phase P4; review D3-risk)
· L · depends: WP-D, WP-E · gate **G4 = H3**

**Goal.** Establish that NCD/BDM can measure anything on strings this short
*before* running H3, then run H3 under whichever group-estimand the corpus
supports — **with a kill criterion for the low-volume-bot case, not just a
scoping note [rev1]**.

**Tasks.**

1. **Provenance branch (resolve first).** Inspect the conversion for
   spambot-group identifiers (folder-of-origin in ids, extra label.csv columns,
   manifests under `data/raw/bot/cresci-2015/`).
   - Branch (i) groups recoverable → H3 estimand = group cohesion (mean pairwise
     NCD within true groups vs size-matched random sets).
   - Branch (ii) not recoverable (expected) → H3 estimand = **cluster
     enrichment**: hierarchical clustering by pairwise NCD on high-volume
     accounts; statistic = bot-enrichment of clusters vs the same clustering on
     marginal-preserving shuffled DNA (per-account symbol shuffles — the
     informative null here, unlike the temporal case, S4.1).
   Choice by corpus fact, recorded in the JSON config echo.
2. **Viability pre-flight (blocking):** per Definition D7 — NCD(zlib-9)
   monotone in shared-prefix fraction ρ at L ∈ {500, 2000}; L = 23 failure mode
   recorded as the documented floor length; compressors {zlib, bz2, lzma}
   tabulated. If monotonicity fails at L = 500 for all compressors ⇒ NCD leg
   **cancelled** (negative write-up); BDM legs may continue only on concatenated
   long timelines with the estimand change recorded.
3. **Branch-(ii) kill criterion (pre-registered here, [rev1]).** Restriction to
   strings long enough for NCD means restricting to high-volume accounts. Define
   the working threshold: accounts with ≥ 100 events (DNA length ≥ 99). If that
   leaves **fewer than 200 bots, or fewer than 10 % of the bot class, whichever
   is larger**, then H3 is **untestable on Cresci-2015**: cancel the H3 leg,
   write the negative with the measured survivor table, and record that
   coordination testing requires a higher-volume corpus (candidate: TwiBot-20
   timelines — flagged, out of this plan's scope). The threshold's rationale
   (≈200 accounts minimum for the enrichment statistic to resolve 0.1
   enrichment at σ ≈ 0.05) is stated in the bitácora alongside the measured
   power sketch — it is a pre-registered choice, not a derivation.
4. **R10 cross-validation:** `pybdm` vs `acss.data` CTM tables on ≥ 200 random
   short strings (alphabets 2 and 4); disagreements are findings; pin versions
   in `02-ext-research/requirements-frozen.txt` (new file, allowed).
5. Implement `renyiext/dna.py` (action-DNA 4-symbol incl. quote; temporal-DNA
   4 bins by **train-fold quartiles** — leakage rule) and `renyiext/ait.py`
   (gzip ratio floor, block-entropy floor at BDM's block size, BDM 1.0 via
   pybdm, NCD cohesion per branch estimand). Properties implemented now:
   **P9** (periodic string: BDM 1.0 ≪ its block entropy), **P12** (NCD(x,x) ≈ 0;
   independent pair high). P10/P11 stay with P7.
6. Run H3 with floors 4 (gzip ratio) and 5 (block entropy) mandatory,
   dim-matched arms; output `results/p4l_ait.json`; figures: DNA renders for one
   bot/human account, NCD matrices (sample vs null). **Length reality check
   rendered:** DNA length distribution per class; if the high-volume restriction
   leaves mostly humans, the kill criterion in task 3 governs — scoping language
   alone is not an outcome.

**Datasaurus:** G1 — actual strings drawn; G2 — viability tables elementwise;
G3 — block size, quartile edges, compressor levels printed, interior-checked;
G4 — shuffled-DNA null and size-matched group null in the same run.

**Acceptance.**

- [ ] Provenance branch decided with evidence quoted.
- [ ] Viability table complete; cancel-condition evaluated explicitly.
- [ ] **Kill-criterion survivor table computed; fired/unfired stated** — if
      fired, H3 leg cancelled and negative written up (this box then reads
      "executed the cancellation").
- [ ] P9, P12 pass (P16 still passing); versions pinned.
- [ ] H3 verdict (if run) with floors, dim-matched arms, majority baseline,
      TPR@1%FPR.
- [ ] HANDOFF/EVIDENCE-INDEX/bitácora updated.

---

### WP-M — Ledger, findings document, hand-off quality (recurring) · S ·
depends: none · runs last in every session and once more at plan completion

**Tasks.**

1. Maintain `02-ext-research/EVIDENCE-INDEX.md`: every number quoted anywhere in
   `02-ext-research/` → artefact + command. Audit at each WP close: pick three
   numbers at random from the newest bitácora and click through elementwise.
2. Maintain `02-ext-research/docs/07-FINDINGS.md` (DISCREPANCIES-analogue): one
   split-interpretation section per finding (decode bug; retention failure;
   H1 amendment; P2 pass/fail split; dim-matched downgrade if fired; probe
   ceiling; TB20 volume verdict; …).
3. `HANDOFF.md` always reflects current WP, last gate, next action, open items
   tagged with WP letters.

**Acceptance (per instance).**

- [ ] Three-random-numbers audit passed elementwise.
- [ ] FINDINGS sections current; HANDOFF current.

---

### WP-N — Protocol C execution: H4, the primary claim **[rev1: new package]**
· L · depends: WP-B (framing), WP-E, WP-F, WP-G, WP-H, WP-I ·
gate **G6 = H4 or H4′ per WP-B's amendment**

**Goal.** Actually run the transfer experiment the project exists for, under the
framing selected in WP-B, with the controls the review demanded.

**Tasks.**

1. Build the Protocol-C feature matrix for every family available from the done
   deps: META (aligned per D8/R8), COUNT, BURST, SHAN, SPEC_T (+ SHAPE arm),
   SPEC_B (if WP-H passed its gate), SPEC_X (exploratory), TAIL/SURV (if WP-J
   ran). Commensurability enforced exactly as protocol §4: spectra corpus-
   independent at fixed definitions; standardisation fitted on source only;
   `META` reduced to the four overlapping fields, drop reported.
2. Execute Protocol C (fit Cresci-2015 → test TwiBot-20), 10 seeds, under the
   WP-B-selected framing (H4 or H4′ incremental-over-volume), with Definition
   D6's degradation estimator (subsample draws + paired bootstrap over test
   users, B = 1000).
3. Mandatory R8 control: META degradation reported **both** with and without
   schema alignment; if the effect exists only unaligned, it is an artefact and
   is reported as one. Dim-matched arms per WP-E apply to the transferred
   comparisons as well.
4. Era-shift caveat quantified, not waved at: report TwiBot-20 label-source
   heterogeneity (bot categories if available) and state that a 2011–13
   fake-follower bot and a 2020 bot are different populations — the claim tested
   is feature-family robustness, not bot identity.
5. Output `results/p6n_transfer.json`; figures: the degradation figure (within
   vs transferred AUC per family, CIs, floors annotated), and the
   alignment-with/without panel.
6. Verdict per the framing selected in WP-B; split interpretation; consequences
   for the charter's success criteria stated in FINDINGS.

**Datasaurus:** G1 — per-family within/transferred bars are statistics; render
alongside example transferred-score distributions per family (objects);
G2 — feature columns elementwise-compared across corpora (names, scales,
missingness table); G3 — every alignment knob printed; G4 — a within-Cresci
pseudo-transfer (train/test split by account id hash, same machinery) must show
Δ ≈ 0 — run it as the sanity null; if it doesn't, the estimator is broken, not
the hypothesis.

**Acceptance.**

- [ ] Transfer matrix complete for every available family; byte-stable JSON.
- [ ] Framing (H4/H4′) matches WP-B's amendment; verdict stated with CI.
- [ ] R8 alignment-with/without table present; artefact rule applied.
- [ ] Sanity null (pseudo-transfer Δ ≈ 0) demonstrated.
- [ ] FINDINGS/HANDOFF/EVIDENCE-INDEX/bitácora updated; charter success
      criteria revisited in FINDINGS.

---

## 6. Dependency graph

```
A(censoring probe) ─────────────┐
B(TB20 preflight) ──► N ─┐      │
C(evidence repair) ► D ◄─┼──────┘  (D re-runs A as a fidelity check)
                    D ► E ├─► F ─► N
                          ├─► G ─► H ─► N
                          ├─► I ─► N
                          ├─► J        (feeds F's JSON; independent)
                          ├─► K
                          └─► L
M(recurring, every session close)
```

Single-threaded order = alphabetical among ready packages:
**A → B → C → D → E → F → G → H → I → J → K → L → N**, M throughout.
Parallelisable after E: {F, G, I, J, K, L} mutually independent;
H waits on G; N waits on {B, E, F, G, H, I} and prefers J/K/L done.

---

## 7. Kill criteria and amendment triggers (in addition to `docs/03-PHASES.md`)

1. **Censoring trigger (WP-A):** any probe `COUNT+SPEC_T` cell AUC ≥ 0.85 ⇒ the
   censoring amendment fires; SPEC-family claims stated net of the probe
   ceiling. Reinterpretation with teeth, not a kill.
2. **NCD viability trigger (WP-L task 2):** monotonicity fails at L = 500 for
   all compressors ⇒ NCD leg cancelled; negative written up; BDM continues only
   on concatenated long timelines with the estimand change recorded.
3. **Overflow regress trigger (WP-D):** clause (i)/(ii) below floor after the
   fix ⇒ H1 re-stated as failed-as-amended; no floor adjustment; other fronts
   unaffected; write-up leads with the correction.
4. **Low-volume-bot kill (WP-L task 3) [rev1]:** fewer than max(200, 10 % of
   bots) survivors at the ≥ 100-event working threshold ⇒ H3 untestable on this
   corpus; leg cancelled with the survivor table published; coordination testing
   flagged for a higher-volume corpus.
5. Existing project kill criteria (two of G2/G3/G4 failing; P8′ unmeetable)
   are unchanged and continue to bind.

Amendments are always **new numbered bitácora entries** naming what they
supersede (bitacora 02 is the template).

---

## 8. Pre-registered definitions and constants

Fixed v1.0 2026-08-24 unless dated otherwise. All randomness derives from
`np.random.default_rng(seed)`; derived streams add distinct integer salts so
arms never share noise draws.

- **D1 · Standard evaluation.** Seeds 42–51; StratifiedKFold(5, shuffle,
  random_state=seed); HGB(max_iter=200, early_stopping=False,
  random_state=seed); metrics: AUC, TPR@1%FPR (pooled + per-fold mean),
  macro-F1, accuracy + majority baseline; floors: majority, META(-lite),
  Shannon slice, gzip ratio (AIT only), block entropy (AIT only), BURST (every
  front, D10), COUNT (every front, amendment 02), plus the dim-matched noise
  arm (D2) wherever dim(family) > dim(floor). Effect-size floor 0.02 +
  Wilcoxon p < 0.05 two-sided, unchanged.
- **D2 · Noise padding.** `X_noisy = [X_floor ‖ N]`, N ~
  `default_rng(seed*1000 + arm_index).standard_normal((n_accounts, k))`,
  `k = dim(family) − dim(floor)`. Failure semantics per WP-E task 3.
- **D3 · `sigma_config`.** Population SD (ddof=0) of a comparison's delta across
  the full published config sweep; reported beside every floor verdict.
- **D4 · Count-caliper strata.** Deciles of `log1p(event_count)` within the kept
  sample (n = 4,770 at headline config); per-decile class histograms; ≥ 20 per
  class per decile required, else merge adjacent deciles and say so.
- **D5 · Censoring probe.** *(Reworded v1.1 for unambiguity.)* For each
  generator g ∈ {periodic(jitter 0.5), poisson, bursty(tail 1.2)} independently:
  draw 120 class-A and 120 class-B accounts from **g's own renewal process**
  (periodic ⇒ near-fixed period; poisson ⇒ exponential; bursty ⇒ Pareto — the
  probe generator always matches g; there is no separate Poisson process).
  Class A is observed over the full 900-day timeline; class B is the **same
  process** truncated to its first W days, W ∈ {30, 90, 400}. Identical feature
  pipeline; binary AUC(B vs A) per metric {SPEC_T, COUNT+SPEC_T, SHAPE};
  trigger evaluated on `COUNT+SPEC_T` at 0.85.
- **D6 · Degradation with uncertainty (WP-K/N).** R = 20 seeded draws: fit on
  80 % of the train partition, within-estimate on held-out 20 %, transfer-
  estimate on the full test partition; Δ_r = within_r − transfer_r; mean ± SD
  over r **plus** paired bootstrap over test users (B = 1000, same resamples
  across families) for the transfer term.
- **D7 · NCD viability.** Alphabet-4 uniform base string x of length L; y flips
  x's first ⌊ρ·L⌋ symbols; ρ ∈ {0, .25, .5, .75, 1}; NCD(x,y) with
  C ∈ {zlib-9, bz2, lzma} on bytes; viability = Spearman-monotone
  non-decreasing across ρ with a strict increase somewhere, per (compressor, L);
  L ∈ {23, 500, 2000}.
- **D8 · Hill estimator.** Sort Δt ascending; k = max(3, ⌊n_dt/4⌋); j = n−k;
  H = (1/k)·Σ_{i=j+1..n}[ln dt₍ᵢ₎ − ln dt₍ⱼ₎]; α̂ = 1/H clipped to [0.3, 20];
  k recorded. Bias stance: plug-in, upward-biased at small n, stated in the
  docstring (house rule).
- **D9 · Overflow cell.** `log_bin_counts` returns `[zero_count] + h +
  [overflow_count]`; `zero_count = (x <= 0).sum()`; `overflow_count =
  (x > hi).sum()`; interior bins from `np.histogram` on `[lo, hi]`. Corpus-wide
  `hi` pinned at 400 d for temporal features; sweeps vary it openly.

---

## 9. Prohibited actions (hard no's)

1. Editing `dtwre/`, `disinfo/`, `botsage/` (charter non-goal 4). Read-only imports.
2. Editing bitácora history or any committed `.ipynb`; editing notebooks instead
   of builders (S3).
3. Changing any floor, threshold, hypothesis, or the classifier to make a result
   pass. Amendments only, as new entries, before seeing the affected data.
4. Selecting sweep argmax as a headline (protocol defaults only; sweeps in full).
5. Committing anything under `data/` or `results/` (gitignored by design);
   committing secrets; skipping the two-run byte-equality check.
6. Writing any number into any document without its EVIDENCE-INDEX row and its
   datasaurus artefact.
7. Re-running a closed phase "to get a better number" (S2.4). Negative results
   are results. (Exception: WP-A's own fidelity re-run after WP-D, which is
   pre-registered in WP-A task 4.)

---

## 10. Changelog of this plan

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-24 | Initial issue. Pre-registered decisions D1–D9 (§8) and the WP-A…WP-L programme, derived from the 2026-08-24 adversarial review. |
| 1.1 | 2026-08-24 | Peer-review amendments **[rev1]**: (1) reordered — censoring probe is now WP-A (first, cheapest, headline-invalidating), TwiBot-20 preflight WP-B (second); all packages relettered accordingly; (2) added **WP-N** — Protocol C / H4 is now executed by this plan, not deferred; (3) WP-E dim-matched control given pre-committed failure semantics (downgrade executed, not "noted"); (4) WP-L branch-(ii) kill criterion with numeric threshold (max(200, 10 % of bots)); (5) §8 D5 reworded — the probe generator always matches g; (6) WP-C extended: notebook-04 inherits A3's error (§6.1 "declines monotonically"), so builders are audited against artefacts and notebook constants become JSON-derived (new standing rule in §2). |

*(Agents: append rows; never rewrite history. Sub-resolution of ambiguity →
also add a line here per §0.)*

---

## Appendix R — Review findings this plan remediates (one line each)

- **A1** Four quoted P2 qualification numbers (SHAPE decomposition +0.0273;
  SPEC_T−H₀ +0.0104; COUNT+SHAPE 0.9673; COUNT+BURST+SPEC_T vs COUNT+BURST
  +0.0192) had no committed code or artefact → WP-C.
- **A2** Measurement tables living only in bitácora prose → EVIDENCE-INDEX rule
  (WP-C, enforced by every WP).
- **A3** bitacora 04 §2 called the n_bins trend "monotone declining"; it peaks
  at 12 (0.0626) — **and the error propagated into notebook 04 §6.1** →
  corrected by entry + builder audit (WP-C).
- **B1** Floor 6 pre-registered as CV+Fano; executed as B/M/CV, unrecorded →
  acknowledged in WP-E's retroactive rows; substitution ledgered (WP-D
  bitácora); Fano stays unimplemented by design unless a reviewer demands it.
- **B2** P14 window widened in config vs METHODS without a decision entry →
  ledgered in WP-D's bitácora.
- **C1** `log_bin_counts` silently dropped intervals above `hi` (reviewer
  measured: 0.035 % of bot, 0.0005 % of human intervals; 2.21 % vs 0.68 % of
  accounts affected — to be confirmed by WP-D's render, not copied) → WP-D
  (P16, overflow cell, mass render).
- **C2** Dimensionality confound in family-vs-floor comparisons → WP-E (D2,
  with pre-committed failure semantics).
- **C3** Pooled OOF TPR@1%FPR mixes calibrations → WP-E (per-fold TPR).
- **C4** P8′ used the account-dependent grid its own docs ban → WP-D.
- **C5** Dead code, stale exports/stale INFERRED_PARAMETERS → WP-D.
- **D1** Observation-window truncation uncontrolled → WP-A (probe, first) +
  WP-F (equal-window arms, amendment trigger).
- **D2** API-cap censoring of humans unrendered/uncontrolled → WP-F.
- **D3** NCD on ~23-symbol strings near-noise; group provenance uncertain →
  WP-L (viability pre-flight, estimand branches, kill criterion).
- **D4** Effect-size floor's warrant (seed σ) is not its binding uncertainty
  (config σ) → WP-E (sigma_config reported; floor unchanged by design).
- **D5** Circadian anomaly unrendered; UTC/local offset unexplored → WP-G.
