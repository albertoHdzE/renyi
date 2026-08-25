# Bitácora 18 — WP-L, the H3 cancellation: the kill criterion fires exactly as written

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** G4 = H3 — **not run; the leg is CANCELLED by the pre-registered
branch-(ii) kill criterion** (plan WP-L task 3 [rev1]). Property checks
**12/12** (P9, P12 added). Two runs byte-identical (S2.6).
**Artefacts:** `renyiext/dna.py`, `renyiext/ait.py` (new),
`renyiext/checks.py` (P9, P12), `scripts/run_p4l_ait.py` (new),
`02-ext-research/requirements-frozen.txt` (new),
`results/p4l_ait.json`, `figures/p4l_dna_render.png`,
`figures/p4l_dna_lengths.png`.

---

## 1. Provenance branch (resolved first, evidence quoted)

**Branch (ii): groups NOT recoverable.** Evidence: `label.csv` header is
exactly `id,label` (2 columns); account ids are flat `u*`/`t*` with no
folder-of-origin; `split.csv` is `id,split` (the official split, not
groups); and the source archive `d.tar.gz` unpacks to the SAME five flat
files (edge.csv, label.csv, node.json, split.csv, user_info.pt) — no group
folders anywhere in the conversion or its source. Estimand would have been
cluster enrichment vs marginal-preserving shuffled DNA.

## 2. D7 viability: the NCD cancel-condition does NOT fire

All nine (compressor, L) cells — {zlib-9, bz2, lzma} × {23, 500, 2000} —
are Spearman-monotone (ρ = 1.000) with a strict increase over 20 replicates
each. L = 23 is the documented floor length: monotone but range-compressed
(NCD 0.03–0.55, compressed dynamic range near compressor overhead).
Recorded per-cell in the JSON. The NCD machinery is sound; what dies is not
the tool but the population (§3).

## 3. The kill criterion FIRES — H3 cancelled

Restrict to accounts with ≥ 100 events (DNA length ≥ 99):

| | survivors | share of class |
|---|---|---|
| bots | **162** / 3,351 | 4.83 % |
| humans | 1,687 / 1,950 | 86.5 % |

Threshold = max(200, 10 % of 3,351) = **335.1**; 162 < 335.1 → **FIRED**.
The restriction leaves a 8.8 % bot / 91.2 % human population — exactly the
failure mode the criterion was written for (scoping language alone is not
an outcome; the outcome is the cancellation). **H3 is untestable on
Cresci-2015.** Coordination testing is flagged for a higher-volume corpus
(candidate: TwiBot-20 timelines — out of this plan's scope). The G1 length
render is the kill evidence: the bot mass sits left of the threshold, the
survivor region right of it is almost purely human (with the ~3200 API-cap
spike). The rationale for 200 (resolving 0.1 enrichment at σ ≈ 0.05) is
pre-registered in the JSON, not derived.

## 4. R10: blocked by a PyPI name collision — recorded, not substituted

`pip install acss` yields 0.5.2, a **Python-2 web-service project**
(`acss.client` imports `urllib2`) — not the CTM-tables library
(`acss.data`) the plan names. No trustworthy CTM source is installed, so
the pybdm-vs-CTM cross-validation is **BLOCKED** and recorded as a finding;
it is not silently substituted. pybdm 0.1.0 is installed and pinned in
`requirements-frozen.txt` (with the acss caveat as a comment). Its measured
CTM coverage: 1D alphabets {2, 4, 5, 6, 9} at 12-symbol blocks; 2D binary
at 4×4 only — **no 2D alphabet-4 table** (the config note's expectation
does not ship; alphabet-4 DNA uses 1D BDM).

## 5. Implementations and properties

`renyiext/dna.py` (action-DNA ORTQ incl. quote per D5; temporal-DNA binned
by caller-supplied train-side quartile edges — the leakage rule) and
`renyiext/ait.py` (gzip ratio, block entropy, 1D BDM via pybdm, NCD per D7).
**P9**: periodic 12×12 BDM = 0.122× random (structure far below noise).
**P12**: NCD(x,x) ≤ 0.058, NCD(independent) ≥ 0.833. Checks **12/12**.

Two findings recorded inside P9's own message: (a) the plan's shorthand
"BDM 1.0 ≪ its block entropy" does **not** hold under the standard
definition — the periodic string's block entropy (1.0 bit) and its BDM
(33.4 bits) diverge the *other* way; what P9 asserts is the R6 intent
(BDM is not block entropy; it orders structure ≪ noise); (b) pybdm's
`shape` argument is the *block* shape — the first draft's 12×12 "blocks"
hit untabulated CTM and the zero-BDM guard on degenerate patterns; the
working configuration (4×4 blocks) is documented.

## 6. Consequences for the programme

H3's gate (G4 = H3) resolves as **cancelled-by-kill-criterion** — an
honourable outcome per the charter. WP-N's family list proceeds WITHOUT an
AIT family; the coordination question is explicitly deferred to a
higher-volume corpus. No other front is affected.

## 7. What failed and was not fixed

- The `acss` name collision (§4): unfixable within this plan's dependency
  rules; recorded and pinned around.
- P9's first draft asserted block entropy ≈ 0 for the checkerboard tiling
  and failed on its own (1D flatten has two 12-block phases); the check was
  rewritten to assert the R6 intent and report the divergence — the
  datasaurus loop refusing an unsatisfiable literal.
- One wording bug in the cancellation string ("8.8 % human" for the bot
  share) — caught on review, fixed before the artefact was committed to
  git (the JSON was regenerated).

## 8. Multiple comparisons counted

D7: 9 cells × 5 ρ × 20 reps; P9/P12 thresholds set from 30-string
measurements; kill table 2 classes × 1 threshold; R10 attempted once. No
hypothesis-level claims (the gate's outcome is a cancellation).
