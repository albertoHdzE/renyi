# HANDOFF

**Updated:** 2026-08-18
**Branch:** `main`
**State:** design frozen, no experiment run, no package written.

## Read first

1. [bitacora/00_kickoff.md](bitacora/00_kickoff.md) — why this exists, what changed, what
   is open
2. [docs/00-CHARTER.md](docs/00-CHARTER.md) — the four pre-registered hypotheses
3. [docs/06-STANDING-RULES.md](docs/06-STANDING-RULES.md) — the datasaurus gate

## Confirmed state

Design documents complete (7 docs, 1 bitácora entry). `datasaurus` skill ported to
`.claude/skills/datasaurus/`. `renyiext/` does not exist. `02-ext-research/results/` does
not exist.

## Single next action

**P0 — data layer.** Decode all 2,827,757 Cresci-2015 snowflake IDs, build per-account
event series, pass gate G0.

**Blocking sub-item:** the snowflake claim in bitácora 00 §4 is currently supported only
by counts and a date range, which G2 explicitly says is not agreement. Render the
inter-arrival distribution and per-account trajectories, and answer G4's question — what
would this look like if IDs were a counter rather than a clock — *before* any downstream
number is quoted.

## Open items

1. Datasaurus gate on D1 (above). Blocking.
2. Post-type classification into the 4-symbol alphabet is unspecified; Cresci's
   conversion has no type field.
3. `n_events = 128` is a guess; G0 may force it down.
4. `pybdm` v0.1.0 "Pre-Alpha" needs cross-validation against `acss.data` before any AIT
   number (R10).
