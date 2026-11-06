# Madden 27 Play Knowledge v1 — implemented research component

This implementation is **not** a new live playcaller. It is a safe first step in building
play-level structural intelligence on top of PR #104. Nothing is merged, installed,
or promoted automatically.

## Existing catalog

The repository already has `cfb_coach/data/madden27/playbooks_full.json` from Huddle.gg.
It currently contains **49 offensive and 37 defensive source books**, about **25,569
listed offensive play entries and 12,201 defensive entries**. Repeated play names and
formation records from separate source books are distinct entries. These values are
catalog records, **not** counts of fully annotated or unique play designs.

## New files

- `cfb_coach/madden/model/play_knowledge.py`: normalized SQLite play/assignment
  store, route taxonomy, individually sourced player jobs, verified transformations,
  macro declarations, and conservative read/user advice.
- `cfb_coach/madden/model/assignment_coordinator.py`: calls existing
  `madden.adjustments.executable` for *legal, user-confirmed* actual Madden actions
  and returns exactly one adjustment or **NO ADJUSTMENT** in research mode.
- `scripts/madden_ml/build_play_knowledge.py`: builds a separate research database
  from catalog names; refuses to overwrite a file or write into `~/.cfb-coach`.
- `tests/test_play_knowledge.py`: synthetic catalog and assignment safety tests.
- `cfb_coach/data/madden27/play_knowledge_seeds.json`: one **source-documented**, not Xbox-verified, offensive route-table example from Madden Tools (`Balanced / Gun Tight Slots / Mesh`). It is intentionally partial. The published page's narrative and route/read table contain discrepancies; no read order is imported.

## Build the database

From the checkout, with the package installed or working directory on `PYTHONPATH`:

```bash
python scripts/madden_ml/build_play_knowledge.py --output /tmp/madden27-play-knowledge.db
```

The importer first creates **names only**, then (by default) applies one small
public-source example annotation if the seed file is present. Without that
example (`--no-seeds`), annotated assignments are **zero**. It cannot infer WR
routes, blocking patterns, or defensive zones from a label like `Mesh` or
`Cover 3 Sky`. The seed remains `source_documented`, not `user_verified`.

Add a verified annotation for a known play using `PlayKnowledgeDB.annotate` with
`PlayerJob(position, kind, task, evidence, source, params)`. Positions are
play-relative IDs (`WR1`, `WR2`, `TE1`, `MLB`, `CB1`), not fabricated roster names.
Use a source document or player-confirmed screenshot; never mark a guess as verified.
Routes are a reusable taxonomy, with depth/direction optionally captured in `params`.

**Completeness levels**: `catalog_only` → `partial` → `fully_verified`
(requires jobs for all eleven distinct players explicitly confirmed by the user).

## What can and cannot yet be said

- **Supported**: query a play's recorded jobs; change one sourced WR route to a
  separately confirmed Madden hot route; change a sourced defender job through
  a verified assignment; apply a declared, active, user-verified macro with
  known changes; preserve unaffected assignments; reject unknown or conflicting
  changes; choose a specific eligible action or `NO ADJUSTMENT`.
- **Supported**: only emit a quarterback read order if that actual play has a
  sourced progression and relevant routes; only provide a user-defender duty
  if the selected position has a sourced defensive job.
- **Not supported yet**: automatic diagram interpretation, complete 11-man
  annotations for all catalog entries, unsourced Cover 2 matchups, opponent
  user behavior from unseen footage, accurate play-specific adjustment effects,
  or causal proof an adjusted call will perform better than stock.
- **Important**: confirmation that a macro can be executed is distinct from
  confirmation of how that macro changes each player. Global coverage and
  alignment adjustments with unknown assignment effects are labeled as such.

## Next engineering milestone

1. Obtain appropriately licensed/permitted diagrams or *user-provided* screenshots
   for the actual installed books. Build a reviewable diagram annotation tool;
   never scrape restricted materials or claim a name gives the complete play art.
2. Verify a representative set of whole plays, including each route/block/zone,
   positional alignment and conditional match rule.
3. Extend the `executable` action registry and macro editor to store concrete,
   confirmed assignment deltas, not generic scoring bonuses.
4. Compare structure-aware and old name-based predictors on genuinely held-out
   Madden games, with in-game execution confirmations and no future-snap leakage.
5. Add actual pre-snap reads/user advice as an opt-in display only after
   annotation coverage is sufficient; don't invent unknown reads or jobs.

This v1 is **implemented infrastructure, not a trained football-understanding model**.