# Researched offense primitive pairs

Every unordered pair of usable `offense_adjustments` rows in `cfb_coach/data/madden27/research_db.json` was evaluated by `rejected_offense_pairs()` in `cfb_coach/madden/adjustments/synthesis.py`. That function calls `conflicts()` in `cfb_coach/madden/model/offense_macro_lab.py`. The four usable rows are `hot_slant_vs_man`, `hot_hb_flat_vs_pressure`, `audible_run_vs_two_high`, and `max_protect_vs_pressure`. Six pairs. No two-action offense macro is emitted because each pair returns a reason.

QA and Meta Researcher: **every conflict below is inferred, not sourced.** The research_db row and the URL or cited file document each primitive on its own. None of those sources states the pairwise conflict. The conflict text comes from the lab function, which describes itself as compatibility rules over sourced primitives, not a cited research sentence. Do not treat "every pair conflicts" as final until a source is attached or the rule is withdrawn.

## hot_slant_vs_man + hot_hb_flat_vs_pressure

- Primitives: `hot_slant_vs_man` and `hot_hb_flat_vs_pressure`
- Conflict: primitives do not share a defensive look
- Status: INFERRED — not sourced
- Conflict rule file: `cfb_coach/madden/model/offense_macro_lab.py`, function `conflicts`. The `vs` sets are `man` and `pressure`. The function returns this reason when those sets do not intersect. That intersection test is code. It is not a sentence in the research DB.
- Primitive sources (each adjustment alone; they do not state this conflict):
  - research_db `offense_adjustments` id `hot_slant_vs_man`, source id `prodigy-beat-man`, https://www.maddenprodigy.com/how-to-beat-man-coverage-in-madden-27/ — why: "slants separate instantly vs pressed / tight man"; vs `man`; target WR1; route Slant
  - research_db `offense_adjustments` id `hot_hb_flat_vs_pressure`, source id `prodigy-beat-man`, https://www.maddenprodigy.com/how-to-beat-man-coverage-in-madden-27/ — why: "fast outlet vs a blitz — on man blitzes take the back"; vs `pressure`; target HB; route Flat

## hot_slant_vs_man + audible_run_vs_two_high

- Primitives: `hot_slant_vs_man` and `audible_run_vs_two_high`
- Conflict: an audible replaces the play and is not composed with other adjustments
- Status: INFERRED — not sourced
- Conflict rule file: `cfb_coach/madden/model/offense_macro_lab.py`, function `conflicts`. The docstring says audibles change the play, so they are not composed with hot routes or protections. No research row says a slant and a run audible cannot be one macro.
- Primitive sources (each adjustment alone; they do not state this conflict):
  - research_db `offense_adjustments` id `hot_slant_vs_man`, source id `prodigy-beat-man`, https://www.maddenprodigy.com/how-to-beat-man-coverage-in-madden-27/ — why: "slants separate instantly vs pressed / tight man"; vs `man`
  - research_db `offense_adjustments` id `audible_run_vs_two_high`, source id `meta-baseline`, cited file `cfb_coach/data/madden27/meta_baseline.json` — why: "two-high shell = light box: check to a zone run"; vs `two_high`

## hot_slant_vs_man + max_protect_vs_pressure

- Primitives: `hot_slant_vs_man` and `max_protect_vs_pressure`
- Conflict: primitives do not share a defensive look
- Status: INFERRED — not sourced
- Conflict rule file: `cfb_coach/madden/model/offense_macro_lab.py`, function `conflicts`. The `vs` sets are `man` and `pressure`, so the intersection test fails. Neither source says these two adjustments conflict.
- Primitive sources (each adjustment alone; they do not state this conflict):
  - research_db `offense_adjustments` id `hot_slant_vs_man`, source id `prodigy-beat-man`, https://www.maddenprodigy.com/how-to-beat-man-coverage-in-madden-27/ — why: "slants separate instantly vs pressed / tight man"; vs `man`
  - research_db `offense_adjustments` id `max_protect_vs_pressure`, source id `realsport-controls`, https://realsport101.com/article/madden-27-controls-guide-how-to-master-offense-and-defense — why: "keep extra blockers in vs repeated pressure"; vs `pressure`; route "Max protect"

## hot_hb_flat_vs_pressure + audible_run_vs_two_high

- Primitives: `hot_hb_flat_vs_pressure` and `audible_run_vs_two_high`
- Conflict: an audible replaces the play and is not composed with other adjustments
- Status: INFERRED — not sourced
- Conflict rule file: `cfb_coach/madden/model/offense_macro_lab.py`, function `conflicts`. The audible rule fires because one part has type `audible`. The sources do not say an HB flat and a two-high run check cannot be composed.
- Primitive sources (each adjustment alone; they do not state this conflict):
  - research_db `offense_adjustments` id `hot_hb_flat_vs_pressure`, source id `prodigy-beat-man`, https://www.maddenprodigy.com/how-to-beat-man-coverage-in-madden-27/ — why: "fast outlet vs a blitz — on man blitzes take the back"; vs `pressure`
  - research_db `offense_adjustments` id `audible_run_vs_two_high`, source id `meta-baseline`, cited file `cfb_coach/data/madden27/meta_baseline.json` — why: "two-high shell = light box: check to a zone run"; vs `two_high`

## hot_hb_flat_vs_pressure + max_protect_vs_pressure

- Primitives: `hot_hb_flat_vs_pressure` and `max_protect_vs_pressure`
- Conflict: max protect conflicts with an HB hot route
- Status: INFERRED — not sourced
- Conflict rule file: `cfb_coach/madden/model/offense_macro_lab.py`, function `conflicts`. The docstring says "Max protect keeps the back in, so it conflicts with an HB hot route." The research rows share the look `pressure`, so the defensive-look rule does not fire. The HB rule is the lab's inference from "keep extra blockers in" plus an HB flat. Neither source states that pairing as a conflict.
- Primitive sources (each adjustment alone; they do not state this conflict):
  - research_db `offense_adjustments` id `hot_hb_flat_vs_pressure`, source id `prodigy-beat-man`, https://www.maddenprodigy.com/how-to-beat-man-coverage-in-madden-27/ — why: "fast outlet vs a blitz — on man blitzes take the back"; vs `pressure`; target HB; route Flat
  - research_db `offense_adjustments` id `max_protect_vs_pressure`, source id `realsport-controls`, https://realsport101.com/article/madden-27-controls-guide-how-to-master-offense-and-defense — why: "keep extra blockers in vs repeated pressure"; vs `pressure`; route "Max protect"

## audible_run_vs_two_high + max_protect_vs_pressure

- Primitives: `audible_run_vs_two_high` and `max_protect_vs_pressure`
- Conflict: an audible replaces the play and is not composed with other adjustments
- Status: INFERRED — not sourced
- Conflict rule file: `cfb_coach/madden/model/offense_macro_lab.py`, function `conflicts`. The audible rule fires before the defensive-look check. `two_high` and `pressure` also do not intersect, but that is not the reason returned. No source says this pair conflicts.
- Primitive sources (each adjustment alone; they do not state this conflict):
  - research_db `offense_adjustments` id `audible_run_vs_two_high`, source id `meta-baseline`, cited file `cfb_coach/data/madden27/meta_baseline.json` — why: "two-high shell = light box: check to a zone run"; vs `two_high`
  - research_db `offense_adjustments` id `max_protect_vs_pressure`, source id `realsport-controls`, https://realsport101.com/article/madden-27-controls-guide-how-to-master-offense-and-defense — why: "keep extra blockers in vs repeated pressure"; vs `pressure`

## Not paired

These researched offense controls were not part of the pair evaluation. They are listed so an audit can see they were not dropped from the six-pair claim in silence.

- `control:offense:motion` in the designer registry, from `controls_xbox.offense.motion` in `research_db.json`. Confidence is `conflict`. Sources: `civil-offense` (https://www.civil.gg/tips/madden-27-complete-offense-guide) and `realsport-controls` (https://realsport101.com/article/madden-27-controls-guide-how-to-master-offense-and-defense). The button note in that row is "VERIFY: hold B then left stick (Civil.GG) vs right stick left/right (Realsport101)". The control itself is a sourced disagreement, so the primitive is not usable and was not paired. That block is sourced. It is not one of the six pair conflicts above.
- `fake_snap` and `flip_run` are `controls_xbox.offense` entries (`realsport-controls` and `civil-offense`). They are not `offense_adjustments` rows, so `rejected_offense_pairs()` never saw them.
