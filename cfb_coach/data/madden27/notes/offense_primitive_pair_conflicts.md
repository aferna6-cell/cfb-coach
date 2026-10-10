# Researched offense primitive pairs

Meta Researcher verified these six pairs on 2026-10-10 (patch 1.007). The evaluator is `offense_pair_verdicts()` in `cfb_coach/madden/adjustments/synthesis.py`. The four researched rows are `hot_slant_vs_man`, `hot_hb_flat_vs_pressure`, `audible_run_vs_two_high`, and `max_protect_vs_pressure`.

Pairs marked legal are generated on installed pass plays as DRAFT. They are not auto-eligible. Pairs with a conflict are not generated.

## hot_slant_vs_man + hot_hb_flat_vs_pressure

- Primitives: `hot_slant_vs_man` and `hot_hb_flat_vs_pressure`
- Conflict: no conflict
- Status: VERIFIED-M27
- Confidence: high
- Situational tag: `man_blitz` (both fit a man blitz). Same defensive look is a tactics note, not a game rule.
- Note: Hot routes on different players stack.
- Source: https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay

## hot_slant_vs_man + audible_run_vs_two_high

- Primitives: `hot_slant_vs_man` and `audible_run_vs_two_high`
- Conflict: audible replaces the play
- Status: HYPOTHESIS
- Confidence: medium
- Note: Audible replaces the play. No M27 source documents that an audible clears the slant.
- Source: https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay

## hot_slant_vs_man + max_protect_vs_pressure

- Primitives: `hot_slant_vs_man` and `max_protect_vs_pressure`
- Conflict: no conflict
- Status: VERIFIED-M27
- Confidence: high
- Note: Protection (LT) and hot routes (Y) are separate menus. EA says both can be saved in one Custom Adjustment.
- Sources: https://www.ea.com/games/madden-nfl/madden-nfl-27/controls-hub/m27-xb-preplay-offense and https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay

## hot_hb_flat_vs_pressure + audible_run_vs_two_high

- Primitives: `hot_hb_flat_vs_pressure` and `audible_run_vs_two_high`
- Conflict: audible replaces the play
- Status: HYPOTHESIS
- Confidence: medium
- Note: Audible replaces the play. The HB would be the ball carrier on the run check.
- Source: https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay

## hot_hb_flat_vs_pressure + max_protect_vs_pressure

- Primitives: `hot_hb_flat_vs_pressure` and `max_protect_vs_pressure`
- Conflict: conditional: unknown precedence; max protect and an HB flat are not emitted together
- Status: CONDITIONAL
- Confidence: low to medium
- Note: CARRIED-FORWARD/HYPOTHESIS. Which assignment the HB obeys is not sourced. The sanctioned alternative is HB Chip and Release (VERIFIED-M27). This combo is not generated.
- Sources: https://fenixbazaar.com/2025/08/10/madden-26-leverage-system/ , https://madden.tools/playbooks/book/madden-franchise-offense-maxprotect , https://www.818madden.com/tools/position-adjustments , https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay

## audible_run_vs_two_high + max_protect_vs_pressure

- Primitives: `audible_run_vs_two_high` and `max_protect_vs_pressure`
- Conflict: audible replaces the play
- Status: HYPOTHESIS
- Confidence: medium
- Note: Audible replaces the play. Max protect is pass-only. ID Mike is the run-side equivalent (VERIFIED-M27).
- Source: https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay

## HB Chip and Release

Sanctioned alternative to `hot_hb_flat_vs_pressure` + `max_protect_vs_pressure`. Added as primitive `hot_hb_chip_and_release` even though `research_db.json` has no row for it. No vs-look, button, or other detail was invented beyond the two sources.

- Status: VERIFIED-M27
- Confidence: high
- Route name: Chip and Release. Target: HB. The hot-route menu item is what the write-up names.
- EA: "Any route can be assigned a chip." https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay
- 818 Madden: the HB hot-route menu includes Chip and Release. https://www.818madden.com/tools/position-adjustments
- Generated with max protect on installed pass plays, as a DRAFT. The flat-plus-max-protect combo is not generated.

## Not paired

Motion stays off. No Xbox control is verified yet.

- EA's official Xbox preplay page has no motion entry: https://www.ea.com/games/madden-nfl/madden-nfl-27/controls-hub/m27-xb-preplay-offense
- Realsport101 maps motion to the right stick: https://realsport101.com/article/madden-27-controls-guide-how-to-master-offense-and-defense
- Civil.GG: select the player (hold B), then the left stick: https://www.civil.gg/tips/madden-27-complete-offense-guide
- ClutchPoints: "Motion Player – Hold LS": https://clutchpoints.com/gaming/madden-27-passing-guide
- Those three creator pages conflict with each other and with the missing EA entry. Best estimate: select the player, then left stick left/right, medium confidence, pending a Practice-mode test.
- Registry id `control:offense:motion` stays unusable.

`fake_snap` and `flip_run` are `controls_xbox.offense` entries. They are not `offense_adjustments` rows, so they were not part of the six-pair evaluation.
