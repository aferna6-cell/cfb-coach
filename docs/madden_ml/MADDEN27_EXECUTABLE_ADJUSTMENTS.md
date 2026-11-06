# Madden 27 Executable Adjustments (Xbox)

Rule: a recommendation is ready for live use only if you can perform it in Madden NFL 27 on
Xbox from the instructions CFB Coach prints. Generic advice ("defend underneath") is never output.

This work is research-only. The live caller, heuristic fallback, `cfb_coach/madden/adjustments/__init__.py`
and the `research_db.json` control table are unchanged. No live adjustment is enabled.

## Pieces

| File | Role |
| --- | --- |
| `cfb_coach/data/madden27/executable_actions.json` | Action registry: exact names, side, category, parameters, requirements, Xbox input steps, kind (stock / individual / preset), compatibility groups, timing, sources, grades |
| `cfb_coach/madden/adjustments/executable.py` | Loadout and confirmation checks, input validation, plan validation, candidate generator, recommendation, coaching card, pre-game loadout planner, catalog report |
| `scripts/madden_ml/executable_adjustment_examples.py` | Prints example offense and defense cards, the catalog status and the audit of the old control table |
| `tests/test_executable_adjustments.py` | Rejection and decision tests |

## Sources

- EA official Madden 27 Xbox controls hub: pre-play offense and pre-play defense images. These are the
  only source for entry buttons. Image hashes are recorded in the registry.
- EA Madden 27 gameplay deep dive: the hold-LT coverage hub (press/back off, leverage, pass commit,
  Smart Zone strategy, deep zone focus), double team (hold LT, pick the receiver, press X), Custom
  Adjustments (create 20, bring 10 active), the rule that players must get set between adjustments,
  and the Defend Screen Pass preset.
- EA title update of Sep 16, 2026: Half Slide / Full Slide protections and QB Contain exist.
- Secondary sources (Madden27.wiki, Madden Prodigy, Magic Game World): preset names Play Short Routes,
  No Deep Passes and QB Scramble, and the offensive preset list. These are graded `secondary` in the registry.

Sub-menu choices EA does not document are printed as "choose '<label>' on the on-screen menu". No button is guessed.
Saved Custom Adjustments are printed as "Press LB -> choose '<name>' in your confirmed slot (<your slot text>)".
No face-button mapping is assumed.

## Status today (no in-game confirmations recorded)

**Fully verified: none.** No action has been confirmed by you on Xbox yet.

**EA documents the full input sequence, but it still needs your test:** Fake Snap (RB), Double Team
(hold LT, choose the receiver, press X).

**EA documents the entry button; the menu step still needs your test:** Hot Route (Y), Audible (X),
Pass Protection (LT; Half Slide / Full Slide), Motion Player (B to select, hold the left stick
left/right), Global Coverage hub (LT), Coverage Audible (Y), DEF Hot Route (A), DL / LB Shift
(right stick left/right), Defensive Keys (RB).

**Not installed (presets and saved adjustments):** every preset (Go Vertical, Slants, Mesh, Outs, Smart Routes,
Comebacks, Double Moves, Zigs; Defend Screen Pass, Play Short Routes, No Deep Passes, QB Scramble)
and every saved macro. Each one stays unusable until you record it as installed, active and in a slot.

**Unsupported (no verified Madden 27 input):** Chip Block (EA documents the mechanic but not its input), QB Contain,
Show Blitz.

## Conflicts in the existing live control table (needs your approval to fix)

`research_db.json` `controls_xbox`, used by the current live adjustment text, disagrees with EA's Madden 27 map:

- Defense individual assignment is listed as "Y then defender". On EA's map Y is Coverage Audible and A is DEF Hot Route.
- Show Blitz is listed as "RB". On EA's map RB is Defensive Keys.
- QB Contain is listed as "RB then LB". On EA's map those are Defensive Keys, then Custom Adjustments.
- Stunt is listed as "RS left then RB". That input is not on EA's map.
- Flip run is not on EA's map.
- Press/back off, shade, commit and zone strategy are listed with separate inputs, but EA puts them in the hold-LT hub.

These are reported in `RESEARCH_CONTROL_AUDIT`. They have not been changed in the live path.

## How decisions are made

For each snap the generator lists every executable option for this formation and play. The unadjusted
play is always first. Each option's expected effect comes from the opponent threat distribution
(defense: run, rpo, vert, flood, cross, stack, scram; offense: man, pressure, two_high, single_high, cover2):

    score = (threat it answers - threat it exposes) x evidence weight x (1 - 0.5 x uncertainty) - timing cost

Evidence weights are measured 1.0, prior 0.35 and hypothesis 0.15. The registry has no measured
entries yet, so every confidence prints as LOW or VERY LOW, never MEASURED. An adjustment is chosen
only if it beats the unadjusted play by at least 0.01. Otherwise the card says to run the called
play. In live mode only live-ready actions are eligible. Research mode can also show actions that
still need your test, and labels them that way. Not-installed and unsupported actions are never
chosen, but the card notes when one of them would have been the best option in theory.

Every plan is rejected if it has any of these problems:
- an unknown name;
- a wrong side;
- an invalid or out-of-map input;
- a preset or macro that isn't installed and active;
- a receiver who isn't eligible, or a defender who isn't on the field;
- a route that isn't legal for that player;
- a hot route or protection on a run play;
- an audible that isn't in the formation;
- two actions that touch the same thing;
- a conflict with a setting already applied this snap;
- not enough play clock (play clock minus 3 s, minus 15 s in Coach Mode, with about 1.5 s for players to get set after each step).

## Recording what you verified in-game

Loadout for a game (presets and saved macros):

```json
{
  "game_label": "week 11",
  "active_limit_per_side_confirmed": false,
  "offense": [
    {"name": "Slants", "preset_id": "off.preset.slants", "slot": "LB menu, 2nd item",
     "installed": true, "active": true, "confirmed_by_user": true, "confirmed_at": "2026-11-07"}
  ],
  "defense": [
    {"name": "TAMPA MABLE", "slot": "LB menu, 1st item",
     "installed": true, "active": true, "confirmed_by_user": true, "confirmed_at": "2026-11-07"}
  ]
}
```

`slot` is whatever you see on screen. It is printed back to you exactly as written.

Direct-control confirmations (after you perform the action on Xbox):

```json
{
  "def.double_team": {"confirmed": true, "on": "2026-11-07"},
  "off.hot_route": {"confirmed": true},
  "off.hot_route|Slant": {"confirmed": true}
}
```

A menu option such as a route name or a hub option needs its own `action|option` entry. An audible
only needs the action entry, because the plays come from your own playbook.

## Next steps

1. On Xbox, test Fake Snap and Double Team first, because EA documents them end to end. Then test
   Hot Route with Slant, Flat and Streak, and the LT hub options.
2. Record your real loadout and slots for the next game.
3. Decide whether the live control table conflicts above should be fixed. That needs your approval.
4. Measure adjusted versus unadjusted outcomes from logged games before any confidence moves above LOW.
