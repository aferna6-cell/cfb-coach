# Defensive Coordinator V2 — test franchise runbook

## Current status

Draft PR #139 is a **testing-franchise-only experimental implementation**, not a
verified production release. The default Madden database, main franchise and
offensive coordinator have not been changed. The V2 artifact is never promoted
by the tool. The decision pipeline is opt-in and fails back to the existing
defensive call if its identity, artifact or installed-playbook gates fail.

The first version was shadow-only. That description is obsolete: this branch
now includes an opt-in LIVE Madden 27 defensive selection route, an offline
playbook proposal, individual adjustment search, USER defender reads and
training/evaluation commands.

## What the model now does

- Uses only actually verified executed defensive plays for outcome training,
  including CPU and human opponents, and separates them from displayed calls.
- Estimates offensive success plus available yards, conversions, TDs,
  explosives and turnover risks using a shrinkage learner.
- Fits extra defensive candidate-structure features only after 200 verified
  snaps in at least 8 games with sufficient catalog support.
- Updates its opponent concept probability mixture from completed **earlier**
  defensive snaps, not guessed live coverage or same-snap results.
- Ranks only stock plays in the confirmed installed 15-or-fewer-formation
  defensive book.
- Compares unadjusted calls against up to two compatible individual
  adjustments using source-documented Madden 27 options. The option must have
  an expected advantage that clears a conservative threshold. A diffuse
  opponent concept belief produces NO ADJUSTMENT.
- **No Xbox buttons are required.** The user wants the recommendation, not a
  controller tutorial. The catalog establishes that the named adjustment
  actually exists in Madden 27. User-confirmed execution can teach the model
  about the action's observed performance.
- Gives a pre-snap user-controlled defender read, such as run fit first and
  then carry a crosser in the assigned zone, or play the middle seam against
  verticals. Guidance is conditional on the model's opponent concept beliefs.
- Keeps football coverage scheme templates separate from exact Madden
  11-player assignments. General Cover 1/2/3/4/6/9, Tampa 2, man, match and
  pressure responsibilities are available. Unverified exact player roles
  remain UNKNOWN; they are NOT fabricated from the stock play name.
- Generates up to 15 model-selected complete stock defensive formations as
  a proposal, with source playbooks and inventory fingerprint; the game editor
  installation remains a separately confirmed action.
- Saves postgame challengers outside the live game directory and compares
  them to the saved champion on games beyond its training cutoff; no automatic
  promotion.

### Named individual adjustments supported

The proposal search uses the project's permitted first-party EA Madden NFL 27
action catalog: Global Coverage Press, Back Off, Inside/Outside Leverage, Pass
Commit Inside/Outside, Smart Zone Aggressive/Conservative variants and related
documented named choices. A specific Double Team option is generated only
when an actual receiver target is supplied. A DEF Hot Route man-match can be
represented when both defender and receiver targets are identified, but is not
automatically selected until it has a defensible estimated effect.

These are in-game adjustment names, not saved macros and not Xbox instructions.
A stock play can absolutely win NO ADJUSTMENT. Effect magnitudes are
conservative football hypotheses at cold start, NOT empirically proven
advantages. After enough *confirmed executed* adjustments and matched
no-adjustment snaps, observational effects contribute to the model. This
cannot remove selection bias without controlled tests.

When multiple changes are proposed, they must have distinct adjustment
groups. The model is capped at two and requires an additional marginal benefit
for the second; this cap prevents speculative adjustment spam.

## Source evidence

- EA Madden NFL 27 Gameplay Deep Dive, 2026-06-04:
  https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay
- Existing defense knowledge catalog:
  cfb_coach/madden/model/defense_semantics.py
- Game action names and first-party provenance:
  cfb_coach/data/madden27/executable_actions.json

## Read-only development commands

Always use a separate test franchise DB. Never make a live DB a model-training
output. Fill in the actual path and dynasty identity recorded in the DB.

    PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py inventory --db /PATH/TO/TEST/madden27.db

    PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py audit \
      --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE"

    PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py evaluate \
      --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE"

    PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py train \
      --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE" \
      --out /tmp/test-defense-v2-shadow.json

    PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py design \
      --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE" \
      --model /tmp/test-defense-v2-shadow.json \
      --out /tmp/test-defense-v2-playbook-proposal.json

Design is a proposal, NOT a confirmed installed Madden game book.

## Explicit, testing-franchise-only live opt-in

Only after the artifact contains verified defensive training from earlier games
and the installed defensive book is confirmed in Madden and CFB Coach:

    export CFB_COACH_DEFENSE_V2_TEST_FRANCHISE="EXACT_DYNASTY_VALUE"
    export CFB_COACH_DEFENSE_V2_ARTIFACT=/tmp/test-defense-v2-shadow.json
    python3 -m cfb_coach play --game madden27 --defense-model v2 --cpu-defense ...

Use the actual opponent and game arguments from play --help in place of the
placeholder. Select side d in terminal for defense. No clocks, pre-snap
showing prompts or punting decisions are added.

Rollback: omit --defense-model v2 in the next play session. The global caller
mode is not modified by this flag.

### After game

    PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py postgame \
      --db /PATH/TO/TEST/madden27.db --franchise "EXACT_DYNASTY_VALUE" \
      --model /tmp/test-defense-v2-shadow.json \
      --out /tmp/test-defense-v2-candidate.json

This writes a new candidate; no promotion occurs. Compare the saved champion
and challenger on genuinely later games. A classification/forecast metric
improvement does NOT establish that the new call selection policy wins more.

## Remaining game-readiness gates

1. Local owner WSL and actual test-franchise DB audit (real verified executed
   snaps, exact dynasty, distinct DB path, opponent IDs and chronological
   snap sequences). Without verified executions V2 must remain inactive.
2. Confirm the physically installed defensive formation inventory matches the
   locked CFB Coach book. The ML portfolio proposal does not install itself.
3. Test a genuine play -> V2 adjustment or NO ADJUSTMENT -> user execution
   confirmation -> next updated opponent belief -> next call loop.
4. Run the real installed-book decision benchmark at or under 150 ms, including
   the joint adjustment search and USER read generation.
5. Review full Madden regression suite; previous CPU2 parent branches had known
   broader-suite failures, independent of these targeted V2 tests.
6. Expand individual slot-level assignments from first-party Madden diagrams
   or user-captured verified play art. Current schematics do not identify the
   exact eleven player assignments for every stock play.
7. Demonstrate prospective policy value with controlled CPU games. Do not
   promote an unproven challenger or assume opponent-success correlations are
   causal adjustment effects.

**Game-ready status remains NO-GO until owner laptop and real-game checks
succeed.** The branch is a useful guarded experimental implementation,
not an automatic green light to replace your trusted caller.
