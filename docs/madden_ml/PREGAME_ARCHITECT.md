# Pregame Playbook Architect (Madden 27)

Before each game, `ml game-prep` designs the offensive and defensive playbooks
from the full Madden 27 catalog, explains every change, and walks you through
building them in Madden. The coach starts calling from a new book only after
you confirm that side in Madden.

```
Generate  ->  Review  ->  Install in Madden  ->  Confirm  ->  Ready to Play
(no writes)   (page)      (you, in the editor)   (per side)   (both sides, or offense for CPU)
```

## One command

```bash
python3 -m cfb_coach ml game-prep --opponent ryan --franchise primary
```

Optional inputs: `--nfl-team CIN`, `--opponent-type human|cpu` (default comes
from the opponent record), `--roster-json roster.json`, `--injury "WR1 out"`
(repeatable), `--scouting "lives in Cover 3 on 3rd down"`, `--max-offense N`,
`--max-defense N` (default and ceiling 15), `--out DIR`, `--no-open`.

The command reads the database read-only and writes three files to
`~/.cfb-coach/game_prep/<opponent>/`, plus one proposal JSON per side:

- `game_prep.html` opens in the browser. It has the executive summary, KEEP/ADD/REMOVE/SOURCE per
  formation with the reason, the exact Madden build steps, each formation's complete play list
  (with check boxes), the confirm commands, recent results, limitations, and catalog coverage.
- `game_prep_checklist.md` is the same content, ready to print.
- `game_prep.json` holds everything above as data.

`--status` prints each side's state: `no_proposal`, `awaiting_install`,
`stale` (the applied book changed, so run prep again), or `ready`.

## What gets designed

**Catalog.** `cfb_coach/data/madden27/playbooks_full.json` is parsed from
Huddle.gg. It holds 49 offensive books, 1,809 book/formation pairs, 25,569
play listings, and 427 distinct formations. On defense it holds 37 books, 824
pairs, 12,201 play listings, and 88 formations. Every formation in Huddle's
index appears in at least one parsed book (426/426 offense, 88/88 defense),
and there were no parse failures. A formation always carries the complete
play list from one source book, and plays from different books are never
mixed. `catalog_full.completeness_audit()` reports this, and the page shows it.

**Choice.** Both sides use the same portfolio search, not a top-15 by
individual score.

- **Situation coverage.** For each situation (offense: normal, short yardage,
  third and long, red zone, goal line, two-minute, backed up, explosive, and
  run balance; defense: early, short, medium, long, red zone, goal line,
  two-minute), the best formation counts fully. The second counts 0.3 as a
  counter, and the third counts 0.12 against a human or 0.05 against the CPU.
- **Depth and variety.** Credit for play depth, concept variety, and
  personnel variety. Defense also gets credit for package and coverage-family
  diversity.
- **Install cost.** Every formation costs 0.004, so 15 is a ceiling, not a
  quota. Against the CPU the books stay small.
- **Evidence.** Verified results, opponent tendencies, and roster fit shift
  a formation's situation values. They never add a flat bonus per formation.
  Defensive evidence uses identified, verified executions only. Today there
  are none, so the defense is designed from priors and tendencies.
- **Search.** The search starts from the installed book and tries add,
  remove, and swap moves. A move must gain more than a switch threshold: 0.004
  with empirical evidence, 0.012 when the evidence is prior-driven, plus 0.003
  when there are fewer than 30 evidence rows. An "ideal" book is also computed
  from scratch and shown, with the gain each unchosen formation would add.

**No rotation.** Nothing changes on a timer, a game count, a snap count, or a
win or loss. A formation enters, leaves, or comes back only when the portfolio
score says so. Run prep twice on unchanged data and you get the same
fingerprint; the tests check this across a simulated week.

**Status labels.**
- KEEP: the formation stays as installed.
- ADD: a new formation.
- REMOVE: the formation leaves the book.
- SOURCE: the same formation switches to another book's play list.

Each label comes with the football reason, plus the score change.

## Install in Madden, then confirm

Follow the page's "Build it in Madden" steps:
1. Creation Center > Playbooks > Create Offense/Defense, and pick the
   suggested base book.
2. Delete the listed formations, add the others from All Plays, and match
   every play list exactly.
3. Set audibles and save with Start.
4. In Franchise, select the book on Gameday > Advanced Settings before Ready.

Then confirm each side on its own:

```bash
python3 -m cfb_coach ml game-prep --opponent ryan \
  --confirm defense --proposal-id <id from page> --fingerprint DEF-XXXX-XXXX \
  --attest "I built ML DEF XXXXXXXX in Madden and selected it on Gameday" \
  --db ~/.cfb-coach/madden27.db
```

Confirmation is refused, and nothing is written, if:
- the proposal file was edited,
- a newer proposal exists,
- the applied book changed since generation,
- the fingerprint is wrong,
- the attestation is shorter than 30 characters,
- a formation no longer matches its source play list, or
- the book breaks the editor limits.

A confirmation does three things:
1. It snapshots the database to `game_prep/snapshots/`.
2. It writes the new book and its history in one transaction.
3. It revokes generated Custom Adjustment approvals for that side.

The coach cannot see Madden. A confirmation only records your statement that
you built the book.

Ordinary `prep` does not rewrite a confirmed architect book. Only a new
confirmation or an explicit `--o-book` / `--d-book` changes it.

## Roll back

```bash
python3 -m cfb_coach ml game-prep --opponent ryan --rollback defense \
  --attest "I reselected my previous defensive book in Madden" --db ~/.cfb-coach/madden27.db
```

Rollback restores the book that was applied before the confirmation, or
removes the side if there was none. Each side rolls back on its own. The
database snapshots are a second safety net: copy one back over
`~/.cfb-coach/madden27.db` while the coach is stopped.

## Playing with both books

```bash
# Human opponent: ML offense + model defense from the confirmed books
python3 -m cfb_coach play --game madden27 -o ryan --ml-human-control --defense-model v1

# CPU: offense only (unchanged default)
python3 -m cfb_coach play --game madden27 -o cpu

# CPU with defense too (opt-in); add --defense-model v1 for the model
python3 -m cfb_coach play --game madden27 -o cpu --cpu-defense --defense-model v1
```

The header prints one line per side with the caller, book name, revision, and
architect proposal and fingerprint. No game clock or pre-snap `showing` input
is needed.

When the defensive book came from a confirmed proposal, the defense model:

- ranks only installed calls;
- falls back to the established caller when nothing it knows is installed,
  or on a short-yardage or goal-line down with no known heavy front
  installed. Fallback reason: `installed_gap`;
- penalizes a formation that passes 40% of the last ten calls, so a close
  installed alternative gets through and a clear leader still wins. In a
  replay of the Ryan game this took the model from 29 of 37 Dime 3-2 Odd
  calls to 17 of 37, using four formations. That is selection behavior only,
  not a claim about the result;
- stays inside the 150 ms budget.

## Replay (read-only)

```bash
python3 scripts/replay_pregame_architect.py --db ~/.cfb-coach/madden27.db
```

The script replays the Ryan loss and the latest CPU win through the live
pickers, once with the current books and once with the proposed books. It
reports formation usage, legality, latency, and fallback rates. Designs use
only evidence logged before each game. It writes nothing.

## Data notes

- The CPU game stored as 42-7 is read as 35-7 (W) through
  `data/madden27/result_corrections.json`. The correction applies only while
  the stored score is still 42-7.
- The Ryan game (28-20 loss) was called by the heuristic offense and the
  experimental defense. 37 defensive snaps were logged and none have verified
  executions, so they shape tendencies but never label a call.
- The installed 428-row offense model is unchanged. The architect only reads
  its predictions.

## Known limits

- Play lists are from Huddle.gg, a third-party site. They are not verified
  in game, and title updates may change them.
- Hail Mary in the Falcons, Eagles, and Chargers books differs from the
  older core catalog.
- The editor limits (15 formations, 56 sets, 500 plays) are from
  madden.tools and are not verified in game.
- Menu labels in the steps come from ClutchPoints and EA Help and may differ
  slightly in Madden 27.
- The defense model knows only the formations it was trained on. Installed
  formations it does not know are called by the established caller through
  the fallback.
