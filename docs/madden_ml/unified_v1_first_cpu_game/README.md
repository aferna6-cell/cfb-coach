# First CPU game with the unified coordinator — owner runbook

Everything here runs on **your** WSL/laptop. Nothing writes to `~/.cfb-coach` (your primary franchise, installed-book state or champion models): the source
database is only ever opened read-only, copied with SQLite's backup API into an isolated work folder, and all training and new game records use that copy.

Branch: `architecture/unified-v1-live-session` (draft PR on top of #143). Read `../UNIFIED_COORDINATOR_V1.md` for the architecture and limits.

## 0. Get the code and find your database (read-only)
```bash
cd ~/cfb-coach && git fetch origin && git checkout architecture/unified-v1-live-session && export PYTHONPATH=.
python3 -m cfb_coach.madden.unified discover          # lists candidate DBs + model artifacts, flags which sit in ~/.cfb-coach (primary)
```

## 1–5. Backup, isolated test franchise, audit, history, both playbooks  (stops before installation)
```bash
W=~/unified_v1_cpu1
python3 -m cfb_coach.madden.unified local-prep --source-db "<path discover showed>" --workdir $W --opponent cpu
```
It (1) checksums and backs up the protected files, (2) makes `$W/testing_franchise/madden27_test.db`, (3) audits it, (4) imports **only verified executions**
read-only and compares learned vs cold-start vs baselines by whole game, (5) writes both proposed books to `$W/prep/`. It prints `decision`:
`CANDIDATE_SUPPORTED` (learned weights beat the baselines on held-out games) or `COLD_START_RETAINED` (use the labelled uncalibrated prior). Either way a
real model file is written (`$W/MODEL_CHOICE.json`). **No model is promoted for you.**

## Physical steps in Madden (only you can do these)
1. Open `$W/prep/INSTALL_GUIDE.md`. For each side, build the book in the Playbook Editor exactly as listed (family → set → source playbook → keep all plays).
2. Any set that is missing/has a different play count in the editor: stop and tell me — do **not** attest that book. Re-run `local-prep` after removing the set from the catalog.
3. Only after you have **checked every set in the editor**, attest each side:
```bash
python3 -m cfb_coach.madden.unified confirm-install --proposal $W/prep/proposal_offense.json --attested-by "<your name>" --confirmed-books $W/books.json
python3 -m cfb_coach.madden.unified confirm-install --proposal $W/prep/proposal_defense.json --attested-by "<your name>" --confirmed-books $W/books.json
```
4. (Optional) Adjustments: `python3 -m cfb_coach.madden.unified adjustments --template $W/adjustments.json`, then set a key to `true` only after you performed that exact adjustment
   on your Xbox in Madden 27. With nothing confirmed the coordinator says **NO ADJUSTMENT** every snap. `adjustments --confirmations $W/adjustments.json` shows what became recommendable.

## 6–8. Live-path benchmark, two-sided smoke, final preflight
```bash
python3 -m cfb_coach.madden.unified local-finish --workdir $W --confirmed-books $W/books.json --calls 100 [--adjustment-confirmations $W/adjustments.json]
```
Runs the real continuous game loop 100 snaps per side on **your** books + model file with persistence and a cold reload. **Do not start the game unless it prints `VERDICT: GO`.**
It also prints the exact launch and postgame commands.

## Play
```bash
python3 -m cfb_coach.madden.unified game --opponent cpu --confirmed-books $W/books.json --store $W/game/unified.db --model <model from $W/MODEL_CHOICE.json>
```
One window for the whole game; **no relaunch on possession changes**. Each snap:
1. `sit>` type who has the ball + situation: `o 1&10 my 22` (we have it) or `d 3&7 opp 40` (they have it). Later lines may omit the `o`/`d` (same side as last).
2. Read the card (call, adjustments or NO ADJUSTMENT, QB reads or USER job). Run the play.
3. Answer three prompts: **ran it?** `y` (as called) / `y noadj` (ran the play but skipped the adjustment) / `n` / `u` (unknown) / `formation | play` (what you really ran);
   **result** (the offense's result: `+7`, `inc`, `sack -6`, `int`, `fumble`, `td`); **opponent's call** (`4-3 Over | Cover 3 Sky` or just `cover 3 sky`; blank if you did not see it).
4. Type the next situation. `status` shows counters; `end` closes the game; `quit` leaves it open (resume with `--game-id <id> --resume`).

Clock text, punts and pre-snap "showing" text are accepted and ignored. Nothing is assumed: a blank "ran it?" is stored as **unknown** and never trains.

## After the game
```bash
python3 -m cfb_coach.madden.unified postgame --store $W/game/unified.db --out $W/postgame_001 --champion <the model you played>
```
Writes an immutable snapshot, a challenger, a whole-game comparison and `report.json`. The champion file is read-only; promotion is a manual decision.

---
# Update — play in the FAMILIAR HTML window (PR stacked on #144)

The same page you used in earlier Madden games now runs on Unified V1. Layout, buttons and flow are unchanged; one extra green panel shows the model, artifact,
learning status, fallback, adjustment and (on defense) the USER job. The terminal loop remains as a rollback.

## Open it (your configuration)
```bash
cd ~/cfb-coach && git fetch origin && git checkout architecture/unified-v1-html-interface && export PYTHONPATH=.
W=/home/aidan/unified_v1_cpu1
python3 -m cfb_coach play --game madden27 --opponent cpu --coordinator unified_v1 \
  --unified-confirmed-books $W/books.json \
  --unified-store $W/game/unified.db \
  --unified-model $W/history/candidate_model.json
```
It opens your browser (`--html-port N` to pick a port, `--unified-no-open` to skip opening). Rollbacks: add `--terminal` for the terminal loop, or omit `--coordinator` for the established caller.
Add `--unified-adjustment-confirmations $W/adjustments.json` only if you created that file.

## Resume after a crash / closed tab
Refreshing or reopening the tab just reloads the same game from the server. If the server itself stopped, it prints `NOTE: unfinished game(s) …`; continue with
`--unified-game-id <id> --unified-resume`: earlier observations are replayed once, and a snap that was waiting for its result is restored (no new snap is created).

## Using the window
* **Side** selector O/D = who has the ball. Fill down / distance / yard line (my/opp) and press **Submit → log + new PLAY**.
* The previous snap's fields are logged on that same Submit: outcome buttons or free text, **they ran** (their coverage, or their concept when you were on defense),
  and **What I ran** (used recommended / used different + the formation and play / unknown). **Nothing is marked executed unless you tick it**; untouched = unknown.
* **RESULT ENTRY banner**: on offense it is *your* yards; on defense it is **the opponent offense's yards**. Incomplete/INT/sack/TD buttons mean what the offense did.
* **Changed play** / used different must be a play in your confirmed book, otherwise you get an error and nothing is written.
* **Call only (no log)** produces a fresh recommendation for a corrected situation; the earlier waiting snap stays decided but unlearned (its id is not reused).
* **Undo last snap** appends corrections that make the last snap train nothing and drops the waiting call; ids are never reused.
* **Game over** closes the unified game; it does NOT run the legacy retrain (that would touch the primary database). Evaluate with the `postgame` command shown.
* The yellow/green panel always says: model `unified_v1`, the artifact file (+hash), whether it is learned weights or the cold-start prior, how many rows were learned, the opponent-belief source,
  and FALLBACK (there is no heuristic fallback in this window; an error is shown as an error).

## Validate on YOUR laptop before kickoff (required for an HTML GO)
```bash
python3 -m cfb_coach.madden.unified html-check --confirmed-books $W/books.json --model $W/history/candidate_model.json --calls 100 --browser
python3 -m cfb_coach.madden.unified preflight --html --confirmed-books $W/books.json --model $W/history/candidate_model.json \
   --history-report $W/history/history_report.json --db $W/testing_franchise/madden27_test.db --calls 100
```
`html-check` must print `HTML CHECK: PASS` (the `--browser` part needs node + playwright; if missing it says NOT_RUN and the HTTP check still counts).
`preflight --html` must show `HTML VERDICT: GO`. (`local-finish` now runs the same gates and prints both launch commands.)
