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
