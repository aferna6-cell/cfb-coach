# First CPU game: exact launch, verify, roll back, report

Everything below runs **on your computer** (the code review/tests ran in a cloud sandbox with no access to your database or Madden).
Nothing was enabled for you, and no CPU game has been played by me. The existing coordinator stays the automatic fallback; the defensive
coordinator is untouched. Your gameplay form is unchanged.

## 0. Get the code (PR #117 contains #113 and #115)
```bash
cd ~/cfb-coach && git fetch origin claude/defense-knowledge-116 && git checkout claude/defense-knowledge-116
python -m pip install -e . 2>/dev/null || true
python -m cfb_coach ml find-db                 # prints the DB the app uses (expected ~/.cfb-coach/madden27.db)
```

## 1. Prepare: backup, work on a copy, real-data evaluation (does not touch the live DB)
```bash
python scripts/madden_ml/cpu_experiment.py prepare            # backup -> ~/cfb_cpu_experiment/backup, copy -> experiment_copy.db
python scripts/madden_ml/cpu_experiment.py summary            # verdict, every gate, artifact, 93/93 art records, shadow replay
```
`prepare` runs `offense_route_aware_readiness.py --run-tests` (retrains the offense artifact from your recorded games using all 93
approved play-art records; checks model validity, candidate legality, 150 ms latency) and `defense_knowledge_eval.py heldout` (knowledge
model vs the route-aware coordinator on whole held-out games). Read the summary. **Lack of a significant improvement does not block an
experimental CPU test. `prepare` exits non-zero (and writes no usable manifest) if any evaluation step fails or if any of the 8 integration/safety/performance gates is not PASS, and it discards earlier reports first. `enable` re-verifies, from a manifest with checksums, that the report, artifact and 93-play art bank are the untampered product of that one `prepare` of YOUR database; a model shown to be *worse* is refused, `NOT EVALUATED` or a non-significant lift is allowed.**

## 2. Enable (explicit, CPU opponents only, reversible)
```bash
python -m cfb_coach ml experimental --retrain            # only if no experimental offense artifact exists yet (writes the artifact + mode)
python scripts/madden_ml/cpu_experiment.py enable --artifact ~/cfb_cpu_experiment/readiness/offense_route_aware.json --work ~/cfb_cpu_experiment --i-approve-cpu-experiment
python scripts/madden_ml/cpu_experiment.py status       # verify: ranker route_aware, artifact loads, playbook coverage, fallbacks, adjustments
```
Want the model to *show* its pick while the old coordinator still calls? Add `--preview` to `enable` for the first few snaps.
`status` prints `applied offense playbook contains N/93 of the plays the model scores` and lists any missing play; the model can only
choose plays in the applied book. To change the book, use the normal prep flow (`python -m cfb_coach play --game madden27 -o cpu` prints it).

## 3. Play
```bash
python -m cfb_coach play --game madden27 -o cpu         # the normal HTML window
```
Follow the recommended play. After each snap type the outcome and, in the existing "they ran" field, the defense as you see it
(any name; it is stored exactly). Opponent offensive play names you type on defense are also stored exactly. Between snaps the model
re-reads the coverages you reported in this game. If anything technical fails (missing/stale artifact, over 150 ms, exception) that snap
uses the existing coordinator and the reason is shown and logged.

Check at any time: `python scripts/madden_ml/cpu_experiment.py status` (fallback list, p95 latency, custom adjustments created).
Expect "custom adjustments: 0": hot-route adjustments cannot be scored yet (no first-party route assignments), so every card says NO ADJUSTMENT.

## 4. Roll back immediately
```bash
python scripts/madden_ml/cpu_experiment.py rollback           # restores the EXACT saved ranker, artifact and ML mode (e.g. a prior preview setup)
python scripts/madden_ml/cpu_experiment.py rollback --safe    # only force the ranker to 'experimental', touch nothing else
# or, for the current shell only:  unset CFB_COACH_OFFENSE_RANKER
# or restore everything from the backup: copy ~/cfb_cpu_experiment/backup/<file>.db over ~/.cfb-coach/madden27.db (app closed)
```

## 5. After the game
```bash
python scripts/madden_ml/cpu_experiment.py report --out ~/cpu_game_report.json
```
Covers: plays and diversity, yards per play / success rate overall and vs man, zone and blitz looks (by what you reported), exact defensive
calls and (on defense) exact opponent calls logged, defensive results, execution attestations, illegal recommendations (must be 0),
latency p50/p95/max and over-150 ms count, fallbacks with reasons, adjustments, whether the expected coverage moved with your reports,
and route-aware vs existing-coordinator top picks for every snap from the shadow log (`madden_ml_experimental/route_aware_shadow.jsonl`).
One game is descriptive only: the comparison shows *what each would have called*, not how the unplayed call would have gone.

## What is and is not established
* Tested here (synthetic data): the capture of exact names (form -> DB -> next decision), the driver's gates/enable/status/rollback/report,
  legality, latency, and fallback.
* Not evaluated: your real games. `prepare` is what evaluates them.
* Not verified: eleven-defender assignments for any play.
