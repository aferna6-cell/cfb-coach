# Wild Card John — dual coordinators

**Offense: CONDITIONAL GO**

**Defense: NO-GO**

This does not authorize a merge to `main`. The pull request targets `madden-ml/coordinators`.

Base: `madden-ml/coordinators` at `510ba67f079bdd62db9e3e3d9a07c9297ded810a`. Branch: `madden-ml/wildcard-dual-ml`. The two full-suite runs below were executed on this branch's working tree immediately before the commit that adds this file. No franchise database and no real game rows are in the commit.

John stays a Madden-only cold start (`unknown` / `none`, empty offense, empty coverages, no borrowed film). He learns from observations logged against his id. `accepted` stays `unknown` until verify is touched. `ML_LATENCY_BUDGET_MS` stays 150.

## Audit

Checked on `510ba67` before the edits in this branch.

| Capability | Offense | Defense |
| --- | --- | --- |
| Human-game primary control | Present. `--ml-human-control` plus persistent `ml experimental`. Without the session flag, a human stays on the heuristic even when the DB mode is experimental. | Missing for John. `maybe_apply_defense` runs for other human opponents when `ml defense-experimental` is on, including with no trained file. `make_call` skipped that path for a cold start, so John always kept the established caller. |
| Independent mode | `ml experimental` / `ml heuristic` write the offense mode only. | `ml defense-experimental --enable/--disable` writes `ml_defense_coordinator_mode.v1` only. |
| Automatic fallback | Timeout, exception, and an illegal candidate already returned the heuristic call. A missing offense artifact trained and wrote `offense.json` from the live call. The HTML line still said "ML experimental" on a fallback. | Timeout and exception already returned the established caller for the non-cold-start path. A missing file did not fall back; the prior ranker still picked. John never entered the path. |

The two mode keys already failed separately. What John lacked was a defense session switch, a trained-artifact gate that does not fit a model, truthful per-side labels, and a live-path fallback that does not train.

## What this branch adds

- `--ml-defense-control` is the defense session switch. It does not change the offense mode. `--ml-human-control` does not change the defense mode.
- John reaches experimental defense only when that flag is on, the defense mode is `experimental`, and the installed artifact has `supervised_defensive_snaps` greater than 0. Otherwise the established caller is shown and labeled `HEURISTIC`.
- A missing offense artifact, a timeout, an exception, an illegal play, or an invalid adjustment returns the heuristic call immediately. The decision records `fallback_reason` and `fallback_source` (`heuristic_offense` or `select_defense`). The live call does not train either side.
- A macro is shown only when it is installed, armed, and executable (catalog loadout, or a defense lab sheet that is `verified_armed`). An invented, draft, or unarmed sheet fails the call closed to the heuristic.
- Every `make_call` line ends with `LABEL: EXPERIMENTAL` or `LABEL: HEURISTIC`. A fallback is never labeled experimental.
- `ml heuristic` rolls offense back on the next snap. `ml defense-experimental --disable --db "$CFB_COACH_MADDEN_DB"` rolls defense back. Neither deletes snaps, `ml_decisions`, or the open session. Final score and session close still go through `end_game`.

Gavin's existing defense path is unchanged when `--ml-defense-control` is absent. Preflight still fails while the defense mode is experimental, which matches the defense NO-GO.

## Validation

Isolated SQLite database. Books are the trimmed Detroit Lions primary books from `playbooks.json` (`source_book` Lions, mode custom): offense Gun Doubles Clamp Stack, I Form Y Off Close, Singleback Deuce Close, Maryland I Close (66 plays); defense 4-3 Over, Nickel 3-3 Over, Dime 2-3, Dollar 3-2 Mug (108 plays). 24 situations per side: formation changes, coverage looks, 4th down, red zone, goal line, two-minute and clock, and unfamiliar shells (`bear front`, `some unfamiliar shell`). No `--retrain`. `train_defense` was not called during the 48 calls.

### Artifacts

| Side | Path | sha256 | Training rows | Used for the calls below |
| --- | --- | --- | --- | --- |
| Offense | temporary `offense.json` inside the test directory (not committed). Launch path on the laptop is `data_dir()/madden_ml_experimental/offense.json`. | `cf249f8f404070762fa82f77b6b97cb4efe637f2f5a02128e95db60db1b6c044` | 0 (`train_experimental([], side="offense")`, prior-driven). `trained_at` is inside the file, so a later training pass will not repeat this digest. | Yes. 24 model picks. |
| Defense | none. `ml_defense_coordinator_artifact.v1` was unset. | none | 0 | No. Not activated and not trained. |

There is no defense artifact with real verified training in this repository. A one-row synthetic file is used only by the unit test that checks timeout, exception, and illegal-play fallback. It is not the artifact in the table and it is not installed for the 24 defense situations.

### Latency

Budget 150 ms. Timeouts in the 48-situation run: 0. Offense model time is `CoachingDecision.latency_ms_model`. Defense model time is 0 because every snap returned before a model rank, with `fallback_reason=model_missing`.

| Clock | p50 | p95 | max |
| --- | --- | --- | --- |
| Offense wall ms | 16.773 | 28.685 | 30.586 |
| Offense model ms | 12.736 | 16.222 | 26.004 |
| Defense wall ms | 1.692 | 1.875 | 2.135 |
| Defense model ms | 0 | 0 | 0 |

Offense fallback rate: **0 / 24**. Defense fallback rate: **24 / 24**. Fallback source on every defense snap: `select_defense`.

### Offense decisions (all EXPERIMENTAL)

| Situation | Formation | Play | Model ms |
| --- | --- | --- | --- |
| 1&10 my 25 | I Form Y Off Close | DBL Outs | 16.353 |
| 2&6 my 31 showing cover 3 | Singleback Deuce Close | Stick | 15.080 |
| 3&8 my 40 showing cover 2 | I Form Y Off Close | Mtn Flood | 10.999 |
| 1&10 opp 40 showing cover 4 | Gun Doubles Clamp Stack | Flood Seam | 15.385 |
| 2&9 opp 33 showing cover 1 | Maryland I Close | Mesh HB Rail | 26.004 |
| 3&2 opp 28 | Gun Doubles Clamp Stack | Inside Zone | 12.713 |
| 4&1 opp 42 | Singleback Deuce Close | Counter Y | 12.736 |
| 4&3 my 48 | Gun Doubles Clamp Stack | Mtn Shuffle Verts Smash | 12.736 |
| 4&12 opp 45 showing cover 6 | Gun Doubles Clamp Stack | Flood Seam | 11.546 |
| 1&10 opp 18 | I Form Y Off Close | Double Dig | 12.334 |
| 2&7 opp 14 showing cover 3 | Singleback Deuce Close | PA Drag Wheel | 13.988 |
| 3&goal opp 4 | Singleback Deuce Close | Mtn Zone Split | 11.988 |
| 1&goal opp 2 showing cover 0 | Maryland I Close | Mtn PA Sprint HB Flat | 14.738 |
| 2&8 opp 22 two minute clock 1:20 | Singleback Deuce Close | Stick | 12.717 |
| 3&6 my 35 hurry clock 0:35 | Maryland I Close | Mtn HB Zone WK | 12.643 |
| 1&10 opp 30 showing quarters | Gun Doubles Clamp Stack | Flood Seam | 15.197 |
| 2&5 opp 16 showing tampa 2 | Maryland I Close | Mesh HB Rail | 13.809 |
| 1&10 my 40 showing bear front | I Form Y Off Close | DBL Outs | 12.850 |
| 2&11 opp 37 showing invert cover 2 | Gun Doubles Clamp Stack | Flood Seam | 15.481 |
| 3&1 opp 9 | Maryland I Close | HB Toss | 12.251 |
| 1&15 my 12 clock 9:40 | Singleback Deuce Close | PA Drag Wheel | 11.966 |
| 4&2 opp 11 two minute | Maryland I Close | Mtn HB Zone WK | 12.236 |
| 2&2 opp 6 showing cover 9 palms | Gun Doubles Clamp Stack | HB Mid Zone | 15.127 |
| 3&14 opp 44 showing some unfamiliar shell | I Form Y Off Close | PA X Deep Out | 9.780 |

Formations used: Gun Doubles Clamp Stack, I Form Y Off Close, Maryland I Close, Singleback Deuce Close. Every play is in that formation's Lions list. No macro was attached on these 24 snaps. Armed catalog ids `O-RUN` and `SAFE DEEP` pass `macro_is_live`. An invented id and an unarmed `O-RUN` do not, and that failure returns the heuristic call.

### Defense decisions (all HEURISTIC)

These are the established caller. They are not experimental picks.

| Situation | Formation | Play | Fallback |
| --- | --- | --- | --- |
| d 1&10 my 25 | 4-3 Over | Cover 1 Contain | model_missing / select_defense |
| d 2&6 my 31 showing cover 3 | 4-3 Over | Cover 1 Contain | model_missing / select_defense |
| d 3&8 my 40 showing cover 2 | Nickel 3-3 Over | Bracket Switch Willie | model_missing / select_defense |
| d 1&10 opp 40 showing cover 4 | 4-3 Over | Cover 1 Contain | model_missing / select_defense |
| d 2&9 opp 33 showing cover 1 | 4-3 Over | Cover 1 Contain | model_missing / select_defense |
| d 3&2 opp 28 | 4-3 Over | OLB Fire Man | model_missing / select_defense |
| d 4&1 opp 42 | 4-3 Over | OLB Fire Man | model_missing / select_defense |
| d 4&3 my 48 | 4-3 Over | OLB Fire Man | model_missing / select_defense |
| d 4&12 opp 45 showing cover 6 | Nickel 3-3 Over | Bracket Switch Willie | model_missing / select_defense |
| d 1&10 opp 18 | 4-3 Over | LB Sting I | model_missing / select_defense |
| d 2&7 opp 14 showing cover 3 | 4-3 Over | LB Sting I | model_missing / select_defense |
| d 3&goal opp 4 | 4-3 Over | OLB Fire Man | model_missing / select_defense |
| d 1&goal opp 2 showing cover 0 | 4-3 Over | OLB Fire Man | model_missing / select_defense |
| d 2&8 opp 22 two minute clock 1:20 | Dime 2-3 | Cover 3 Sky | model_missing / select_defense |
| d 3&6 my 35 hurry clock 0:35 | Dime 2-3 | Cover 3 Sky | model_missing / select_defense |
| d 1&10 opp 30 showing quarters | 4-3 Over | Cover 1 Contain | model_missing / select_defense |
| d 2&5 opp 16 showing tampa 2 | 4-3 Over | LB Sting I | model_missing / select_defense |
| d 1&10 my 40 showing bear front | 4-3 Over | Cover 1 Contain | model_missing / select_defense |
| d 2&11 opp 37 showing invert cover 2 | 4-3 Over | Cover 1 Contain | model_missing / select_defense |
| d 3&1 opp 9 | 4-3 Over | LB Sting I | model_missing / select_defense |
| d 1&15 my 12 clock 9:40 | 4-3 Over | Cover 1 Contain | model_missing / select_defense |
| d 4&2 opp 11 two minute | 4-3 Over | LB Sting I | model_missing / select_defense |
| d 2&2 opp 6 showing cover 9 palms | 4-3 Over | LB Sting I | model_missing / select_defense |
| d 3&14 opp 44 showing some unfamiliar shell | Nickel 3-3 Over | Bracket Switch Willie | model_missing / select_defense |

Formations used: 4-3 Over, Nickel 3-3 Over, Dime 2-3. Every play is in the Lions defense book. None of these lines contain `DEFENSE ML` or `LABEL: EXPERIMENTAL`.

## Verdicts

**Offense — CONDITIONAL GO.** The human session flag, the 150 ms budget, truthful labels, and the Lions-book guard all held, and all 24 snaps were in-book model picks. The artifact scored here has zero supervised rows. It is prior-driven. It is not a claim that Franchise history trained it, and it is not playtested. Use it only when `offense.json` is already installed. Confirm each call. If that file is missing, leave the mode heuristic and do not start the pilot.

**Defense — NO-GO.** No valid defense artifact with real training is installed. The coordinator was not activated and was not trained on zero snaps to produce a passing number. With the defense flag on and the mode experimental, all 24 snaps fell back to the established caller and were labeled `HEURISTIC`. Do not pass `--ml-defense-control`. Do not run `ml defense-experimental --enable`.

## Regression

Collected **909** tests (baseline 902, plus 7 in `tests/madden_ml/test_wildcard_dual_ml.py`). None of the baseline ids were removed. The skip is `tests/madden_ml/test_browser_idempotency_playwright.py` (`playwright not installed`).

`python3 -m pytest -q` twice on this branch:

| Run | Result |
| --- | --- |
| 1 | 908 passed, 1 skipped, 73.05s |
| 2 | 908 passed, 1 skipped, 73.07s |

## Laptop commands

One shell. Absolute paths only. A leading `~` is rejected by `ml backup-db`. Do not pass `--opp-team`. Do not pass `--retrain`. Export `CFB_COACH_MADDEN_DB` before the backup and use that same path everywhere. `ml backup-db`, `preflight`, `ml defense-experimental --disable`, and `play` take `--db`. `ml experimental`, `ml heuristic`, and `ml defense-experimental` with no write flag read `CFB_COACH_MADDEN_DB`.

`tests/madden_ml/test_wildcard_dual_ml.py::DualCliTests::test_commands_exist_in_one_shell_without_retrain` runs this sequence against an isolated database: backup, defense status, defense disable, preflight PASS, `ml experimental` with an already installed offense file, one experimental offense snap, `ml heuristic`, a refused `--ml-human-control` start, a heuristic offense snap, a defense snap that stays `HEURISTIC` when no trained artifact exists, defense disable, and a refused `--ml-defense-control` start. `ml_decisions` from the experimental snap were still present afterward.

```bash
export CFB_COACH_MADDEN_DB=/ABSOLUTE/PATH/madden27.db
python3 -m cfb_coach ml backup-db --db "$CFB_COACH_MADDEN_DB" --out /ABSOLUTE/PATH/backups

git fetch origin madden-ml/coordinators
git checkout madden-ml/coordinators
git pull origin madden-ml/coordinators
python3 -m pip install -e .

python3 -m cfb_coach ml defense-experimental
python3 -m cfb_coach ml defense-experimental --disable --db "$CFB_COACH_MADDEN_DB"
python3 -m cfb_coach preflight --game madden27 --mode human -o john --no-vod-prior --db "$CFB_COACH_MADDEN_DB"

python3 -m cfb_coach prep --game madden27 -o john --franchise primary --no-vod-prior
# If the printed book says PENDING, build those sheet changes in Madden first.
# --mark-applied only records that those changes were made.
python3 -m cfb_coach prep --game madden27 -o john --franchise primary --no-vod-prior --mark-applied
```

If the offense artifact is missing, stay heuristic. Do not run `ml experimental` in that case: with no file, that command trains.

```bash
python3 -m cfb_coach ml heuristic
python3 -m cfb_coach play --game madden27 -o john --franchise primary --no-vod-prior --db "$CFB_COACH_MADDEN_DB"
```

If `offense.json` is already installed:

```bash
ls "$(python3 -c 'from cfb_coach.games import data_dir; print(data_dir() / "madden_ml_experimental" / "offense.json")')"
python3 -m cfb_coach ml experimental
python3 -m cfb_coach play --game madden27 -o john --franchise primary --ml-human-control --no-vod-prior --db "$CFB_COACH_MADDEN_DB"
```

Defense stays off. Do not add `--ml-defense-control`.

### Per-side rollback

From another terminal, with the same `CFB_COACH_MADDEN_DB`, while the play session is still open. The next snap on that side follows the new mode. The other side is left alone. Snaps, decisions, and the session row stay.

Offense only:

```bash
python3 -m cfb_coach ml heuristic
```

Defense only:

```bash
python3 -m cfb_coach ml defense-experimental --disable --db "$CFB_COACH_MADDEN_DB"
```

To resume offense after a rollback, `python3 -m cfb_coach ml experimental` with the artifact still in place. A new `play --ml-human-control` refuses until that mode is experimental again. The already-open session does not need to be restarted.
