# Wild Card John — GO / NO-GO

**Verdict: CONDITIONAL GO**

Issue #72. Branch `madden-ml/wildcard-john`, cut from `madden-ml/coordinators` at `c5e1ebec5cbfc3384a7fbdc7a20aa8df160b8337` (merge of #71). This document does not authorize a merge to `main`.

Heuristic offense and the established defense caller are GO for a human game against John. Experimental defense is NO-GO. Experimental offense is an opt-in pilot behind `--ml-human-control` only. It is not claimed playtested. A human must confirm each call: `accepted` stays `unknown` until verify is touched.

## What is gated

John is a Madden-only cold start. Exact ids `john` and `John` resolve. `jo` and `johnny` do not. He is not in the CFB seed, not in the Madden seed, and not an alias of Gavin, Tiano, or any other persona. `register_cold_start` refuses `gavin` and `auburn`. His profile starts at archetype `unknown`, confidence `none`, empty offense, empty coverages, no NFL team, and no tendency rows. CFB coverages for Gavin (`split_field_zone`) and Tiano (`c2_c3_mixer`) are not copied onto him. In-game learning uses the existing `their_coverage` bump and updates only John.

The product default stays heuristic. `ML_LATENCY_BUDGET_MS` stays 150. A model latency over 150 ms, or a timeout, must fall back to the heuristic. That fallback is unchanged.

## Evidence

Synthetic data only. No franchise database and no real game rows are in this commit.

| Check | Result |
| --- | --- |
| John priors neutral, not shared | pass (`tests/test_wildcard_john.py`) |
| In-game learning updates only John | pass |
| `prep -o john` and `play -o john` | pass, with `--game madden27` |
| Preflight without `--no-vod-prior` | exit 2, `no_vod_prior` FAIL |
| Preflight with `--no-vod-prior` | exit 0, `verdict: PASS` |
| Offense without `--ml-human-control` | heuristic only, even if the DB mode is experimental |
| Offense with `--ml-human-control` and mode still heuristic | exit 2, `requires experimental mode` |
| Offense with the flag and experimental mode | banner requires a human confirm on each call |
| Experimental defense vs John | not called; Gavin's path still calls it |
| Preflight while defense mode is experimental | FAIL on `defense_mode_off`; sample defense call stays the established caller |
| Invented, unarmed, DRAFT, and INSTALLED-but-not-armed adjustments | rejected |
| Three synthetic games vs John | sessions closed; `decision_json.macro` is `O-RUN` when a catalog macro qualifies; `orphan_reason` is `superseded` and `no_result`; `accepted` is `unknown` unless verify is touched; `drive_id` is set only from an explicit `drive id` |

Collected test ids: **902** (baseline 894, plus 8). None of the baseline ids were removed.

`python3 -m pytest -q` was run twice on this branch:

| Run | Result |
| --- | --- |
| 1 | 901 passed, 1 skipped, 71.49s |
| 2 | 901 passed, 1 skipped, 68.93s |

The skip is `tests/madden_ml/test_browser_idempotency_playwright.py` (`playwright not installed`).

### Latency (three synthetic games, 27 experimental decisions)

Measured by `tests/madden_ml/test_wildcard_john_games.py` on this branch. Wall time is the live `call_only` / `result_and_call` path. Model time is `ml_decisions.latency_ms_model`. Budget is 150 ms. Timeout count is 0. Every decision whose model time would exceed 150 ms must have `fell_back`; none did.

| Clock | p50 | p95 | max |
| --- | --- | --- | --- |
| Wall ms | 11.475 | 19.125 | 19.894 |
| Model ms | 2.518 | 3.226 | 3.472 |

Timeouts: **0**.

## Laptop commands, in order

Run these on the machine that holds the real Madden database, after this pull request is merged into `madden-ml/coordinators`. Do not check out `main`. Paths for `--db` and `--out` must be absolute. A leading `~` is rejected by `ml backup-db`. Do not pass `--opp-team`. John's NFL team is unknown. Do not pass `--retrain`. That flag overwrites `offense.json` with no eval gate.

Use one absolute database for every step that can take it. Copy the `madden_db:` line from `ml find-db` into `CFB_COACH_MADDEN_DB`. Pass that same path as `--db` on every command that accepts the flag: `ml backup-db`, `preflight`, `ml defense-experimental --disable`, and `play`. `prep`, `ml experimental`, `ml heuristic`, and `ml defense-experimental` with no write flag have no `--db`. They open `CFB_COACH_MADDEN_DB` (else `madden_db_path()`). `ml experimental --db` is an unrecognized argument and exits 2.

`ml backup-db` already exists on `madden-ml/coordinators` at `c5e1ebe`, so the backup can run before the fetch. `play --ml-human-control` exits 2 until experimental mode is on, and only when the existing artifact is present. `ml defense-experimental` with no flags is read-only.

```bash
# 1. Back up the live DB first. Source is opened read-only.
python3 -m cfb_coach ml find-db
export CFB_COACH_MADDEN_DB=/ABSOLUTE/PATH/madden27.db
python3 -m cfb_coach ml backup-db --db "$CFB_COACH_MADDEN_DB" --out /ABSOLUTE/PATH/backups

# 2. Coordinators at the merged head. Never main.
git fetch origin madden-ml/coordinators
git checkout madden-ml/coordinators
git pull origin madden-ml/coordinators

# 3. Install. Console script is cfb-coach; the module form is the same entry point.
python3 -m pip install -e .

# 4. Human preflight. --no-vod-prior also freezes the book (book_frozen is true when the prior is off).
python3 -m cfb_coach preflight --game madden27 --mode human -o john --no-vod-prior --db "$CFB_COACH_MADDEN_DB"

# 4b. If that preflight fails on defense_mode_off, turn the experimental defense caller off, then rerun step 4.
#     The status line has no --db; it reads CFB_COACH_MADDEN_DB. --disable requires --db.
python3 -m cfb_coach ml defense-experimental
python3 -m cfb_coach ml defense-experimental --disable --db "$CFB_COACH_MADDEN_DB"

# 5. Prep. No --db flag; CFB_COACH_MADDEN_DB is the file.
python3 -m cfb_coach prep --game madden27 -o john --franchise primary --no-vod-prior
# If the printed book says PENDING, build those sheet changes in Madden first.
# Run --mark-applied only after those changes are actually made. The flag only records that they were applied.
python3 -m cfb_coach prep --game madden27 -o john --franchise primary --no-vod-prior --mark-applied

# 6. Confirm the existing experimental artifact. This is the path ml experimental writes
#    when the file is missing (data_dir()/madden_ml_experimental/offense.json).
#    find-db prints data_dir. The default data_dir is ~/.cfb-coach.
ls "$(python3 -c 'from cfb_coach.games import data_dir; print(data_dir() / "madden_ml_experimental" / "offense.json")')"
```

If that `ls` fails, the artifact is missing. Do not run `ml experimental`. With no artifact it trains and overwrites `offense.json` even without `--retrain`. Fall back to heuristic offense: leave the mode heuristic and play without `--ml-human-control`.

```bash
python3 -m cfb_coach ml heuristic
python3 -m cfb_coach play --game madden27 -o john --franchise primary --no-vod-prior --db "$CFB_COACH_MADDEN_DB"
```

If that `ls` succeeds, enable the existing artifact. No `--retrain`. If `play` later prints `Run python -m cfb_coach ml experimental --retrain first`, ignore that hint. Do not add `--retrain`.

```bash
python3 -m cfb_coach ml experimental
python3 -m cfb_coach play --game madden27 -o john --franchise primary --ml-human-control --no-vod-prior --db "$CFB_COACH_MADDEN_DB"
```

Confirm each call in the live window. `accepted` stays `unknown` until verify is touched. `drive_id` stays NULL unless the situation text contains an explicit drive id. Defense stays the established caller. Do not pass `ml defense-experimental --enable`. To leave the pilot, stop the session and run `python3 -m cfb_coach ml heuristic` with the same `CFB_COACH_MADDEN_DB`.

## Out of scope

Do not enable experimental defense. Do not activate unreviewed VOD. Do not treat v0.23/v6 VOD as the CPU-tested experimental offense model. The hybrid defense gate stays at 40 verified snaps. CPU remains offense-only.
