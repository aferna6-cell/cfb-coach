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

Collected test ids after these tests: **902** (baseline 894, plus 8). The Playwright browser test remains a skip when Playwright is not installed. The two full-suite runs are recorded in the pull request after they finish. None of the baseline ids were removed.

### Latency (three synthetic games, 27 experimental decisions)

Measured by `tests/madden_ml/test_wildcard_john_games.py` on this branch. Wall time is the live `call_only` / `result_and_call` path. Model time is `ml_decisions.latency_ms_model`. Budget is 150 ms. Timeout count is 0. Every decision whose model time would exceed 150 ms must have `fell_back`; none did.

| Clock | p50 | p95 | max |
| --- | --- | --- | --- |
| Wall ms | 11.475 | 19.125 | 19.894 |
| Model ms | 2.518 | 3.226 | 3.472 |

Timeouts: **0**.

## Laptop commands, in order

Run these on the machine that holds the real Madden database, after this pull request is merged into `madden-ml/coordinators`. Do not check out `main`. Paths for `--db` and `--out` must be absolute. A leading `~` is rejected. Do not pass `--opp-team`. John's NFL team is unknown.

`ml backup-db` already exists on `madden-ml/coordinators` at `c5e1ebe`, so the backup can run before the fetch. `ml experimental --retrain` is not in the requested list, but `play --ml-human-control` exits 2 until experimental mode is on. It is inserted after prep and before play. `ml defense-experimental` with no flags is read-only.

```bash
# 1. Back up the live DB first. Source is opened read-only.
python3 -m cfb_coach ml find-db
python3 -m cfb_coach ml backup-db --db /ABSOLUTE/PATH/madden27.db --out /ABSOLUTE/PATH/backups

# 2. Coordinators at the merged head. Never main.
git fetch origin madden-ml/coordinators
git checkout madden-ml/coordinators
git pull origin madden-ml/coordinators

# 3. Install. Console script is cfb-coach; the module form is the same entry point.
python3 -m pip install -e .

# 4. Human preflight. --no-vod-prior also freezes the book (book_frozen is true when the prior is off).
python3 -m cfb_coach preflight --game madden27 --mode human -o john --no-vod-prior

# 4b. If that preflight fails on defense_mode_off, turn the experimental defense caller off, then rerun step 4.
python3 -m cfb_coach ml defense-experimental
python3 -m cfb_coach ml defense-experimental --disable --db /ABSOLUTE/PATH/madden27.db

# 5. Prep. If the printed book says PENDING, run the same command again with --mark-applied before play.
python3 -m cfb_coach prep --game madden27 -o john --franchise primary --no-vod-prior

# 6. Opt in to experimental offense. This is the only mode --ml-human-control will start.
#    Heuristic remains the code default. Rollback for the next call: python3 -m cfb_coach ml heuristic
python3 -m cfb_coach ml experimental --retrain

# 7. Play. Offense: experimental, this session only. Defense: established caller (do not pass --enable).
python3 -m cfb_coach play --game madden27 -o john --franchise primary --ml-human-control --no-vod-prior
```

Confirm each call in the live window. `accepted` stays `unknown` until verify is touched. `drive_id` stays NULL unless the situation text contains an explicit drive id. To leave the pilot, stop the session and run `python3 -m cfb_coach ml heuristic`.

## Out of scope

Do not enable experimental defense. Do not activate unreviewed VOD. Do not treat v0.23/v6 VOD as the CPU-tested experimental offense model. The hybrid defense gate stays at 40 verified snaps. CPU remains offense-only.
