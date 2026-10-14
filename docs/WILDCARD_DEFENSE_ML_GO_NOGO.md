# Wild Card — defensive ML parity and short-yardage offense

**Defense experimental playtesting: NO-GO**

**Short-yardage offense challenger: CONDITIONAL GO (evaluate against immutable offense baseline before promotion)**

Issue #75 follow-up on `madden-ml/coordinators`. This document does not authorize a merge to `main`. Branch: `cursor/defensive-ml-short-yardage-7d53`.

## What changed

1. **Real defensive training-data audit** (`ml audit-defense`): candidate rows vs verified live/VOD counts by game, exclusion reasons, play-specific supervised totals. Aggregates such as “178 defensive snaps” are candidates, not automatic supervised examples.
2. **Independently versioned defense artifact** (`ml train-defense` / `ml eval-defense`): hierarchical shrinkage with stop, expected yards, explosive (≥15), and touchdown risk. Opponent tables stay separate. `auto_promote` is always false. Offense artifact paths and modes are untouched.
3. **Offline model vs heuristic / situation baseline**: match holdout Brier difference with bootstrap CI and p95 latency. Enable refuses unless the report beats baseline with CI entirely above 0 and p95 under 150 ms.
4. **Short-yardage offense**: conversion passes, play-action, and RPOs now receive competing selection priors with runs; mild run-family exposure on 3rd/4th-and-short. No fixed run/pass quotas or forced alternation. Third-and-long run filtering and deep-shot short-yardage penalties are preserved.
5. **Wild Card postgame audit** (`ml wildcard-postgame`): reconciles the user-reported fourth interception with orphan INT decisions (no invented snaps); reports empty `drive_id` / `game_clock` / score gaps; flags unarmed macros. Attribution stays `unknown` unless evidence exists.
6. **Macro legality**: shown macros must be armed and installed (from PR #76 cherry-pick).

## Evidence gates

| Gate | Result |
| --- | --- |
| Verified defensive training present in this environment | **No real franchise DB.** Synthetic fixtures only. |
| `ml defense-experimental --enable` without a passing eval report | Refuse; mode stays off |
| Missing / insufficient verified snaps | Refuse promotion; heuristic remains active |
| Latency budget | 150 ms; timeout/exception → labeled `HEURISTIC` |
| Offense baseline | Existing `offense.json` is not retrained by these commands |
| Causal claims from unexecuted counterfactuals | Not claimed |

## Issue #75 reconciliation

| Finding | Handling |
| --- | --- |
| User: 4 INTs vs issue body: 3 | `ml wildcard-postgame` counts offense-snap INTs and orphan INT decisions/outcomes. A superseded/no-snap INT is a candidate for the missing event. No snap is invented. |
| INT attribution | Remains `unknown` unless supporting evidence exists. Individual reads/execution mistakes are not assumed to be bad playcalls. |
| Empty `drive_id` / `game_clock` | Still only recorded from explicit situation text (`drive id X`, `drive:X`, labeled clock). Not invented. |
| Final 45–42 vs last snap 42–42 | Session `end_game` / stored final score; not a fabricated snap. |
| Unarmed defense macros | Dropped; audit lists any shown-but-unarmed ids. |

## Launch and rollback

Absolute paths only. Export `CFB_COACH_MADDEN_DB` first. Do **not** pass `--retrain` on the offense experimental command. Do **not** enable defense until `ml audit-defense` shows enough verified play-specific snaps **and** `ml eval-defense` prints `gate: pass`.

```bash
export CFB_COACH_MADDEN_DB=/ABSOLUTE/PATH/madden27.db
: "${CFB_COACH_MADDEN_DB:?export CFB_COACH_MADDEN_DB first}"

git fetch origin madden-ml/coordinators
# Work from a PR into coordinators — never main.
git checkout cursor/defensive-ml-short-yardage-7d53   # or the merged coordinators head after review
python3 -m pip install -e .

python3 -m cfb_coach ml backup-db --db "$CFB_COACH_MADDEN_DB" --out /ABSOLUTE/PATH/backups
python3 -m cfb_coach ml audit-defense --db "$CFB_COACH_MADDEN_DB"
python3 -m cfb_coach ml wildcard-postgame --db "$CFB_COACH_MADDEN_DB" \
  --issue-ints 3 --user-ints 4 --final-us 42 --final-them 45
```

### Defense (NO-GO until gates pass)

```bash
: "${CFB_COACH_MADDEN_DB:?export CFB_COACH_MADDEN_DB first}"
python3 -m cfb_coach ml train-defense --db "$CFB_COACH_MADDEN_DB" \
  --out /ABSOLUTE/PATH/madden_ml_experimental/defense_coordinator.json
python3 -m cfb_coach ml eval-defense --db "$CFB_COACH_MADDEN_DB" \
  --model /ABSOLUTE/PATH/madden_ml_experimental/defense_coordinator.json
# Only if gate: pass AND p95 under 150:
# python3 -m cfb_coach ml defense-experimental --enable --db "$CFB_COACH_MADDEN_DB"
# python3 -m cfb_coach play --game madden27 -o john --franchise primary \
#   --ml-defense-control --no-vod-prior --db "$CFB_COACH_MADDEN_DB"
```

Rollback defense instantly (leaves offense alone):

```bash
: "${CFB_COACH_MADDEN_DB:?export CFB_COACH_MADDEN_DB first}"
python3 -m cfb_coach ml defense-experimental --disable --db "$CFB_COACH_MADDEN_DB"
```

### Offense short-yardage challenger

Keep the existing successful `offense.json` as the immutable baseline. The short-yardage policy change is code-path only; it does not overwrite the artifact. Evaluate against that baseline before any promotion. Launch with the installed offense file:

```bash
: "${CFB_COACH_MADDEN_DB:?export CFB_COACH_MADDEN_DB first}"
python3 -m cfb_coach ml experimental          # enables existing offense.json; no train
python3 -m cfb_coach play --game madden27 -o john --franchise primary \
  --ml-human-control --no-vod-prior --db "$CFB_COACH_MADDEN_DB"
```

Rollback offense:

```bash
: "${CFB_COACH_MADDEN_DB:?export CFB_COACH_MADDEN_DB first}"
python3 -m cfb_coach ml heuristic
```

## Verdict

| Side | Verdict | Reason |
| --- | --- | --- |
| Defense experimental | **NO-GO** | No verified franchise training in-repo; enable gate refuses without a passing eval report; heuristic stays default. |
| Defense tooling | Ready for laptop audit/train/eval on the real DB | Does not auto-promote. |
| Offense short-yardage | **CONDITIONAL GO** | Competing short-yardage pass/PA/RPO priors + exposure; baseline artifact untouched; confirm conversion quality in playtest before claiming improvement. |
| Wild Card #75 fields | Audit tooling ready | Fourth INT reconcilable only from logged orphan evidence; never invented. |
