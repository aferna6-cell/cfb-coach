# Wild Card — defensive ML GO/NO-GO

**Defense experimental playtesting: NO-GO in this repository**

**Offense short-yardage policy: CONDITIONAL GO (code path only; baseline artifact unchanged)**

This does not authorize a merge to `main`. The pull request targets `madden-ml/coordinators`.

A passing test suite and a JSON file are not a live defense coordinator. Promotion requires verified play-specific training in the operator database, an offline compare that does not claim causal effects, and an explicit `--enable` after that compare.

## Audit (deliverable 1)

Logged defensive snaps are candidates. They are not automatically supervised examples.

| Figure | Status |
| --- | --- |
| Issue #75 defensive snaps (established caller) | 45 candidates on that session, not a training count |
| Reported “178 defensive snaps” | Candidates only. This checkout has no franchise database, so 178 cannot be verified here |
| Supervised play-specific in this commit | 0 real human games. Unit tests use synthetic verified rows |
| John vs general | Opponent-specific tables are stored separately and capped. John is not the general prior |

`python3 -m cfb_coach ml audit-defense --db "$CFB_COACH_MADDEN_DB"` prints candidate vs supervised counts by game and exclusion reasons (CPU, unverified execution, unlabeled outcome, missing play, unrecognized family). Uncertain executions stay out of play-specific training.

### Wild Card interceptions

Issue #75 recorded **3** offensive interceptions among 41 logged snaps. The user reported **4**. Eighteen decisions had no snap (17 superseded). The last logged snap is 42-42 against a session final of 45-42. A fourth INT is **not** reconstructed. Attribution stays **unknown**. An interception is not assumed to be a bad playcall.

## Artifact (deliverable 2)

- Schema remains `madden.defense.coordinator.v1` so `load_model` still fail-closes unknown files.
- New payload: `artifact_version=madden.defense.shrinkage.v1`, `model_kind=hierarchical_shrinkage.v1`, `objective=situational_defense_utility.v1`.
- Mode key `ml_defense_coordinator_mode.v1` and model key `ml_defense_coordinator_artifact.v1` stay independent of the offense artifact.
- Train: `python3 -m cfb_coach ml defense-experimental --retrain --db "$CFB_COACH_MADDEN_DB" --out /path/defense_coordinator.json`
- `--enable` does **not** auto-train. Zero verified snaps refuse enable and leave the heuristic active.

The objective penalizes touchdowns and 15+/40-yard explosives harder than a routine conversion. Personnel and offensive formation enter only when observed pre-snap. Post-snap concepts train historical tendencies only.

## Offline eval (deliverable 3)

`python3 -m cfb_coach ml compare-defense --db "$CFB_COACH_MADDEN_DB" --model /path/defense_coordinator.json`

Held-out when two or more games exist. Limitations are printed: observational labels, no causal claims on unexecuted plays, prior-driven artifacts cannot promote, latency is offline rank time. Hybrid still requires 40 verified snaps.

In this checkout `promote_ok` is false.

## Short-yardage offense (deliverable 4)

The experimental sampler was run-heavy on 3rd/4th-and-short because only runs got a +0.025 selection prior (deep −0.035). Legitimate quick passes, play-action, and RPOs now receive competing conversion priors. No run/pass quota and no forced alternation. The heuristic `SIT_BONUS` (0.12) is unchanged. Optional conversion buckets on a **new** offensive artifact are backward-compatible: existing `offense.json` files without those keys rank as before on early downs.

Do not replace the working offensive artifact unless a challenger passes regression. This branch does not write `offense.json`.

## Live integration (deliverable 5)

- 150 ms budget; timeout/exception/illegal play → established caller, labeled fallback (`fallback_source=select_defense`).
- Experimental defense with a missing file or `supervised_defensive_snaps < 1` → heuristic.
- Unarmed / already-armed researched macros are not jointly replaced; the established call is kept (`CIRCUIT_OPEN`).
- Clock: last entered `game_clock` carries forward until a new value is supplied. Drive IDs stay NULL unless explicit. `end_game` keeps an explicit final score (for example 45-42) instead of overwriting it with the last snap.
- Outcome JSON stores `kind` (including `int`) with `attribution: unknown` unless the operator set an explicit attribution.

## Launch and rollback (deliverable 6)

Prerequisite: a physically confirmed defensive playbook and `audit-defense` showing `supervised_play_specific >= 1`.

```bash
# 1. Audit (read-only)
python3 -m cfb_coach ml audit-defense --db "$CFB_COACH_MADDEN_DB"

# 2. Train a defense artifact (does not enable live mode, does not touch offense)
python3 -m cfb_coach ml defense-experimental --retrain \
  --db "$CFB_COACH_MADDEN_DB" \
  --out "$PWD/defense_coordinator.json"

# 3. Offline compare; exit 2 means NO-GO
python3 -m cfb_coach ml compare-defense --db "$CFB_COACH_MADDEN_DB" \
  --model "$PWD/defense_coordinator.json"

# 4. Enable only after compare prints promote_ok true
python3 -m cfb_coach ml defense-experimental --enable --db "$CFB_COACH_MADDEN_DB"

# 5. Live session: defense flag is independent of --ml-human-control
#    python3 -m cfb_coach play --ml-defense-control ...
```

Rollback (next snap; does not delete snaps, decisions, or the session):

```bash
python3 -m cfb_coach ml defense-experimental --disable --db "$CFB_COACH_MADDEN_DB"
```

Offense rollback remains `python3 -m cfb_coach ml heuristic`.

`--enable` with no artifact or zero verified snaps exits 2, prints `refused: true`, and leaves the heuristic on.

## Verdict

| Gate | Result |
| --- | --- |
| Real verified defensive training in this repo | Missing → **NO-GO** for experimental defensive playtesting |
| Artifact + reproducible train path | Present; independent of offense |
| Offline eval with limitations | Present; will not promote a prior-driven file |
| Short-yardage competing passes | Present in selection policy; baseline offense file not replaced |
| Live fallback, legality, logging, macros | Covered by tests |
| Auto-live on `--enable` | Refused |

Playtest defense ML only after `audit-defense` and `compare-defense` succeed on the operator database.
