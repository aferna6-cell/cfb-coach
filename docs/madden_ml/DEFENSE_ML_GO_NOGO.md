# Defensive ML + short-yardage — GO / NO-GO

**Experimental defense playtesting: NO-GO until the laptop audit prints at least 10 verified human defensive executions.**

**Short-yardage offense policy: CONDITIONAL GO** against the existing experimental offense artifact. Do not retrain or replace `offense.json`.

This document does not authorize a merge to `main`. The pull request targets `madden-ml/coordinators`.

Issue #75 (Wild Card vs John, build `8ddbc29`, loss 45–42). Offense ran 100% experimental with 0 fallbacks. Defense used the established caller.

## What was audited

No franchise database and no real game rows are in this repository. The reported **178 defensive snaps are candidates**, not 178 supervised examples. `ml audit-defense` is the laptop command that prints verified counts **by game** and every exclusion reason.

| Source | Supervised defensive executions | Notes |
| --- | --- | --- |
| This repository | 0 | Nothing to train on. A model file trained here would be prior-driven and must not be enabled. |
| Issue #75 session | 45 defensive snaps logged, **0% experimental**, execution/accepted unknown on all | Candidates only. `drive_id` empty; `game_clock` mostly empty. |
| User-reported 178 | Not in git | Run `ml audit-defense --db "$CFB_COACH_MADDEN_DB"` on the franchise laptop. |

Uncertain executions are excluded from play-specific training. CPU rows, recommendation-only rows, and unverified `executed_status` never enter `train_defense`.

## Interceptions (do not invent a snap)

| Count | Origin |
| --- | --- |
| 3 | Issue #75 logged offensive interceptions |
| 4 | User report |
| 1 | Gap. Last logged snap 42–42; session final 45–42. Eighteen of 59 offensive decisions had no snap. |

This branch does **not** invent a fourth interception row. Attribution of any INT to playcall versus a read/execution mistake stays **unknown** unless a verified note exists.

## Independent defensive artifact

- Path: `{db.parent}/madden_ml_experimental/defense_coordinator.json` (operator `--out` on train)
- Meta: `ml_defense_coordinator_artifact.v1`, mode `ml_defense_coordinator_mode.v1`
- Version: `madden-ml.defense.coordinator.{n}.{quality}`
- Offense `offense.json` / `ml_mode` are never written by defense commands
- Hierarchical shrinkage on verified executions: family, down/distance, field, clock, score, personnel (when observed), coverage vs pressure
- Objective: stop rate + conversion prevention + explosive (≥15) + house (≥40) + TD. A 40-yard touchdown outweighs a routine first down
- Opponent tables shrink toward the general prior (`OPP_PRIOR_STRENGTH = 12`). John does not overwrite general knowledge
- Post-snap `concept_seen` is historical tendency only. `concept_source=last` is not a live look

## Promotion and fallback

`ml train-defense` writes `defense_coordinator.json` and records that path in `ml_defense_coordinator_artifact.v1`. It does **not** set experimental mode.

`ml defense-experimental --enable` **does not train**. `--enable --retrain` together is refused. Enable refuses (exit 2) unless the installed artifact has at least **10** verified defensive executions and `evidence_quality` is not `prior_driven`. The heuristic caller stays active.

Live path (150 ms budget): insufficient verified training, timeout, exception, illegal play → established `select_defense`, labeled `HEURISTIC`, `fallback_source=select_defense`. An unarmed or unverified macro is dropped; the in-book play remains. If the established caller already armed a physically created macro, that snap stays on the heuristic (circuit open) rather than swapping the play under an uncalibrated overlay.

## Short-yardage offense

The Wild Card user saw 3rd/4th-and-short run-heavy. The pool already contained passes; ranking gave runs a conversion bonus and gave conversion passes nothing.

This branch:

- Keeps every zone-fit short-yardage play, including PA, RPO, Stick, slant, smash, and screens
- Gives conversion passes the same situational bonus as runs (`+0.12` pool / `+0.025` selection). Deep shots stay penalized
- Applies recent-run **exposure** cost on 3rd/4th-and-short. No run/pass quota and no forced alternation
- Leaves 3rd/4th-and-long as pass-only when a pass exists
- Does **not** retrain `offense.json`. Challenger policy is evaluated against the existing artifact

## Live logging fixes (future snaps)

Historical Wild Card rows stay empty; we do not backfill invented drives.

Going forward:

- Session sequential `D{n}` on possession change when the operator does not type a drive id
- HTML drive box writes `drive id N` into the situation line
- Clock from the HTML field **or** from a parsed `clock m:ss` on the sit
- `end_game` records `score_gap:{session}` when the final score differs from the last snap, without inventing a play
- Unarmed defense macros are not shown and not suggested

## Offline eval limitations

`ml eval-defense` is observational. Family match and score of the **executed** play are not proof that an unchosen look would have stopped the conversion. Small-n CIs are not a causal Wild Card claim.

## Launch and rollback

Absolute paths only. Do not pass `--retrain` on enable. Do not merge to `main`.

```bash
: "${CFB_COACH_MADDEN_DB:?export CFB_COACH_MADDEN_DB first}"
python3 -m cfb_coach ml backup-db --db "$CFB_COACH_MADDEN_DB" --out /ABSOLUTE/PATH/backups

# 1. Audit. Exit 2 means zero verified defensive executions — stay heuristic.
python3 -m cfb_coach ml audit-defense --db "$CFB_COACH_MADDEN_DB"

# 2. Train a defense-only file. Records the path in the DB. Does not enable live mode. Does not touch offense.json.
python3 -m cfb_coach ml train-defense --db "$CFB_COACH_MADDEN_DB" --out /ABSOLUTE/PATH/defense_coordinator.json

# 3. Observational eval (not a causal claim).
python3 -m cfb_coach ml eval-defense --db "$CFB_COACH_MADDEN_DB" --model /ABSOLUTE/PATH/defense_coordinator.json

# 4. Enable only if the artifact has ≥10 verified executions. Refuse leaves heuristic on.
python3 -m cfb_coach ml defense-experimental --enable --db "$CFB_COACH_MADDEN_DB"

# 5. Play. Defense ML needs the session flag. Offense keeps the existing artifact.
python3 -m cfb_coach play --game madden27 -o john --no-vod-prior \
  --ml-human-control --ml-defense-control --db "$CFB_COACH_MADDEN_DB"
```

Rollback (next snap, history kept):

```bash
python3 -m cfb_coach ml defense-experimental --disable --db "$CFB_COACH_MADDEN_DB"
python3 -m cfb_coach ml heuristic   # offense only; does not change defense mode
```

If `audit-defense` or `train-defense` reports fewer than 10 verified defensive executions, **do not enable**. Play heuristic defense.

## Verdict

| Gate | Verdict |
| --- | --- |
| Code path, 150 ms fallback, legality, unarmed-macro drop, independent versioning | GO for further testing |
| Enable experimental defense on the franchise laptop | **NO-GO** until `ml audit-defense` shows ≥10 verified human defensive executions by game |
| Replace working offensive coordinator / retrain `offense.json` | **NO-GO** |
| Short-yardage conversion-pass competition (existing offense artifact) | CONDITIONAL GO |
| Claim the Wild Card would have been won with these calls | **NO-GO** (unexecuted counterfactuals) |
