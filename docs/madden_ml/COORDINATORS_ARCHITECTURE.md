# Madden coordinators architecture

Status: integration note, 2026-10-10. This file records the branch plan, the shared contract versions, and one canonical module for each duplicated concept. It does not delete code, change the default heuristic, or enable an experimental model.

The older coach note is `docs/madden_ml/ARCHITECTURE.md`. Where this file names a canonical module, that choice is the one later edits should follow. The modules it does not choose stay in place until a later change migrates their callers.

## Branch and pull-request plan

Base: `cursor/sprint17-defensive-intelligence-four-user-games` (head `5d9ea45` when this branch was cut). That head already contains PR #20 (`madden-ml/coach`), PR #36, PR #37, PR #39, and PR #40. It did not contain PR #38.

Integration branch: `madden-ml/coordinators`, opened as a draft pull request into the sprint 17 branch. Title: `[WIP] Madden coordinators integration (do not merge)`. Do not merge it, do not retarget it at `main`, and do not close PR #20 from this work.

PR #38 is one commit, `3eb8596`, branched from PR #36 head `a6c60f2`. It is on this branch by cherry-pick. `cli.py` and `expert_film.py` auto-merged with no conflict markers. The commit registers remote elite-VOD candidates and an unsent permission draft. It does not download media and it does not add a database file.

Role branches are cut from `madden-ml/coordinators`, not from `main` and not from `madden-ml/coach`:

| Branch | Draft PR target |
| --- | --- |
| `madden-ml/data` | `madden-ml/coordinators` |
| `madden-ml/policy` | `madden-ml/coordinators` |
| `madden-ml/live` | `madden-ml/coordinators` |
| `madden-ml/train` | `madden-ml/coordinators` |
| `madden-ml/qa` | `madden-ml/coordinators` |
| `madden-ml/adjustments` | `madden-ml/coordinators` |
| `madden-ml/vod` | `madden-ml/coordinators` |

Each role PR stays a draft until its own suite is green. Cross-role types land in `cfb_coach/madden/model/schema.py` on this integration branch first. A role branch does not invent a second copy of those types.

`madden-ml/coach` is not a working branch anymore. Its commits are ancestors of sprint 17. New work does not push there.

## Shared contracts and versions

All of these live in `cfb_coach/madden/model/schema.py`. Unknown is `None` in Python and JSON `null`. Every enum has `UNKNOWN`. CSV missing cells stay `__UNKNOWN__`.

| Wire version | Records |
| --- | --- |
| `madden-ml.schema.1` | Stable records (`CONTRACT_VERSION`): game state, observation, outcome, decision support types, registry entry, config |
| `madden-ml.provisional.1` | `PlayRanking`, `OpponentContext` |
| `madden-ml.features.2` | `FeatureVector` / `FeatureValue` (`FEATURE_SCHEMA_VERSION`) |
| `madden-ml.decision.1` | Field `CoachingDecision.decision_schema_version` |
| `madden-ml.macro.1` | `MacroCandidate`, `EditorFieldAssignment`, `AdjustmentPrimitive`, `MacroConflict`, `MacroJointFeatures` |

`schema_version_for` writes the matching version into `__schema__` on `to_dict`. `from_dict` rejects a payload whose `__schema__` is some other version.

### Macro lifecycle

`MacroState`: `DRAFT`, `EDITOR_VERIFIED`, `INSTALLED`, `ARMED`, `ELIGIBLE`, `EVALUATED`, `RETIRED`, `UNKNOWN`.

`is_live_selectable(state)` is true only for `ARMED`, `ELIGIBLE`, and `EVALUATED`. That matches the 2026-10-10 live ruling that a macro is shown only at `ARMED` or later. Installed, editor-verified, draft, retired, and unknown are not live-selectable.

The macro labs still store `draft`, `verified`, and `armed`, plus the status string `DRAFT_NEEDS_IN_GAME_VERIFICATION` (`defense_macro_lab.py`, `offense_macro_lab.py`). `adapt_macro_state` reads those labels: `draft` and the draft status string become `DRAFT`, `verified` becomes `EDITOR_VERIFIED`, `armed` becomes `ARMED`. The old strings stay valid inputs. This commit does not rewrite the lab rows.

### MacroCandidate

`MacroCandidate` is the wire contract for one Custom Adjustment candidate. The Gameplan Coach (`cfb_coach/madden/adjustments/`, not built here) is the producer. This module defines the type, the validation, and the round-trip tests only.

Fields: `macro_id`, `version` (the candidate's own revision, not the wire version), `side` (`OFF` / `DEF` / `UNKNOWN`), `state`, `formation`, `play`, `adjustment_primitives`, `editor_fields`, `tactical_purpose`, `situations`, per-field `provenance` and `confidence` on `EditorFieldAssignment`, `conflicts`, `is_no_adjustment`, and joint-ranking features (`compatibility`, `situation`, `action_count`, `control_reliability`, `empirical_results`, `uncertainty`).

`editor_fields` maps each field name to `{value, source}`. `source` is `researched`, `default`, or `unknown`. A defense candidate that is not the no-adjustment option must name every field in `cfb_coach/data/madden27/editor_fields.json` (51 unique defense fields, sections General, DL/LB, Secondary, Zone Drops, Strategy, Coverage Checks, Individuals). A missing field, an extra field, an unknown source, or an empty value fails validation. A no-adjustment candidate has no editor sheet. A non-empty `conflicts` tuple cannot sit on a live-selectable state. `conflicts is None` means the check was not recorded.

### Defensive propensity log

`defense_coordinator.live_pick` and `maybe_apply_defense` attach a `CoachingDecision` with `propensity_method`, `behavior_propensity`, `propensity_override_reason` when the method is `override`, `policy_source`, `shadow_status`, and `fallback_reason`. The near-tie draw in `_stable_choice` is unchanged: the log computes the weight of the formation and play that draw already returned and does not draw again. The call `maybe_apply_defense` returns is still the deterministic joint sort of the top of the book, so that shown call is logged as `propensity_method=deterministic` and `behavior_propensity=1.0`. Fallback (timeout, exception, empty options, or an already-armed action that the model must not replace) uses `policy_source=heuristic`, `fell_back=true`, and an `MLStatus` other than `ok`.

The decision is persisted with `CoachDB.log_ml_decision` only when `extras["snap_id"]` is set. Mode `off`, a missing database, and a CPU opponent return the incoming call with no decision attached. Default heuristic behavior is unchanged.

## Canonical module for each duplicated concept

Nothing below is deleted. The deprecation path is "new code uses the canonical module; the other modules keep working until their callers are moved."

### Opponent profiles

**Canonical module:** `OpponentContext` in `cfb_coach/madden/model/schema.py` (wire `madden-ml.provisional.1`), keyed by `OpponentContextKey`.

Evidence:

- `policy.rank_plays` takes `opponent_context: OpponentContext | None` (`cfb_coach/madden/model/policy.py`). `features.py` reads the same type. `CoachingDecision.opponent_context_key` stores the key.
- `opponent_learning.fit_opponent_model` is the offense estimator. `offense_joint_decision.py` calls it and then `learning_adjustment`. It keeps its own dict, with `_shrink` and within-game `_change` (`opponent_learning.py`). It does not construct an `OpponentContext`.
- `defense_intelligence.fit_opponent_offense` is the defense estimator. `defense_coordinator.train_defense` is the caller that stores that dict on the defense artifact. It is a separate shape from both `OpponentContext` and the offense learner.
- `cfb_coach/madden/model/opponent.py` is not in the tree. The 2026-10-08 note that the Strategist would build `OpponentContext` there was not implemented. The type that did land is the schema record.

Deprecation path: leave `opponent_learning.py` and `defense_intelligence.fit_opponent_offense` in place. A later change adds a pure adapter from each dict into `OpponentContext` and teaches new readers to take the schema type. Do not merge the two estimators into one function, and do not delete either, until that adapter has tests on both the offense joint path and `train_defense`.

### Defense selection

**Canonical live module:** `select_defense` in `cfb_coach/madden/defense_select.py`.

Evidence:

- `playcaller._select_base_defense` calls `select_defense` for the whole locked defensive book (`cfb_coach/madden/playcaller.py`). That is the call the user gets when the defensive coordinator mode is off.
- `maybe_apply_defense` runs only after that pick, and only when `mode(db) == "experimental"` and the opponent is not CPU (`defense_coordinator.py`). `mode` defaults to `"off"`. On timeout or any other exception it returns the call `select_defense` already made.
- `defense_coordinator.rank_defense` scores the installed book for the experimental path. It is not what `make_call` uses in the default mode. Tests in `tests/madden_ml/test_sprint16_defensive_coordinator.py` cover the experimental selector; the heuristic sampler stays in `defense_select.py`.

Deprecation path: do not delete `defense_select.py` and do not make `experimental` the default. When a model later passes the registry gate, `playcaller` can prefer the coordinator behind the existing opt-in flag. `select_defense` remains the fallback the coordinator already returns on failure. `defense_action_policy.select_defensive_action` stays the adjustment chooser for that experimental path; it is not a second play selector.

### Shadow-model stacks

Three stacks write model artifacts. They do not share a file format.

| Stack | What it writes | Who loads it for a call |
| --- | --- | --- |
| `model/train.py` + `model/registry.py` | `ModelRegistryEntry` plus `model_artifact.json`, with `CURRENT` pointing at one version | `policy.py` and `inference.py` call `train.load_artifact` |
| `expert_policy.py` + `personalization.py` | `policy.json` and `personalization.json` under the learning store | `expert_signal.py` and `learning_reports.py`. `learning_eval.py` fits with `persist=False` so an eval cannot replace `policy.json` |
| `user_game_learning.train_shadow_models` | `offense_four_human_games_shadow.json` via `experimental_model`, and `defense_four_human_games_shadow.json` via `defense_coordinator.save_model` | Nothing on the live path. The function sets `installed: False` and `live_mode_changed: False` |

**Canonical module:** `cfb_coach/madden/model/registry.py`. It is the only store whose `CURRENT` link is allowed to become the model `policy.py` will load. The other two stacks are shadow producers.

Deprecation path: do not delete `expert_policy.py`, `personalization.py`, `experimental_model.py`, or `user_game_learning.py`. A later change may have them also emit a `ModelRegistryEntry` with `gate=not_run` in a directory that is not the live registry root. They must not write `CURRENT` on the registry the heuristic/hybrid path reads, and they must not flip `ml_defense_coordinator_mode` or the offense experimental mode. Promotion stays the registry gate already described in `docs/madden_ml/ARCHITECTURE.md`.

### Decision and outcome log

**Canonical log:** the `ml_decisions` and `ml_outcomes` tables, written by `CoachDB.log_ml_decision` and `CoachDB.log_ml_outcome` in `cfb_coach/db.py`. **Canonical payload:** `CoachingDecision` (and `SnapOutcome` / executed fields) in `schema.py`. These are one log. `CoachingDecision` is not a second store.

Evidence they are the same log, which is why the 2026-10-10 note's preference for the #40 tables applies:

- `log_ml_decision` persists `to_dict(CoachingDecision)` in `ml_decisions.decision_json` and copies mode, policy source, formation, play, and `shadow_status` into columns (`db.py`).
- `dataset.build_rows` joins `snaps` to `ml_decisions` and `ml_outcomes` on `ml_snap_id` (`dataset.py`). `ml four-user-games` calls `dataset.build_rows` and then `user_game_learning` (`model/cli.py`). That is the #40 reader. It does not define another decision schema.
- `ml report` reads the same two tables (`cmd_ml_report`).
- Propensity fields (`propensity_method`, `behavior_propensity`, `propensity_override_reason`, `fallback_reason`) are inside `decision_json`. They are not their own columns. Readers that need them parse the JSON. That projection gap does not make `CoachingDecision` a competing log.

Deprecation path: new writers, including the defensive coordinator, build a `CoachingDecision` and call `log_ml_decision` when a snap id exists. Do not add a third table. Do not stop writing `CoachingDecision` into `decision_json`. A later additive migration may index propensity columns from that JSON; it should not change the row identity (`snap_id`) or the outcome table. Do not treat a logged recommendation as a verified execution: `ml_outcomes.executed_status` stays `unknown` until a confirmation write, which is the same rule `CoachingDecision.executed` already enforces.
