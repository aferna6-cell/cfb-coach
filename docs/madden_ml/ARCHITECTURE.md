# Madden ML coach architecture

Phase 1 adds the shared contracts under `cfb_coach/madden/model/` and does not change live play-calling. `cfb_coach.madden.playcaller.make_call` still returns the deterministic call, then `cfb_coach.vod_model.live.apply_madden_call` may replace the formation and play. The default coaching mode stays heuristic.

## What main does today

Audited at `2def811` (`Use a trained VOD model as the primary call prior`, PR #18).

`make_call` builds one `MaddenCall` from the locked book in `active_playbook_json`, the eight-per-side Custom Adjustment loadout (`LOADOUT_N` is 8), the situation parser, and the opponent profile. Offense and defense each pick inside that book. CPU opponents are offense-only. After the call, the reads, and the macro are already chosen, `apply_madden_call` may overwrite `formation` and `play` when a medium or high VOD cell names an in-book beater. If the armed macro does not contain the new play, the macro is cleared.

That late swap is the problem this package is here to replace. The displayed play can disagree with the reads and the adjustment that were chosen for the previous play. A model score applied the same way would have the same split.

PR #19 (`cursor/vod-play-mappings-d4df`, open, not on this branch) teaches prep to place a mapped formation in the book. It does not change `live.py`. Its stable entry points, once that branch is merged, are `load_mappings` and `lookup`. This package must call those, not copy them, and must not treat a mapped raw VOD name as a concept hint.

## One decision pipeline

The live path, when it is wired later, is one sealed `CoachingDecision`:

1. Parse the live line into the existing `Situation`. Do not copy parser defaults such as `side="offense"` into ML fields that were not observed.
2. `adapters.situation_to_state` is the only builder of `GameState` and `PreSnapObservation` from that situation.
3. Candidates come from the applied book via `candidate_in_book`. Names that are not in the book are not ranked. `legal_candidates` drops anything that is not confirmed in-book.
4. The deterministic caller still produces `heuristic_pick`, including reads, macro, pivot, and the book guard.
5. `policy.rank_plays(game_state, observation, candidate_plays, roster_context, opponent_context, recent_history)` scores that pool. `observation` is required so feature construction can run without reading post-snap labels (contract revision this sprint). `recent_history` is an argument, not a field on `GameState`. Each entry is a `(CoachingDecision, SnapOutcome | None)` pair for a snap that already ended.
6. Mode decides what is shown. Guards stay authoritative. A VOD override is an input prior inside this sequence, not a mutation after the call is sealed.
7. The sealed record stores the shown play, the policy that actually sampled it, and the propensity of that policy. Post-snap labels stay on `SnapOutcome` and `ExecutedPlay`.

`PreSnapObservation` is the decision-time boundary. `LEAKAGE_FIELDS` lists outcome and post-snap names that must not appear on it. Feature code reads the observation, the candidate, roster, opponent context, and ended snaps. It does not read the current snap's yards, success, sack, or verified look.

## Modes, fallback, and the shadow-game counter

`CoachingDecision.mode` is the coaching mode. Legacy payloads that omit it load as heuristic. `policy_source` on a new heuristic record is heuristic.

| Mode | Shown call | `fell_back` | `fallback_reason` | `policy_source` | `shadow_status` |
| --- | --- | --- | --- | --- | --- |
| heuristic | heuristic | `False` | unset | heuristic | unset |
| shadow | heuristic | `False` | unset | heuristic | required |
| hybrid, model used | model, after guards | `False` | unset | model, or whatever policy sampled the shown play | required |
| hybrid, fallback | heuristic | `True` | `MLStatus` other than `OK` | heuristic | required |

`MLStatus` values are lowercase on the wire: `ok`, `model_missing`, `worker_busy`, `timeout`, `exception`, `invalid_output`, `circuit_open`, `low_confidence`, `unknown`. `OK` is a successful model attempt. It is never a fallback reason. `CoachingDecision.validate` rejects the other combinations and runs from `__post_init__` and from `from_dict`.

The shadow-game counter counts only rows with `shadow_status == OK`. A timeout or other non-OK status is a logged model result, not a counted shadow game. Shadow mode still shows the heuristic call when the model fails, so `fell_back` stays false.

`behavior_propensity` is the shown policy's end-to-end probability of the shown play after anti-repeat, pivot, the book guard, and a VOD override. It is not a model probability. Model scores live only in `shadow_scores` and `shadow_pick`. A hard VOD override is `propensity_method=DETERMINISTIC` and `behavior_propensity=1.0`, and those snaps are excluded from off-policy evaluation. `propensity_override` is true only when the method is `OVERRIDE`, and then `propensity_override_reason` is set.

## Contracts

Stable records use `madden-ml.schema.1` (`CONTRACT_VERSION`). `PlayRanking` and `OpponentContext` are provisional (`madden-ml.provisional.1`). The Strategist may add fields to those two. That bump does not change the stable version. A stable payload stamped with the provisional version is rejected, and the reverse.

`PlayRanking.status` is `ok`, `low_confidence`, `model_missing`, or `invalid_output`. `status_detail` is optional text. The ranking has no fallback reason; hybrid fallback stays on `CoachingDecision`. `OpponentContext` has no stale flag. `seeded_from_key` and `seed_shrink_weight` are both unset or both set, the weight is in `[0, 1]`, and the seed key shares the row's opponent id without equaling the row's own key.

Unknown is `None` in Python, JSON `null`, SQLite `NULL`, and the CSV cell `__UNKNOWN__` (`CSV_UNKNOWN`). Every enum has `UNKNOWN`. Do not write `0`, `False`, or a guessed play name for a missing fact.

Fixed bounds, changed only by editing `schema.py`:

- yards, and expected yards when a number is set, lie in `[-99, 99]`
- `ML_LOW_CONFIDENCE` is `0.6`
- `ML_LATENCY_BUDGET_MS` is `150`, and it is not rewritten from measured latency

Offensive success uses the same rule as `learning.SUCCESS_NEED`: 40% of yards to go on 1st, 60% on 2nd, 100% on 3rd and 4th. A turnover is a failure. `stop` is the defensive side of that rule. A sack is the sack flag, negative yards when yards are known, and an offensive failure. An accepted penalty leaves `success` unknown. A declined penalty keeps the play's label. A no-play is excluded and leaves `success` unknown. `SnapOutcome` does not grade a snap by itself.

`accepted` stays unknown until the user confirms. Showing a call does not accept it. `executed` is set only when `executed_status` is `identified` and the play is verified.

## Rollout

Existing installs stay on heuristic. Heuristic mode does not call `inference.rank_live`. Shadow mode may score and log, and the heuristic call remains what is displayed. Hybrid mode may let a validated model choose the candidate before the final reads and adjustments. Deterministic guards still win. Hybrid stays off until `registry.promotion_allowed` can see a passed gate. Phase 1 implements none of that runtime.

The model budget is 150 ms and covers only the model call. A miss, a busy worker, a timeout, a bad output, an open circuit, or confidence under 0.6 is an `MLStatus` on the decision. In hybrid that status is the fallback reason and the heuristic call is shown. In shadow the same status is `shadow_status` and the call does not change.

## Determinism and seeds

`train` and `evaluate` take `seed` with no default. The seed that was used is the seed that is recorded. `CoachingDecision.ml_seed` is optional. `None` means that decision was unseeded. Do not invent a seed when the run did not pass one.

Equal ranking scores break ties on formation, then play, then adjustment slot. The heuristic caller keeps the `rng` it was given. A shadow or hybrid failure must reproduce the heuristic call that would have been made without the model. A VOD override that forces one play is a point mass at propensity 1.0, not a new random draw.

## Concept names and PR #19

Three different "concept" ideas already exist. Do not merge them.

- `Situation.concept_hint` and `PreSnapObservation.concept_prediction` are pre-snap play-name tells, with a source (`live`, `last`, or unknown).
- `ObservedLook.concept` is a post-snap label. It is leakage if it is copied onto the observation of the snap being decided.
- Vision family tags (`cfb_coach.vision.play_family`) and VOD raw call names are not those fields.

PR #19 resolves a raw VOD name to one formation and play through `lookup`. That result is a candidate identity, not a concept prediction. This branch does not edit `cfb_coach/vod_model/live.py`, prep, or the mapping files. When the pipeline is wired, it should call PR #19's `lookup` after that PR lands, and it should pass the mapped play in as a candidate prior.

## File ownership

Phase 1 owns the schema and the stubs. Later work stays inside the module named here. Do not retune `make_call` from a feature or training change.

| Owner | May edit | Leaves alone |
| --- | --- | --- |
| Technical lead | `schema.py`, `model/__init__.py`, `tests/madden_ml/test_schema.py`, this document | live caller, VOD live swap |
| Data engineer | `adapters.py`, `dataset.py` | `GameState` construction anywhere else |
| ML training | `features.py`, `train.py`, `evaluate.py` | optional ML imports at package import |
| Registry | `registry.py` | promoting a model whose gate has not passed |
| Strategist | `policy.py`, and additive fields on provisional `PlayRanking` and `OpponentContext` | stable `madden-ml.schema.1` records |
| Live engineer | `inference.py`, and later the single call site inside `madden/playcaller.py` | heuristic behavior when mode is heuristic or the model fails |

`import cfb_coach.madden.model` exports the schema only. It must not import numpy, scikit-learn, LightGBM, CatBoost, XGBoost, torch, or pandas. Those learners, when they exist, stay inside `train.py`.

## Sprint 3 / 3.1 status (real-game data collection)

Implemented on `madden-ml/coach` without activating hybrid:

- Shared `evaluate_live_shadow` for HTML, terminal, and one-shot (`identity.LiveDecisionTracker` + `game_sessions.session_id` as `game_id`)
- Optional HTML execution verification: unknown / used recommended / used different (never auto-verify)
- `ml_decisions` ↔ `ml_outcomes` ↔ `snaps.ml_snap_id` linkage with undo audit + HTTP `idempotency_key`
- Supervised training eligibility (`verified_execution`, `trusted_vod` only by default)
- **Action attribution:** `_row_vector` / `action_*` use executed (or trusted VOD) plays — never the recommendation when execution is unknown
- Canonical snap dedupe across `snaps` / `play_records` / ML tables
- Feature schema `madden-ml.features.2` (football family categoricals; no numeric name hashes)
- Promotion gate: train-only baseline (incl. rate 0.0); mins 80/30/4; +0.02 log-loss; no hybrid activation
- Safe laptop DB: SQLite `backup()` API; inspect/export/train/eval open gameplay DB **read-only**; explicit `migrate-db --backup`
- Sealed pipeline module behind `CFB_COACH_SEALED_PIPELINE` (default off; see `SEALED_PIPELINE.md`)

Hybrid remains off. PR #19 mappings stay an optional boundary (`optional_vod_lookup`).

## Sprint 15A — expert / personal learning

See `SPRINT15A_EXPERT_LEARNING.md`. Expert VODs and personal verified gameplay are retained as separate evidence stores. They enter `choose_joint_action` only through the capped `expert_signal` path (shadow by default). `vod_model.apply_madden_call` remains off the experimental joint path so older VOD priors cannot double-count or override model-primary scoring.
