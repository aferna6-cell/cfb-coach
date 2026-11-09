# Unified football coordinator — `unified_v1`

Branch `architecture/unified-coordinator-v1`, based on the tip of the draft stack (PR #142 `madden-ml/defense-v2-live-test-franchise`, which contains #137–#139, #142 and everything beneath them). **Draft. Not merged, no weights promoted, opt-in only.**

> **What this is, honestly.** A single coordinator implementation for offense *and* defense: one typed game-state, one hierarchical outcome model, one Bayesian opponent model, one robust solver that picks a stock play and its individual adjustments jointly. It ships with a **cold-start, explicitly uncalibrated prior** and learns from verified executions. It is **not** a trained or proven-superior coordinator. See `UNIFIED_V1_VALIDATION_REPORT.md` for what was and was not measured.

## Architecture

```
                         ┌──────────────── GameState (typed, versioned, same for both sides) ───────────────┐
                         │ identity · down/dist/yardline · score · optional clock · installed-book fingerprint │
                         └───────────────────────────────────────┬───────────────────────────────────────────┘
                                                                  │
   installed book (owner-attested)                                ▼
   ── plays ─▶ knowledge.py ─▶ FootballPlay (concept path, variants, assignments, provenance, UNKNOWN fields)
                    │                         │
                    │      adjustments.py ────┤  stock play × registry adjustments (confirmed+legal) ; NO ADJUSTMENT always
                    ▼                         ▼
              LegalCandidates (FootballAction)          OpponentModel (opponent.py) ◀── closed, verified, strictly-earlier snaps
                    │                                   general prior→class→opp history→this game→situation→sequence→reaction
                    │                                                    │ Belief (distribution + evidence + TV radius)
                    └───────────────┬────────────────────────────────────┘
                                    ▼
                         WorldModel.predict(state, offAction, defAction)   (world_model.py — shared by BOTH sides)
                         cold-start prior + hierarchical residual L0..L5 (situation → type → concept×family → variant → exact play → exact×exact)
                                    ▼
                         Solver (solver.py)  Stage A concept screen (interval) → Stage B exact scoring
                         score = robust EV(TV-ball) − λ·uncertainty − exec cost − predictability(human only)
                         adjusted variants must beat their own NO-ADJ by z·sd (weak prior 1.64, learned 1.0)
                                    ▼
                         RecommendationCard (coordinator.py)  ── shows exactly the argmax; no later substitution
                                    ▼
        record_execution (what WE ran) · record_opponent (what THEY ran) · close_snap (outcome) ── store.py: 4 separate tables
                                    ▼
        immediate: opponent beliefs, shared world-model rows (verified executions only)       postgame.py: immutable snapshot → challenger → report
```

Code: `cfb_coach/madden/unified/` — `types.py`, `knowledge.py`, `world_model.py`, `opponent.py`, `adjustments.py`, `personal.py`, `solver.py`, `reads.py`, `coordinator.py`, `store.py`, `guard.py`, `pregame.py`, `postgame.py`, `preflight.py`, `sim.py` (synthetic truth for validation only), `cli.py`.

## Technical audit of the existing system (what was read, not assumed)

| # | Question | Finding |
|---|---|---|
| 1 | Who makes the final offensive decision? | `playcaller.make_call` first runs the **heuristic** `_pick_offense` (anti-repeat via `gameplan.anti_repeat_penalty`, `active_pivot`, VOD `apply_madden_call`), *then* `maybe_apply_experimental` (`model/experimental_live.py`) may replace it when that mode is on. The final call can be either; the caller must read attribution flags to know. |
| 2 | …defensive? | `_pick_defense`/`_select_base_defense` (`defense_select.select_defense`) first, then `maybe_apply_defense` (`defense_coordinator`, opt-in) or, with `--defense-model v2`, `defense_v2_live` (isolated test DB, 150 ms deadline, fallback to the heuristic call). |
| 3 | Legacy overrides that can replace learned picks | anti-repeat, pivot, book guard, VOD override, near-tie `_stable_choice` draw, `joint_override_guard`, post-selection macro/adjustment attachment (`rebuild_offense_attachments`, `_resolve_model_adjustment`). `CoachingDecision` docs list these as part of the logged propensity. |
| 4 | Recommended vs executed | `CoachingDecision.executed/executed_status/accepted` and `ml_outcomes.executed_status` exist and default to *unknown*; V2 learns only from verified executions. Good foundation — reused as a rule, re-implemented as separate tables. |
| 5 | Leakage of future snaps | PR #138 ("fix chronological learning leakage") and V2's "strictly earlier, closed, session-sequenced" fencing address this on the existing path. The unified path fences by `seq < state.seq` + closed-game ordering and has a test that mutation-fails if the fence is removed. |
| 6 | Structural features really available | Offense: `offense_assignments.json` has **93** plays, only **8** with any documented route, **0** verified route; `playbooks_full.json` has 2,135 distinct offense (formation, play) pairs known by name only. Defense: `defense_play_catalog.json` has 150 entries, **0 plays with a verified defender**, 55 whose exact variant bucket is `unknown`. No trained weights ship in the repo. |
| 7 | Reliable assignments | None are first-party verified in-repo. The unified layer keeps every unverified assignment `UNKNOWN` (`knowledge.knowledge_coverage_report()`). |
| 8 | Reusable | Catalogs (`playbooks_full`, `defense_play_catalog`, `offense_assignments`), `football_knowledge.CONCEPTS`, `offense_taxonomy.ROUTE_FAMILIES`, `adjustments.executable` registry + `validate_plan/readiness`, `defense_v2_roles` (owner role files), `defense_v2_live.installed_fingerprint`, `catalog_full` editor limits, `playbook_architect.family_ingame`. |
| 9 | What blocks genuine joint play+adjustment optimisation | Heuristic base pick → model overlay → adjustment attached *afterwards*; offense and defense are separate coordinators with separate artifacts and no shared matchup knowledge (issue #140); defense adjustments default to NO ADJUSTMENT without execution data; QB reads are derived from the already-chosen call. |
| 10 | Tests currently failing | Baseline (before this work, same base commit): `tests/` core 471 passed / 1 skipped; `tests/madden_ml` **15–16 failures** (one flaky) — see "Test results". |

### Migration map

| Existing | Status in `unified_v1` |
|---|---|
| `playcaller.make_call`, `_pick_offense/_pick_defense`, anti-repeat/pivot/VOD | **Untouched; remain the default and the rollback.** Unified path has none of the overrides. |
| `experimental_live`, `defense_coordinator`, `defense_v2_live` | **Untouched**, still selectable. |
| `catalog_full`, `defense_play_catalog`, `offense_assignments`, `football_knowledge`, `offense_taxonomy` | **Reused** (read-only adapters in `knowledge.py`). |
| `adjustments.executable` (registry, `validate_plan`, `readiness`) | **Reused** for legality/confirmation of every adjustment. |
| `defense_v2_roles.verified_roles` | **Reused** for first-party USER assignments. |
| `CoachingDecision` / `ml_decisions` tables | **Not modified.** Unified writes its own isolated DB (`store.py`); migration is additive (new file, no schema replacement). |
| `playbook_architect` (DB-bound) | Superseded for unified games by `pregame.py` (same editor limits, same catalog); the old one stays for the legacy flow. |
| Hard-coded counter tables / random change-ups | **Not used** in the unified path. |

## Using it

### Enable / disable
```bash
# enable (per session, one side per session; isolated store required)
PYTHONPATH=. python3 -m cfb_coach play --game madden27 --opponent cpu --coordinator unified_v1 \
   --unified-side offense --unified-confirmed-books books.json --unified-store /ISOLATED/unified.db [--unified-model challenger.json]
# or call it directly
PYTHONPATH=. python3 -m cfb_coach.madden.unified live --side defense --opponent cpu --confirmed-books books.json --store /ISOLATED/unified.db
# disable / rollback: run `play` again WITHOUT --coordinator. Nothing else to undo; the established caller is unchanged.
```
The store path must not be inside `~/.cfb-coach` (the guard raises `ProtectedPathError`).

### Game prep (both books) → install → confirm
```bash
PYTHONPATH=. python3 -m cfb_coach.madden.unified game-prep --side both --opponent cpu --out /ISOLATED/prep
#  -> proposal_offense.json, proposal_defense.json, install_offense.md, install_defense.md  (status PROPOSAL_NOT_INSTALLED)
# build the books in Madden's playbook editor (formation → set → source playbook → keep all listed plays), then CHECK them in game, then:
PYTHONPATH=. python3 -m cfb_coach.madden.unified confirm-install --proposal /ISOLATED/prep/proposal_offense.json --attested-by YOU --confirmed-books /ISOLATED/books.json
PYTHONPATH=. python3 -m cfb_coach.madden.unified confirm-install --proposal /ISOLATED/prep/proposal_defense.json --attested-by YOU --confirmed-books /ISOLATED/books.json
```
Nothing is ever marked installed automatically; live play refuses without the attestation and checks the fingerprint.

### CPU preflight
```bash
PYTHONPATH=. python3 -m cfb_coach.madden.unified preflight --confirmed-books /ISOLATED/books.json --db /ISOLATED/test-franchise/coach.db \
   [--model challenger.json] --calls 100 --out /ISOLATED/preflight.md
```
Verdicts: **GO** (all gates pass, ready for an *experimental* CPU test only), **SHADOW ONLY** (runs, but a book/DB/latency gate is unconfirmed), **NO-GO** (critical correctness, legality, integrity or execution failure). Run it **on the machine that plays**; the in-repo run cannot see your Xbox, DB or installed books.

### Postgame challenger (never promotes)
```bash
PYTHONPATH=. python3 -m cfb_coach.madden.unified postgame --store /ISOLATED/unified.db --out /ISOLATED/post_001 [--champion champion.json]
```
Validates labels (only *verified executions + revealed opponent call + outcome* become training rows; the rest are counted with reasons), writes an immutable checksummed snapshot, fits a challenger, evaluates by **whole game, expanding window** against the cold prior and the frozen champion, with a game-clustered bootstrap, and writes `report.json`. The champion file is only read (sha256 before/after is recorded) and there is no promotion function: promoting is a deliberate manual act.

## What the card shows
Decision model (`unified_v1`), solver/objective version, artifact, football-evidence confidence and what is unknown, opponent-prediction source (`general_prior`, `hist(n)`, `game(n)`…), fallback (occurred/reason), the call, `NO ADJUSTMENT` or the adjustment(s), and for offense two QB reads from the **final adjusted play** / for defense the USER position with primary and secondary responsibility (marked VERIFIED only when it comes from a verified ledger or owner role file).

## Honest limitations
* **Cold start.** The prior numbers (base yards, completion rates, matchup shifts, provisional value weights) are football-plausible placeholders, flagged `calibrated=False`. They are replaced by evidence only as verified executions accumulate.
* **No EPA.** `fit_state_value` refuses below 200 real possessions; the provisional objective is used.
* **Assignments.** 0 verified defenders and 0 verified offensive routes exist in the repo. Reads are concept-level unless a documented route exists; USER advice is a labelled football prior unless the owner supplies verified roles.
* **Adjustments.** Every registry action is `user_confirmed=false` in the repo. With no confirmations the coordinator recommends NO ADJUSTMENT. Even confirmed ones need a benefit that survives uncertainty; effects are weak registry priors until executed-adjustment data exist.
* **Opponent identity at signature level.** Beliefs are over coverage/concept signatures (type, family, exact variant), not over unseen exact plays.
* **Not done:** neural/offline-RL models, boosted-tree challenger (no data, no sklearn), sequencing beyond the human-only predictability term, controller-input instructions, full catalog annotation, owner-DB integration beyond a read-only audit.

## Test results (this branch, run in the build environment)

| Suite | Result |
|---|---|
| `tests/unified` (new: 20 acceptance + 15 scenario/learning/preflight tests) | **35 passed** |
| `tests/` core (no `madden_ml`), baseline at base commit | 471 passed, 1 skipped |
| Full `tests/` with this branch | 2,113 passed, 5 skipped, **16 failed** |
| Same 13 failing test IDs on the untouched base (git-stashed) | **12 of 13 fail identically on the base** — pre-existing, not caused by this work. The 13th (`test_defense_typed_capture … Fire Zone 3-None`) passes alone with this branch and appeared only in one full-suite order; treated as an order-dependent flake. |

Pre-existing failures (all outside `cfb_coach/madden/unified/`): `test_data_layer_synthetic::test_backup_manifest_precedes_mutation_and_failure_aborts`, `test_football_priors::test_qb_reads_are_recomputed_from_the_coverage_and_the_transformed_play`, five in `test_oc_model_adjustments`, two in `test_offense_intel_model`, `test_route_aware_live_integration::test_primary_mode_falls_back…`, `test_vod_eligibility::test_flag_off_export_matches_5d9c566_bytes`, `test_wildcard_john_games…game2_fields`. They were **not** fixed here (out of scope; this is the draft stack's existing state) and remain blockers for merging the stack, not for this opt-in path.

Mutation checks done by hand: removing the `seq < state.seq` fence, auto-promoting a recommendation to an execution, and disabling the adjustment gate each make the corresponding acceptance test fail.

Protected assets: `sha256(cfb_coach/data/madden27/*.json)` is identical before and after all work; no `~/.cfb-coach` primary database, installed-book state or champion model exists in the build environment, so none could be touched (the `protected_assets_unchanged` preflight gate checksums them wherever they exist).

## Cross-side learning (issue #140)
Implemented as the issue's "required distinction": a revealed opponent defense updates only `OpponentModel` (semantic knowledge about what they run) and is never turned into *our* execution; our own executions update the shared `WorldModel` tables used by both sides. Weak, shrunk transfer of "this coverage did well against our offense" into our own defense choices is **not** wired (needs real data and the issue's holdout evaluation).

---
# Sprint 2 — first-CPU-game readiness (branch `architecture/unified-v1-live-session`)

| Area | What changed | Proof |
|---|---|---|
| One continuous game | `unified game` = `UnifiedSession`: both coordinators on **shared** snap history, one game id, one monotone sequence, one world model, one personal-execution model, one adjustment planner; side taken from the possession you type (`o …` / `d …`). `play --coordinator unified_v1` now opens this session. | `tests/unified/test_session.py` (incl. a scripted full `cmd_game` run across possession changes and a second game that starts from the first's evidence) |
| IDs / overwrites | Store is append-only: decision rows unique per snap id and per (game, seq); execution/opponent/outcome facts refuse a second write unless `amend=True`, which appends an audited revision; memory is only changed after the write succeeds. Game ids can't be reused without `resume`. | `test_duplicate_ids_and_overwritten_rows_are_refused` |
| History across possession changes | "Previous snap" for sequence/reaction context is the last snap **on the same side**, so a defensive snap in between no longer corrupts offensive context. | `test_previous_snap_context_is_same_side…` |
| Shared learning | Verified rows from BOTH sides update one world model (keyed by offense concept × defense coverage, so what our offense learns about Cover 1 is used when our defense considers Cover 1). Provenance is counted separately (`our_offense_executed`, `our_defense_executed`); a revealed opponent call updates only that side's opponent model and never counts as our execution; side-specific execution effects stay in `(side, opponent class)` offsets. `ingest_history` no longer drops the other side. | `test_shared_learning_crosses_sides_but_provenance_stays_separate`, `test_history_replay_feeds_both_sides…` |
| Historical data | `unified history --db …`: READ-ONLY (`mode=ro`) import through the repo's own `dataset.build_rows` eligibility (recommendations are never executions), skips and **counts** unverified adjustments, uncatalogued plays, unclassifiable/absent opponent calls. Trains a candidate, evaluates by whole game against cold start, a global-mean baseline and a situational baseline, keeps the cold start unless the candidate wins with a game-clustered CI. Never promotes. | `tests/unified/test_history.py` |
| Preflight can't be gamed | GO now requires: owner-attested BOTH books, a real model file consistent with the history evaluation (a file the history run did not produce → NO-GO), an audited isolated DB, the local parser gate, **and** a clean run of the real `cmd_game` loop (≥100 snaps/side) on those exact books+model with persistence and a cold-reload audit (no duplicates, no uninstalled calls, executed==what was typed, attribution, legal adjustments, no fallback — `playcaller.make_call` is booby-trapped during the run — and previous-play learning replayed from disk). | `test_go_is_impossible_without_each_owner_supplied_ingredient`, `test_live_audit_catches_a_corrupted_execution_log` |
| Latency | Found and fixed: with the larger proposed books the first version ran p95 ≈125 ms with search-deadline fallbacks. Predictions are now memoised per feature class, exact-level plays use branch-and-bound pruning (3 sd of the model's own effect-size prior), stage B is ordered most-promising-first so a deadline can only drop the weakest plays. | see report §3 |
| Playbooks | `game-prep` now emits an install guide with per-set source playbook, editor status (`PREVIOUSLY_INSTALLED_FROM_THIS_SOURCE` / `UNVERIFIED_IN_EDITOR`), editor-limit checks, proposed fingerprint, `confirmed: NOT CONFIRMED`, and a comparison with the previously installed book. | `docs/madden_ml/unified_v1_first_cpu_game/` |
| Adjustments | `unified adjustments` lists every registry action: exists in Madden 27 (evidence), owner-verified?, and `LIVE_RECOMMENDABLE / NOT_OWNER_VERIFIED / UNAVAILABLE` with why. A confirmations template (all `false`) is generated for the owner. | `ADJUSTMENT_STATUS.md` |
| Owner machine | `discover`, `local-prep`, `local-finish` implement the 8-step procedure (protected backup + checksums, SQLite-backup test copy, audit, history, game-prep, install stop, benchmark, preflight). | `tests/unified/test_localrun.py` |

## Known behaviours to expect in the first game
* **Cold start is repetitive.** With no learned evidence many calls tie; ties break deterministically (higher play-knowledge confidence, then name). Expect the same defensive call often until games accumulate evidence. There is deliberately no random change-up.
* **Reads are concept-level** for almost every play (only 8 catalog plays have documented routes); USER advice is a labelled prior (0 verified defenders).
* **Result convention:** the result you type is always the *offense's* result on that snap (on defense, what the other team's offense got).
* **Search deadline:** if the decision would run past 85% of 150 ms it stops searching the weakest plays and flags `deadline_partial_search`; preflight treats any occurrence as a failure.

## Pre-existing failures investigated (not caused by this work)
12 distinct tests (15 counting parametrised/duplicate IDs) fail identically on the untouched base; none touch `cfb_coach/madden/unified/`. Gameplay relevance:
| Tests | Cause (read from the failure) | Can it affect the unified game? |
|---|---|---|
| `test_oc_model_adjustments` ×5, `test_wildcard_john_games` | the legacy experimental offense path no longer attaches the catalog macro (`None != 'O-RUN'`) | Only the **legacy** `--ml-human-control` macro path. Unified never uses macros. |
| `test_offense_intel_model` ×2, `test_football_priors` | the legacy offense card/QB-read ordering for documented plays changed (`WR1 drag…` vs expected `TE1…`; hot-route text) | Legacy card only. Unified reads come from `unified.reads`. |
| `test_route_aware_live_integration` | legacy primary-mode fallback picks a play outside the expected set | Legacy fallback only. |
| `test_data_layer_synthetic` | expects `OSError` from a permission-based failure; does not raise when running as root | Environment (root), not gameplay. |
| `test_vod_eligibility` | needs a git commit that a shallow clone does not contain | Environment (shallow clone). |
| `test_defense_typed_capture[Fire Zone 3-None]` | passes alone; failed once in one full-suite order | Order-dependent flake, legacy defense capture. |
Recommendation: do not use the legacy macro path in the same session as the unified coordinator until those are fixed on the stack.

---
# Sprint 3 — the familiar HTML window on Unified V1 (branch `architecture/unified-v1-html-interface`)

`play --game madden27 --coordinator unified_v1` now opens the ESTABLISHED live page (same layout, buttons and flow); `--terminal` keeps the terminal loop as a rollback.

**How it is wired (nothing legacy is on the decision path):**
```
browser (live_server.render_live_html page)  ──HTTP──▶  live_server.make_handler  ──▶  UnifiedLiveController  ──▶  UnifiedSession ──▶ both coordinators
                                                                                         │                          (shared world model, one game id, one seq)
                                                                                         └─▶ isolated UnifiedStore (append-only; idempotency keys, voids, cards)
```
* `html_controller.UnifiedLiveController` implements only the surface the page/handler use (`state`, `result_and_call`, `call_only`, `undo_last`, `end_game`). It never imports `CoachDB` or the legacy playcaller, and `play` dispatches to it before any database is opened.
* `live_server.py` got ONE additive change: when the controller says `unified_ui`, the page renders a status panel (model, artifact, learned-vs-cold-start, learning counters, opponent-belief source, fallback, adjustment, USER job) and a result-entry banner (offense → our yards; defense → OPPONENT OFFENSE's yards). Legacy pages are byte-identical in behaviour (28 legacy HTML tests unchanged).
* Every request is validated before anything is written; each mutating request is idempotent via the browser's key, persisted in the store, so a retried/duplicated POST never creates a second snap.
* Execution is recorded only from the page's explicit "What I ran" choice (`used recommended` → ran as called, or `y noadj` if the card had an adjustment and "applied" was not ticked; `used different` → a play from the confirmed book; untouched → **unknown**).
* Resume: `--unified-game-id ID --unified-resume` replays the game from disk once and restores a snap that was still waiting for its result. Refresh/reconnect with the server alive just re-reads state.
* Undo: append-only. The last snap with facts is "voided" (execution→unknown, outcome→no yardage, opponent call→unverified), the waiting call is dropped, the session is rebuilt from disk; ids are never reused and the voided snap trains nothing.

**Validation added**
| Check | Where |
|---|---|
| Real HTTP, offense → result → defense → result → offense; one recommendation per snap; card/reads/USER; persistence; belief + shared-model updates; provenance; contiguous ids; idempotent retry; bad input writes nothing; used-different legality; undo; restart + resume | `tests/unified/test_html.py` |
| Real headless Chromium (Playwright) drives the actual page through the same sequence, reads the DOM (panel, QB reads, USER box, result banner, log), reloads, and submits bad input | `test_real_browser_drives_the_established_page`, `cfb_coach/madden/unified/browser_drive.js` |
| Real `python -m cfb_coach play …` process with a sentinel "primary" home: directory byte-identical afterwards | `test_play_command_opens_the_html_window_and_never_touches_the_primary_home` |
| 100 snaps per side over real HTTP with cold-reload audit + traps on `playcaller.make_call` and `CoachDB` | `htmlcheck.py`, `unified html-check` |
| Preflight: `html_verdict` is `GO` only if the terminal gates are GO **and** the browser gates ran and passed; legacy hit → NO-GO; not run → SHADOW ONLY | `test_html_verdict_cannot_be_go_without_validated_browser_integration` |
