# Offense Play Intelligence v1 (research + CPU-pilot candidate)

Offense only, the 93 installed plays (`OFF-3C6D-E543`). Nothing here is imported by the live caller, writes a game
database, or changes the installed playbook. The pilot is **inert until you run `approve`**.

## What exists

| Piece | File |
|---|---|
| Route / blocking / concept / evidence taxonomy | `cfb_coach/madden/model/offense_taxonomy.py` (+ `data/madden27/offense_taxonomy.json`) |
| 11-slot assignment database + overlay merge + audit | `offense_assignments.py`, `data/madden27/offense_assignments.json` |
| Structural features, hot-route transform, coverage-matchup priors | `offense_play_features.py` |
| Structure-aware model (residual on the existing coordinator) | `offense_structural_model.py` |
| Joint play + adjustment decision (registry-validated) | `offense_decision.py` |
| Pre-snap QB card | `offense_presnap_advice.py` |
| Offline A/B/C/D evaluation | `offense_structural_eval.py`, `scripts/madden_ml/offense_structural_eval.py` |
| CPU pilot harness + CLI | `offense_pilot.py`, `scripts/madden_ml/offense_pilot_cli.py` |
| Annotation workflow | `scripts/madden_ml/offense_annotation.py` |
| Tests | `tests/madden_ml/test_offense_intel_{assignments,model}.py` |

## Evidence tiers (strongest first)

`in_game_verified` > `diagram_verified` > `source_documented` > `football_inference` > `unknown`.

* A slot with no evidence is `UNKNOWN` (category `unknown`, task `UNKNOWN`). The validator rejects an unknown slot that carries a task.
* **Verified** means the user's own evidence: a verified tier must cite `user_screenshot:` / `user_recording:` / `user_ingame:`. A third-party URL can never be verified.
* A play is `fully_verified` only if all 11 slots are verified. Play-**name** hints (`name_hint`) are always `football_inference`, confidence 0.35, and never create a route or blocking assignment.
* Stronger evidence replaces weaker; equal-tier conflicts are recorded as disputes, not overwritten.

## Current coverage (generated: `OFFENSE_ASSIGNMENT_AUDIT.md`)

* 93 plays x 11 slots = 1023 slot records: 40 `source_documented`, 983 `unknown`, **0 verified**.
* 8 plays have sourced receiver/back jobs (5 of 11 slots each) from public Madden Tools pages (PR #106 annotations). QB and all five OL slots are unknown on **every** play (558 slots).
* Sources: the 8 annotated plays come from individual public Madden Tools pages (PR #106). Madden Tools and 818 Madden restrict systematic extraction for building another product, so nothing was scraped; a survey of other sites found play lists but no source that permits reusing per-play assignments for a model. The remaining 85 plays need your own play art (below).
* Formation personnel: slot names (`HB1, TE1, WR1, WR2, SLWR|WR3`) are only known for the three formations with a sourced play; `Gun 5WR Tight` and `Pistol Deuce Close` have unnamed `SKILL_1..5` slots.

## Annotating the missing plays from your own game

```bash
python scripts/madden_ml/offense_annotation.py template --out ~/offense_slots.csv     # 1023 rows, pre-filled
python scripts/madden_ml/offense_annotation.py manifest --images ~/playart            # files named <Formation>__<Play>.png
python scripts/madden_ml/offense_annotation.py import   --csv ~/offense_slots.csv     # validated; writes the overlay
python scripts/madden_ml/offense_annotation.py audit    --write docs/madden_ml/OFFENSE_ASSIGNMENT_AUDIT.md
```

1. In the play-call screen capture each play's art (screenshot or clip). Name it `Gun_Trips_X_Nasty__Hi_Lo_Cross.png`.
2. Fill `category/task/side/depth/tier/source` **only** for slots you can read; leave `tier` empty otherwise.
3. `import` writes `cfb_coach/data/madden27/offense_user_annotations.json`. Every consumer loads base + overlay (`load_effective`), so a retrain picks it up; artifacts record the database digest and refuse to load if it changed.

## Model (not a rule-based caller)

`logit(p) = logit(p_experimental) + alpha * w . x_structure`, with `x` = formation, play-type/run-scheme/concept **name-hint channel** (low trust, strongest L2), route geometry from *known* routes only, structural concept flags (mesh, flood, levels, drive, smash, dagger, mills, stick, spacing, four verticals, post/dig, slant-flat), and coverage-matchup priors (P(shell) x route stress; Palms, Quarters, Drop, Hole/Robber, Cover 6/9 stay separate). Group-wise L2, unpenalized bias only, `alpha = n/(n+30)`, Laplace uncertainty `x'H^-1x`, residual disabled below 25 verified rows. Coverage mix comes from the opponent's **earlier** games; the current snap's coverage/result are rejected (`PostSnapLeak`).

Adjustments: a hot route is applied as an exact one-slot transform (original must be known), then scored by structural extrapolation. Because training never saw hot-routed plays, the gain is halved, capped at 0.30 logit, rejected if any feature leaves the training range, and accepted only if modeled gain >= 0.02 **and** its lower bound (z=1) is positive **and** Madden's registry validates it live (confirmed in-game). With nothing confirmed, every snap is `ADJUSTMENT: NO ADJUSTMENT`. Verified macros are scored only with a user-verified effect map; none exists today.

## Evaluation

```bash
python scripts/madden_ml/offense_structural_eval.py --db /path/to/COPY_of_coach.db --out /tmp/offense-eval
```

Chronological whole-game folds (no game in both train and test), paired game-clustered bootstrap vs the current coordinator, penalty sensitivity (x0.3, x3) and a no-matchup ablation, plus decision-behavior metrics (legality, concept/formation diversity, repeat rate, latency vs 150 ms). **The game database is not in this environment, so no real result exists yet**; with < 8 verified games the report refuses to claim anything. C/D can only be measured as behavior: no hot-route executions were ever logged.

## CPU pilot walkthrough

```bash
W=~/.cfb-coach-pilot                       # deliberately NOT ~/.cfb-coach
python scripts/madden_ml/offense_pilot_cli.py --workdir $W config-init
python scripts/madden_ml/offense_structural_eval.py --db COPY.db --out /tmp/offense-eval     # 1. shadow evaluation: read it
python scripts/madden_ml/offense_pilot_cli.py --workdir $W train --db COPY.db               # 2. build artifact
python scripts/madden_ml/offense_pilot_cli.py --workdir $W validate                         # 3. digests, fingerprint, paths
python scripts/madden_ml/offense_pilot_cli.py --workdir $W shadow --down 3 --distance 7 --yardline 45   # 4. cards, logged, no approval
python scripts/madden_ml/offense_pilot_cli.py --workdir $W confirm --rec-id ID --formation F --play P    # after each snap
python scripts/madden_ml/offense_pilot_cli.py --workdir $W report
python scripts/madden_ml/offense_pilot_cli.py --workdir $W approve --by "<your name>"       # 5. ONLY when you decide
python scripts/madden_ml/offense_pilot_cli.py --workdir $W pilot --down 3 --distance 7 --yardline 45    # 6. CPU games
```

Human opponents are refused. A recommendation is logged as `executed: not_confirmed`; only `confirm` records an execution. The baseline coordinator's top 5 is logged next to every call; it replaces the model only on a technical exception (logged as `fallback_reason`).

## Known limits

* 0 verified assignments; 85 plays have no sourced route; no OL/QB job on any play.
* Structural benefit is unproven on real games; on pure-noise labels the structural variants correctly do **not** beat the base coordinator.
* The sourced `side` field is the published side, not a verified alignment; depths are family defaults (`DEPTH_BASIS`).
* Hot-route menu items are `needs_user_test` in the registry (PR #104); confirm them in-game via the confirmations file before any adjustment can go live.
