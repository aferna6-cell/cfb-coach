# Route-aware offense in the normal CFB Coach live loop

Draft, stacked on #112. **Default behaviour is unchanged**: nothing runs unless you select the ranker.

## What changed

| Piece | Where |
|---|---|
| Selectable ranker (`experimental` default / `route_aware_preview` / `route_aware`) | `cfb_coach/madden/model/offense_route_ranker.py`, hook in `experimental_live.apply_experimental_offense` + `maybe_apply_experimental` |
| Time-correct coverage for training and evaluation (same blend as live) | `offense_time_correct.py`, `offense_live_coverage.blend_counts` |
| Cross-fitted offsets, nested game-level model selection ("do no harm") | `offense_structural_model.train` |
| Situation x route-shape interaction features | `offense_play_features.situation_features` |
| Data audit, shadow replay, scenarios A-F, readiness gates | `offense_readiness.py`, `scripts/madden_ml/offense_route_aware_readiness.py` |
| Display in HTML live window and terminal loop | `live_server._format_call_text`, `madden/cli._route_aware_text` |
| CLI switch | `python -m cfb_coach ml offense-ranker [mode] [--artifact PATH]` |
| Planted-signal generator (self-check only, never evidence) | `offense_synthetic.py` |

The post-play form is **not** duplicated. `LastSnapBook.close_from_form` already writes `coverage_seen`; before the next
call, `coverage_for_next_snap` reads the closed snaps of *this game and opponent* and blends them into the historical prior
(`PRIOR_STRENGTH = 8`). Training uses the same formula on strictly earlier snaps.

## Modes

| Mode | The call | Route-aware model |
|---|---|---|
| `experimental` (default) | current ranker | off |
| `route_aware_preview` | **current ranker (unchanged)** | scores all 93 installed plays on the same state; shows its pick, expected coverage, reads, adjustment; logs to a separate shadow file |
| `route_aware` | route-aware ranking of **all 93 installed plays** (no situational shortlist); anti-repeat selection, verified-macro policy, rebuild, logging, 150 ms budget are the existing ones | primary |

Failure of the route-aware model (missing/stale artifact, over budget, exception) falls back to the current ranker for that snap and
prints/logs the reason. It never silently substitutes a rule.

Preview also works while the DB is in `heuristic` mode (it never touches the call). CPU opponents only.

## Enable / disable / roll back

```bash
# status (also prints the active artifact)
python -m cfb_coach ml offense-ranker

# install a validated artifact, then PREVIEW (recommended first)
python -m cfb_coach ml offense-ranker route_aware_preview --artifact ~/route_aware_readiness/offense_route_aware.json

# make it the primary ranker (only after the readiness report supports it)
python -m cfb_coach ml offense-ranker route_aware

# ROLL BACK (instant, next call)
python -m cfb_coach ml offense-ranker experimental
```

One-session override without touching the DB: `CFB_COACH_OFFENSE_RANKER=route_aware_preview python -m cfb_coach play --game madden27 -o cpu`.
Artifact search order: `CFB_COACH_ROUTE_AWARE_ARTIFACT`, DB meta (`--artifact`), `<data dir>/madden_ml_experimental/offense_route_aware.json`.
Stale artifacts (feature schema, assignment DB or route-art bank changed) are refused.
`route_aware` needs the existing experimental mode (`python -m cfb_coach ml experimental`); `route_aware_preview` does not.

## One-command readiness (your machine, on a COPY of the database)

```bash
cp ~/.cfb-coach/madden27.db ~/madden27_copy.db
python scripts/madden_ml/offense_route_aware_readiness.py --db ~/madden27_copy.db --out ~/route_aware_readiness --run-tests
# add --install to copy the artifact where the coach looks (still does not enable it)
```

It audits the data, retrains, evaluates on held-out whole games against the existing coordinator (slices, train-vs-held-out gap,
calibration, coverage sensitivity, diversity, latency), replays held-out games as shadow decisions, runs scenarios A-F through the real
live loop on a temporary DB, and prints GO / CONDITIONAL GO / NO-GO with each gate. `--selfcheck` runs the same pipeline on planted
synthetic data and says so; it proves plumbing and learnability, not model quality.

## Scenarios

A repeated Cover 2 Man (beliefs rise monotonically, counts follow the form entries, rankings recomputed each snap);
B coverage change (beliefs move off the previous look); C ten situations incl. 4th down, red zone, goal line, backed up (all pre-snap
inputs; punts/field goals are yours); D diversity (raw model and through the live loop; measured, never forced);
E adjustment legality (unconfirmed adjustments never offered live; NO ADJUSTMENT valid); F latency.

## Limits you should know

* Score, game clock and timeouts are **not** inputs to the learned residual: the dataset rows do not carry them. Two-minute play is
  modeled only through down/distance/field/quarter.
* Hot-route adjustments cannot currently be scored: the only route assignments are third-party and excluded from training (#110), and
  there are no logged hot-route outcomes. Every recommendation is therefore `NO ADJUSTMENT` until first-party assignments and in-game
  confirmations exist.
* Pre-snap "showing" coverage typed in the form is not used by the route-aware model.
* Reads on the card come only from routes that exist in the recommended play; 85 of 93 plays have none documented, so their reads say UNKNOWN.
* Coverage labels in the database are manual entries, not independently verified.

## What was actually executed in the development environment (2026-11-07)

**The ~30-game database was not available**, so no real training, held-out comparison or live CPU session happened here.
Executed instead (all synthetic, planted-signal; proves plumbing and learnability only, NOT model quality):

* Full suite: 1504 passed, 4 skipped, 1 failed (pre-existing, root/chmod environment test). Untouched #112 head: 4 failed (2 inherited
  test bugs from #111, 1 from #108/#110, 1 environment); this PR fixes the three real ones.
* `offense_route_aware_readiness.py --selfcheck` (30 synthetic games, 1,800 verified rows, planted coverage x shallow-route effect that
  CONTRADICTS football convention):
  * learned the planted effect (`am_man_shallow` strongly negative); nested selection switched the residual ON (penalty x1);
  * held-out whole games, rolling origin: log loss 0.6909 vs existing coordinator 0.6951 (diff -0.0042, 95% CI [-0.0092, +0.0002]);
    a fixed-penalty variant was worse (+0.0075) and train/held-out gap 0.058 -> 0.027 with selection;
  * on pure-noise labels, selection switches the residual OFF (tested), so the existing coordinator stays in charge;
  * 1,620 replayed shadow decisions: 0 illegal, 0 errors, p95 15 ms; agrees with the existing coordinator 63% of the time (planted data);
  * scenarios A/B/C/E/F pass through the real live controller; live-path diversity 17 distinct plays / 40 snaps (route-aware) vs 32 (current ranker);
    the raw model alone is concentrated, so the existing anti-repeat policy matters.
* Real-data gate `model_quality_vs_existing_coordinator`: **NOT EVALUATED**.

Findings about the existing features worth knowing: `art_horizontal_share` has almost no spread across plays (sd 0.03), so it cannot carry signal.

## Defensive knowledge features (issue #116, opt-in)

An artifact trained with `defense_knowledge=True` adds a `dk_*` feature block (catalog-derived structure of the opponent's defense) and is
refused if the catalog changed. The default artifact and vector are unchanged. See `DEFENSE_KNOWLEDGE_BASE_V1.md` and
`DEFENSE_KNOWLEDGE_READINESS.md`; `offense_route_aware_readiness.py --defense-knowledge` trains and replays it (and refuses `--install`).
