# civil_scheme_only — the 26-variant Lions offensive scheme mode

Status: **draft, off by default, not live-ready.** Nothing in this mode has been run in a Madden 27 game yet.
All 26 variants are catalog-approved, and **none is verified executable**. Live calls stay blocked until you verify them one by one in Practice (see the checklist below).

## What it does

The mode is selected per session or per DB, and the existing coordinator stays separately selectable:

| How | Offense coordinator |
|---|---|
| default (nothing set) | the existing unrestricted coordinator, unchanged |
| `ml civil-scheme enable` (DB meta) or `play ... --civil-scheme-only` (this session) | **civil_scheme_only** |
| `play ... --unrestricted-offense` (this session) or `ml civil-scheme disable` | the existing unrestricted coordinator |

In civil_scheme_only, every offensive call is exactly one of the 26 catalog variant IDs. The call shows:

- the formation, base play, variation name and ID;
- **every** adjustment, in order and verbatim;
- verification warnings, an explanation, and P(success), first down, explosive and sack/turnover/loss chances, plus uncertainty.

Reads are shown only when a verified read progression exists, and none does today.

The mode never calls the heuristic `_pick_offense` or the experimental offense path. It never substitutes another play, never offers an unadjusted version (`unadjusted_allowed` is false for all 26), and never uses the 2 red-zone packages or the unnamed 5WR setup.

If nothing is legal, the answer is `NO_LEGAL_SCHEME_PLAY` with the reasons. There is no hidden fallback. An internal error also returns `NO_LEGAL_SCHEME_PLAY`.

## Pipeline per snap (`cfb_coach/madden/scheme/`)

1. **Legal filter** (`legality.py`). This one filter runs before scoring and again before display (`validate_live_call` → `revalidate_call`). A variant passes only if:
   - it is a catalog variant ID;
   - its formation and base play are in the in-repo Buccaneers stock data, matched by exact normalized name;
   - both are in the applied (installed) offense book;
   - you have verified it executable with `ml civil-scheme verify`.
2. **Opponent belief** (`beliefs.py`). Fine coverage labels come from the existing normalizer, and pressure is estimated separately. The belief is a hierarchy: opponent → CPU/human → everyone, built from earlier games only. Earlier games are recency weighted ×0.9 per game. This game's closed snaps are weighted ×0.85 per snap, so the last play counts most. A live pre-snap look (`showing cover 3`) counts once. Coverages that were unreported or unrecognized add nothing: unknown stays unknown.
3. **Score** (`coordinator.py`, `model.py`):
   - The **existing coordinator's model** (`experimental_model` shrinkage) is the base logit for the base play.
   - Learned heads add to it (see Evaluation for which ones).
   - The variant's adjustments are applied to the base geometry, the features are recomputed, and the change versus the stock play is transferred at 50% (`VARIANT_TRANSFER`), because no verified variant outcome exists yet. The size of that transferred change is added to the uncertainty.
   - Then the expert situation prior, the **coverage-beater term** (below), a per-variant learned residual from verified variant outcomes, and a repetition penalty are added.
   - Utility = P(success) + 0.10·P(explosive) − 0.20·P(sack/turnover/loss). These are fixed weights with no situation rules.
4. **Pick and display.** The highest utility wins. The chosen variant is re-validated, then shown and logged.

### Coverage beaters (your rule: "if we are seeing a certain type of coverage over and over the model should go to its beater")

Each variant has "beats Cover X / Man" tags, taken from the catalog plus the Civil.GG sheet's "also beats" column. These become a strong term that is gated by the belief:

```
beater = 1.2 · min(1, Σ_T gate_T · tagged_T) / sqrt(#tags)
gate_T = confidence · clip((share_T − 0.35) / 0.40, 0, 1)
confidence = clip((effective observations − 1) / 3, 0, 1)
```

- `share_T` is the belief that the next coverage is in tag group T (Cover 0/1/2/3/4, Man).
- Effective observations are this game's observations (recency weighted) plus recency-weighted history.

How it behaves:

- **When one coverage keeps showing up**, its beaters get up to +1.2 logit (single-tag beaters) and normally rise to the top. Specific beaters get more than plays tagged for every coverage.
- **With mixed or thin evidence** (one observation, or an even spread), the gate is about 0, so the tags barely matter.
- **It is a term inside the model score, not an if-then rule.** Down and distance, field position, the sack/turnover head and the legal filter still apply.
- **Variety:** a repetition penalty (0.25 logit per recent call of the same variant, decaying ×0.6) spreads calls across the tagged beaters.
- **Untagged plays:** all Gun 5WR Tight variants have no tags. They get the *average* beater term of the tagged variants, so they are treated as unknown, not negative.
- **Fading:** the beater term fades slowly (K=40) as verified outcomes for that variant arrive. The weak situation prior (≤0.12 logit, K=10) fades faster.

Tests `test_beater_a/b/c` show the following:

- Repeated Cover 3 moves the top picks to Cover-3-tagged variants.
- Switching to repeated man coverage moves them to Man-tagged variants.
- Mixed or low evidence keeps every beater term below 0.2 logit.
- After 3 repeats of the same variant, a different tagged beater takes the top spot.

## Data and training

- **Training data:** verified offensive snaps only. Those are the eligibility `verified_execution`-class rows from `dataset.build_rows`, read from a backup copy opened read-only. A simulated snap is never verified history.
- **Snapshot used:** `madden27_full_20261107.db`, sha256 `fddaa613…`. It is a copy of `~/.cfb-coach/madden27.db` taken from the laptop on 2026-11-07, and its newest snap is 2026-11-05 6:09 PM ET. The sha256 was the same before and after training and evaluation.
- **Contents:** **655 verified offensive snaps from 16 games.**
  - 572 against CPU and 83 against human opponents.
  - Basis: 508 standing user attestation, 147 per-snap or legacy.
- **Geometry:** history rows use the provisional AceMadden-derived geometry of the play that was run. That is a low-confidence prior and never a verified assignment. 369 rows have high-confidence extracted geometry, 200 partial, and 86 name only. The geometry export is a local file, kept out of git. Set `CFB_COACH_PROVISIONAL_EXPORT` or place it at `~/.cfb-coach/catalog/provisional_catalog_offense.v1.jsonl`. Without it the model still runs, but every design is `name_only`.
- **Variant outcomes:** zero verified variant outcomes exist. Variant effects are transferred from base plays and route shapes, with 50% shrinkage and added uncertainty.
- **Expert tags:** these are the catalog tags and the Civil.GG sheet (`datasets/madden27/civil_bucs_offense_sheet_expert_tags.json`, labeled "expert_tag, unverified"). They are never used as outcomes.
- **Neural challenger:** not trained. 655 snaps and 16 games are far below the gate of 5,000 snaps and 60 games.

### Features used to score the 26 variants

| Group | Features |
|---|---|
| Situation | down, distance (linear, log, short/medium/long), field position, red zone, inside 10, goal-to-go, backed up, late down, CPU vs human |
| Opponent belief | P(man), P(spot zone), P(match), one-high, two-high, zero, pressure, information, entropy |
| Formation and play | formation-name tokens; play type from the name (all 26 are passes) |
| Adjusted design | route families, depth bands and weights after adjustments: vertical, breaking, underneath, crossing, curl/sit, mean depth, depth spread, levels, side balance, man-beater and zone-beater shares, quick, deep, deep middle, flat, horizontal, motion, number of commanded or unrepresented adjustments, geometry status |
| Interactions (learned) | man × man-beater routes, zone × curl, match × crossers, pressure × quick or deep, two-high × deep middle, one-high × verticals, short yardage × quick, goal-to-go × quick, and others |
| Existing knowledge | `football_priors` concept-versus-coverage matchup edge; the existing model's logit (offset) |

## Evaluation (chronological, game-grouped, no leakage)

The evaluation is rolling-origin by whole game: 13 test games, each trained only on earlier games. The existing-model offset and the beliefs for each row also use only earlier data. Heads were chosen on the first 6 test games and then reported on the last 7.

Held-out log loss on all 13 test games (lower is better). Δ is the game-bootstrap 95% CI of the difference versus the existing model.

| Target | Existing model / base rate | Best new head | Δ vs existing (95% CI) | Chosen |
|---|---|---|---|---|
| success (n=599) | **0.676** (AUC 0.566, ECE 0.010) | strong-L2 logistic 0.679 | +0.003 (−0.013, +0.024) | **existing model** (no head beat it) |
| first down (n=599) | 0.686 (AUC 0.47) | strong-L2 logistic **0.664** (AUC 0.61, ECE 0.013) | −0.023 (−0.034, −0.011) | strong-L2 logistic |
| explosive (n=544) | 0.455 | strong-L2 logistic **0.440** (AUC 0.61) | −0.015 (−0.024, −0.007) | strong-L2 logistic |
| sack/turnover/loss (n=599) | 0.386 | strong-L2 logistic **0.382** (AUC 0.55) | −0.004 (−0.011, +0.004) | strong-L2 logistic (passed the selection margin; overall difference not significant) |
| 3rd/4th-down conversion (n=117) | 0.705 | 0.655 (AUC 0.66) | — | — |

- Plain L2 logistic and GBT success heads *hurt* held-out success: Δ +0.047 and +0.029, both significant. The do-no-harm rule therefore kept the existing model for success.
- For base plays in the catalog, expected versus observed on held-out games: Texas Y-Stutter Wheel predicted 0.61 vs observed 0.63 (15/24). Mtn Shuffle Verts Smash predicted 0.55 vs observed 0.27 (3/11, 95% CI 0.10–0.57). Clear Deep and Mesh Post have n=3 each.
- Uncertainty terciles for the auxiliary heads are in the report. The spread is calibrated only roughly.

Coordinator simulation on the 599 held-out situations, playing within each game. These are **predictions, not observed outcomes**:

- 17 distinct variants were picked (entropy 2.32 out of a possible 3.26 nats). The top pick was Texas Y Stutter Wheel 1 at 24%.
- Adjustments changed the top pick in 13.7% of situations compared with ranking the unadjusted base plays. Variants of the same base play take turns at the top: Mesh is split 199/164/120/115 across 4 variants, and Texas Y-Stutter Wheel 375/154/70.
- When the beater gate was high (41 situations), the pick was a tagged beater for the dominant coverage 93% of the time. With a low gate (175 situations) it was 34%.
- The existing model's own P(success) for the restricted pick's base play averaged 0.604. Its best play from the whole installed book averaged 0.626, so **the restriction costs about 0.02 predicted P(success)** by the existing model's estimate. The restricted mode's own 0.658 includes the expert and beater terms, which are unvalidated, and is **not** proof of a gain.
- **Short yardage (distance ≤ 2), 65 snaps:** historically 40% were runs. Runs succeeded 18/26 and passes 26/39.
- **Goal-to-go, 75 snaps:** 49% were runs. Runs succeeded 13/37 and passes 20/38.
- The restricted mode cannot run, QB-sneak or use the red-zone packages. That loss of options is real and has not been measured in a game.

## Playbook status

- All 4 formations exist in the in-repo Buccaneers stock data.
- 10 of 12 base plays match by exact name. **`Z Spot` and `Z Spot Shake` do not**: the repo's stock data has `L Spot` and `Zl Spot Shake`. PR #115 approved those names as *art aliases* only, so `civil_5wr_z_spot_1` and `civil_5wr_z_spot_shake_1` stay blocked until you confirm the in-game name with `ml civil-scheme verify --variant civil_5wr_z_spot_1 --book-play "L Spot"`.
- Catalog-valid: **24 of 26** (all except the 2 Z Spot variants). Verified executable: **0 of 26**.
- On the 2026-11-07 snapshot, the installed offense book (`ML Designed Offense`, 5 formations) **lacks Gun Bunch TE and Gun Tight Doubles**. `ml civil-scheme prepare-book --stage` stages a pending merged book.
- Civil.GG sheet cross-check: all 17 non-5WR variants match by name or probable name.
  - `Mtn Shuffle Verts Smash 2` is probably the catalog's "Return Motion (name unverified)".
  - `Mesh 1` is probably "Mesh - Outside Bunch WR Post".
  - `MTN Cross HB Sneak 3` is probably "Corner/Flat (name unverified)".
  - The sheet shows names `Mesh 2` and `Mesh 4`.
  - No IDs were changed, and every probable match still needs owner confirmation.
  - No Gun 5WR Tight plays are on the sheet. That is unknown, not negative.

## Learning

- **Per play:** the belief is rebuilt from the DB on every call, so each closed snap's coverage moves the next call. A verified variant outcome moves that variant through a shrunk residual (K=10, cap ±0.6 logit). A repetition penalty uses this game's earlier calls.
- **Per game (postgame):** `ml civil-scheme train` refits on verified snaps, including verified variant executions with exactly the adjustments that were run. Then `enable` the new artifact.
- **What counts as evidence.** A variant outcome counts only when all of these hold:
  - the executed formation and play match the variant;
  - `executed_status` is `identified`;
  - the adjustment was confirmed, or you filed `report-adjustments`.
- Skipped adjustments are kept, and the design is rebuilt from what actually ran. Changed adjustments are logged but not trained on.
- An unconfirmed civil recommendation is excluded from training, and a recommendation that wasn't executed is never evidence.
- Logs are kept in the additive tables `civil_scheme_decisions` and `civil_scheme_executions`. `snaps.recommended_adjustment_id` / `executed_adjustment_id` carry `scheme:<variant_id>`. `ml civil-scheme log` shows the per-variant log.

## Safety proofs in this PR

- **Defense unchanged:** `git diff --stat 5bd8dce -- $(git ls-files | grep -i defense)` is empty. The only `make_call` change is a new branch in the offense path, and `test_11` proves a defense call never enters the civil coordinator.
- **No live DB mutation:** training and evaluation open a backup copy with `mode=ro`, and the sha256 is the same before and after (in the report and in `test_10`). Only enable, disable, verify, prepare-book `--stage` and report-adjustments write, and each backs up the DB first.
- **Unrestricted coordinator preserved** (`test_06`). Old heuristic callers are not activated, because civil mode bypasses them.
- **Latency:** 29 ms median, 44 ms max for 30 shadow calls against the 655-snap DB on the build box. The target is 150 ms.

## Commands (from `~/cfb-coach` in WSL)

```bash
git fetch origin && git switch ml/lions-civil-scheme-only
# 0. optional: local provisional geometry (never commit it)
#    export CFB_COACH_PROVISIONAL_EXPORT=~/.cfb-coach/catalog/provisional_catalog_offense.v1.jsonl
# 1. prepare the book (dry run, then stage; install in Madden; then mark applied)
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme prepare-book
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme prepare-book --stage
PYTHONPATH=. python3 -m cfb_coach prep --game madden27 -o cpu --text
PYTHONPATH=. python3 -m cfb_coach prep --game madden27 -o cpu --mark-applied
# 2. train + evaluate (backs up the DB, trains on the read-only copy, writes ~/.cfb-coach/models/civil_scheme/<version>)
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme train
# 3. pregame checklist, then verify each variant you executed exactly in Practice
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme checklist
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme verify --variant civil_bucs_texas_y_stutter_wheel_1
# 4. shadow test (read-only, nothing logged; unverified variants allowed and labeled SHADOW)
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme shadow -o cpu --down 3 --distance 6 --yardline 45
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme shadow -o cpu --down 1 --distance 10 --yardline 25 --look "cover 3"
# 5. enable + prove the live path loads the expected model
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme enable
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme status
# 6. controlled CPU game (session flag works with or without enable)
PYTHONPATH=. python3 -m cfb_coach play --game madden27 -o cpu --civil-scheme-only
#    if you skipped/changed an adjustment on snap N:
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme report-adjustments --snap N --applied 1,3 --skipped 2
#    postgame: retrain, then enable the new artifact
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme train && PYTHONPATH=. python3 -m cfb_coach ml civil-scheme enable
# 7. disable / revert
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme disable
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme disable --revert-artifact
PYTHONPATH=. python3 -m cfb_coach play --game madden27 -o cpu --unrestricted-offense
```
Backups go to `~/.cfb-coach/backups/civil_scheme/` and models to `~/.cfb-coach/models/civil_scheme/`. Neither is in git.

## Limitations before Madden testing

1. 0 of 26 variants are verified executable, so live calls return `NO_LEGAL_SCHEME_PLAY` until you run `verify`.
2. Gun Bunch TE and Gun Tight Doubles are not in the installed book on the 11-07 snapshot.
3. Z Spot and Z Spot Shake need your confirmation of the in-game name.
4. No verified variant outcomes exist. Every adjustment effect is a transferred, uncertain estimate, and the success head is the existing model.
5. Four adjustment kinds are not in the repo's hot-route vocabulary: wheel, short cross, D-pad text, and stems on stock routes. They are flagged and need in-game checking.
6. Pass-only: no runs, sneaks or red-zone packages in this mode.
7. The provisional geometry export may not be on the laptop. Without it every design is `name_only`.
8. The live DB was not reachable from the build machine. The model was trained on the 11-07 snapshot, and games after 2026-11-05 are not included.
9. The coverage-beater weights (1.2 logit and the gate) are set by hand to follow your rule. They are not fitted, because no variant outcomes exist yet.
