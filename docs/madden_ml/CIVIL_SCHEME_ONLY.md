# civil_scheme_only — the 26-variant Lions offensive scheme mode (+ 12 designed runs per t349u, + 7 red-zone plays, field bands per t338u)

Status: **draft, off by default, not live-ready.** Nothing in this mode has been run in a Madden 27 game yet.
**Owner confirmation (2026-11-10, t333u):** Aidan confirmed that every catalog adjustment (wheel, short cross, stems up/down, motion, D-pad text and the rest) exists in game and that he knows how to do them. That is recorded in `adjustments.OWNER_CONFIRMED_ADJUSTMENTS`, pinned to catalog `lions_civil_bucs_5wr_v4@15c120decc4b`. Live calls are no longer blocked for unverified adjustments. Since t337u (below), every dataset formation counts as installed, so every play is live-legal in its field band (t338u): **inside the opponent 10, only the 7 red-zone plays; from the 10 to the 20, the 2 I Form Close Flex red-zone plays plus the 26 pass variants and the 12 designed runs; outside the 20, the 26 pass variants and the 12 designed runs.**

## What it does

The mode is selected per session or per DB, and the existing coordinator stays separately selectable:

| How | Offense coordinator |
|---|---|
| default (nothing set) | the existing unrestricted coordinator, unchanged |
| `ml civil-scheme enable` (DB meta) or `play ... --civil-scheme-only` (this session) | **civil_scheme_only** |
| `play ... --unrestricted-offense` (this session) or `ml civil-scheme disable` | the existing unrestricted coordinator |

In civil_scheme_only, every offensive call is exactly one of the 26 catalog variant IDs, one of the 12 designed runs (t349u) or one of the 7 red-zone plays, within its field band (see "Field bands" below). The call shows:

- the formation, base play, and **the exact variation: its name and ID, with every adjustment** (owner instruction t336u). This appears on the call's first line, which the snap log stores as `our_call`, and in the headline, the live card and the decision log (`prediction_json.variation`). A red-zone stock play reads "ADJUSTMENTS: none (catalog lists no adjustments: stock play)";
- **every** adjustment, in order and verbatim, numbered as in the catalog. Stems are tagged `[STEM UP]`, `[STEM DOWN]` or `[STEM DOWN (all the way)]` and are never dropped or hidden;
- warnings, an explanation, and P(success), first down, explosive and sack/turnover/loss chances, plus uncertainty.

Reads are shown only when a verified read progression exists, and none does today.

The mode never calls the heuristic `_pick_offense` or the experimental offense path. It never substitutes another play, never offers an unadjusted version (`unadjusted_allowed` is false for all 26; the stock designed runs and stock red-zone plays have no adjustments to drop), and never uses the unnamed 5WR setup.

## Designed runs and the run-pass mix (owner instruction 2026-11-11, t348u/t349u)

Aidan (t348u): "I also want the model to be able to call run plays out of the formations we use, we shouldn't be strictly pass pass pass there should be a good balance of running as well." (t349u): "The design runs, no need for screens and rpos besides the goal to go one."

**The 12 designed runs** are in `datasets/madden27/civil_designed_runs_owner_t349u.json`. Each is the exact stock play in `cfb_coach/data/madden27/playbooks_full.json` → `books.offense.Buccaneers.formations`, called as the stock play. Each has no adjustments, and its variation is the stock play name itself, so the call reads e.g. `Gun Bunch TE — Inside Zone · Inside Zone [civil_run_gun_bunch_te_inside_zone]` with "ADJUSTMENTS: none (catalog lists no adjustments: stock play)".

| Formation | Designed runs (variant_id `civil_run_<formation>_<play>`) |
|---|---|
| Gun Doubles Clamp Stack | Inside Zone, Counter Y, HB Mid Zone, Same Side Zone |
| Gun Bunch TE | Inside Zone, Read Option, HB Off Tackle, Boomerang HB Off Tackle, Same Side Zone |
| Gun Tight Doubles | Inside Zone, HB Draw |
| Gun 5WR Tight | QB Draw |

- **Not added (owner):** HB Slip Screen (all three formations), Return RPO Zone Alert, RPO Alert Out. The only RPO is still RPO Alert QB Draw in the Gun Empty Quads red-zone package, inside the opponent 10 only.
- **Loader fails closed** (`catalog.with_designed_runs`). It raises `CatalogError` for any of these: a name that is not an exact stock play in that formation; a formation outside the scheme's 4; a screen or an RPO; any adjustment; a `play_family` other than `run`; or a variation name that differs from the play name (`test_22b`).
- **The 26-variant catalog file is not edited.** The runs live in their own owner file, like the red-zone file. So the catalog version `lions_civil_bucs_5wr_v4@15c120decc4b` is unchanged, and the t333u/t337u confirmations and the trained artifacts pinned to it keep working without a retrain.
- **Field bands:** the runs are regular plays (`field_zone: any`). They are legal from the 10 to the 20 and outside the 20, and **never inside the opponent 10** (`test_23`, `test_18`).
- **Separate calls:** each formation + play is its own `variant_id`. Calling Inside Zone from Bunch TE adds a repetition penalty to that id only; Inside Zone from Doubles Clamp Stack is unchanged (`test_24c`). The existing-model offset is looked up per formation + play.
- `play_family` (`run` / `pass`) is explicit in the data. The 26 are passes, even the ones with "HB Sneak" in the name. In the red-zone file, HB Zone, Speed Option, QB Blast, RPO Alert QB Draw and Stretch are runs; Spot Y Option 1 and Mesh are passes.

**Why a gate was needed.** The untrained stock runs carry no expert situation or coverage tags. With only the 0.08 near-best mixer, they would sit behind 26 pass variants that get expert and beater terms, and could lose on every snap. That already happened to the I Form Close Flex Stretch in the 10–20 band.

**The mix** (`coordinator.pick_mixed`, used live and in the evaluation simulation). Outside the opponent 10, when both a run and a pass are legal:

```
logit P(run) = RUN_PRIOR_LOGIT[down][distance bucket]              soft situation prior
             + 10 x (best run utility - best pass utility)          learned: model heads, per-variant residuals,
                                                                    coverage-beater pull, repetition penalties
             + 0.6 x (passes - runs among this game's last 6 calls) tendency: stay unpredictable
P(run) clipped to [0.03, 0.85]
```

The family is drawn with the snap's seed, so a re-call of the same snap gives the same play. The usual seeded near-best draw (0.08, τ 0.025) then picks among that family's candidates, so runs are also mixed among themselves.

- The prior is soft. With equal utilities it gives 1st & 10 about 45% run, 2nd & 1–2 about 65%, and 3rd & 8–10 about 12%. The learned gap and the tendency term move it.
- If the runs keep failing, their residuals and the existing model pull the gap down and the run share falls toward 3%. If the opponent keeps showing one coverage, the pass beaters widen the gap and the mode passes more.
- Inside the opponent 10 the gate is off, and red-zone selection is exactly as before.
- In 10–20, Stretch competes in the run pool with the designed runs. The +0.35 red-zone lift still applies.
- The decision log stores `mix.family_gate`, `family`, `p_run`, `prior_logit`, `learned_gap` and `tendency_logit`.
- This is not a quota. Nothing forces a run on any down. A clearly worse family is still drawn rarely, by design, for unpredictability. The chance is at least 3% outside the 10 when both families are legal.

**Measured (box, predictions, not game outcomes).** The real artifact `v20261110-184648` was run on a scratch copy of the game-2 snapshot DB. There were 6 simulated games of 60 snaps each, with random spots from the own 15 to the opponent 12 and each call closed as a snap.

| Situation | Run share |
|---|---|
| 1st & 10 | 39/110 = 35% |
| 1st & 5 | 50% |
| 2nd & 3 | 67% |
| 2nd & 7 | 39% |
| 2nd & 12 | 21% |
| 3rd & 2 | 49% |
| 3rd & 6 | 18% |
| 3rd & 10 | 7% |

- All 12 designed runs and Stretch were called.
- Latency after the first call: median 66 ms, p95 73 ms, against the 150 ms target. The first, cold call took 0.68 s; `play` warms up before the first snap.
- `test_24` (test artifact): 60 first-and-10 snaps between the 20s give a run share between 25% and 75%, at least 6 distinct runs and at least 8 distinct pass variants. 3rd & 12 runs at least 15 points less often.
- `test_24b` checks that the learned gap, the situation and the tendency each move P(run), and that the gate is off inside the 10. `test_24d` checks the 10–20 band, and `test_25` checks that a run call shows its variation in html-live.

## Field bands, play mixing and variant independence (owner instruction 2026-11-10, t338u)

Aidan, after game 2: "when we are in goal to go sitations for example inside the 10 I want the model to only call the redzone plays from our book and the other redzone plays I sent from the previous book. These plays are to be called in the redzone only so inside the 10 and 10-20 you can call these and a mix of some other stuff. Also there has to be more variability in the plays … The variants should count as seperate play calls too … and I also still did not see variants next to the play call name in the html browser".

**Field bands** (`catalog.field_band`, `SchemeVariant.allowed_at`). They are enforced in the one legal filter before scoring, and again before display with the live snap's yardline:

| Band | Yardline | Legal plays |
|---|---|---|
| Inside the opponent 10 | ≥ 90 | **Red-zone plays only (7):** the 5 package plays plus the 2 I Form Close Flex plays. No regular variant. |
| 10–20 | 80–89 | The 2 I Form Close Flex plays (tagged 10-20 on the sheet) **plus** the 26 pass variants and the 12 designed runs. The red-zone plays get a +0.35 logit mix lift (`RZ_MIX_BONUS`), so they actually show up. |
| Outside the 20 | < 80, or spot unknown | The 26 pass variants and the 12 designed runs. A red-zone play is never legal here. |

**Red-zone plays added from the 5WR Tight sheet (t338u screenshots, not committed).** The sheet lists "Redzone HB Stretch / Close Flex" and "Redzone Mesh / Close Flex" under both "Redzone (Inside 10)" and "Redzone (10-20)". Both formation and plays are in the Buccaneers stock data (`I Form Close Flex`: `Mesh`, `Stretch`).

| ID | Formation | Play | Setup (verbatim from the lesson) | Band |
|---|---|---|---|---|
| `civil_rz_i_form_close_flex_stretch` | I Form Close Flex | Stretch ("Redzone HB Stretch") | "1. Hike" (stock, no adjustments) | inside the 10 and 10–20 |
| `civil_rz_i_form_close_flex_mesh` | I Form Close Flex | Mesh ("Redzone Mesh") | "1. Crosser Slot WR" → Slot WR: Crosser | inside the 10 and 10–20 |
| the 5 package plays (table below) | Pistol Bunch TE / Gun Empty Quads | | | inside the 10 only |

The sheet gives no coverage tags for the Close Flex plays (unknown, not negative). The unnamed 5WR setup stays blocked. `test_18` checks every yardline 0–100, the live calls, injected out-of-band plays (rejected before display), and the yardline match.

**Mixing** (`coordinator.pick`, `repeat_penalty`):
- **Repetition penalty, per variant_id only:** 0.45 logit × Σ 0.7^i over that variant's own calls in the last 8 snaps of this game, plus 0.10 per earlier call this game (capped at 1.0). Before t338u it was 0.25 × Σ 0.6^i. History now comes from the game's closed snaps (`snaps.recommended_adjustment_id`), so a re-call or preview of the same snap no longer counts twice.
- **Seeded sampling instead of argmax:** the call is drawn among the candidates within 0.08 utility of the best, with Boltzmann weights (τ 0.025). The seed is session + snap count + down/distance/yardline, so a re-call of the same snap gives the same play. A clearly better play still wins: anything more than 0.08 behind is never drawn. The decision log stores the pool and the probabilities (`scheme_variant.mix`). The evaluation simulation uses the same pick.

**Variants are separate calls.** Calling Texas Y Stutter Wheel 1 adds a penalty to that id only. Texas Y Stutter Wheel 2's score is unchanged (`test_18d`). Civil variant snaps also no longer train the shared base-play offset (the existing model, which is per formation+play). A variant's own outcomes move only its own residual (K=10, cap ±0.6). Route-feature heads still generalize across similar designs; that is learned from features, not from the base-play name.

**html-live.** The compact Madden page used to render only `formation — play`. It now shows `formation — play · <variation name> [<variant_id>]` in the main call line. Under it, a `#call-variation` block shows "VARIATION: … ADJUSTMENTS:" with every adjustment numbered. Both the server-rendered page and the JS refresh (`pending_recommendation.variation` / `variation_block`) do this, and so does the snap history list. `test_21` checks it.

## Every dataset play is installed (owner instruction 2026-11-10, t337u)

Aidan: "we can assume all formations in the dataset are in my book because they are so all plays are legal to call." This is recorded in `adjustments.OWNER_CONFIRMED_BOOK` and pinned to catalog `lions_civil_bucs_5wr_v4@15c120decc4b`.

- The stock-name check and the applied-book check no longer block any variant. They still run and are recorded in the verdict (`stock_play`, `applied_book_record_has_*`) for information.
- **All 31 plays are live-legal:** the 26 core variants anywhere, and the 5 red-zone package plays inside the opponent 10 only. This includes Gun Bunch TE, Gun Tight Doubles, both Z Spot variants and the red-zone formations.
- `prepare-book` is now optional and informational. `--stage` only syncs the app's book record.
- **Z Spot was a naming mismatch, not a book problem.** The repo's in-house Buccaneers stock data spells those plays `L Spot` / `Zl Spot Shake`, so an exact-name check failed. The Z Spot calls now use the dataset name and carry a warning saying so. `verify --book-play` remains as an optional alias.
- Unchanged:
  - the red-zone field gate;
  - the blocked unnamed 5WR setup;
  - the one legal filter, which runs before scoring and again before display;
  - only the listed IDs can be called;
  - every call shows its variation with all of its adjustments;
  - the mode is off by default.
- A different catalog version falls back to the book check, which needs a new confirmation (`test_03`).

## Red zone (owner instruction 2026-11-10, t335u; bands superseded by t338u above)

Aidan: "I want the model to call the redzone offense that is in the book." The catalog's two red-zone packages appear on the Civil.GG sheet as "Redzone - Pistol Bunch TE" and "Redzone - Gun Empty Quads" (bucket "Redzone (Inside 10)"). They are now candidates, defined in `datasets/madden27/civil_redzone_owner_t335u.json`. The unnamed 5WR setup stays blocked.

| ID | Formation | Play | Adjustments (verbatim from the catalog) |
|---|---|---|---|
| `civil_rz_pistol_bunch_te_spot_y_option_1` | Pistol Bunch TE | Spot Y Option ("Spot Y Option 1") | 1) Inside bunch WR: Drag; 2) TE: Drag; 3) HB: Swing toward TE; 4) Outside bunch WR: Crosser |
| `civil_rz_pistol_bunch_te_hb_zone` | Pistol Bunch TE | HB Zone | none in the catalog (stock play) |
| `civil_rz_pistol_bunch_te_speed_option` | Pistol Bunch TE | Speed Option | none in the catalog (stock play) |
| `civil_rz_gun_empty_quads_qb_blast` | Gun Empty Quads | QB Blast | none in the catalog (stock play) |
| `civil_rz_gun_empty_quads_rpo_alert_qb_draw` | Gun Empty Quads | RPO Alert QB Draw | none in the catalog (stock play) |

- **Where:** only inside the opponent 10, meaning 10 or fewer yards to the goal line (`yardline >= 90`). If the spot is unknown, they are not eligible.
  - The 10–20 does not count. The catalog tags both packages only "Redzone Inside 10", and the sheet's separate "Redzone 10-20" bucket holds only Mesh 1, which is already a regular variant.
  - The gate is in the one legal filter. It runs before scoring (`legal_set(..., yardline=sit.yardline)`) and again before display: `revalidate_call` checks the live snap's yardline, which `live_server` passes through `validate_live_call`, and it must equal the yardline the call was scored at.
  - The evaluation simulation applies the same gate.
- **Since t338u:** inside the 10 only red-zone plays are legal, and they are scored and mixed among themselves (`test_18c`).
- **Where the formations come from.** The catalog has `formation: null` for both packages. The formations come from the sheet names plus the in-repo Buccaneers stock book:
  - HB Zone, Speed Option and Spot Y Option appear together only in Pistol Bunch TE.
  - QB Blast and RPO Alert QB Draw appear only in Gun Empty Quads.
- **What the catalog does not have.** Nothing was invented to fill these gaps:
  1. a formation for either package;
  2. any adjustments for the QB-run package (the lesson says to watch it for the blocking setup, which was never transcribed);
  3. any adjustments for HB Zone or Speed Option (the 4-step setup applies to Spot Y Option 1);
  4. whether the package plays are alternatives or a sequence (each is scored as its own candidate).
- **Book:** on the 11-10 postgame snapshot, the applied book has neither Pistol Bunch TE nor Gun Empty Quads, and it also lacks Gun Bunch TE. Since t337u that record no longer gates anything. `prepare-book --stage` optionally syncs it.
- **Adjustment vocabulary:** swing and crosser were added. t333u covers every catalog adjustment, and these are catalog adjustments.

If nothing is legal, the answer is `NO_LEGAL_SCHEME_PLAY` with the reasons. There is no hidden fallback. An internal error also returns `NO_LEGAL_SCHEME_PLAY`.

## Pipeline per snap (`cfb_coach/madden/scheme/`)

1. **Legal filter** (`legality.py`). This one filter runs before scoring and again before display (`validate_live_call` → `revalidate_call`). A variant passes only if:
   - it is a catalog variant ID (26), or a red-zone package play (5) **and** the snap is inside the opponent 10;
   - its formation and base play are in the in-repo Buccaneers stock data, matched by exact normalized name;
   - both are in the applied (installed) offense book;
   - its adjustments are owner-confirmed executable. The blanket confirmation from 2026-11-10 (t333u) covers this catalog version. A new catalog version needs a new confirmation; `ml civil-scheme verify` still records per-variant confirmations and the `--book-play` name alias.

   **Only the 26 IDs can be output.** `test_15` runs 60 random situations and coverage looks and asserts every call is one of the 26. It injects an invented ID, a variant with one adjustment dropped, and a substituted base play into the scorer; all three are rejected before display (`NO_LEGAL_SCHEME_PLAY`). Blocked catalog entries never pass `check`.
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

### Stems and other adjustments in the derived design

All catalog adjustment kinds are representable: route changes, wheel, short cross, motion, D-pad text, stems on commanded routes, and stems on stock routes. `test_16` covers all 26 variants and checks:

- the adjustments appear verbatim and in order in the call text, `scheme_variant["adjustments"]` and the adjustment buttons;
- each of the 11 stems is tagged;
- each stem changes the derived design's route features.

**How stems are applied.** A stem on a commanded route shifts that route's depth band. A stem on a stock route (for example "Existing corner route: Stem down all the way" or "TE post: Custom stem up one D-pad click") shifts the band of the matching route family when the provisional geometry identifies exactly one such route. Of the 11 stems, 8 are resolved this way.

If no single route matches, the stem is applied **directionally**: the mean route depth moves about one band per step (−2 bands for "all the way down"), and stem-up/down counts become features. That applies to the "Outside doubles WR: Custom stem up one press on D-pad" stems on the 3 Mtn Cross variants. Each directional stem adds 0.10 logit sd to the uncertainty, and the call shows a warning.

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
| Adjusted design | stem-up / stem-down / directional-stem counts, route families, depth bands and weights after adjustments: vertical, breaking, underneath, crossing, curl/sit, mean depth, depth spread, levels, side balance, man-beater and zone-beater shares, quick, deep, deep middle, flat, horizontal, motion, number of commanded or unrepresented adjustments, geometry status |
| Interactions (learned) | man × man-beater routes, zone × curl, match × crossers, pressure × quick or deep, two-high × deep middle, one-high × verticals, short yardage × quick, goal-to-go × quick, and others |
| Existing knowledge | `football_priors` concept-versus-coverage matchup edge; the existing model's logit (offset) |

## Controlled CPU game, 2026-11-10 (session `703697284cd445de`)

The data comes from a read-only copy of the laptop DB taken after the game (sha `8c47e6d6…`). The game ran in civil_scheme_only:

- DB meta `civil_scheme_mode_json` was enabled at 3:51 PM ET with Aidan's artifact `v20261110-154950`;
- all 70 logged decisions say `civil_scheme_only` / `CALL`;
- all 53 offensive snaps carry `scheme:<variant>`.

| Item | Result |
|---|---|
| Opponent and result | CPU, **won 42–21**. The snap log has 41–21, a 1-point reconcile gap. |
| Offensive snaps | 53. All are identified, verified and accepted. The executed play, formation and variant equal the recommendation on every snap. No adjustment reports were filed. |
| Passing | 32 of 53 complete, 294 yards, 22 first downs or TDs, 6 TDs, 2 INTs |
| Variants used | 15 distinct (the applied book had Clamp Stack, Tight Doubles and 5WR, but no Bunch TE). Mesh Post 1 was called 15 times, Texas Y Stutter Wheel 3 11 times, Mtn Cross HB Sneak 2 5 times. |
| Red zone, inside the 20 | 5 trips, 4 TDs. The stalled drive was 1&6 at the 6: Mesh Post 1, Mtn Shuffle Verts Smash 1, then Mesh Post 1, all incomplete. |
| Errors | No `NO_LEGAL_SCHEME_PLAY` and no errors. Latency was not logged in this version; it is logged from this PR on. |

## Controlled CPU game 2, 2026-11-10 (session `8dbda393948d4392`, 5:38–6:20 PM ET)

Data: read-only copy of the laptop DB taken after the game (sha `e13acb02b55bc1be…`). civil_scheme_only was enabled with Aidan's artifact `v20261110-173853`. All 75 decisions are `civil_scheme_only` / `CALL` (catalog `lions_civil_bucs_5wr_v4@15c120decc4b`), with no `NO_LEGAL_SCHEME_PLAY` and no errors.

| Item | Result |
|---|---|
| Result | CPU, **won 49–28**. The snap log has 48–28, a 1-point reconcile gap. |
| Offensive snaps | 60. All are identified, verified and accepted, and the executed play equals the recommendation on every snap. No adjustment reports. |
| Production | 42 completions or positive runs / 60, 380 yards, 26 first downs or TDs, 7 TDs, 3 INTs |
| Latency | logged for 75 decisions: median 33.5 ms, max 170.6 ms, 1 over 150 ms |
| Variety | 12 distinct variants. Mesh Post 1 was called 21 times (17 completions, 148 yds, 3 TD, 1 INT), including 4 in a row. Texas Y Stutter Wheel 3 12, Z Spot Shake 1 9 (8 completions, 81 yds, 2 TD). 11 back-to-back repeats of the same variant. |
| Coverage | Mixed: Cover 3 Sky, Cover 2 Man, Cover 4 Quarters, Tampa 2, Cover 6. The belief's top label matched the seen coverage on 12 of 48 snaps; the gate stayed low (top share 0.17–0.31), so the beater term rarely fired. |
| Red zone | 8 trips inside the 20, 7 TDs. Of 6 snaps inside the 10, only one was a red-zone package play (RPO Alert QB Draw, 1&9 at the 9, −2; the drive stalled). The other 5 were regular passes (TYSW3, Mesh Post 1 ×2, Z Spot 1, Z Spot Shake 1). This is what t338u changes. |

## Evaluation (chronological, game-grouped, no leakage)

**Retrain on the game-2 snapshot, t338u** (artifact `v20261110-184648`, model sha `8955eb87e632fcad…`, report `civil_scheme_eval_20261110-184645.json`): 820 verified offensive snaps from 19 games, 113 of them verified variant outcomes (53 + 60). Copy sha unchanged before and after. Chosen heads are unchanged: success = existing, first down and explosive = strong-L2 logistic, negative = existing. Held-out log loss, all test games:

| Target | Existing | strong-L2 logistic | previous artifact (existing / strong-L2) |
|---|---|---|---|
| success (n=764) | 0.6785 | 0.6782 | 0.6788 / 0.6811 |
| first down (n=764) | 0.6836 | **0.6607** | 0.6835 / 0.6622 |
| explosive (n=694) | 0.4334 | **0.4219** | 0.4416 / 0.4293 |
| negative (n=764) | **0.3806** | 0.3727 (not chosen: failed the early-fold selection margin) | 0.3876 / 0.3817 |

Simulation with the t338u bands and mixing (predicted, not observed): 31 distinct picks (was 20), entropy 3.32 nats (was 2.28), 0 back-to-back repeats, the top variant at 6.7% (TYSW1, 51/764). By band: outside the 20, 593 regular; 10–20, 44 red-zone / 33 regular; inside the 10, 94/94 red-zone. The costs: the tagged-beater share at a high gate fell from 0.90 to 0.66 (n=50), and the existing model's P(success) for the pick's base play fell from 0.603 to 0.589. Variety is bought with about 0.014 predicted success.

Replaying game 2's 60 real situations through the live path on a copy, in a fresh session: 29 distinct variants, 0 back-to-back, top variant 8/60 (Z Spot Shake 1). Inside the 10: 5 different red-zone plays in 6 snaps. 10–20: 3 red-zone plays and 3 regular in 6 snaps.

Variant scores, 1st & 10 at own 30 with no coverage info: Z Spot Shake 1 now ranks 1 (P 0.593 → 0.720; 9 outcomes, residual +0.59 at the cap). Mesh Post 1 is 0.638 → 0.651 (36 outcomes). Texas Y Stutter Wheel 3 is 0.671 → 0.610: the shared base-play offset no longer learns from civil snaps, and its own residual is −0.01 over 23 outcomes. Bunch TE Mesh Post is 0.610 → 0.568 (6 outcomes, 3 completions, 1 INT).

Previous retrain (11-10 game-1 snapshot):



**Retrain on the 11-10 postgame snapshot** (artifact `v20261110-165046`, model sha `232a153a7e0d8355…`): 760 verified offensive snaps from 18 games. That adds the 11-08 game and today's game. 53 of the snaps are verified variant outcomes.

- The success head is still the existing model: log loss 0.679 vs 0.681 for strong-L2 logistic.
- First down: strong-L2 logistic is better by 0.021 (95% CI 0.011 to 0.031).
- Explosive: strong-L2 logistic is better by 0.012 (95% CI 0.001 to 0.022).
- Sack/turnover/loss: now kept on the existing model. The difference is −0.006 (−0.012 to +0.001), not significant, and it failed the selection margin.
- Simulation: 20 distinct picks. A tagged beater was picked when the gate was high 90% of the time (n=48).
- Red-zone package plays were picked 3 times in the simulation: RPO Alert QB Draw twice, HB Zone once.
- One game is a small sample. The per-variant corrections move by up to ±0.25 logit, and single snaps move a variant by about 0.2.

The numbers below are from the earlier 11-07 snapshot (artifact `v20261110-153235`).

The model was retrained after the stem features were added (artifact `v20261110-153235`, model sha256 `8fe4858b2c2965b9…`, report `civil_scheme_eval_20261110-153232.json`). The chosen heads are the same, and every head's held-out log loss is unchanged to 3 decimals, except GBT (first down 0.669 → 0.666 and explosive 0.454 → 0.452 on the confirmation folds; not chosen).

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
- Adjustments changed the top pick in 13.9% of situations compared with ranking the unadjusted base plays. Variants of the same base play take turns at the top: Mesh is split 199/164/120/115 across 4 variants, and Texas Y-Stutter Wheel 375/154/70.
- When the beater gate was high (41 situations), the pick was a tagged beater for the dominant coverage 93% of the time. With a low gate (175 situations) it was 34%.
- The existing model's own P(success) for the restricted pick's base play averaged 0.604. Its best play from the whole installed book averaged 0.626, so **the restriction costs about 0.02 predicted P(success)** by the existing model's estimate. The restricted mode's own 0.658 includes the expert and beater terms, which are unvalidated, and is **not** proof of a gain.
- **Short yardage (distance ≤ 2), 65 snaps:** historically 40% were runs. Runs succeeded 18/26 and passes 26/39.
- **Goal-to-go, 75 snaps:** 49% were runs. Runs succeeded 13/37 and passes 20/38.
- (Before t349u) The restricted mode could not run outside the red zone. Since t349u it calls the 12 designed runs; the simulation summary now reports `run_pass_mix` and `pick_run_share`. The numbers in this section predate it.

## Playbook status

- All 4 formations exist in the in-repo Buccaneers stock data.
- 10 of 12 base plays match by exact name. **`Z Spot` and `Z Spot Shake` do not**: the repo's stock data has `L Spot` and `Zl Spot Shake`. PR #115 approved those names as *art aliases* only, This was a naming mismatch, not book membership. Since t337u the Z Spot variants are live-legal under the dataset name, and `verify --book-play` is an optional alias.
- Exact name match in repo stock data: **24 of 26** (the 2 Z Spot names differ). Adjustments executable: **26 of 26** (t333u). Installed: **all** (t337u). **Live-legal: 26 of 26 anywhere, plus 5 of 5 red-zone package plays inside the opponent 10.**
- On the 2026-11-07 snapshot, the installed offense book (`ML Designed Offense`, 5 formations) **lacks Gun Bunch TE and Gun Tight Doubles**. Since t337u this no longer gates live calls.
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
- **Latency:** `play` loads the model and geometry before the first snap (`warm_up`, about 0.6 s, once). Then, on 40 live calls against a copy of the 655-snap DB on the build box: first call 79 ms, median 35 ms, max 79 ms. The target is 150 ms. Without the warm-up, the first call took 0.73 s.
- **Label:** civil calls show `LABEL: CIVIL_SCHEME_ONLY`, and the `play` banner shows `OFFENSE CONTROLLER: CIVIL_SCHEME_ONLY`. Before this fix they showed `HEURISTIC` even though the heuristic picker never ran (`test_17`).

## Commands (from `~/cfb-coach` in WSL)

```bash
git fetch origin && git switch ml/lions-civil-scheme-only
# 0. optional: local provisional geometry (never commit it)
#    export CFB_COACH_PROVISIONAL_EXPORT=~/.cfb-coach/catalog/provisional_catalog_offense.v1.jsonl
# 1. (optional since t337u) compare the app's book record with the scheme formations
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme prepare-book
# 2. train + evaluate (backs up the DB, trains on the read-only copy, writes ~/.cfb-coach/models/civil_scheme/<version>)
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme train
# 3. pregame checklist (every adjustment listed, stems tagged; all LIVE-LEGAL; red-zone plays "INSIDE THE OPPONENT 10 ONLY")
PYTHONPATH=. python3 -m cfb_coach ml civil-scheme checklist
# 4. shadow test (read-only, nothing logged; --live-legal-only applies the live filter)
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

1. Two controlled games have been played (113 verified variant outcomes). The I Form Close Flex plays and most red-zone package plays have no outcomes yet.
2. Installation is owner-confirmed (t337u), not read from the game.
3. The Z Spot names differ from the repo's stock data (`L Spot` / `Zl Spot Shake`). They are called by the dataset name.
4. Variant outcomes are still few. Every adjustment effect is a transferred, uncertain estimate, and the success head is the existing model.
5. Stem depths are not measured. 8 of 11 stems shift a specific route band on provisional geometry; 3 are applied directionally with extra uncertainty.
6. Since t349u the 12 designed runs are called outside the 10, but none has a verified outcome yet: their scores are the existing model's estimate for the stock play plus the shared heads. The run-pass prior and weights (`RUN_PRIOR_LOGIT`, `RUN_LEARN_W` 10, `RUN_TEND_W` 0.6, clip 0.03–0.85) are set by hand, not fitted. Inside the 20 the red-zone plays add HB Stretch (10–20 too), and inside the 10 HB Zone, Speed Option, QB Blast and RPO Alert QB Draw. Only RPO Alert QB Draw has a verified outcome (1 snap, −2).
10. The mixing constants (0.45/0.7/0.10, margin 0.08, τ 0.025, 10–20 lift 0.35) are set by hand to follow your variety rule. They are not fitted.
7. The provisional geometry export may not be on the laptop. Without it every design is `name_only`.
8. The live DB was not reachable from the build machine. The model was trained on the 11-07 snapshot, and games after 2026-11-05 are not included.
9. The coverage-beater weights (1.2 logit and the gate) are set by hand to follow your rule. They are not fitted, because no variant outcomes exist yet.
