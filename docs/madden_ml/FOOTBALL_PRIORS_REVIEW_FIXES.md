# Review fixes for the football-priors stack (follow-up to #125)

| # | Problem | Fix | Test |
|---|---|---|---|
| 1 | `ace_catalog.py art` wrote `<side>__<formation>__<play>.png`; the ingester split it as formation=`<side>`, play=`<formation>__<play>`, so nothing matched | the ingester reads `<side>__<f>__<p>` (side from the name) or `<f>__<p>` (+ `--side`), matches by catalogue identity **and** by the site's art-file name (which can differ from the link slug), reports unmatched files instead of guessing, and writes `review_queue_offense.csv` / `review_queue_defense.csv` | `test_end_to_end_download_then_extract_matches_identity_and_routes_to_the_right_queue` (fixture site -> discover -> sync -> real script download path -> extract with no `--side` -> geometry -> per-side queue) and the art-slug test |
| 2 | learned cells **replaced** the whole variant prior, flattening Robber/Man/Spot/Match/Palms | learned values are RESIDUALS added to the variant-specific prior at two levels: coverage group (Cover 1 and Cover 2 Man are separate groups) and variant (robber, hole, man, trap, cloud, buzz, match, palms, quarters, ...), each with its own shrinkage (sd 0.07 / 0.06) | `test_learning_adds_a_residual_and_keeps_the_variant_distinctions` (the four named pairs keep >50% of their gap after 40 contrary snaps), variant-residual test |
| 3 | an opponent's snaps entered the Madden-wide posterior and again as the opponent posterior's evidence | leave-one-opponent-out: for a known opponent the Madden level is fit on the OTHER opponents; its own snaps enter once, at the opponent level (sd 0.10). An unseen opponent uses all snaps. Group residuals use each snap once, then variant residuals use what the group did not explain | closed-form test (`_fit` once, and a strictly smaller magnitude than the double-count answer), multi-opponent test, monotone/bounded repeated-outcome test |
| 4 | every route-derived concept got trust 1.0 regardless of evidence | trust = weakest tier among the routes forming the concept: in-game 1.0, diagram 0.9, source-documented 0.6, inferred (incl. hot-route transforms) 0.5, unknown 0; name hint 0.35. The real 8 annotated plays are source-documented, so they now count 0.6 | tier-ordering and weakest-route tests |
| 5 | the winning adjusted play was found by play NAME only | `PlanScore` carries (formation, play); the winner and its stock probability are looked up by full identity | duplicate-name test (same play name in two formations, only the second gains) |
| 6 | any protection call removed the back from the checkdown | a back leaves the checkdown/reads only when a **chip block names that player**; a Half/Full Slide protection removes nobody and the card says its effect on the backs is undocumented | chip vs protection vs none test |
| 7 | intermittent `test_live_card_adjustment_is_legal_...` failure | see section below | |

## Traced decisions (synthetic artifact; hot-route gain stubbed because the real model rejects hot routes as outside training support)
`Cover 1 70% / Cover 2 Man 30%` vs `Quarters` vs `Blitz` through `decide` -> `_apply_actions` -> `build_card` (see `test_full_trace_of_one_live_offensive_decision`): the
ranking, matchup edge, adjustment set, transformed route map and the read order all change with the look, and the card reads the same transformed structure the decision selected.
`test_full_trace_through_the_ui_state_and_the_logged_result` follows the same chain through the real live controller: card and compact-UI state for the call, then the typed defense and
result stored with the call (exact `coverage_typed`, attested execution) and the shadow record with the football-priors payload. One defensive shadow decision
(`defense_shadow.py`, 123 installed calls, applied=False, < 150 ms) ranks differently for a Mesh-heavy, a Smash-heavy and a screen/quick-game opponent; calls no prior speaks to abstain instead of ranking as "neutral".

## Real-game evaluation (item 8)
**Not run.** The sandbox cannot reach your computer or `~/.cfb-coach/madden27.db` (no folders are shared with this session). Run on your machine:
```bash
cp ~/.cfb-coach/madden27.db ~/madden27_copy.db
python scripts/madden_ml/football_priors_eval.py --db ~/madden27_copy.db --out ~/priors_eval
```
It prints name-only vs structured-prior vs structured+learned log loss, the game-clustered intervals and the champion/challenger gate (KEEP_CHAMPION unless the interval clears zero over >= 8 games).
No synthetic number is a substitute and none is reported as one.
