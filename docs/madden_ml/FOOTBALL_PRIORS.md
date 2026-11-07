# Football priors, coverage-conditioned decisions and adaptive learning (issues #124 / #121)

Draft, stacked on #123. **Opt-in and default OFF** so the running CPU experiment is unchanged. Real-game evidence: **none yet**; everything numeric
below comes from synthetic or fixed inputs and shows that the knowledge changes decisions, not that the hypotheses are true in Madden.

## 1. The prior layer (`football_priors.py`, extends `football_knowledge`, `coverage_intelligence`, `offense_taxonomy`)
* 68 conditional priors over 18 offensive concepts and 17 hierarchical defensive profiles (Cover 0; man one-high with robber/hole deltas; Cover 2 zone / man / trap; Cover 3
  spot / match / cloud / buzz; Cover 4 with quarters / palms; Cover 6; pressure with zone-blitz / man-blitz deltas). Each entry: a signed logit lift (|lift| <= 0.35, mostly 0.03-0.18),
  confidence, mechanism, counters, failure conditions, evidence ids, and whether it needs route-level assignments.
* Profiles are predicates over the existing coverage axes, so a variant adds a delta to its family (Cover 3 match vs spot, Cloud, Palms vs Quarters); `cover_2_man` is not
  `cover_2_spot`. Unknown coverages (`cover_9`, `unknown`) **abstain** (edge 0).
* Concepts from route geometry count 1.0; concepts guessed from a play NAME count 0.35.
* `matchup_edge(structure, coverage_mixture)` is the expected lift under the opponent's coverage MIXTURE (capped at 0.6 logit). `route_read_scores` orders QB reads by coverage group.
  `defense_call_edge` reads the same table from the defensive side (shadow only). Nothing selects a play.

## 2. Where it changes decisions (all gated by `CFB_COACH_FOOTBALL_PRIORS=1` or `ml_football_priors=1`)
| Path | Change |
|---|---|
| Live ranking (`offense_route_ranker.apply`) | each of the 93 plays' score gets `+edge` before ranking; the edge and its top cells are logged on the card payload |
| Decision/adjustment search (`offense_decision.decide`) | candidate plays re-scored with the edge; every transformed (hot-routed) play gets its OWN edge, so an adjustment's gain depends on the coverage; pass protection gains vs pressure mass |
| QB reads (`offense_presnap_advice.build_card`) | inferred order and a new COVERAGE-CONDITIONED ORDER (beside the documented order, never replacing it) come from route priors under the mixture; a back kept in to protect is not offered as the checkdown |
| Pregame / defensive live | **not wired** (see 6) |

Evidence (synthetic artifact, fixed situation 2&7 at the 45): Cover 1 vs Quarters moves **14** plays by >= 3 ranks with priors ON versus **3** OFF; Hi Lo Cross is 6th vs Cover 1
(13th OFF) and 12th vs Quarters; its conditioned read order changes with the look (`WR1, SLWR, TE1...` vs `WR1, SLWR, WR2...` vs `HB1, WR1...` under pressure). Through the real live loop
(three reported defenses fed through the form) a Cover 1 history and a Quarters history give different `route_aware_top` lists and probabilities with priors on. The top-1 play often
stays the same on this synthetic artifact: the priors are a few percent of logit by design, they initialise and do not dominate.

## 3. Adaptive learning (`adaptive_priors.py`)
Three levels per (coverage group, concept) cell: general prior -> Madden posterior (all closed snaps) -> opponent posterior (partial pooling, sd 0.10). Normal-normal updates on the
lift estimate; trust 1.0 route-derived / 0.35 name-only; attested execution 1.0, unattested 0.5; penalties ignored; sacks/turnovers half. Prior sd 0.07-0.2 by confidence: **one failed
snap moves a cell about 0.01, twenty move it about 0.1-0.2**; a concept is never abandoned after one play (tested). Revealed coverage is used as outcome context only. Cautious exploration
(`CFB_COACH_EXPLORE=1`): among near-tied candidates prefer the less-tested estimate (deterministic UCB), never on 4th down, the goal line, a late lead, or when one option clearly leads;
the choice and reason are logged. p0 is the pooled success rate (a documented approximation, not a context-specific baseline).

## 4. Champion / challenger (`football_priors_eval.py`, `scripts/madden_ml/football_priors_eval.py`)
Whole-game chronological folds; compares name-only (current) vs structured prior vs structured + learned; game-clustered paired bootstrap; **KEEP_CHAMPION unless the interval clears zero
over >= 8 games**. Synthetic planted worlds (22 games): null, priors-right and priors-reversed all return KEEP_CHAMPION with differences of +/-0.0003 log loss. So the held-out value of the
priors is **not demonstrated**: the edge acts through the opponent's *expected* coverage, and its worth scales with how predictable the opponent is. `cpu_experiment.py prepare` now runs this
on your database copy (informational, not a gate).

## 5. Pilot switches
`python scripts/madden_ml/cpu_experiment.py enable ... --football-priors` (after `prepare`); `status` prints ON/off; `rollback` restores the previous value.

## 6. Not done / needs validation
* Pregame architecture does not yet use the priors (formation proposals still score as before); original-macro generation and macro-vs-model choice were not reworked.
* Defensive live path: only the shadow function `defense_call_edge` exists and is tested; it is not yet called from the live defensive shadow.
* Real held-out games, real latency with priors on (unit test: ranking 93 plays < 150 ms), and a CPU game with priors on: not run.
* Hot-route transforms still extrapolate outside training support (the model rejects them), so most selected actions will remain NO ADJUSTMENT until first-party route assignments exist.
* Only 8 of 93 plays have route assignments; the rest use name hints at 0.35 trust.
* Catalogue: structured AceMadden authorization record + scope checks were added (`ace_catalog.py authorize`); no bulk download was run because the email text has not been provided.
