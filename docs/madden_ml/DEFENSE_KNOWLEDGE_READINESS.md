# Defensive knowledge (issue #116): readiness report

**Status: draft, research-grade. Nothing here is merged, installed, enabled or promoted.** Your live database and every trained
model were not touched. All evaluation ran on synthetic data or read-only copies, with `HOME` pointed at a temp directory.

## 1. Verdict

| Question | Answer |
|---|---|
| Is there a source-backed knowledge entry for every required play? | **Yes: 150/150** (123 installed + 27 external), each with provenance, a status per claim, and an explicit unknowns list. |
| Are the eleven defender roles of any play verified? | **No: 0/150.** No first-party Xbox evidence exists yet. The ledger is `unknown` everywhere. |
| Does it improve play-calling on your real games? | **NOT EVALUATED.** No game database exists in this environment. The commands to find out are in section 7. |
| Does it improve a *synthetic* world with a planted effect? | Not demonstrably (section 5.3). 16 games: no gain, one small harm. 30 games: −0.0116 with an interval that still includes zero. |
| Is the exact typed call now kept? | **Yes, going forward** (section 3a): additive `snaps.coverage_typed`, no new form. Older rows stay ambiguous. |
| Ready to activate / promote / use as a primary caller? | **No-go.** Preview-only, and only after section 7's `heldout` command returns GO on a copy of your real database. |
| **GO / NO-GO for the first CPU game** | **NO-GO to run the knowledge model as the caller. GO to play the first CPU game on the existing route-aware coordinator while logging with the new capture**, so the data the knowledge model needs starts accumulating. Nothing here is activated. |
| Ready for review as a draft research PR? | **Yes.** |

## 2. What is now understood, name-only, or unknown

| | Count | Notes |
|---|---:|---|
| Entries that exist and resolve | 150 / 150 | the old recognizer named a coverage for 38 / 123 installed and 0 / 27 external |
| EA-documented general mechanic applies | 32 | match families (Cover 3 Match, Quarters, Palms, Cover 6), Zone Pressure / sim style, press |
| Name decomposed, meaning = football hypothesis or unknown | 111 | tokens parsed; assignment detail unverified |
| Catalog-only (nothing decomposable) | 4 | `Thunder Smoke`, `Free Fire`, `Engage Eight`, `Over Storm Brave` (audit-classified only) |
| Third-party per-position jobs exist (not trainable) | 3 | `Cover 3 Sky`, `Tampa 2`, `Cover 4 Quarters` in Nickel Over |
| **Eleven roles verified from first-party evidence** | **0** | |
| Deep shell known / coverage type known (installed) | 73 / 71 of 123 | the rest are pressure names whose coverage behind them is genuinely unknown |
| Pressure typed: zone blitz / man blitz / sim / stunt / untyped | 3 / 3 / 7 / 3 / 31 installed | a rusher count is verified for none |
| Token meanings that are **unknown** | `Willie`, `Sammie`, `Sting`, `Thief`, `Coffee House`, `Okie`, `Eagle`, `Storm Brave`, `Thunder Smoke`, `Free Fire`, `Engage Eight`, `Overload`, `DBL`, `Go`, `Lock` (this call), `23` | recorded as unknown, never guessed |

## 3. The finding that bounds everything else: the post-play form is a lossy channel

The existing post-play form turns "they ran" into `coverage_seen` with keyword rules. Run on the 109 distinct call spellings in the
catalog (the real parser): **7 stored as typed, 77 rewritten to a coarser label, 25 stored as nothing.** Of your 123 installed plays,
**25 are dropped, 69 are stored under a label shared with five or more other calls, 27 under a label shared with one to four, and only 2
unambiguously.** Examples: all 15 "Cover 3 …" calls become `Cover 3 Sky`; every Cover 1 variant becomes `Cover 1`; 31 pressure names
become `pressure`; `Cover 2 Match` becomes `Cover 3 Match`.

Consequences: (1) a stored `Cover 3 Sky` is not evidence of Sky, (2) the Sky / Cloud / Buzz / Hole / Robber / Palms distinctions this
work documents are **invisible to the model** unless the typed call survives, and (3) the history already used by the current model
carries the same ambiguity. The code handles this by reading a stored label as a uniform mixture over the calls the *current parser*
maps to it (family structure kept, variant blurred) and reading a label the form can never produce as exact. **Capture is now fixed going forward (section 3a); history logged
before the fix stays ambiguous and is read as above.**

### 3a. Capture fix (issue #116, priority 1)

`snaps` gets one **additive, nullable** column, `coverage_typed`, added by the existing guarded `ALTER TABLE` migration. The post-play form
is unchanged (same field, same workflow); `LastSnapBook.close_from_form` now also stores the exact "they ran" text there while
`coverage_seen` keeps the same normalized value it always had. Rows logged earlier keep `NULL`; nothing is back-filled or guessed, so they
remain ambiguous. Readers (knowledge path only) use the typed text when it resolves to a catalog name or untyped pressure and otherwise fall
back to the stored label exactly as before. The default model path ignores the column. Undo, corrections and an older database file
without the column are covered by `tests/madden_ml/test_defense_typed_capture.py` (12 tests), which drive HTML-form submissions
(`LivePlayController.result_and_call`) for Cover 3 Cloud, Fire Zone 3, Cover 3 Buzz Match, Cover 4 Quarters, Cover 4 Palms, Zero Blitz,
Cover 2 Man and Cover 1 Robber through storage to the next offensive decision's coverage belief. Tradeoff: only calls logged after this
change benefit; the first games after it are a prior-only period for the new distinctions.

## 4. Delta against prior work

| Prior work | What it had | What this adds |
|---|---|---|
| #107 `coverage_intelligence` | scheme class for 72 / 123 installed (51 `unknown`) | cross-checked: where both commit they **never disagree** (69 / 69 deep shells, 59 / 59 coverage types). Coverage-axis breadth is similar (73 / 71 vs 72); the additions are the typed pressure axis (47 vs 36 installed plays), status per claim, an unknowns list, siblings, 27 external variants, the ledger, and the pipeline |
| #111 baseline and 20-entry knowledge | `ALIASES` recognizer (38 / 123), historical coverage baseline | catalog resolves 123 / 123; `ALIASES` is untouched and every alias the old recognizer accepts maps to the same parent class (tested). The standalone historical *predictor* (`predict_coverage`) is not extended |
| #113 route-aware live loop | selectable ranker, time-correct coverage, nested "do no harm" selection | reused unchanged. A knowledge artifact flows through the same loop (route_aware / preview), refused if stale; default artifacts are byte-identical |
| #115 aliases + gap audit | user-approved art aliases (digest changed); audit of 85 unrecognized names | merged into this branch; the audit's name-based pressure/coverage groups are the `repo_audit_115` basis for four low-evidence entries |

## 5. Results

### 5.1 Name recognition (deterministic)
Installed 38 → 123 of 123; external 0 → 27 of 27; legacy disagreements 0; unresolved 0. This measures **name recognition**, not football
value and not winning.

### 5.2 Tests
* 100 new tests (catalog, resolver, pipeline, live loop). Full `tests/madden_ml`: **1132 passed, 1 failed, 3 skipped.** The one failure,
  `test_backup_manifest_precedes_mutation_and_failure_aborts`, also fails on the unmodified #113 + #115 stack (running as root defeats its
  chmod-based error injection) and is unrelated.
* The consolidated #113 + #115 stack alone had 2 failures; the second was the integration-file guard, whose allow-list pinned to #112's head
  rejected #115's four reviewed files. The allow-list was extended explicitly (with a comment) rather than the guard being bypassed.
* Rest of the repository suite (`tests/` excluding `madden_ml`): exit 0, no failures.

### 5.3 Synthetic controls (PLUMBING ONLY: they say nothing about real games)

A planted world: the opponent's call, not what the form stored, decides success (pressure calls reward shallow route-art geometry,
coverage calls mildly punish it). Games differ in how much pressure they show, so earlier entries carry information. Negative = the
knowledge model is better. "Today's capture" passes each typed call through the real form parser; "lossless" stores the exact call.

**Fixed penalty, 18 games (train 12 / test 6), three seeds, `log-loss(knowledge) − log-loss(identical model without it)`:**

| World | seed 7 | seed 11 | seed 23 |
|---|---:|---:|---:|
| Planted, today's capture | −0.026 | −0.035 | −0.019 |
| Planted, lossless capture | −0.034 | −0.040 | −0.028 |
| Null (no effect) | +0.005 | −0.004 | +0.005 |
| Planted, labels shuffled across all rows (valid null control) | −0.001 | +0.003 | −0.002 |

A first version of the shuffle control (within each game) still showed a gain; that is not a bug but a diagnostic: it preserves each game's
pressure *tendency*, which is exactly what the model learns. The valid control shuffles across all rows. Both are in the CLI, labelled.

**Production protocol (nested game-level selection with the "do no harm" fallback), chronological held-out folds, game-clustered bootstrap:**

**16 games × 40 snaps (13 held-out folds), effect strength 2.0**

| World | Verified games | Held-out folds | Δ log loss (knowledge − baseline) | 95% interval | Verdict |
|---|---:|---:|---:|---|---|
| A planted signal TODAYS CAPTURE | 16 | 13 | +0.0009 | +0.0000 .. +0.0027 | NO-GO |
| B planted signal LOSSLESS CAPTURE counterfactual | 16 | 13 | +0.0057 | +0.0000 .. +0.0171 | NO-GO |
| C null no effect | 16 | 13 | +0.0001 | -0.0017 .. +0.0016 | NO-GO |
| D planted labels shuffled globally NULL CONTROL | 16 | 13 | -0.0001 | -0.0002 .. +0.0000 | NO-GO |
| E planted labels shuffled within game DIAGNOSTIC | 16 | 13 | +0.0068 | +0.0002 .. +0.0189 | NO-GO |

**30 games × 45 snaps (27 held-out folds), effect strength 2.0; worlds A and C only**

| World | Verified games | Held-out folds | Δ log loss (knowledge − baseline) | 95% interval | Verdict |
|---|---:|---:|---:|---|---|
| A planted signal TODAYS CAPTURE | 30 | 27 | -0.0116 | -0.0300 .. +0.0044 | NO-GO |
| C null no effect | 30 | 27 | +0.0003 | -0.0000 .. +0.0009 | NO-GO |

At 30 games the planted-signal world moves in the right direction (−0.0116, about 1.7% of a 0.69 log loss) but its interval still includes zero, and the null world
stays at zero; so even in a world built to contain the effect, this protocol has not yet shown a demonstrated gain. These worlds were
generated by the same code that reads them: they test plumbing and control behaviour, not football.

Reading: with the production protocol the knowledge residual needs enough games before it switches on. Direct inspection of the
selection record: at 16 games both residuals stay OFF; at 24 games the knowledge residual switches ON (inner held-out 0.6900 vs the
base coordinator's 0.7035) while the default one stays OFF. Where it is ON with too little evidence it can cost a little (see B above).

### 5.4 The offline defensive-candidate gate
The live trainer's thresholds alone were too weak: on pure coin-flip data they passed 2 of 24 runs at 6 games. The knowledge comparison
therefore also requires **>= 8 games and a game-clustered bootstrap interval that excludes zero**. Measured at 10 games × 20 snaps of
pure noise: **0 of 24 passed** (the old thresholds alone: 0 of 24). At 6 games the interval-only version still passed 1 of 24, hence the
8-game floor. A pass on real data would allow an offline experiment, never a promotion; the artifact's schema is rejected by the live
defensive loader.

### 5.5 Latency
Scoring all 93 installed plays against one opponent mix: default artifact about 17–31 ms, knowledge artifact about 25–47 ms (budget 150 ms; ranges across
repeated runs on a shared 4-core machine).
First-use costs: catalog load and validation 16 ms, form-channel table 10 ms.

### 5.6 Real data
**NOT EVALUATED.** The only labelled data in the repository is synthetic. The `train` and `heldout` commands were exercised end to end on
a synthetic database built with the real schema (`CoachDB`): both artifacts validated, no file was created under the real home directory,
and `heldout` produced a verdict. That verifies plumbing, not value.

## 6. What exactly needs first-party evidence

`DEFENSE_SCREENSHOT_NEEDS.csv` lists all 150 entries in priority order. **Tier 1 (101 installed plays)** are those the form drops or
collapses with five or more other calls, i.e. where verified detail would add the most: 32 stored as `pressure`, 25 stored as nothing, 12 as
`Cover 1`, 11 as `Cover 3 Sky`, 9 as `Cover 4 Quarters`, 7 as `Cover 3 Match`, 5 as `Cover 2`. Per play, one Xbox pre-snap play-art
screenshot showing all eleven defenders with their zone / man / rush paths and the play and formation name. The catalog records the file
hash; nothing is marked verified before that. Until then each entry's `unknowns` and `missing` lists say exactly what is not known.

## 7. Running it on your real data (exact commands, all read-only)

```bash
# 0. work on COPIES; every command refuses a path under ~/.cfb-coach
cp ~/.cfb-coach/madden27.db ~/madden27_copy.db
# 1. the default route-aware artifact must be retrained anyway: #115 changed the route-art digest
python scripts/madden_ml/offense_route_aware_readiness.py --db ~/madden27_copy.db --out ~/ra_default --run-tests
# 2. how much of YOUR data is collapsed, and the knowledge-vs-baseline held-out comparison with an explicit GO / NO-GO
python scripts/madden_ml/defense_knowledge_eval.py audit   --out ~/dk_eval
python scripts/madden_ml/defense_knowledge_eval.py heldout --db ~/madden27_copy.db --out ~/dk_eval
# 3. only if heldout says GO: train the knowledge artifact and replay recorded games as shadow decisions (never acts)
python scripts/madden_ml/defense_knowledge_eval.py train --db ~/madden27_copy.db --out ~/dk_art
python scripts/madden_ml/offense_route_aware_readiness.py --db ~/madden27_copy.db --out ~/ra_dk --defense-knowledge
# 4. CPU shadow experiment: PREVIEW only (the current ranker still makes the call; the knowledge pick is logged separately)
python -m cfb_coach ml offense-ranker route_aware_preview --artifact ~/dk_art/offense_route_aware_defense_knowledge.json
# roll back instantly
python -m cfb_coach ml offense-ranker experimental
# our own defense, offline (needs the live defensive artifact JSON only as a read-only vocabulary)
python scripts/madden_ml/defense_knowledge_eval.py defense-offline --db ~/madden27_copy.db --current <defense artifact> --out ~/dk_def
```
`--defense-knowledge` deliberately refuses `--install`. `route_aware` (primary) is not recommended for a knowledge artifact until a
preview run has been reviewed.

## 8. Decisions for you

1. **Lossless capture of the typed call: done in this PR** (section 3a; additive column, no workflow change). Review the migration before
   merging. The "lossless capture" rows in 5.3 are now the relevant model of what new games will store.
2. **Screenshots** for tier 1 (section 6).
3. **Run section 7** on a copy of your real database and read the `heldout` verdict. Nothing else should be acted on first.

## 9. Limits and caveats

* Excerpts from EA's page were relayed through a summarizing fetch tool; re-check wording before quoting publicly. EA's deep dive is dated
  2026-06-04 and does not define Sky / Cloud / Buzz, Tampa 2, Robber / Hole, Cover 9, fire zone or spy.
* The Cover 1 Hole / Robber / Contain article is dated 2026-01-09, before Madden 27 shipped, and names no version. It is `third_party_unverified`
  and not a feature source. The Buzz / Sky / Cloud / Mable naming rule appeared only in a search-result summary; the article it was attributed
  to was fetched and does not contain it, and two other candidate sources returned HTTP 403. It is a labelled hypothesis, and `Mable` is left
  zone-vs-match `unknown` because the two descriptions conflict.
* Madden Prodigy's catalog was read partially (first 100k of about 228k characters); an absent name is not proven absent. Madden Tools, 818 Madden
  and Huddle were not ingested (their terms restrict extraction / ML reuse).
* The uniform mixture for collapsed labels is a maximum-entropy assumption, not a measured frequency.
* Synthetic worlds were designed by the same author as the features; they test the machinery, not the football.
* False-positive rates were measured on a few dozen small synthetic runs; they are indications, not guarantees.
* Opponent pre-snap adjustments are unobservable on offense. None is inferred, asked for, stored or fed to a model.
