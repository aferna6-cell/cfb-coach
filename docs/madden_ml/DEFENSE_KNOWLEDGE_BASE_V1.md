# Madden 27 defensive knowledge base v1 (issue #116)

A source-backed, machine-readable catalog of every defensive formation/play in the recorded 123-entry book (`DEF-F1CF-1862`) plus the
27 external variants named in the issue, wired into the offense model as **additive, opt-in features** and into an **offline**
candidate scorer for our own defense. **Nothing is activated, promoted or merged by this change.**

> **What it is not.** It is not a coverage-beater table, not a heuristic play caller, and not defender-level knowledge. No entry has a
> verified eleven-defender assignment (`ledger_known_slots = 0`). Opponent pre-snap adjustments are unobservable on offense and are
> never inferred, assumed absent, or requested from the user.

## Files

| File | Role |
|---|---|
| `cfb_coach/data/madden27/defense_play_catalog.json` | The catalog (150 entries, sources, lexicon). Generated; digest-pinned. |
| `cfb_coach/madden/model/defense_lexicon.py` | Data only: sources, per-token meanings with evidence status, per-name specs, stress hypotheses. |
| `cfb_coach/madden/model/defense_catalog.py` | Builds and **validates** the catalog (completeness, provenance, no-fabrication gates). |
| `cfb_coach/madden/model/defense_semantics.py` | Runtime: label resolution (incl. the post-play form channel), mixes, the `dk_*` feature block, explanations. |
| `cfb_coach/madden/model/defense_knowledge_eval.py`, `scripts/madden_ml/defense_knowledge_eval.py` | Audit, synthetic controls, held-out comparison, latency. |
| `cfb_coach/madden/model/defense_knowledge_train.py` | Offline candidate scoring for our own calls (schema the live defense caller rejects). |
| `docs/madden_ml/DEFENSE_KNOWLEDGE_MATRIX.csv` | One reviewable row per entry: structure, evidence status, nearest sibling and the cited difference, unknown count. |
| `docs/madden_ml/DEFENSE_SCREENSHOT_NEEDS.csv` | The first-party captures that would verify the most, in priority order. |
| `scripts/madden_ml/build_defense_catalog.py` | Deterministic rebuild; `--check` fails if the committed catalog differs. |

## Evidence vocabulary (every claim carries one)

| Status | Meaning | May reach model features? |
|---|---|---|
| `xbox_verified` | first-party play-art screenshot with a recorded hash. **No entry has this.** The validator rejects it without a hash. | yes |
| `ea_documented` | EA's official Madden 27 text states the general mechanic (match checks, Smart Zones, Zone Pressure style, press depth) | yes |
| `name_derived` | the label *contains* a token or digit. Says nothing about what the token means. | yes (presence only) |
| `football_hypothesis` | a standard-football meaning from a CC BY-SA reference or the author's convention; Madden's exact implementation unverified | yes (as structure, with the evidence features beside it) |
| `third_party_unverified` | an educational site states it; version/primary evidence unverified | **no** |
| `unknown` | nothing usable; the catalog records what is missing | n/a (zero-information) |

Sources whose terms restrict extraction or ML reuse (Madden Tools, 818 Madden, Huddle) are **not ingested**. The three per-position
jobs already in the repo from Madden Tools are shown for research only (`model_input: false`, validator-enforced).

## What a catalog entry holds

`raw_name` and `raw_formation` exactly as recorded; `name_key` and aliases; separate **coverage** (`deep_shell`, `type`, `variants`) and
**pressure** (`concept`, `rusher_count`, `dropping_defenders`, `origin_tokens`, `digit_hint`) axes, each value with status, sources,
confidence and `feature_eligible`; a legacy parent class (so the old model's behaviour is preserved); an **11-slot ledger**, all
`unknown`; nearest siblings with the *cited* difference and an explicit "any further difference is unknown"; offensive-stress
**hypotheses** (never model inputs, each with its limitation); an `unknowns` list; evidence; verification level; review status.

Two things the schema keeps deliberately apart:

* **Catalog coverage vs responsibility verification.** 150/150 entries exist and resolve. `role_verified` is 0/150.
* **A trailing digit is a hint, not a coverage claim.** `Fire Zone 3` keeps `deep_shell: unknown` and `digit_hint: 3`.

### Verification levels (150 entries)

| Level | Entries | Meaning |
|---|---:|---|
| `catalog_only` | 4 | exists; no usable decomposition (e.g. `Thunder Smoke`, `Free Fire`) |
| `name_decomposed` | 111 | tokens parsed; meanings are hypotheses or unknown |
| `mechanic_documented` | 32 | at least one EA-documented general mechanic applies (match families, Cover 6, Palms, Sim style, press) |
| `role_partially_documented` | 3 | published per-position jobs exist (third-party; not trainable) |
| `role_verified` | **0** | eleven roles proven from first-party evidence |

## The distinctions asked for, and what is actually established

| Pair / group | Established | Hypothesis (unverified) | Unknown |
|---|---|---|---|
| Cloud / Sky / Buzz | all are Cover 3 (three-deep zone) | a corner / safety / linebacker takes the flat (a single unverified summary; two independent sources returned 403) | which side, when, who is in deep thirds |
| Cover 3 Match | EA documents Cover 3 Match and its checks | | the exact release rules for this call |
| Cover 3 Mable | none | match version (one summary) *or* zone/man mix (another) | zone vs match: left as `unknown`, not chosen |
| Cover 2 Man / Invert / Trap / Tampa 2 | Man is man-under (name); Tampa 2 = Cover 2 with the middle linebacker deep on a read pass (Wikipedia) | Invert swaps roles; Trap baits a flat throw | which defenders swap; trap triggers |
| Quarters / Palms | both four-deep; EA documents match checks for both | | the corner/safety rules that differ |
| Cover 6 / Cover 9 | Cover 6 is a match family per EA | split field (quarters one side, half the other) is convention | Cover 9 is not defined by EA at all; side/zones unknown |
| Cover 1 Hole / Robber / Contain | all single-high man | hook defender: linebacker (Hole) vs safety (Robber); Contain = QB spy + contain rush. **One article dated 2026-01-09, before Madden 27 shipped, version not named.** | who the "bonus defender" is in a given formation |
| Zone blitz / man blitz / sim / stunt | EA documents the Zone Pressure *style* (simulated pressure with zone behind it) | fire zone = rush five, 3-3 behind (Wikipedia); `0` / `Zero` = no deep help | rusher count, which defender drops, for **every** play |

## The post-play form is a lossy channel (the finding that bounds everything else)

The existing post-play form sends "they ran" through keyword rules before storing `coverage_seen`. Run against the catalog's 109
distinct call spellings (the real parser, not a model of it):

* **7** are stored as typed, **77** are rewritten to a coarser label, **25** store nothing.
* Every "Cover 3 …" call (15 names) becomes `Cover 3 Sky`; every Cover 1 variant becomes `Cover 1`; every Cover 4 variant becomes
  `Cover 4 Quarters`; 31 pressure names become `pressure`; `Cover 2 Match` is stored as `Cover 3 Match`.
* Of the 123 **installed** plays: 25 are stored as nothing, 69 under a label shared with five or more other calls, 27 under a label
  shared with one to four others, and only 2 under an unambiguous label.

So a stored `Cover 3 Sky` is **not** evidence of Sky. `defense_semantics.stored_label_weights` reads a stored label as the uniform
mixture over the catalog names the *current parser* maps to it (maximum-entropy; revisited when capture changes). A label the form
cannot produce (for example `Cover 3 Cloud`) is read exactly, so lossless capture is picked up automatically. Family structure
survives; variant detail is blurred rather than invented. An untyped `pressure` stays untyped.

**Capture fix:** the exact typed call is now also stored in `snaps.coverage_typed` (additive nullable column; `coverage_seen` unchanged).
`stored_label_weights(label, typed)` reads the typed text exactly when it resolves; rows without it stay ambiguous as described above.

## How it enters the model

```
post-play form -> snaps.coverage_seen (+ coverage_typed) -> stored_label_weights -> dk: mix (same-game earlier snaps + prior from earlier games)
   -> scheme_distribution:  legacy projection (unchanged classes)  +  family_mix
   -> full_vector(names=artifact.names):  135 legacy features  [+ 71 dk_* features only for a knowledge artifact]
   -> learned residual (same trainer, same nested game-level selection, 'do no harm' fallback)
```

* **Opt-in and artifact-gated.** The default vector is still 135 features at `offense_play_features.v4_situation_interactions`; a
  default artifact serializes and validates exactly as before. A knowledge artifact records `…v5_defense_knowledge` and the catalog
  digest; a stale catalog or edited schema is refused and the live ranker falls back with the reason.
* **`dk_*` features (71):** deep-shell, coverage-type, 21 variant-presence, pressure-concept, digit-hint and evidence axes
  (expectation under the opponent mix, **not renormalized**, so an uninformative mix yields small values) and 23 interactions with
  the play's **screenshot-derived route-art geometry** (the only play-structure signal available for all 93 plays; none has a
  first-party route assignment). They get their own L2 group (`dk`, 30), as shrinkage as the name-hint group.
* **No pre-snap inference.** Only closed, earlier, same-game, same-opponent entries count; the current snap's label never enters its own
  features; a different game or opponent starts clean.
* **Our defense (offline only).** `defense_semantics.candidate_features(formation, play)` describes each of our 123 calls
  (structure + formation personnel). `defense_knowledge_train` fits the existing logistic model plus those columns and compares them
  under the existing leave-one-game-out protocol with a stricter gate (>= 8 games and a game-clustered bootstrap interval that excludes
  zero). Its artifact's schema is rejected by the live defensive loader by construction.

## Commands (read-only; every one runs with `HOME` pointed at a temp directory)

```bash
python scripts/madden_ml/build_defense_catalog.py --check                       # catalog == fresh build
python scripts/madden_ml/defense_knowledge_eval.py audit     --out OUT          # recognition + form-channel audit
python scripts/madden_ml/defense_knowledge_eval.py selfcheck --out OUT          # SYNTHETIC controls (plumbing only)
python scripts/madden_ml/defense_knowledge_eval.py latency   --out OUT
# on a COPY of the database (refuses a path under ~/.cfb-coach):
cp ~/.cfb-coach/madden27.db ~/madden27_copy.db
python scripts/madden_ml/defense_knowledge_eval.py heldout   --db ~/madden27_copy.db --out OUT
python scripts/madden_ml/defense_knowledge_eval.py train     --db ~/madden27_copy.db --out OUT   # writes artifacts to OUT only
python scripts/madden_ml/defense_knowledge_eval.py defense-offline --db ~/madden27_copy.db --current <live defense artifact> --out OUT
```

See `DEFENSE_KNOWLEDGE_READINESS.md` for results, the go / no-go and the CPU shadow-experiment steps.
