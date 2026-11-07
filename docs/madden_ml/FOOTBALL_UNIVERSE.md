# Football universe (issue #121): PR 1 of the plan

Versioned, searchable, read-only library of every offensive and defensive formation/play we can **lawfully identify today**, with
four evidence units kept apart. Regenerate with `python -c "from cfb_coach.madden.model import football_universe as fu; fu.write()"`;
the test suite fails if the committed `cfb_coach/data/madden27/football_universe.json` differs from a fresh build.

## Measured coverage (digest `830fb06170dfcc4e9589`)

| Unit | Offense | Defense |
|---|---:|---:|
| source books | 11 | 7 |
| formation families | 184 | 66 |
| formation-play **listings** across books | 6372 | 2320 |
| unique formation/play **pairs** (indexed) | 3432 | 1080 |
| spelling-variant groups kept separate | 11 | 0 |
| installed (legal to call) | 93 | 123 |
| name decomposed (soft, hypothesis-level) | 2885 (84.06%) | 726 (67.22%) |
| **image geometry** (first-party/licensed art) | 93 (2.71%) | 0 (0%) |
| **some assignments known** | 8 (0.23%) | 0 |
| **roles verified (first-party)** | 0 | 0 |
| offense slots: documented / total | 40 / 1023 | defender slots verified: 0 |
| plays with a documented read order | 7 | n/a |

Different units are never added together: external sites count listings, memberships, formation entries and "plays" differently
(AceMadden's 8,574 offense listings, Madden Prodigy's 10,323 plays and 37,643 memberships, MaddenGuides' 512 formation entries are
**not** reconciled to these numbers; reconciling them needs the permissioned feeds below). A name is not a diagram, a diagram is not an
assignment, an assignment is not a read progression.

## Licensing audit (2026-11-07)

* **AceMadden** `/terms` (revised 2026-06-03): play art, guides, ebooks and tools are copyrighted by AceMadden; purchased content may not be
  shared, resold, redistributed or publicly reproduced. The terms say **nothing** about automated collection, data mining or ML training.
  `robots.txt` allows crawling (`Allow: /`). **Permission to bulk-ingest or train on their art is not established**, so nothing was
  ingested from the site; `sources.acemadden_site.train_rights = NOT_ESTABLISHED` and a test asserts no entry cites it.
  Next step (needs you): ask `support@acemadden.com` for a written licence/export for **catalogue metadata** and, separately, for use of
  play art in model training. Do not scrape in the meantime.
* The 93 offensive art records come from **your own screenshots** (private analysis; images are not in the repository, only derived features).
* Madden Tools / 818 Madden / Huddle terms restrict extraction or ML reuse (see #116); not used.
* Safe path to detailed art for the rest of the library: your own Xbox captures (the #120 capture workflow), which also gives defensive art,
  where AceMadden's own defensive pages say "Defense coming soon".

## What this PR connects

* `ml universe report|search` (CLI, read-only); search finds **non-installed** plays; installed flags come from your attested book only.
* Offline-cached per-play feature rows (`feature_row`, O(1), sub-millisecond) with evidence flags so a model knows how little is known.
* Pregame designer proposals now carry `universe_evidence` (per formation: how many plays have art / assignments / verification) and the
  universe digest. **Informational only: no score, ranking or live call changed.** The live CPU experiment path is untouched.

## Not in this PR (follow-ups, each separately reviewable)

1. Licensed/permissioned catalogue import + reconciliation report across sources (blocked on permission).
2. Defense candidate-book search over the whole universe feeding a defensive pregame proposal (additive, 15 cap, owner install + lock).
3. Wiring `feature_row` into the offense/defense model features **with a held-out comparison** (not enabled without evidence).
4. First-party capture batches for non-installed formations and for defense; QA queue; per-slot annotation import.
5. Live adjustment-effect/read-plan integration for composed adjustments (#120 PR) over universe plays that gain assignments.
