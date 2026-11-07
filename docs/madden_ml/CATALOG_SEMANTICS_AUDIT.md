# Catalog semantics audit (issue #127), 2026-11-07

Base: `claude/football-priors-fixes` (PR #126) at 1943e66. Branch: `grok/catalog-semantics`. No live caller, ranker, training, promotion,
gameplay DB, CPU pilot, defense activation or champion model is touched.

## Catalog process status
- No AceMadden ingestion process is running on the box, and the box has no private store (`~/cfb_acemadden`, `authorization.json`). The
  8,554 + 20 offensive / 1,578 defensive listing crawl from PR #126 lives only in that sandbox and was not committed.
- Sync was **not resumed**: bulk sync and art download both need an `authorization.json` on the store that records the AceMadden grant (sender, date,
  scopes from the email). That record does not exist here, and I do not invent one. To resume: run `ace_catalog.py authorize ...` using the real email, then run `sync`, then run `art`.
- Playbook memberships ("Playbooks With") are still unparsed for the full catalogue.

## Coverage funnel: installed books (from `python scripts/madden_ml/audit_catalog_semantics.py`)
| side | indexed | art_available | downloaded | geometry_extracted | human_reviewed | assignment_verified | concept_verified | model_feature_eligible |
|---|---|---|---|---|---|---|---|---|
| offense | 93 | 93 (owner screenshots, hashed, off-host) | 0 | 0 (93 have 16-d occupancy features only) | 0 | 0 | 0 | 0 |
| defense | 123 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Full repository universe (PR #123/#126, unchanged): offense 3,432 unique pairs and defense 1,080. 40 of 1,023 offensive slots are third-party documented
(madden.tools text, 8 plays). No slot is verified.

## First validated batch
- **Membership (verified):** all 93 installed offensive plays (5 formations) are in the owner-attested installed book, and every one also appears on
  an independent second listing (huddle.gg Buccaneers formation pages from the private cache, names only). The two site-name differences
  (`L Spot`, `Zl Spot Shake`) match the installed names. Result: `cfb_coach/data/madden27/installed_membership_corroboration.json`. Defense has no
  second-source cache, so it is owner-attested only.
- **Assignments: none verified.** Nothing on this host is first-party play art I can review, so no player responsibility is promoted. The 40
  madden.tools slots (8 plays) are at the top of the review queue as quick confirm/correct items (priority 1).
- **Real accuracy check (small):** name-token concept guesses overlap the independently documented concept on only 2 of 8 plays. That is why names never certify.

## Review queue
`docs/madden_ml/catalog_review_queue_installed.csv` has 453 items. Priority 1 is 40 confirm/correct offensive slots. Priority 2 is 123 defensive coverage-shell confirmations
(proposal = name + glossary). Priority 3 is 290 offensive skill slots to annotate. Fill `decision/assignment/reviewer/reviewed_at/evidence_ref` (evidence must be
`user_screenshot:`, `user_recording:`, `licensed_export:` or `in_game:`), then run `audit_catalog_semantics.py --import-reviews file.csv`.

## Rights / blockers
- AceMadden: metadata_bulk, images_download and images_local_storage are owner-reported (PR #126). **ml_training and redistribution are not granted.** No raw art is committed.
- huddle.gg: names are used only as a cross-check. Rights are not established, and the HTML is not committed.
- Defender-level responsibilities, man-coverage art, fronts and pressures: no evidence on host. They stay `unknown`.

## Handoff to the #124 agent (interface)
`cfb_coach.madden.model.catalog_semantics.export()` returns `madden27.catalog_semantics.v1`, which contains `funnel`, `rights` and `feature_rows[{key, stage, eligible, features}]`.
Keys match `football_universe.play_key`. `features` is non-empty **only** when `eligible` (stage `model_feature_eligible`: concept verified by rule from at least 3
human-reviewed skill slots, with no disagreements). Today every row is ineligible. Consume it offline and additively. Do not treat a missing row as negative evidence.

## Full AceMadden listing sync (2026-11-07, counts only; data stays in the private store)
Sync took 10m41s with 0 quarantined. Offense: 8,574 plays, 8,554 with art URLs and 20 without, across 420 unique formations. Defense: 1,578 plays, all with art URLs,
across 90 unique formations. The unique-play listing has no team-book membership, so playbook membership stays `unknown` until a detail or team-book pass.
Art download is running (resumable, 1.5 s delay). The `art` command now checkpoints every 200 images and records a per-play `art_error` instead
of losing the run. It still stops on 401/403.
