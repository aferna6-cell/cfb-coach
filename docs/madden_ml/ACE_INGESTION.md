# AceMadden-compatible ingestion (issue #121 / the Phase 1 directive)

Status: **tooling complete and tested on synthetic fixtures; the full extraction has NOT been run.** A two-page live smoke test of public
metadata discovery (5 requests) worked for both sides and is the only live traffic made.

## What AceMadden publishes without login (observed 2026-11-07, read-only requests, no credentials)

| Field | Where | Notes |
|---|---|---|
| play name, personnel-group label (e.g. `3WR`) | listing tiles `/playbooks/plays/madden27[?page=N][&side=defense]` | 24 plays/page (each tile is linked twice in the HTML); about 358 pages offense (the page states 8,574 listings); the defense view exists and paginates the same way |
| formation slug, play slug (identity) | tile link `/playbooks/plays/madden27/{side}/{formation-slug}/{play-slug}` | the link slug is used as identity |
| play-art PNG (556x258 diagram) | `/play-art/madden27/{side}/{formation-slug}__{play-slug}.png` | public URL, `robots.txt` allows it; the art file name can differ from the link slug (observed typo-style differences), so it is stored as an attribute, not identity |
| formation label | detail page title (`Play — Formation Label`) | |
| playbook memberships | detail page "Playbooks With" block | structure not yet understood: kept only as `detail_hint_playbooks_unverified`, never as membership |
| route / coverage assignments, read orders | **not published as data** | only the picture; assignments must be extracted from the picture and reviewed |

Behind login/membership (not touched, not bypassed): dashboard, "Download Card", finder/tools, anything paid. Rate limits are respected
(1.5 s default delay, backoff on 429/5xx, hard stop on 401/403). `robots.txt`: `User-agent: * / Allow: /`.

## Permission
Terms (rev. 2026-06-03): AceMadden owns the play art/guides/tools; purchased content may not be shared/resold/redistributed/publicly
reproduced; the terms are silent on automated collection and ML training. The owner reports that AceMadden said bulk use is fine. The tools
therefore **require that written permission be recorded**: `--permission-note "<paste the message / date>"` is stored on every record
(`permitted_use`), art download refuses without it, and the audit prints the status. Contact for any new request: support@acemadden.com.
Raw images/overlays go only to the private store (`~/cfb_acemadden`, refused inside a git work tree); nothing third-party is committed.

## Commands
```bash
python scripts/madden_ml/ace_catalog.py discover --max-pages 2            # public metadata smoke test, writes no catalogue
python scripts/madden_ml/ace_catalog.py --permission-note "<text>" sync   # ~700 requests at 1.5 s (about 20 min): all listings, both sides
python scripts/madden_ml/ace_catalog.py --permission-note "<text>" art --side offense   # play art into the private store (resumable)
python scripts/madden_ml/ace_catalog.py import --file export.jsonl --permission-note "<licence>"   # authorized export (jsonl/csv)
python scripts/madden_ml/ace_catalog.py validate && python scripts/madden_ml/ace_catalog.py audit
python scripts/madden_ml/ingest_first_party_art.py extract --images ~/cfb_acemadden/raw --side offense
python scripts/madden_ml/ingest_first_party_art.py review     # review_queue.csv + overlay pictures
python scripts/madden_ml/ingest_first_party_art.py approve --file reviewed.csv
python scripts/madden_ml/audit_football_universe.py --out audit.json   # repository universe + ingestion store, units kept apart
```
Export format (jsonl or csv): `side, formation, play, playbooks ("A|B"), personnel_group, art_url`, and optionally `assignments` which is **rejected
unless** `assignment_tier` and `assignment_source` are present.

## Safety against layout changes
Parsing is staged and validated as a whole before one atomic write; tiles whose image/link disagree go to `quarantine.jsonl`; a run aborts
(writes nothing) if a page has no tiles or over 20% bad tiles; sync never deletes (marks `not_seen`, appends `history.jsonl`) and never
overwrites earned evidence with a bare listing. This guard already did its job on the real site: it stopped the first live smoke test when the
art file name differed from the link slug, and the rule was refined (formation must match; play-slug difference is flagged).

## Extraction from art (the "assign the routes" step)
`play_art_extract.py` segments colour lines, markers, zones and rush arrows from a panel and outputs geometry: marker positions (indexed left to
right), route polylines owned by their start marker, coarse shape suggestions, defensive zones (colour, centroid, size), rush arrows, defender links.
It is **tier `diagram_extracted_unreviewed`**: it does not assign slot names (the picture does not say WR1 vs TE), does not decide what a colour
means, does not convert to yards, and does not produce a QB read order. A person reviews overlay vs original and `approve` promotes only rows with an
allowed source (`user_screenshot:`, `user_recording:`, `licensed_export:`). Measured on three real panels: a pass play's verticals/curl/diagonal
routes were located correctly with markers owned; complex routes are labelled `unknown_complex` for review; man-coverage defensive art has not been tested.

## Not done / blocked
Full catalogue sync and art download (waiting for you to run them with the permission note); detail-page playbook memberships (structure
unclear); defensive man-coverage art; slot-name mapping and QB progressions (need formation knowledge + review); everything in the directive's
Phases 2-4 (matchup priors, scoring integration, adaptive learning, promotion pipeline) is not part of this PR.
