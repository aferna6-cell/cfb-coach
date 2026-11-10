# Madden 27 Lions — offensive scheme dataset (review only)

This folder contains the user-curated **Lions offensive scheme catalog** assembled from screenshots provided November 10, 2026.

- **26 named variants** built from **12 distinct formation + base-play identities** across **4 formations**: Gun Doubles Clamp Stack (6 variants), Gun Bunch TE (7), Gun Tight Doubles (4), Gun 5WR Tight (9).
- Each named variant has a stable `variant_id`, base play, formation, adjustment instructions, expert situation tags, and coverage tags where supplied. These are **expert guidance, not measured success rates**.
- The file also records **2 unverified red-zone packages** and **1 unnamed 5WR setup**. They are **not** among the callable `plays[]` variants.
- The 5WR entries have **no coverage labels in the supplied material**. Empty lists mean **unknown**, not bad performance.
- Some variant names and setup associations are provisional; review `data_quality_flags` and per-play verification fields.

## File

[`lions_offensive_scheme_catalog.json`](lions_offensive_scheme_catalog.json)

## Integration and model safety

This commit is **dataset only**. It does not train weights, install formations, alter live selection, or write to the franchise SQLite database.

The experimental scheme-only caller is designed to rank only the catalog's 26 `plays[].variant_id` entries when explicitly enabled. To enforce that safely in a future integration:

1. Confirm every `formation` + `base_play` matches a physically installed Madden 27 playbook. Do not silently rename, substitute, or fuzzy-match missing items.
2. Verify each recommended setup/hot route works in game before treating it as executable.
3. Require both an explicit variant marker and a verified executed play with an actual outcome to learn variant-specific effectiveness. Never invent successful snaps from expert tags.
4. Do not fall back to unapproved plays when scheme-only mode has zero legal candidates. Defensively fail closed.
5. Keep defense unchanged unless separately reviewed.

**Known gaps:** the previous Lions applied book was missing Gun Bunch TE and Gun Tight Doubles; the scheme has no confirmed designed runs, so a strict scheme-only offense is a pass-only experiment until resolved. No live gameplay integration or local database retraining is established by this PR.

## Provenance and redistribution

This file is structured user-provided metadata derived from screenshots of third-party instructional material; **no videos or screenshots are included**. Verify that any redistribution is permitted by the content provider before merging into a public release.
