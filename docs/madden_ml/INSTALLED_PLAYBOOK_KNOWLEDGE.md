# Madden 27 — Installed Offense + Defense Catalog (research v1)

## Exact scope and evidence

The immutable `cfb_coach/data/madden27/installed_playbook_snapshot.json` records the
**applied playbook state in a read-only November 6, 2026 database backup**, recorded
November 5. It is not independent evidence that every play is presently visible
on an Xbox. Never silently rewrite this snapshot to match a newer installed book.

| Side | Formations | Play entries | Source books | Recorded fingerprint |
| --- | ---: | ---: | --- | --- |
| Offense | 5 | 93 | Buccaneers | `OFF-3C6D-E543` |
| Defense | 6 | 123 | 49ers, Chargers | `DEF-F1CF-1862` |

The full repository `playbooks_full.json` was cross-checked on Nov 6: all
216 recorded book/formation/play combinations were present in that source
catalog. `installed_source_catalog_reference.json` is a reproducible **subset**
of those source records, not a separate third-party data scrape.

`installed_play_annotations.json` contains **8 offense and 3 defense** partially
annotated Madden 27 plays from individual public Madden Tools play pages. Those
annotations provide **53 named player jobs**, **7 published quarterback read
orders**, and 23 broad concept tags. They are `source_documented`, **never**
`user_verified`. Published route tables and page narratives sometimes differ;
only the tables' explicit route/assignment facts were transcribed, and ambiguous
route names are preserved in `published_route` parameters rather than guessed.

No offensive or defensive play has all eleven player responsibilities verified.
Most plays are **catalog_only**. This is **not a claim the model understands all
216 designs or can yet adjust them safely**.

## Reproduce the separate research database

From the repository root with `cfb_coach` importable:

```bash
python scripts/madden_ml/build_installed_play_knowledge.py \
  --output /tmp/madden27-installed-knowledge.db
```

The script defaults to the repository's existing `playbooks_full.json` for
independent source-membership checks. It also loads the snapshot and the
sourced annotation subset. Generates alongside the DB:

- `madden27-installed-knowledge.review.csv`: **216 rows**, one per recorded play,
  with source reference, known/unknown job count and review status.
- `madden27-installed-knowledge.report.json`: counts, fingerprint checks,
  source checks, annotation coverage and a warning about Xbox verification.

Optional:

```bash
python scripts/madden_ml/build_installed_play_knowledge.py \
  --output /tmp/madden27-installed-with-usage.db \
  --history-db /path/to/a/local/madden27.backup.db
```

`--history-db` is read-only and only counts **logged calls**, not proven
executions. Do not commit personal/game DB backups or raw private histories.
For a purely metadata-driven view, use `--no-annotations`.

The tool refuses to overwrite preexisting files or write any research output
under `~/.cfb-coach`. It does not modify game settings, running playcallers,
installed playbooks or the saved model registry.

## Verify in the actual game

1. Open the recorded custom offensive playbook, verify its five formations and
   the listed play names against the CSV/checklist. Repeat for the six defensive
   formations. A newer installation may differ from the backup: record that
   difference separately, do not mutate the historic record.
2. For the 11 partially annotated plays, view their **actual play art** on your
   Xbox. Verify each explicitly recorded route/assignment and which player
   occupies each position slot, then confirm or correct the published read
   progression. A route being present does **not** mean it is selectable as a
   standalone hot route.
3. Screenshot/video or manually annotate missing player routes, offensive-line
   blocks, defensive run fits, match rules and rotation. Unverified jobs remain
   unknown; do not infer them from the name or stock football theory.
4. After coverage and execution verification, compare the structure-aware policy
   against the present ML caller in **shadow/offline** evaluation before
   enabling adjustments or changing playbooks.

## Implementation

- `cfb_coach/madden/model/installed_playbook_catalog.py` validates source
  mappings and recorded playbook fingerprints and imports installed membership
  as normalized foreign-key records in `PlayKnowledgeDB`. It never silently
  discards a recorded play that doesn't match an external source; mismatch is
  separately reported for investigation.
- `scripts/madden_ml/build_installed_play_knowledge.py` builds the **new** DB,
  optional read-only history-frequency priority queue and review outputs.
- `tests/test_installed_playbook_catalog.py` checks fingerprints, exact counts,
  source mismatches, idempotence, no false user verification, CSV coverage,
  and all-record rejection on malformed snapshots.

**Next milestone:** Verify currently installed inventory and actual play art,
then scale the same structure to all Madden 27 plays and use verified structure
for the existing playbook architect and assignment-aware coordinators.