# Installed Madden NFL 27 offense — evidence reality check

Research-only. Source inventory: 93 plays / 5 formations, `OFF-3C6D-E543`.

## What was independently checked

* The installed snapshot contains 93 identifiable formation/play records, each matched to the source playbook catalog.
* A read-only user game backup contains 655 offense snap logs marked `identified` and `verified` whose formation/play matches those installed records. These **prove recorded execution**, not route/OL assignments; do not treat them as play-art evidence.
* Formation personnel are explicit for `Gun 5WR Tight` (5 WR, 0 TE, 0 HB) and `Pistol Deuce Close` (2 WR, 2 TE, 1 HB). The other three are standard 11 personnel (3 WR, 1 TE, 1 HB). The player-to-screen layout and assignment are **not proven by those counts**.
* PR #108's 93×11 ledger has 40 `source_documented` route records across eight plays, and zero player roles independently verified using in-game or screenshot evidence. Third-party sources cannot be promoted to `in_game_verified`.
* The backup has `vision_observations = 0` and `play_records = 0`. There are no recoverable play-art frames inside it.

## Critical blockers to calling the exact database complete

1. We have no original first-party Madden 27 diagram capture for the 93 plays. The public formation/play listings are not substitutes for named WR, TE, RB, QB, and OL assignments.
2. Conventional pass protection and blocking rules can depend on the defensive alignment and post-snap behavior; the actual static play art alone may not settle every assignment.
3. Madden Tools' published terms explicitly disallow use of its content to train or improve machine learning models without written permission: <https://madden.tools/policies/terms>. Existing 40 source-documented entries must remain outside training until rights are established or replaced by first-party play-art verification.
4. Some third-party route tables are internally inconsistent, including a backfield `HB1` slot shown in a documented 00-personnel, five-WR formation. The model must not equate such slots with exact Madden lineup identities.

## Concrete data-completion protocol

1. Capture one Madden 27 pre-snap play-art screenshot from **each of the 93 plays** in the user's custom offensive book, with the formation and play name visible. Record any motion or alternative route screens if needed. Gameplay/console capture, not external restricted datasets, is the evidence of record.
2. Store screenshot file paths/sha256 in the completion queue. The existing PR #108 `offense_annotation.py` supplies a ready-to-fill, 1,023-slot CSV template and an image manifest importer.
3. For each screenshot, annotate visible player starting point and path; map it to its actual position. Do not confuse Xbox button icons with personnel labels. Capture RPO/option prompts and individual play action if the static art omits them.
4. For blocking, record the visible blocker direction **and** the applicable assignment rule (zone, man, slide, protection, combo, pull, read defender), marking conditional parts separately.
5. Validate every non-unknown claim with its screenshot and review it; commit only the *structured assignments and evidence locators*, not a third-party site's copyrighted play images.
6. Run holdout tests comparing the existing coordinator and the structural model on the actual backup. Do not claim model improvement simply from adding assignments.

## Priority

`OFFENSE_93_completion_queue.csv` is sorted by previously verified execution frequency from the backup. This gets the plays used most often annotated first, but **all 93 still require verification**.

## Present completion status

- Names/formation/source membership: **93/93**
- Known formation-level personnel composition: **5/5**
- Full player-responsibility annotation verified: **0/93**
- Third-party partial route records: **8/93** (not sufficient for training permissions or an exact assignment claim)
- Quarterback jobs verified: **0/93**
- Offensive-line jobs verified: **0/465**
- Xbox play-art screenshots available in the uploaded backup: **0/93**

This data-completion report is **not** a claim that the exact offensive database was finished.