# VOD meta pilot

Can the coach learn meta from public Madden 27 / College Football 27 VODs, instead of only the games Aidan logs?

On this sample, no. Down and distance on a clean scorebug are readable. Yards gained are readable when the next down is the same drive and the clock only moves a little. Formation, the called play, and coverage are not. The play-call screen lists the whole tab for a fraction of a second, OCR cannot see which row is highlighted, and the broadcast camera does not show the coverage shell. Nothing here is wired into prep or the live caller, and nothing was inserted into `coach.db`.

## Sources

YouTube from this VM returns a sign-in wall (`yt-dlp` and the player clients: `LOGIN_REQUIRED` / "Sign in to confirm you're not a bot"). `python -m vod_pilot.fetch` records that and stops. The three prefixes below were saved outside the repo (not committed) from public pages. Each file is a 720p H.264 prefix of about 55 MB. The moov atom describes the full upload, but decode stops early, so only the opening minutes were tagged.

| Video | Channel | Game | Published (relative to 2026-10-07) | Decoded | Snaps |
| --- | --- | --- | --- | --- | --- |
| [z5ne-uBM0YI](https://www.youtube.com/watch?v=z5ne-uBM0YI) | TheNewEra23, ranked Top 100 | Madden 27 | about 2 days | 176 s, 176 frames | 9 |
| [NyoqCpRffsk](https://www.youtube.com/watch?v=NyoqCpRffsk) | Sports Gaming Universe, Raiders vs 49ers | Madden 27 | about 1 month | 231 s, 231 frames | 7 |
| [YlwwSquPtVY](https://www.youtube.com/watch?v=YlwwSquPtVY) | Gamer Ability, Dynasty Part 26 | College Football 27 | about 20 hours | 281 s, 281 frames | 0 |

The CFB prefix is the dynasty hub (Recruiting, Create Savepoint, Play Game) for the whole decoded stretch. The pipeline emitted no snaps. That is the right answer for this file, and it means this pilot never saw a CFB snap.

No vision-model API key was available. Tagging is RapidOCR plus the playbook lexicon. Tesseract cannot read this HUD.

## Snaps

16 snaps across the two Madden prefixes. Rows use the `CoachDB.log_snap` columns (`cfb_coach.snaps.v1`). `opponent_id` is `vod:{channel}`, not one of Aidan's opponents. `our_call` and `macro` are empty. JSON extras (`t_start`, `result_kind`, `confidence`) are omitted from the CSV.

Sample rows:

| t | situation | play | coverage_seen | result | result_kind |
| --- | --- | --- | --- | --- | --- |
| TheNewEra 34s | 1st & 10 q1 3:56 | HB Slam |  | +0 | no_gain |
| TheNewEra 88s | 3rd & 10 q1 3:23 |  | Cover 3 Buzz Match | convert | convert |
| TheNewEra 102s | 1st & 10 q1 2:57 |  | Cover 1 Contain | td | td |
| SGU 105s | 1st & 15 own 22 q1 5:53 |  |  | +1 | gain |
| SGU 161s | 3rd & 14 q1 5:11 |  |  | -8 | sack |

Full files: `vod_pilot/out/z5ne-uBM0YI.csv`, `vod_pilot/out/NyoqCpRffsk.csv`, `vod_pilot/out/YlwwSquPtVY.csv`.

A sack stores the signed yards in `result` (what `parse_outcome` will read as a loss unless something maps `result_kind`) and `sack` in the JSON-only `result_kind`. The 3rd & 14 snap is the one case: the next bug is 4th & 22 with a `1 SACK` line.

## Hand-check

All 16 emitted snaps were labeled. Labels are in `vod_pilot/eval/handcheck.json`, scores in `vod_pilot/eval/accuracy.json`. A null label means the frames did not show that field. Filling it counts as a false fill. Recall is only where the human read a value. This is the whole decoded sample, not a draw of 20: the prefixes did not contain 20 snaps.

| Field | Known | Recall | Filled | Precision | False fills | Abstained when unknown |
| --- | --- | --- | --- | --- | --- | --- |
| down | 14 | 14/14 | 14 | 14/14 | 0 | 2 (kickoffs) |
| distance | 14 | 14/14 | 14 | 14/14 | 0 | 2 |
| quarter | 16 | 16/16 | 16 | 16/16 | 0 | 0 |
| yardline | 2 | 2/2 | 2 | 2/2 | 0 | 14 |
| formation | 0 | n/a | 0 | n/a | 0 | 16 |
| play | 3 | 3/3 | 3 | 3/3 | 0 | 13 |
| coverage_seen | 2 | 2/2 | 2 | 2/2 | 0 | 14 |
| result | 8 | 8/8 | 9 | 8/9 | 1 | 7 |

Read those rates against how small the known set is.

- Formation recall does not exist. Every play-call frame listed several formations (`BUNCHSTRNASTY`, `CLUSTER`, `DOUBLESCLAMPSTACK`, …). The pipeline abstains. An earlier pass tagged `Gun Cluster` because that was the only tail the lexicon could resolve; that was a false fill and it is fixed. Abstaining is the correct output, and it is not a formation model.
- Play 3/3 is `HB Slam` (the tooltip sentence on the call screen) plus two `Kickoff` labels. No other play name was visible as the selected row. Hot-route wheels (`Deep Cross`, and the route list behind `concept_seen` = "corner post streak wheel") are not the called play. `concept_seen` on that snap is a false fill and is not in the table above.
- Coverage 2/2 is the `PREVIOUS PLAY` line on the next call screen (`Cover 3 Buzz Match`, `Cover 1 Contain`), attached backward onto the snap that just ended. It is UI text, not a read of the shell. n = 2.
- Yard line 2/2 is the penalty card (`own 22`, `own 35`), and only when that line's down matched the live down. Bare field numbers (`35`, `48`, `27`) were left blank. n = 2.
- Result 8/8 recall, 8/9 precision. The miss is TheNewEra at 154s: `2nd & 5` was labeled `convert` because the next bug is `1st & 10` and the clock did not jump. The clock never moved off 2:36, and the next offense is Detroit (Goff) with the penalty overlay still up. That is a possession change, not a measured first down. The SGU drive is the clean case: 1st & 15 → 2nd & 14 is `+1`, 2nd & 14 → 3rd & 14 is `+0`, 3rd & 14 → 4th & 22 with the sack glyph is `-8`.

## What failed

- The highlighted row is a color, not a unique string. OCR of the formation tab cannot name the call. Play art is up for about a second and then the view is the field.
- Coverage shells are not on a competitive camera. The only coverage strings that survived were `PREVIOUS PLAY` labels, and a penalty screen prints the whole defensive menu (`SAM WILL BLITZ`, `4-3 EVEN 6-1`, `COVER 1`) which must be ignored.
- The scorebug `&` is read as `8` (`1ST810` = 1st & 10). A clock token (`1st 3:56`) must not become a down. Both are handled, and both will break on a new HUD font.
- "Sack the QB 0/3" is a challenge tracker. A bare `SACK` match false-flags it. The result bug `1SACK` is real. The filter is a special case, not a detector.
- One-frame downs (the 2nd & 10 at 87s) and penalty accept/decline screens create extra snaps. Yards inferred across them are wrong when the next "down" is a different possession.
- Dynasty and menu footage looks nothing like a snap, but a long CFB VOD can be mostly that. Gameplay has to be detected before spending OCR on it.
- Download is the other blocker. This VM cannot fetch YouTube. Scaling depends on a machine that can.

## Cost to 100 VOD hours

This run OCR'd 688 frames in about 5 minutes of wall time on a 4-core VM (RapidOCR near 370% CPU), roughly 0.4 s/frame. At 1 fps, 100 hours is 360,000 frames, about 40 wall-hours on this box, on the order of 150–170 core-hours. API cost is $0: there was no vision model. A vision pass on one keyframe per snap, at roughly one snap per 20–30 seconds of gameplay, is on the order of 12,000–18,000 images per 100 hours. That cost is whatever the vision API charges per image; it was not measured here.

720p for these uploads is about 0.9 GB/hour (the full TheNewEra file is ~299 MB for 19 minutes). 100 hours is on the order of 90 GB, and it is not fetchable from this VM today.

Human review does not scale with the OCR. Formation and play still need a person or a vision model that can see the highlight. Reviewing every snap from 100 hours is a different project from running OCR.

## Recommendation

Do not scale this into the coach, and do not point prep or live play at it.

Keep the prototype as a side path that emits `log_snap`-shaped rows for a later importer. If it is picked up again, the only columns worth trusting without a vision model are down, distance, quarter, and a yard delta when the next down is exactly one later and the clock stays in the same quarter. Even those need the possession-change guard that this sample missed once. Yard line only when the frame says own or opp. Coverage only from an explicit previous-play line, stored as low confidence. Formation and play name stay blank until something can see the highlighted row.

Store that corpus under its own key (`vod:{channel}` / `vod:{video_id}`), never mixed into Aidan's opponent snaps. `our_call` stays empty: a streamer's call is not Aidan's call. A vision model is the thing that would change the recommendation, and only after a new hand-check on formation, play, and coverage, on footage that is actually gameplay.
