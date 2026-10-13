# VOD snap pilot

Feasibility prototype. It reads a local Madden 27 or College Football 27 gameplay file, finds stretches that share one down-and-distance, and writes one row per stretch in the `snaps` column layout from `cfb_coach.db.CoachDB.log_snap`.

It does not import prep, the live caller, or the coach database, and it does not insert rows into `coach.db`.

## Run

```bash
pip install -r vod_pilot/requirements.txt
python -m vod_pilot run \
  --video /path/to/clip.mp4 \
  --video-id VIDEO_ID \
  --game madden27 \
  --channel "Channel Name" \
  --title "Video title" \
  --fps 1 \
  --max-seconds 180 \
  --out vod_pilot/out/VIDEO_ID
```

`--game` is `madden27` or `cfb27`. Outputs are `VIDEO_ID.json` (full row plus confidence) and `VIDEO_ID.csv` (the `log_snap` columns only).

Score a hand-check file (snaps matched on `t_start` within 8 seconds):

```bash
python -m vod_pilot score \
  --pred vod_pilot/out/VIDEO_ID.json \
  --hand vod_pilot/eval/handcheck.json \
  --out vod_pilot/eval/accuracy.json
```

A label field set to `null` means the human could not read it. Filling that field is a false fill. Recall is computed only where the human read a value.

## What a row means

`situation_raw` is a string `parse_situation` can read (`1st & 10`). `result` is a string `parse_outcome` can read (`+4`, `-8`, `incomplete`, `td`, `int`, `fumble`, `convert`, `+0`). A sack keeps `result` as the signed yards when the next down shows them, and `result_kind` (JSON only) as `sack`.

`yardline` uses the coach's 0–100 scale from the offense's own goal, and only when OCR saw an explicit `own` or `opp`. A bare field number stays out of the column.

`opponent_id` is `vod:{channel}`. That is the broadcast, not one of Aidan's opponents. `our_call` and `macro` are empty.

## How tags are chosen

- Down and distance come from RapidOCR on the scorebug. The stylized `&` is read as `8` (`1ST810` → 1st & 10). A real `1st & 8` (`1ST8`) is left alone. Clock tokens (`1st 3:56`) are not downs.
- Yards are `previous distance − next distance` when the down advances by exactly one and the two clocks are the same quarter, a few seconds apart. Anything else is left blank.
- Formation and coverage are filled only when the play-call window agrees on exactly one lexicon name. A formation tab, a "previous play" strip, and a penalty screen list every row, so those frames do not donate a name to the next snap. A single coverage on a "previous play" strip is attached to the snap that just ended.
- Play name is filled from a long tooltip line when the menu lists more than one play. Tesseract cannot read this HUD. There is no vision-model pass: no vision API key was available in the environment.

## Fetching video

YouTube from this environment returns a sign-in wall (`yt-dlp` and the player clients). `python -m vod_pilot.fetch` records that failure and does not try to route around it. The clips used for the write-up in `REPORT.md` were saved outside the repo and are not committed.
