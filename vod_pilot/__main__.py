"""CLI: tag a local VOD, or score a hand-check file.

Examples
--------
python -m vod_pilot run \\
    --video /tmp/clip.mp4 --video-id ABC --game madden27 \\
    --channel TheNewEra23 --title "Top 100" --out vod_pilot/out/ABC

python -m vod_pilot score \\
    --pred vod_pilot/out/ABC.json --hand vod_pilot/eval/handcheck.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vod_pilot.accuracy import score_files
from vod_pilot.pipeline import run_to_disk


def _run(args: argparse.Namespace) -> int:
    payload = run_to_disk(
        Path(args.video),
        Path(args.out),
        video_id=args.video_id,
        game=args.game,
        channel=args.channel,
        title=args.title,
        sample_fps=args.fps,
        max_seconds=args.max_seconds,
        source_url=args.url,
        published=args.published,
    )
    print(f"snaps {payload['snap_count']}  frames {payload['source']['frames_read']}")
    print(f"wrote {args.out}.json and {args.out}.csv")
    return 0


def _score(args: argparse.Namespace) -> int:
    report = score_files(Path(args.pred), Path(args.hand))
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vod_pilot", description="Tag gameplay VOD snaps")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="OCR a local video and write snap CSV/JSON")
    run.add_argument("--video", required=True)
    run.add_argument("--video-id", required=True)
    run.add_argument("--game", required=True, choices=("madden27", "cfb27"))
    run.add_argument("--channel", required=True)
    run.add_argument("--title", default="")
    run.add_argument("--url", default="")
    run.add_argument("--published", default="")
    run.add_argument("--fps", type=float, default=1.0)
    run.add_argument("--max-seconds", type=float, default=None)
    run.add_argument("--out", required=True, help="path stem, without .json/.csv")
    run.set_defaults(func=_run)

    score = sub.add_parser("score", help="compare predictions to a hand-check JSON")
    score.add_argument("--pred", required=True)
    score.add_argument("--hand", required=True)
    score.add_argument("--out", default="")
    score.set_defaults(func=_score)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
