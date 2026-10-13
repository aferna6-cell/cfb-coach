"""Run OCR + snap grouping on one local video file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vod_pilot.lexicon import Lexicon, load_lexicon
from vod_pilot.ocr_frames import FrameOCR, green_ratio, iter_samples
from vod_pilot.schema import SCHEMA_ID, write_outputs
from vod_pilot.segment import build_snaps, classify_frame


def analyze_video(
    video_path: Path,
    *,
    video_id: str,
    game: str,
    channel: str,
    title: str = "",
    sample_fps: float = 1.0,
    max_seconds: float | None = None,
    lexicon: Lexicon | None = None,
    source_url: str = "",
    published: str = "",
) -> dict[str, Any]:
    lexicon = lexicon or load_lexicon()
    reader = FrameOCR()
    frames = []
    n = 0
    for t, frame in iter_samples(str(video_path), sample_fps=sample_fps, max_seconds=max_seconds):
        texts = reader.read(frame)
        frames.append(classify_frame(t, texts, green_ratio(frame), lexicon))
        n += 1
    snaps = build_snaps(frames, video_id=video_id, channel=channel, game=game)
    return {
        "schema": SCHEMA_ID,
        "compatible_with": ["snaps", "play_records"],
        "import": (
            "Rows use CoachDB.log_snap column names. Do not insert them into "
            "coach.db from this pilot: opponent_id is the channel, side is the "
            "broadcast offense/defense, and our_call is empty."
        ),
        "game": game,
        "source": {
            "video_id": video_id,
            "title": title,
            "channel": channel,
            "url": source_url or f"https://www.youtube.com/watch?v={video_id}",
            "published": published,
            "file": str(video_path),
            "sample_fps": sample_fps,
            "frames_read": n,
        },
        "snap_count": len(snaps),
        "snaps": [snap.to_full_dict() for snap in snaps],
    }


def run_to_disk(video_path: Path, out_stem: Path, **kwargs: Any) -> dict[str, Any]:
    payload = analyze_video(video_path, **kwargs)
    write_outputs(out_stem, payload)
    return payload
