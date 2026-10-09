"""Offline video ingestion. Nothing here runs during a live playcall.

The importer reads a local recording, keeps source timestamps, and can write
a manifest beside the coach database. It does not upload the file and it
does not copy the recording into Git.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

FILM_IMPORT_VERSION = "film_import.v1"
SUPPORTED_SUFFIXES = frozenset({".mp4", ".mkv"})
_SAMPLE_FPS = 4
_SAMPLE_WIDTH = 32
_SAMPLE_HEIGHT = 18


def fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _ratio(value: str | None) -> float | None:
    if not value or value in ("0/0", "N/A"):
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def probe_video(path: Path) -> dict[str, Any]:
    """Read container metadata. A failure is a report, not a guessed video."""
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return {
            "ok": False,
            "error": "unsupported_container",
            "suffix": path.suffix.lower(),
            "supported": sorted(SUPPORTED_SUFFIXES),
        }
    if not path.is_file() or path.stat().st_size == 0:
        return {"ok": False, "error": "incomplete_or_missing_recording", "bytes": path.stat().st_size if path.is_file() else 0}
    probe = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    if probe.returncode != 0 or not probe.stdout.strip():
        return {
            "ok": False,
            "error": "corrupt_or_unsupported_codec",
            "detail": (probe.stderr or probe.stdout or "")[:500],
        }
    try:
        payload = json.loads(probe.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "corrupt_or_unsupported_codec", "detail": "ffprobe returned non-json"}
    video = next((row for row in payload.get("streams") or [] if row.get("codec_type") == "video"), None)
    if not video:
        return {"ok": False, "error": "no_video_stream"}
    fmt = payload.get("format") or {}
    try:
        duration = float(fmt.get("duration") or video.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        return {"ok": False, "error": "incomplete_recording", "duration_s": duration}
    rate = _ratio(video.get("r_frame_rate"))
    average = _ratio(video.get("avg_frame_rate"))
    variable = bool(rate and average and abs(rate - average) > 0.05)
    return {
        "ok": True,
        "duration_s": round(duration, 3),
        "width": video.get("width"),
        "height": video.get("height"),
        "codec": video.get("codec_name"),
        "frame_rate": None if rate is None else round(rate, 3),
        "average_frame_rate": None if average is None else round(average, 3),
        "variable_frame_rate": variable,
        "start_time_s": _ratio(str(fmt.get("start_time"))) if fmt.get("start_time") is not None else 0.0,
        "format_name": fmt.get("format_name"),
    }


def packet_timestamps(path: Path, limit: int = 4000) -> dict[str, Any]:
    """Source packet times. Long recordings keep the ends and the count."""
    probe = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv=p=0", str(path),
    ])
    if probe.returncode != 0:
        return {"ok": False, "error": "timestamps_unavailable", "detail": (probe.stderr or "")[:300]}
    values: list[float] = []
    for line in probe.stdout.splitlines():
        text = line.strip()
        if not text or text == "N/A":
            continue
        try:
            values.append(float(text))
        except ValueError:
            continue
    truncated = len(values) > limit
    kept = values if not truncated else [values[0], values[-1]]
    return {
        "ok": True,
        "count": len(values),
        "truncated": truncated,
        "timestamps_s": [round(value, 3) for value in kept],
        "first_s": None if not values else round(values[0], 3),
        "last_s": None if not values else round(values[-1], 3),
        "basis": "best_effort_timestamp_time",
    }


def sample_frames(path: Path) -> dict[str, Any]:
    """Small RGB samples for segmentation. Times follow the decoder clock."""
    command = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"fps={_SAMPLE_FPS},scale={_SAMPLE_WIDTH}:{_SAMPLE_HEIGHT}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.run(command, capture_output=True, check=False)
    frame_bytes = _SAMPLE_WIDTH * _SAMPLE_HEIGHT * 3
    if proc.returncode != 0 or len(proc.stdout) < frame_bytes:
        return {
            "ok": False,
            "error": "decode_failed",
            "detail": (proc.stderr or b"")[:300].decode("utf-8", "replace"),
        }
    count = len(proc.stdout) // frame_bytes
    frames = []
    for index in range(count):
        start = index * frame_bytes
        frames.append({
            "time_s": round(index / _SAMPLE_FPS, 3),
            "rgb": proc.stdout[start:start + frame_bytes],
            "width": _SAMPLE_WIDTH,
            "height": _SAMPLE_HEIGHT,
        })
    return {"ok": True, "frames": frames, "sample_fps": _SAMPLE_FPS}


def _mean_abs_diff(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    total = 0
    for a, b in zip(left, right):
        total += abs(a - b)
    return total / len(left)


def locate_candidate_snaps(
    frames: Sequence[Mapping[str, Any]],
    manual_anchors: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Find coarse segments. Ambiguous edges stay unresolved.

    Manual anchors are the supported way to correct snap start, snap end,
    the formation screen, the pre-snap window, and the post-snap window.
    """
    auto = _auto_segments(frames)
    corrected = [_anchor_segment(row, index) for index, row in enumerate(manual_anchors or [])]
    return {
        "auto_segments": auto,
        "manual_segments": corrected,
        "segments": corrected or auto,
        "identification_claim": False,
        "note": (
            "Candidate boundaries are not a claim that every visible play was found. "
            "Low-confidence edges stay unresolved until a person corrects them."
        ),
    }


def _auto_segments(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(frames) < 2:
        return [_segment(0.0, 0.0, 0.0, "unresolved", "insufficient_frames", "auto")]
    diffs = [
        (float(frames[index + 1]["time_s"]), _mean_abs_diff(bytes(frames[index]["rgb"]), bytes(frames[index + 1]["rgb"])))
        for index in range(len(frames) - 1)
    ]
    peak = max(value for _, value in diffs)
    start = float(frames[0]["time_s"])
    end = float(frames[-1]["time_s"]) + (1.0 / _SAMPLE_FPS)
    if peak < 8:
        return [_segment(start, end, 0.2, "unresolved", "no_clear_boundary", "auto")]
    threshold = max(12.0, peak * 0.45)
    cuts = [time_s for time_s, value in diffs if value >= threshold]
    bounds = [start]
    for time_s in cuts:
        if time_s - bounds[-1] >= 0.2:
            bounds.append(time_s)
    if end - bounds[-1] >= 0.2:
        bounds.append(end)
    segments = []
    for index, (left, right) in enumerate(zip(bounds, bounds[1:])):
        strength = max((value for time_s, value in diffs if left <= time_s <= right), default=0.0)
        confidence = round(min(0.85, strength / 80.0), 3)
        status = "candidate" if confidence >= 0.45 and (right - left) >= 0.45 else "unresolved"
        reason = "motion_change" if status == "candidate" else "ambiguous_boundary"
        segments.append(_segment(left, right, confidence, status, reason, "auto", index))
    return segments or [_segment(start, end, 0.2, "unresolved", "ambiguous_boundary", "auto")]


def _anchor_segment(anchor: Mapping[str, Any], index: int) -> dict[str, Any]:
    start = anchor.get("snap_start")
    end = anchor.get("snap_end")
    complete = start is not None and end is not None and float(end) > float(start)
    segment = _segment(
        float(start or 0), float(end or 0),
        1.0 if complete else 0.0,
        "corrected" if complete else "unresolved",
        "manual_anchor" if complete else "incomplete_manual_anchor",
        "manual",
        index,
    )
    segment["formation_interval"] = list(anchor.get("formation_interval") or [])
    segment["presnap_interval"] = list(anchor.get("presnap_interval") or [])
    segment["postsnap_interval"] = list(anchor.get("postsnap_interval") or [])
    segment["role"] = anchor.get("role") or "snap"
    return segment


def _segment(
    start: float, end: float, confidence: float, status: str, reason: str,
    source: str, index: int = 0,
) -> dict[str, Any]:
    return {
        "candidate_id": f"{source}-{index}",
        "start_s": round(start, 3),
        "end_s": round(end, 3),
        "confidence": confidence,
        "boundary_status": status,
        "reason": reason,
        "source": source,
        "role": "unknown",
        "formation_interval": [],
        "presnap_interval": [round(start, 3), round(end, 3)] if status != "unresolved" else [],
        "postsnap_interval": [],
    }


def _store_paths(store: Path) -> dict[str, Path]:
    return {
        "index": store / "index.json",
        "recordings": store / "recordings",
        "annotations": store / "annotations",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def coach_game_summary(db_path: Path | None, game_id: str | None) -> dict[str, Any]:
    """Read-only look at an existing coach log. Missing files stay missing."""
    if db_path is None or not db_path.is_file():
        return {"exists": False, "path": None if db_path is None else str(db_path), "history_modified": False}
    import sqlite3

    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        return {"exists": True, "opened": False, "error": str(exc), "history_modified": False}
    try:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        summary: dict[str, Any] = {
            "exists": True, "opened_read_only": True, "game_id": game_id, "history_modified": False,
        }
        if game_id and "game_sessions" in names:
            row = conn.execute(
                "SELECT opponent_id FROM game_sessions WHERE session_id = ?",
                (game_id,),
            ).fetchone()
            summary["opponent_id"] = None if row is None else row[0]
            summary["session_found"] = row is not None
        if game_id and "snaps" in names:
            count = conn.execute(
                "SELECT COUNT(*) FROM snaps WHERE session_id = ?",
                (game_id,),
            ).fetchone()
            summary["logged_snaps"] = 0 if count is None else int(count[0])
        return summary
    except sqlite3.Error as exc:
        return {"exists": True, "opened": False, "error": str(exc), "history_modified": False}
    finally:
        conn.close()


def import_recording(
    path: str | Path,
    *,
    game_id: str | None,
    store: str | Path | None = None,
    dry_run: bool = True,
    manual_anchors: Sequence[Mapping[str, Any]] | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect one local recording. Dry-run writes nothing."""
    source = Path(path)
    started = __import__("time").perf_counter()
    digest = fingerprint_file(source) if source.is_file() and source.stat().st_size else None
    recording_id = None if digest is None else digest[:16]
    probed = probe_video(source)
    result: dict[str, Any] = {
        "version": FILM_IMPORT_VERSION,
        "path": str(source),
        "game_id": game_id,
        "recording_id": recording_id,
        "fingerprint": digest,
        "dry_run": dry_run,
        "uploaded": False,
        "copied_into_git": False,
        "history_modified": False,
        "live_playcaller": False,
        "probe": probed,
        "coach": coach_game_summary(None if db_path is None else Path(db_path), game_id),
    }
    if not probed.get("ok"):
        result["ok"] = False
        result["elapsed_ms"] = round((__import__("time").perf_counter() - started) * 1000.0, 1)
        return result
    stamps = packet_timestamps(source)
    sampled = sample_frames(source)
    if not sampled.get("ok"):
        result["ok"] = False
        result["timestamps"] = stamps
        result["error"] = sampled.get("error")
        result["elapsed_ms"] = round((__import__("time").perf_counter() - started) * 1000.0, 1)
        return result
    located = locate_candidate_snaps(sampled["frames"], manual_anchors)
    result.update({
        "ok": True,
        "timestamps": stamps,
        "segments": located["segments"],
        "auto_segments": located["auto_segments"],
        "manual_segments": located["manual_segments"],
        "identification_claim": False,
        "recognition_accuracy_claim": False,
        "note": located["note"],
    })
    if store is not None and recording_id:
        root = Path(store)
        paths = _store_paths(root)
        index = _read_json(paths["index"]) if paths["index"].is_file() else {}
        previous = (index.get("fingerprints") or {}).get(digest or "")
        result["duplicate"] = bool(previous)
        result["wrote_manifest"] = False
        result["store"] = str(root)
        if not dry_run:
            paths["recordings"].mkdir(parents=True, exist_ok=True)
            paths["annotations"].mkdir(parents=True, exist_ok=True)
            manifest = {
                "version": FILM_IMPORT_VERSION,
                "recording_id": recording_id,
                "fingerprint": digest,
                "path": str(source),
                "game_id": game_id,
                "probe": probed,
                "timestamps": {
                    "first_s": stamps.get("first_s"),
                    "last_s": stamps.get("last_s"),
                    "count": stamps.get("count"),
                    "truncated": stamps.get("truncated"),
                    "basis": stamps.get("basis"),
                },
                "segments": located["segments"],
                "duplicate_of": previous,
            }
            (paths["recordings"] / f"{recording_id}.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8",
            )
            fingerprints = dict(index.get("fingerprints") or {})
            fingerprints[digest or ""] = recording_id
            index["fingerprints"] = fingerprints
            paths["index"].parent.mkdir(parents=True, exist_ok=True)
            paths["index"].write_text(json.dumps(index, indent=2), encoding="utf-8")
            if game_id and not (paths["annotations"] / f"{game_id}.json").is_file():
                from cfb_coach.madden.model.film_review import empty_annotations, save_annotations

                save_annotations(root, empty_annotations(game_id, recording_id, located["segments"]))
            result["wrote_manifest"] = True
    else:
        result["duplicate"] = False
        result["wrote_manifest"] = False
    result["elapsed_ms"] = round((__import__("time").perf_counter() - started) * 1000.0, 1)
    return result
