"""Glue the PR #88 readers into the box vodx.py label path.

The box wrapper calls ``classify_frame`` / ``build_snaps`` with plain strings
and then builds rows from a fixed field list, so the new readers never reached
the labels. This module:

- turns OCR-cache tokens ``[text, conf, x, y]`` into positioned tokens
- passes ``ExtractFlags`` from ``--fixes``
- copies the new fields onto each vodx row
- writes ``source.fixes`` and ``field_coverage`` into the label manifest
- lets a later vodx_refine / vodx_field / vodx_score write win only when that
  stage is strictly more confident (otherwise the PR #88 value stays)
- rebuilds from an OCR cache with ``--cache-only`` and never opens a video

Snap grouping stays on the baseline down, so snap ids stay ``video:t_start``
except menu rows dropped by ``flag-menu``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from vod_pilot.field_coverage import coverage_report
from vod_pilot.flags import ExtractFlags
from vod_pilot.lexicon import Lexicon, load_lexicon
from vod_pilot.segment import build_snaps, classify_frame
from vod_pilot.tokens import Tok

# Cumulative order. Each build adds the next name to --fixes.
FIX_ORDER = (
    "read-defense",
    "read-score",
    "read-offense-formation",
    "read-adjustments",
    "yardline-glyphs",
    "read-result",
    "goal-to-go",
    "dd-norm",
    "presnap-clock",
    "flag-menu",
    "field-coverage",
)

_FLAG_ATTR = {
    "read-defense": "read_defense",
    "read-score": "read_score",
    "read-offense-formation": "read_offense_formation",
    "read-adjustments": "read_adjustments",
    "yardline-glyphs": "yardline_glyphs",
    "read-result": "read_result",
    "goal-to-go": "goal_to_go",
    "dd-norm": "dd_norm",
    "presnap-clock": "presnap_clock",
    "flag-menu": "flag_menu",
    "field-coverage": "field_coverage",
    "book-match": "book_match",
}
_FLAG_ATTR.update({attr: attr for attr in _FLAG_ATTR.values()})

# Row field -> confidence key written by the PR #88 reader.
_CONF_KEY = {
    "def_formation": "def_formation",
    "team_left": "score",
    "team_right": "score",
    "score_left": "score",
    "score_right": "score",
    "score": "score",
    "record_left": "score",
    "record_right": "score",
    "adjustments": "adjustments",
    "audible": "adjustments",
    "hot_routes": "adjustments",
    "off_macros": "adjustments",
    "def_adjustments": "adjustments",
    "goal_to_go": "goal_to_go",
    "result": "result",
    "result_kind": "result",
    "result_flag": "result",
    "yards": "result",
    "clock": "clock",
    "yardline": "yardline",
    "formation": "formation",
    "down": "down",
    "distance": "distance",
    "is_menu_or_lobby": "menu",
    "menu_confidence": "menu",
}


def flags_from_fixes(spec: str) -> ExtractFlags:
    """``read-defense,read-score`` or attribute names. ``all`` is FIX_ORDER."""
    names = [part.strip() for part in (spec or "").split(",") if part.strip()]
    if any(name == "all" for name in names):
        names = list(FIX_ORDER)
    unknown = [name for name in names if name not in _FLAG_ATTR]
    if unknown:
        known = ", ".join(FIX_ORDER)
        raise SystemExit(f"unknown --fixes {unknown}. Known: {known}, book-match, all")
    data = {attr: False for attr in set(_FLAG_ATTR.values())}
    for name in names:
        data[_FLAG_ATTR[name]] = True
    return ExtractFlags(**data)


def parse_token(item: Any) -> str | Tok:
    """Cache token to a positioned token. Plain strings stay strings."""
    if isinstance(item, str):
        match = re.fullmatch(r"(.+)@([0-9.]+),([0-9.]+),([0-9.]+)", item.strip())
        if not match:
            return item
        return Tok(match.group(1), x=float(match.group(2)), y=float(match.group(3)), conf=float(match.group(4)))
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("t") or "")
        return Tok(
            text,
            x=_float(item.get("x")),
            y=_float(item.get("y")),
            conf=_float(item.get("conf", item.get("score"))),
        )
    if isinstance(item, (list, tuple)) and item:
        return Tok(
            str(item[0]),
            conf=_float(item[1]) if len(item) > 1 else None,
            x=_float(item[2]) if len(item) > 2 else None,
            y=_float(item[3]) if len(item) > 3 else None,
        )
    return str(item)


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def normalize_frame(record: dict) -> dict:
    raw = record.get("k", record.get("tokens", record.get("texts", [])))
    green = record.get("g", record.get("green", 0.0))
    return {
        "t": float(record.get("t") or record.get("time") or 0.0),
        "g": float(green or 0.0),
        "tokens": [parse_token(item) for item in raw],
    }


def load_ocr_jsonl(path: Path) -> list[dict]:
    """OCR cache only. Does not open a video."""
    text = path.read_text(encoding="utf-8")
    frames = []
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "frames" in record and isinstance(record["frames"], list):
            frames.extend(normalize_frame(frame) for frame in record["frames"])
        else:
            frames.append(normalize_frame(record))
    frames.sort(key=lambda frame: frame["t"])
    return frames


def cache_files(cache: Path, video_ids: Iterable[str] | None) -> list[tuple[str, Path]]:
    paths = [cache] if cache.is_file() else sorted(cache.glob("*.jsonl"))
    want = {item for item in (video_ids or [])}
    found: list[tuple[str, Path]] = []
    for path in paths:
        video_id = path.name.split(".fps", 1)[0]
        if want and video_id not in want and path.stem not in want:
            continue
        found.append((video_id, path))
    return found


def _clock_text(situation: str) -> str:
    match = re.search(r"(\d{1,2}:\d{2})", situation or "")
    return match.group(1) if match else ""


def _adjustment_fields(labels: list[str]) -> dict[str, str]:
    buckets = {"audible": [], "hot_routes": [], "off_macros": [], "def_adjustments": []}
    for label in labels:
        kind, _, text = label.partition(": ")
        if kind == "hot_route":
            buckets["hot_routes"].append(text)
        elif kind == "audible":
            buckets["audible"].append(text)
        elif kind == "offensive":
            buckets["off_macros"].append(text)
        elif kind == "defensive":
            buckets["def_adjustments"].append(text)
    return {key: "; ".join(values) for key, values in buckets.items() if values}


def rows_from_frames(
    frames: list[dict],
    *,
    video_id: str,
    channel: str,
    game: str,
    extract: ExtractFlags,
    lexicon: Lexicon | None = None,
) -> list[dict]:
    """Classify with positions and flags. Snap ids stay ``video:t_start``."""
    lexicon = lexicon or load_lexicon()

    def _tokens(frame: dict) -> list:
        raw = frame.get("tokens", frame.get("k", []))
        out = []
        for item in raw:
            out.append(item if isinstance(item, (str, Tok)) else parse_token(item))
        return out

    observed = [
        classify_frame(frame["t"], _tokens(frame), frame.get("g", frame.get("green", 0.0)), lexicon, extract, game=game)
        for frame in frames
    ]
    snaps = build_snaps(
        observed,
        video_id=video_id,
        channel=channel,
        game=game,
        extract=extract,
    )
    rows = []
    for snap in snaps:
        raw = snap.to_full_dict()
        conf = dict(raw.get("confidence") or {})
        adjustments = list(raw.get("adjustments") or [])
        adj = _adjustment_fields(adjustments)
        score_left, score_right = raw.get("score_left"), raw.get("score_right")
        score = f"{score_left}-{score_right}" if score_left is not None and score_right is not None else None
        row = {
            "snap_id": f"{video_id}:{float(raw['t_start']):.1f}",
            "video_id": video_id,
            "game": game,
            "t_start": raw.get("t_start"),
            "t_end": raw.get("t_end"),
            "side": raw.get("side") or "",
            "quarter": raw.get("quarter"),
            "clock": _clock_text(raw.get("situation_raw") or ""),
            "down": raw.get("down"),
            "distance": raw.get("distance"),
            "yardline": raw.get("yardline"),
            "situation_raw": raw.get("situation_raw") or "",
            "formation": raw.get("formation") or "",
            "play": raw.get("play") or "",
            "coverage_seen": raw.get("coverage_seen") or "",
            "def_formation": raw.get("def_formation") or "",
            "team_left": raw.get("team_left") or "",
            "team_right": raw.get("team_right") or "",
            "score_left": score_left,
            "score_right": score_right,
            "record_left": raw.get("record_left") or "",
            "record_right": raw.get("record_right") or "",
            "score": score,
            "adjustments": adjustments,
            "goal_to_go": bool(raw.get("goal_to_go")),
            "is_menu_or_lobby": bool(raw.get("is_menu_or_lobby")),
            "menu_confidence": float(raw.get("menu_confidence") or 0.0),
            "result": raw.get("result") or "",
            "result_kind": raw.get("result_kind") or "",
            "result_flag": raw.get("result_flag") or "",
            "yards": raw.get("yards"),
            "book_match": raw.get("book_match") or "",
        }
        row.update(adj)
        owned = {
            "def_formation": float(conf.get("def_formation") or 0.0),
            "score": float(conf.get("score") or 0.0),
            "adjustments": float(conf.get("adjustments") or 0.0),
            "yardline": float(conf.get("yardline") or 0.0),
            "formation": float(conf.get("formation") or 0.0),
            "down": float(conf.get("down") or 0.0),
            "distance": float(conf.get("down") or 0.0),
            "result": float(conf.get("result") or 0.0),
            "clock": 0.85 if row["clock"] else 0.0,
            "goal_to_go": 0.8 if row["goal_to_go"] else 0.0,
            "menu": float(row["menu_confidence"] or 0.0),
        }
        # Only fields this build actually read are protected from a later stage.
        active = {}
        if extract.read_defense:
            active["def_formation"] = owned["def_formation"]
        if extract.read_score:
            active["score"] = owned["score"]
        if extract.read_adjustments:
            active["adjustments"] = owned["adjustments"]
        if extract.yardline_glyphs:
            active["yardline"] = owned["yardline"]
        if extract.read_offense_formation:
            active["formation"] = owned["formation"]
        if extract.dd_norm or extract.goal_to_go:
            active["down"] = owned["down"]
            active["distance"] = owned["distance"]
        if extract.goal_to_go:
            active["goal_to_go"] = owned["goal_to_go"]
        if extract.read_result:
            active["result"] = owned["result"]
        if extract.presnap_clock:
            active["clock"] = owned["clock"]
        if extract.flag_menu:
            active["menu"] = owned["menu"]
        row["conf"] = {**conf, **{key: value for key, value in owned.items() if key in active}}
        row["_pr88_conf"] = active
        rows.append(row)
    return rows


def stage_write(row: dict, field: str, value: Any, conf: float, source: str) -> bool:
    """A vodx_refine / vodx_field / vodx_score write.

    Returns False when the PR #88 reader already holds the field at equal or
    higher confidence, or when the value is already there. That is the
    no-double-write rule: the new reader wins only when it is more confident.
    """
    key = _CONF_KEY.get(field, field)
    held = (row.get("_pr88_conf") or {}).get(key)
    current = row.get(field)
    if current == value:
        return False
    if held is not None and held >= conf and current not in ("", None, [], {}):
        return False
    row[field] = value
    row.setdefault("conf", {})[key] = float(conf)
    row.setdefault("field_sources", {})[field] = source
    if held is None or conf > held:
        row.setdefault("_pr88_conf", {}).pop(key, None)
    return True


def label_manifest(rows: list[dict], *, version: str, fixes: list[str], cache_only: bool) -> dict:
    report = coverage_report(rows)
    return {
        "version": version,
        "source": {"fixes": list(fixes), "cache_only": cache_only},
        "field_coverage": report,
        "counts": {"snaps_labeled": len(rows)},
    }


def write_label_build(out_dir: Path, rows: list[dict], manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snaps.jsonl").write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "source": manifest["source"],
        "field_coverage": manifest["field_coverage"],
        "counts": manifest["counts"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_cache(
    cache: Path,
    *,
    out_dir: Path,
    fixes: str,
    game: str,
    channel: str,
    video_ids: list[str] | None,
    version: str,
) -> dict:
    extract = flags_from_fixes(fixes)
    files = cache_files(cache, video_ids)
    if not files:
        raise SystemExit(f"no OCR cache jsonl under {cache}")
    rows: list[dict] = []
    for video_id, path in files:
        frames = load_ocr_jsonl(path)
        rows.extend(
            rows_from_frames(
                frames,
                video_id=video_id,
                channel=channel,
                game=game,
                extract=extract,
            )
        )
    manifest = label_manifest(
        rows,
        version=version,
        fixes=extract.enabled(),
        cache_only=True,
    )
    write_label_build(out_dir, rows, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vodx_pr88", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="rebuild labels from an OCR cache")
    run.add_argument("--cache", required=True, help="OCR jsonl file or directory of *.jsonl")
    run.add_argument("--cache-only", action="store_true", help="do not open a video; cache is the only input")
    run.add_argument("--video-ids", nargs="*", default=None)
    run.add_argument("--game", default="madden27", choices=("madden27", "cfb27"))
    run.add_argument("--channel", default="")
    run.add_argument("--fixes", default="", help="comma list, or 'all'. See FIX_ORDER.")
    run.add_argument("--label-version", default="v0.97")
    run.add_argument("--out", required=True, help="label directory (manifest.json + summary.json)")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        manifest = run_cache(
            Path(args.cache),
            out_dir=Path(args.out),
            fixes=args.fixes,
            game=args.game,
            channel=args.channel,
            video_ids=args.video_ids,
            version=args.label_version,
        )
        print(
            f"snaps {manifest['counts']['snaps_labeled']}  "
            f"fixes {','.join(manifest['source']['fixes']) or '(none)'}  "
            f"cache_only {manifest['source']['cache_only']}"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
