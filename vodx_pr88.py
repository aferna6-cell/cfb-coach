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
import gzip
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


def as_int(value: Any) -> int | None:
    """Blank, circled, and non-numeric values are None. Never return \"\" and never raise."""
    from vod_pilot.digits import safe_int

    return safe_int(value)


def yards_delta(dist: Any, ndist: Any) -> int | None:
    """``dist - ndist`` only when both distances are real numbers."""
    left, right = as_int(dist), as_int(ndist)
    if left is None or right is None:
        return None
    return left - right


def coerce_row_numbers(row: dict) -> None:
    """In-place: \"\" on a numeric field becomes None. The goal flag is left alone."""
    for key in ("down", "distance", "yardline", "yards", "quarter", "score_left", "score_right"):
        if key in row:
            row[key] = as_int(row.get(key))


def install_quality_guard() -> None:
    """Make vodx_quality.apply_quality treat a blank distance as missing.

    The box function does ``yy = dist - ndist``. A goal-to-go row used to
    carry distance \"\" with an int down, which raises TypeError. Patch the
    loaded function when its source has that subtraction, and always coerce
    blanks before it runs.
    """
    try:
        import vodx_quality
    except ImportError:
        return
    current = getattr(vodx_quality, "apply_quality", None)
    if current is None or getattr(current, "_blank_safe", False):
        return
    import inspect
    import textwrap

    vodx_quality.as_int = as_int
    vodx_quality._yards_delta = yards_delta

    def _coerce_then(fn):
        def apply_quality(rows, *args, **kwargs):
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        coerce_row_numbers(row)
            return fn(rows, *args, **kwargs)

        apply_quality._blank_safe = True
        return apply_quality

    try:
        src = textwrap.dedent(inspect.getsource(current))
    except (OSError, TypeError):
        vodx_quality.apply_quality = _coerce_then(current)
        return
    if "dist - ndist" not in src:
        vodx_quality.apply_quality = _coerce_then(current)
        return
    src = src.replace("dist - ndist", "_yards_delta(dist, ndist)")
    ns = dict(vodx_quality.__dict__)
    ns["_yards_delta"] = yards_delta
    try:
        exec(src, ns)
    except Exception:
        vodx_quality.apply_quality = _coerce_then(current)
        return
    patched = ns.get("apply_quality", current)
    patched._blank_safe = True
    vodx_quality.apply_quality = patched


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


_HR_RX = re.compile(r"^(WR|TE|HB|RB|FB|SLOT|QB|X|Y|Z)?HOTROUTES?$")
_MENU_Y = (0.55, 0.92)


def _norm_alnum(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _token_parts(item: Any) -> tuple[str, float | None, float | None, float | None]:
    parsed = item if isinstance(item, (str, Tok)) else parse_token(item)
    if isinstance(parsed, Tok):
        return parsed.text, parsed.conf, parsed.x, parsed.y
    return str(parsed), None, None, None


def engine_tokens(raw_tokens: list[Any]) -> list[Tok | str]:
    """Same token gate as vodx.py extract_one before classify_frame.

    Confidence under 0.45 is dropped. A hot-route wheel (y 0.55-0.92) is dropped
    when a HOT ROUTES token is on the frame. Positions are kept.
    """
    kept: list[tuple[str, float | None, float | None, float | None]] = []
    for item in raw_tokens:
        text, conf, x, y = _token_parts(item)
        if conf is not None and conf < 0.45:
            continue
        kept.append((text, conf, x, y))
    if any(_HR_RX.search(_norm_alnum(text)) for text, *_ in kept):
        kept = [
            item
            for item in kept
            if item[3] is None or not (_MENU_Y[0] <= item[3] <= _MENU_Y[1])
        ]
    out: list[Tok | str] = []
    for text, conf, x, y in kept:
        if x is None and y is None and conf is None:
            out.append(text)
        else:
            out.append(Tok(text, x=x, y=y, conf=conf))
    return out


def apply_layout(
    frames: list[dict],
    layout: dict | None,
    *,
    section_offset_s: float = 0.0,
) -> list[dict]:
    """Transform, mask, then gameplay window. Same order as vodx.py extract_one.

    ``layout`` comes from tools/layouts.json for one video (masks and
    ``gameplay_window`` in absolute VOD seconds). Token x/y are normalized
    0-1. Frames outside the window are removed. Tokens inside a mask box are
    removed. This is what keeps pre-game and ticker chat out of the snap count.
    """
    prepared = []
    for frame in frames:
        raw = list(frame.get("k", frame.get("tokens", [])))
        prepared.append(
            {
                "t": float(frame.get("t") or frame.get("time") or 0.0),
                "g": float(frame.get("g", frame.get("green", 0.0)) or 0.0),
                "k": [_as_k(item) for item in raw],
            }
        )
    layout = layout or {}
    transform = layout.get("transform")
    if transform:
        sx, sy = float(transform["sx"]), float(transform["sy"])
        ox, oy = float(transform["ox"]), float(transform["oy"])
        for frame in prepared:
            kept = []
            for token in frame["k"]:
                nx = (token[2] - ox) / sx
                ny = (token[3] - oy) / sy
                if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                    kept.append([token[0], token[1], round(nx, 4), round(ny, 4), *token[4:]])
            frame["k"] = kept
    masks = layout.get("mask") or []
    if masks:
        for frame in prepared:
            frame["k"] = [
                token
                for token in frame["k"]
                if not any(
                    box[0] <= token[2] <= box[2] and box[1] <= token[3] <= box[3] for box in masks
                )
            ]
    window = layout.get("gameplay_window")
    if window:
        start = float(window[0] or 0.0)
        end = float(window[1]) if window[1] is not None else float("inf")
        prepared = [
            frame
            for frame in prepared
            if start <= frame["t"] + section_offset_s < end
        ]
    for frame in prepared:
        frame["tokens"] = frame["k"]
    return prepared


def _as_k(item: Any) -> list:
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        return list(item)
    text, conf, x, y = _token_parts(item)
    return [text, 1.0 if conf is None else conf, 0.0 if x is None else x, 0.0 if y is None else y]


def layout_for(record: dict, layouts: dict) -> dict:
    """Creator defaults, then the per-video entry. Video wins."""
    layout = dict((layouts.get("creators") or {}).get(record.get("creator") or "", {}))
    layout.update((layouts.get("videos") or {}).get(str(record.get("video_id") or ""), {}))
    panel = record.get("previous_play_panel")
    layout["calls"] = "panel" if panel is True else ("no_panel" if panel is False else "unknown")
    return layout


def baseline_snap_ids(
    frames: list[dict],
    *,
    video_id: str,
    channel: str,
    game: str,
    lexicon: Lexicon | None = None,
) -> list[str]:
    """v0.96 count logic: engine token gate, plain text, no PR #88 flags."""
    lexicon = lexicon or load_lexicon()
    observed = []
    for frame in frames:
        raw = frame.get("k", frame.get("tokens", []))
        texts = []
        for token in engine_tokens(raw):
            texts.append(token.text if isinstance(token, Tok) else str(token))
        observed.append(classify_frame(frame["t"], texts, frame.get("g", 0.0), lexicon, game=game))
    snaps = build_snaps(
        observed,
        video_id=video_id,
        channel=channel,
        game=game,
        extract=ExtractFlags(),
    )
    return [f"{video_id}:{snap.t_start:.1f}" for snap in snaps]


def snapshot_rows(rows: list[dict]) -> dict[str, dict]:
    """Values and confidences to protect from a later, less-confident stage."""
    saved = {}
    for row in rows:
        saved[row["snap_id"]] = {
            "values": {key: row.get(key) for key in _CONF_KEY},
            "conf": dict(row.get("_pr88_conf") or {}),
        }
    return saved


def restore_if_not_more_confident(row: dict, snapshot: dict) -> None:
    """Put a PR #88 value back when a later stage is not strictly more confident."""
    held = snapshot.get("conf") or {}
    values = snapshot.get("values") or {}
    for field, key in _CONF_KEY.items():
        if key not in held:
            continue
        if row.get(field) == values.get(field):
            continue
        stage_conf = float((row.get("conf") or {}).get(key) or 0.0)
        if held[key] >= stage_conf:
            row[field] = values.get(field)
            row.setdefault("conf", {})[key] = held[key]


def restore_rows(rows: list[dict], snapshots: dict[str, dict]) -> None:
    for row in rows:
        saved = snapshots.get(row.get("snap_id"))
        if saved:
            restore_if_not_more_confident(row, saved)


def observe_frames(
    frames: list[dict],
    *,
    video_id: str,
    channel: str,
    game: str,
    extract: ExtractFlags,
    lexicon: Lexicon | None = None,
):
    """Classify frames that already went through transform, mask, and window."""
    lexicon = lexicon or load_lexicon()
    observed = [
        classify_frame(
            frame["t"],
            engine_tokens(frame.get("k", frame.get("tokens", []))),
            frame.get("g", frame.get("green", 0.0)),
            lexicon,
            extract,
            game=game,
        )
        for frame in frames
    ]
    snaps = build_snaps(
        observed,
        video_id=video_id,
        channel=channel,
        game=game,
        extract=extract,
    )
    return snaps, observed


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
    snaps, _observed = observe_frames(
        frames,
        video_id=video_id,
        channel=channel,
        game=game,
        extract=extract,
        lexicon=lexicon,
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
            "down": as_int(raw.get("down")),
            "distance": as_int(raw.get("distance")),
            "yardline": as_int(raw.get("yardline")),
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
    """Per-VOD ``<video_id>.jsonl.gz`` plus manifest and summary.

    ``source.fixes`` and ``field_coverage`` are in both JSON files. Opponent
    type and side already on the row are left as the pipeline set them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[dict]] = {}
    for row in rows:
        by_video.setdefault(str(row.get("video_id") or "vod"), []).append(row)
    for video_id, group in by_video.items():
        with gzip.open(out_dir / f"{video_id}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for row in group:
                fh.write(json.dumps(row, default=str) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "source": manifest["source"],
        "field_coverage": manifest["field_coverage"],
        "counts": manifest["counts"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def stamp_manifest_row(row: dict, record: dict) -> None:
    """Keep Scout opponent metadata and the engine side. Do not invent either."""
    opponent = record.get("opponent_type") or row.get("opponent_type") or "unknown"
    if opponent not in ("cpu", "human", "unknown"):
        opponent = "unknown"
    row["opponent_type"] = opponent
    row["opponent_type_source"] = record.get("opponent_type_source") or (
        "scout_manifest" if opponent in ("cpu", "human") else "missing_scout_tag"
    )
    row["side"] = row.get("side") or "offense"
    if record.get("game") and record.get("game") != "mixed":
        row["game"] = record["game"]


def run_cache(
    cache: Path,
    *,
    out_dir: Path,
    fixes: str,
    game: str,
    channel: str,
    video_ids: list[str] | None,
    version: str,
    layouts: dict | None = None,
    records: dict[str, dict] | None = None,
) -> dict:
    extract = flags_from_fixes(fixes)
    files = cache_files(cache, video_ids)
    if not files:
        raise SystemExit(f"no OCR cache jsonl under {cache}")
    rows: list[dict] = []
    for video_id, path in files:
        record = dict((records or {}).get(video_id) or {})
        record.setdefault("video_id", video_id)
        record.setdefault("game", game)
        record.setdefault("creator", channel)
        layout = layout_for(record, layouts or {})
        frames = apply_layout(load_ocr_jsonl(path), layout, section_offset_s=float(record.get("section_offset_s") or 0))
        built = rows_from_frames(
            frames,
            video_id=video_id,
            channel=channel or record.get("creator") or "",
            game=record.get("game") if record.get("game") != "mixed" else game,
            extract=extract,
        )
        for row in built:
            stamp_manifest_row(row, record)
        rows.extend(built)
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
    run.add_argument("--layouts", default="", help="tools/layouts.json; masks and gameplay windows apply before classify")
    run.add_argument("--out", required=True, help="label directory (per-VOD jsonl.gz, manifest, summary)")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        layouts = json.loads(Path(args.layouts).read_text(encoding="utf-8")) if args.layouts else {}
        manifest = run_cache(
            Path(args.cache),
            out_dir=Path(args.out),
            fixes=args.fixes,
            game=args.game,
            channel=args.channel,
            video_ids=args.video_ids,
            version=args.label_version,
            layouts=layouts,
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
