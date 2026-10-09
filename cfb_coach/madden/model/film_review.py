"""Local review of candidate snaps. Annotations are a separate dataset.

Confirm, correct, reject, and leave-uncertain update that file only.
Original gameplay rows are not opened for writing.
"""
from __future__ import annotations

import json
import subprocess
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from cfb_coach.madden.model.film_evidence import snap_belongs

ANNOTATION_SCHEMA = "madden.film.annotations.v1"
REVIEW_ACTIONS = frozenset({"confirm", "correct", "reject", "uncertain"})
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
TIMING_VALUES = frozenset({"pre_snap", "at_snap", "post_snap", "unknown"})
EXPLICIT_FIELDS = (
    "formation", "play", "down", "distance", "quarter", "clock_seconds",
    "coverage_shell", "safety_depth", "pressure", "box_count", "front", "leverage",
)
STATE_FIELDS = ("down", "distance", "quarter", "clock_seconds")
DEFENSE_FIELDS = ("coverage_shell", "safety_depth", "pressure", "box_count", "front", "leverage")
MEDIA_SUFFIXES = frozenset({".mp4", ".mkv"})


def annotation_path(store: str | Path, game_id: str) -> Path:
    return Path(store) / "annotations" / f"{game_id}.json"


def empty_annotations(game_id: str, recording_id: str | None, segments: list[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = []
    for segment in segments:
        candidates.append({
            "candidate_id": segment.get("candidate_id"),
            "start_s": segment.get("start_s"),
            "end_s": segment.get("end_s"),
            "formation_interval": segment.get("formation_interval") or [],
            "presnap_interval": segment.get("presnap_interval") or [],
            "postsnap_interval": segment.get("postsnap_interval") or [],
            "boundary_status": segment.get("boundary_status"),
            "role": segment.get("role"),
            "snap_claim": bool(segment.get("snap_claim")),
            "suggested_seq": segment.get("suggested_seq"),
            "review_status": "unreviewed",
            "association_status": segment.get("association_status") or "unresolved",
            "snap_id": None,
            "proposed_snap_id": None,
            "formation": None,
            "play": None,
            "game_state": {},
            "defensive_observation": {},
            "observation_time": None,
            "unknown_fields": [],
            "confidence": segment.get("confidence"),
            "provenance": segment.get("source") or "auto",
            "missing": ["snap_association", "formation", "play", "defensive_alignment"],
            "defense_review": "unreviewed",
            "execution_review": "unreviewed",
            "human_label": None,
        })
    return {
        "schema": ANNOTATION_SCHEMA,
        "game_id": game_id,
        "recording_id": recording_id,
        "history_modified": False,
        "candidates": candidates,
    }


def load_annotations(store: str | Path, game_id: str) -> dict[str, Any]:
    path = annotation_path(store, game_id)
    if not path.is_file():
        return empty_annotations(game_id, None, [])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return empty_annotations(game_id, None, [])
    if not isinstance(payload, dict):
        return empty_annotations(game_id, None, [])
    payload.setdefault("candidates", [])
    payload["history_modified"] = False
    return payload


def save_annotations(store: str | Path, payload: Mapping[str, Any]) -> Path:
    game_id = str(payload.get("game_id") or "unscoped")
    path = annotation_path(store, game_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["schema"] = ANNOTATION_SCHEMA
    body["history_modified"] = False
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def apply_review_action(
    store: str | Path,
    game_id: str,
    candidate_id: str,
    action: str,
    updates: Mapping[str, Any] | None = None,
    log_snaps: list[Mapping[str, Any]] | None = None,
    opponent_id: str | None = None,
) -> dict[str, Any]:
    """Record a review decision. Does not open the gameplay database.

    ``log_snaps is None`` keeps the older direct-call behavior: a supplied
    snap id can be marked confirmed. A list, including an empty list, is
    checked against that log before an association is certified.
    """
    if action not in REVIEW_ACTIONS:
        return {"ok": False, "error": "unknown_review_action", "saved": False, "history_modified": False}
    payload = load_annotations(store, game_id)
    match = next((row for row in payload.get("candidates") or [] if row.get("candidate_id") == candidate_id), None)
    if match is None:
        return {"ok": False, "error": "unknown_candidate", "saved": False, "history_modified": False}
    updates = dict(updates or {})
    if action == "confirm":
        draft = dict(match)
        _apply_updates(draft, updates)
        errors = _confirm_errors(draft, log_snaps, game_id, opponent_id)
        if errors:
            return {
                "ok": False,
                "error": "confirm_rejected",
                "errors": errors,
                "saved": False,
                "history_modified": False,
            }
        _apply_updates(match, updates)
        match["review_status"] = "confirmed"
        match["association_status"] = "confirmed"
        match["defense_review"] = "confirmed"
        if "execution_review" in updates:
            match["execution_review"] = updates["execution_review"]
    elif action == "correct":
        _apply_updates(match, updates)
        match["review_status"] = "corrected"
        match["boundary_status"] = "corrected"
        match["association_status"] = _association_status(
            match.get("snap_id"), log_snaps, game_id, opponent_id,
        )
    elif action == "reject":
        match["review_status"] = "rejected"
        match["association_status"] = "rejected"
    else:
        match["review_status"] = "uncertain"
        match["association_status"] = "unresolved"
    if updates.get("human_label") and action in ("confirm", "correct"):
        match["human_label"] = updates["human_label"]
    match["verified_execution"] = False
    match["missing"] = _missing(match)
    payload["history_modified"] = False
    path = save_annotations(store, payload)
    return {"ok": True, "saved": True, "candidate": match, "path": str(path), "history_modified": False}


def _confirm_errors(
    row: Mapping[str, Any],
    log_snaps: list[Mapping[str, Any]] | None,
    game_id: str,
    opponent_id: str | None,
) -> list[str]:
    errors = []
    association = _association_error(row.get("snap_id"), log_snaps, game_id, opponent_id)
    if association:
        errors.append(association)
    timing = row.get("observation_time")
    if timing not in TIMING_VALUES:
        errors.append("timing_not_explicit")
    unknown = set(row.get("unknown_fields") or [])
    for field in EXPLICIT_FIELDS:
        if field in unknown:
            continue
        if _is_blank(_field_value(row, field)):
            errors.append(f"{field}_not_explicit")
    return errors


def _association_error(
    snap_id: Any,
    log_snaps: list[Mapping[str, Any]] | None,
    game_id: str,
    opponent_id: str | None,
) -> str | None:
    if not snap_id:
        return "snap_association"
    if log_snaps is None:
        return None
    found = next((row for row in log_snaps if str(row.get("snap_id") or "") == str(snap_id)), None)
    if found is None:
        return "snap_not_in_log"
    ok, reason = snap_belongs(found, game_id=game_id, opponent_id=opponent_id)
    return None if ok else reason


def _association_status(
    snap_id: Any,
    log_snaps: list[Mapping[str, Any]] | None,
    game_id: str,
    opponent_id: str | None,
) -> str:
    if _association_error(snap_id, log_snaps, game_id, opponent_id):
        return "unresolved"
    return "confirmed"


def _apply_updates(match: dict[str, Any], updates: Mapping[str, Any]) -> None:
    unknown = [str(item) for item in (updates.get("unknown_fields") or match.get("unknown_fields") or [])]
    for key in ("start_s", "end_s", "formation_interval", "presnap_interval", "postsnap_interval"):
        if key in updates:
            match[key] = updates[key]
    if "snap_id" in updates:
        match["snap_id"] = updates.get("snap_id") or None
    state = dict(match.get("game_state") or {})
    if isinstance(updates.get("game_state"), dict):
        state.update(updates["game_state"])
    defense = dict(match.get("defensive_observation") or {})
    if isinstance(updates.get("defensive_observation"), dict):
        defense.update(updates["defensive_observation"])
    if "formation" in updates:
        match["formation"] = updates.get("formation")
    if "play" in updates:
        match["play"] = updates.get("play")
    if "observation_time" in updates:
        timing = updates.get("observation_time")
        match["observation_time"] = timing if timing in TIMING_VALUES else None
    for field in ("formation", "play"):
        if isinstance(match.get(field), str) and match[field].strip().lower() == "unknown":
            match[field] = None
            unknown.append(field)
    for field in STATE_FIELDS:
        if field in updates and field not in state:
            state[field] = updates[field]
        value = state.get(field)
        if isinstance(value, str) and value.strip().lower() == "unknown":
            state[field] = None
            unknown.append(field)
        elif isinstance(value, str) and value.strip() == "":
            state[field] = None
    for field in DEFENSE_FIELDS:
        value = defense.get(field)
        if field == "pressure":
            value = _coerce_pressure(value)
        if isinstance(value, str) and value.strip().lower() == "unknown":
            defense[field] = None
            unknown.append(field)
        elif isinstance(value, str) and value.strip() == "":
            defense[field] = None
        else:
            defense[field] = value
    for field in set(unknown):
        if field in ("formation", "play"):
            match[field] = None
        elif field in STATE_FIELDS:
            state[field] = None
        elif field in DEFENSE_FIELDS:
            defense[field] = None
    match["game_state"] = state
    match["defensive_observation"] = defense
    match["unknown_fields"] = sorted(set(unknown))


def _coerce_pressure(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
        if text in ("", "unknown"):
            return None if text == "" else "unknown"
    return value


def _field_value(row: Mapping[str, Any], field: str) -> Any:
    if field in ("formation", "play"):
        return row.get(field)
    if field in STATE_FIELDS:
        return (row.get("game_state") or {}).get(field)
    return (row.get("defensive_observation") or {}).get(field)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _missing(row: Mapping[str, Any]) -> list[str]:
    unknown = set(row.get("unknown_fields") or [])
    missing = []
    if row.get("association_status") != "confirmed" or not row.get("snap_id"):
        missing.append("snap_association")
    if "formation" not in unknown and not row.get("formation"):
        missing.append("formation")
    if "play" not in unknown and not row.get("play"):
        missing.append("play")
    defense = row.get("defensive_observation") or {}
    if not defense and not unknown.intersection(DEFENSE_FIELDS):
        missing.append("defensive_alignment")
    if "coverage_shell" not in unknown and defense and not defense.get("coverage_shell"):
        missing.append("coverage_shell")
    if row.get("observation_time") not in TIMING_VALUES:
        missing.append("observation_time")
    return missing


def authorized_recording_path(store: str | Path, game_id: str) -> Path | None:
    """The manifest path for this game's recording. Query strings are ignored."""
    annotations = load_annotations(store, game_id)
    recording_id = annotations.get("recording_id")
    if not recording_id:
        return None
    manifest_path = Path(store) / "recordings" / f"{recording_id}.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    raw = manifest.get("path")
    if not raw:
        return None
    path = Path(str(raw))
    if path.suffix.lower() not in MEDIA_SUFFIXES or not path.is_file():
        return None
    return path


def make_review_server(
    store: str | Path,
    game_id: str,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    log_snaps: list[Mapping[str, Any]] | None = None,
    opponent_id: str | None = None,
) -> ThreadingHTTPServer:
    """Localhost review server. The media route ignores any requested file path."""
    if host not in LOCAL_HOSTS:
        raise ValueError("review_server_localhost_only")
    snaps = list(log_snaps or [])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/media":
                self._media()
                return
            if parsed.path == "/frame":
                self._frame(parse_qs(parsed.query))
                return
            if parsed.path not in ("/", "/review"):
                self.send_error(404)
                return
            page = render_review_html(load_annotations(store, game_id), log_snaps=snaps).encode("utf-8")
            self._bytes(200, "text/html; charset=utf-8", page)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/review":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                body = {}
            result = apply_review_action(
                store, game_id, str(body.get("candidate_id") or ""), str(body.get("action") or ""),
                body.get("updates") if isinstance(body.get("updates"), dict) else None,
                log_snaps=snaps,
                opponent_id=opponent_id,
            )
            encoded = json.dumps(result).encode("utf-8")
            self._bytes(200 if result.get("ok") else 400, "application/json", encoded)

        def _media(self) -> None:
            path = authorized_recording_path(store, game_id)
            if path is None:
                self.send_error(404)
                return
            size = path.stat().st_size
            if size <= 0:
                self.send_error(404)
                return
            parsed = _parse_range(self.headers.get("Range"), size)
            if parsed is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end, partial = parsed
            self._file(path, start, end, 206 if partial else 200, size)

        def _frame(self, query: dict[str, list[str]]) -> None:
            path = authorized_recording_path(store, game_id)
            if path is None:
                self.send_error(404)
                return
            try:
                stamp = float((query.get("t") or ["0"])[0])
            except (TypeError, ValueError):
                stamp = 0.0
            if stamp < 0 or stamp > 24 * 3600:
                stamp = 0.0
            proc = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-ss", f"{stamp:.3f}", "-i", str(path),
                    "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "pipe:1",
                ],
                capture_output=True, check=False,
            )
            if proc.returncode != 0 or not proc.stdout.startswith(b"\xff\xd8"):
                self.send_error(404)
                return
            self._bytes(200, "image/jpeg", proc.stdout)

        def _file(self, path: Path, start: int, end: int, status: int, size: int) -> None:
            length = end - start + 1
            content_type = "video/x-matroska" if path.suffix.lower() == ".mkv" else "video/mp4"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _bytes(self, status: int, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def serve_review(
    store: str | Path,
    game_id: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    log_snaps: list[Mapping[str, Any]] | None = None,
    opponent_id: str | None = None,
) -> None:
    """Serve the review page on localhost until the process stops."""
    server = make_review_server(
        store, game_id, host=host, port=port, log_snaps=log_snaps, opponent_id=opponent_id,
    )
    bound_host, bound_port = server.server_address[:2]
    print(json.dumps({
        "serving": f"http://{bound_host}:{bound_port}",
        "game_id": game_id,
        "history_modified": False,
    }))
    server.serve_forever()


def _parse_range(header: str | None, size: int) -> tuple[int, int, bool] | None:
    if not header:
        return 0, size - 1, False
    if not header.startswith("bytes="):
        return None
    spec = header.split("=", 1)[1].split(",")[0].strip()
    try:
        if spec.startswith("-"):
            tail = int(spec[1:])
            if tail <= 0:
                return None
            start = max(0, size - tail)
            return start, size - 1, True
        start_text, _, end_text = spec.partition("-")
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else size - 1
    except ValueError:
        return None
    if start < 0 or start >= size or end < start:
        return None
    return start, min(end, size - 1), True


def render_review_html(
    payload: Mapping[str, Any],
    log_snaps: list[Mapping[str, Any]] | None = None,
) -> str:
    """One page for several snaps, with playback and editable fields."""
    candidates = [dict(row) for row in payload.get("candidates") or []]
    suggestions = _suggestions(candidates, log_snaps or [])
    cards = []
    for index, row in enumerate(candidates):
        suggestion = suggestions.get(str(row.get("candidate_id"))) or {}
        cards.append(_card(index, row, suggestion, log_snaps or []))
    body = "\n".join(cards) or "<p>No candidate snaps in this annotation file.</p>"
    log_json = json.dumps(list(log_snaps or [])).replace("<", "\\u003c")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Film review {escape(str(payload.get('game_id') or ''))}</title>
  <style>
    body {{ font-family: sans-serif; margin: 1.5rem; background: #111; color: #eee; }}
    article {{ border: 1px solid #444; padding: 1rem; margin-bottom: 1rem; }}
    label {{ display: block; margin: 0.35rem 0; }}
    button, video, img {{ margin-right: 0.4rem; }}
    video, img {{ max-width: 100%; background: #000; }}
    .hidden {{ display: none; }}
    .status {{ min-height: 1.2rem; }}
  </style>
</head>
<body>
  <h1>Film review</h1>
  <p>Game {escape(str(payload.get('game_id') or ''))}. Recording {escape(str(payload.get('recording_id') or ''))}.</p>
  <p>Playback requires <code>film-review --serve</code> on localhost. The player loads <code>/media</code> for the imported recording only.</p>
  <video id="film" controls preload="metadata" src="/media"></video>
  <div class="controls">
    <button type="button" id="seek">Seek to snap</button>
    <button type="button" id="loop">Loop current snap</button>
    <button type="button" id="presnap">Pre-snap frame</button>
    <button type="button" id="postsnap">Post-snap frame</button>
  </div>
  <img id="frame" alt="Selected frame">
  <p id="save-status" class="status"></p>
  {body}
  <script id="log-snaps" type="application/json">{log_json}</script>
  <script>
    const cards = Array.from(document.querySelectorAll('article.snap'));
    const video = document.getElementById('film');
    const frame = document.getElementById('frame');
    const status = document.getElementById('save-status');
    let looping = false;
    function show(index) {{
      cards.forEach((card, i) => card.classList.toggle('hidden', i !== index));
      status.textContent = '';
    }}
    function current() {{
      return cards.find((card) => !card.classList.contains('hidden')) || cards[0];
    }}
    function numberValue(card, name) {{
      const el = card.querySelector('[name="' + name + '"]');
      if (!el || el.value === '') return null;
      const value = Number(el.value);
      return Number.isFinite(value) ? value : null;
    }}
    function textValue(card, name) {{
      const el = card.querySelector('[name="' + name + '"]');
      return el ? el.value : '';
    }}
    function updatesFrom(card) {{
      const unknown = [];
      card.querySelectorAll('[data-unknown]').forEach((box) => {{
        if (box.checked) unknown.push(box.dataset.unknown);
      }});
      ['formation', 'play', 'coverage_shell', 'safety_depth', 'pressure', 'front', 'leverage', 'observation_time'].forEach((name) => {{
        if (textValue(card, name) === 'unknown') unknown.push(name);
      }});
      const pressure = textValue(card, 'pressure');
      return {{
        start_s: numberValue(card, 'start_s'),
        end_s: numberValue(card, 'end_s'),
        presnap_interval: [numberValue(card, 'presnap_start'), numberValue(card, 'presnap_end')],
        postsnap_interval: [numberValue(card, 'postsnap_start'), numberValue(card, 'postsnap_end')],
        snap_id: textValue(card, 'snap_id') || null,
        formation: textValue(card, 'formation'),
        play: textValue(card, 'play'),
        observation_time: textValue(card, 'observation_time'),
        game_state: {{
          down: numberValue(card, 'down'),
          distance: numberValue(card, 'distance'),
          quarter: numberValue(card, 'quarter'),
          clock_seconds: numberValue(card, 'clock_seconds')
        }},
        defensive_observation: {{
          coverage_shell: textValue(card, 'coverage_shell'),
          safety_depth: textValue(card, 'safety_depth'),
          pressure: pressure,
          box_count: numberValue(card, 'box_count'),
          front: textValue(card, 'front'),
          leverage: textValue(card, 'leverage')
        }},
        unknown_fields: unknown
      }};
    }}
    function seekTo(seconds, play) {{
      if (!video) return;
      video.currentTime = Number(seconds) || 0;
      if (play) video.play();
      else video.pause();
      frame.src = '/frame?t=' + encodeURIComponent(String(video.currentTime || 0));
    }}
    if (cards.length) show(0);
    document.getElementById('seek').addEventListener('click', () => {{
      const card = current();
      seekTo(textValue(card, 'start_s'), true);
    }});
    document.getElementById('loop').addEventListener('click', () => {{
      looping = !looping;
      document.getElementById('loop').textContent = looping ? 'Stop loop' : 'Loop current snap';
    }});
    document.getElementById('presnap').addEventListener('click', () => {{
      seekTo(textValue(current(), 'presnap_start'), false);
    }});
    document.getElementById('postsnap').addEventListener('click', () => {{
      seekTo(textValue(current(), 'postsnap_start'), false);
    }});
    if (video) {{
      video.addEventListener('timeupdate', () => {{
        if (!looping) return;
        const card = current();
        const end = numberValue(card, 'end_s');
        const start = numberValue(card, 'start_s') || 0;
        if (end !== null && video.currentTime >= end) video.currentTime = start;
      }});
    }}
    document.body.addEventListener('click', (event) => {{
      const button = event.target.closest('button');
      if (!button) return;
      if (button.dataset.next) {{
        show(Math.min(cards.length - 1, Number(button.dataset.next)));
        return;
      }}
      if (button.dataset.prev) {{
        show(Math.max(0, Number(button.dataset.prev)));
        return;
      }}
      const action = button.dataset.action;
      const id = button.dataset.id;
      if (!action || !id) return;
      const card = button.closest('article');
      const body = {{candidate_id: id, action: action}};
      if (action === 'confirm' || action === 'correct') body.updates = updatesFrom(card);
      fetch('/review', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(body)
      }}).then(async (response) => {{
        let result = {{}};
        try {{ result = await response.json(); }} catch (err) {{ result = {{ok: false, error: 'invalid_response'}}; }}
        if (!response.ok || !result.ok) {{
          const errors = (result.errors || []).join(', ');
          status.textContent = (result.error || 'not saved') + (errors ? ': ' + errors : '');
          return;
        }}
        status.textContent = 'saved';
      }}).catch(() => {{
        status.textContent = 'not served — use film-review --serve';
      }});
    }});
  </script>
</body>
</html>
"""


def _suggestions(
    candidates: list[Mapping[str, Any]], log_snaps: list[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if not log_snaps:
        return {}
    from cfb_coach.madden.model.film_align import align_segments

    aligned = align_segments(candidates, log_snaps)
    return {str(row.get("candidate_id")): row for row in aligned.get("associations") or []}


def _card(
    index: int,
    row: Mapping[str, Any],
    suggestion: Mapping[str, Any],
    log_snaps: list[Mapping[str, Any]],
) -> str:
    state = row.get("game_state") or {}
    defense = row.get("defensive_observation") or {}
    unknown = set(row.get("unknown_fields") or [])
    presnap = list(row.get("presnap_interval") or [])
    postsnap = list(row.get("postsnap_interval") or [])
    proposed = suggestion.get("proposed_snap_id") or row.get("proposed_snap_id")
    options = ['<option value="">unresolved</option>']
    selected = str(row.get("snap_id") or "")
    seen = set()
    for snap in log_snaps:
        snap_id = str(snap.get("snap_id") or "")
        if not snap_id or snap_id in seen:
            continue
        seen.add(snap_id)
        label = f"{snap_id} · Q{snap.get('quarter')} {snap.get('down')} and {snap.get('distance')} · {snap.get('logged_play') or 'play unknown'}"
        options.append(
            f'<option value="{escape(snap_id)}"{" selected" if snap_id == selected else ""}>{escape(label)}</option>'
        )
    if selected and selected not in seen:
        options.append(f'<option value="{escape(selected)}" selected>{escape(selected)} (not in this log)</option>')
    return f"""
            <article class="snap" id="snap-{index}">
              <h2>Candidate {escape(str(row.get('candidate_id')))}</h2>
              <p>Role {escape(str(row.get('role') or 'unknown'))}
                 · snap claim {escape(str(bool(row.get('snap_claim'))))}
                 · boundary {escape(str(row.get('boundary_status')))}</p>
              <p>Confidence {escape(str(row.get('confidence')))}
                 · provenance {escape(str(row.get('provenance')))}</p>
              <p>Suggested snap {escape(str(proposed or 'none'))}
                 · suggestion {escape(str(suggestion.get('reason') or 'none'))}. A suggestion is not a confirmed association.</p>
              <p>Review {escape(str(row.get('review_status')))}
                 · association {escape(str(row.get('association_status')))}</p>
              <p>Missing: {escape(', '.join(row.get('missing') or []) or 'none listed')}</p>
              <label>Snap start <input name="start_s" type="number" step="0.001" value="{escape(str(row.get('start_s') if row.get('start_s') is not None else ''))}"></label>
              <label>Snap end <input name="end_s" type="number" step="0.001" value="{escape(str(row.get('end_s') if row.get('end_s') is not None else ''))}"></label>
              <label>Pre-snap start <input name="presnap_start" type="number" step="0.001" value="{escape(str(presnap[0] if presnap else ''))}"></label>
              <label>Pre-snap end <input name="presnap_end" type="number" step="0.001" value="{escape(str(presnap[1] if len(presnap) > 1 else ''))}"></label>
              <label>Post-snap start <input name="postsnap_start" type="number" step="0.001" value="{escape(str(postsnap[0] if postsnap else ''))}"></label>
              <label>Post-snap end <input name="postsnap_end" type="number" step="0.001" value="{escape(str(postsnap[1] if len(postsnap) > 1 else ''))}"></label>
              <label>Snap association <select name="snap_id">{''.join(options)}</select></label>
              {_text_field('formation', row.get('formation'), unknown)}
              {_text_field('play', row.get('play'), unknown)}
              <label>Observation time <select name="observation_time">{_timing_options(row.get('observation_time'))}</select></label>
              {_number_field('down', state.get('down'), unknown)}
              {_number_field('distance', state.get('distance'), unknown)}
              {_number_field('quarter', state.get('quarter'), unknown)}
              {_number_field('clock_seconds', state.get('clock_seconds'), unknown)}
              {_choice_field('coverage_shell', defense.get('coverage_shell'), unknown, ('cover_0', 'cover_1', 'cover_2', 'cover_3', 'cover_4', 'cover_6'))}
              {_choice_field('safety_depth', defense.get('safety_depth'), unknown, ('single_high', 'two_high'))}
              {_choice_field('pressure', _pressure_choice(defense.get('pressure')), unknown, ('true', 'false'))}
              {_number_field('box_count', defense.get('box_count'), unknown)}
              {_text_field('front', defense.get('front'), unknown)}
              {_text_field('leverage', defense.get('leverage'), unknown)}
              <div class="controls">
                <button type="button" data-action="confirm" data-id="{escape(str(row.get('candidate_id')))}">Confirm</button>
                <button type="button" data-action="correct" data-id="{escape(str(row.get('candidate_id')))}">Correct</button>
                <button type="button" data-action="reject" data-id="{escape(str(row.get('candidate_id')))}">Reject</button>
                <button type="button" data-action="uncertain" data-id="{escape(str(row.get('candidate_id')))}">Leave uncertain</button>
                <button type="button" class="prev" data-prev="{index - 1}">Previous snap</button>
                <button type="button" class="next" data-next="{index + 1}">Next snap</button>
              </div>
            </article>
            """


def _timing_options(selected: Any) -> str:
    options = ['<option value="">not set</option>']
    for value in ("pre_snap", "at_snap", "post_snap", "unknown"):
        mark = " selected" if selected == value else ""
        options.append(f'<option value="{value}"{mark}>{value}</option>')
    return "".join(options)


def _choice_field(name: str, selected: Any, unknown: set[str], choices: tuple[str, ...]) -> str:
    current = "unknown" if name in unknown and _is_blank(selected) else ("" if selected is None else str(selected).lower() if isinstance(selected, bool) else str(selected))
    if isinstance(selected, bool):
        current = "true" if selected else "false"
    if name in unknown and _is_blank(selected):
        current = "unknown"
    options = ['<option value="">not set</option>', '<option value="unknown">unknown</option>']
    for value in choices:
        mark = " selected" if current == value else ""
        options.append(f'<option value="{escape(value)}"{mark}>{escape(value)}</option>')
    if current == "unknown":
        options[1] = '<option value="unknown" selected>unknown</option>'
    return f'<label>{escape(name)} <select name="{escape(name)}">{"".join(options)}</select></label>'


def _text_field(name: str, value: Any, unknown: set[str]) -> str:
    shown = "unknown" if name in unknown and _is_blank(value) else ("" if value is None else str(value))
    checked = " checked" if name in unknown else ""
    return (
        f'<label>{escape(name)} <input name="{escape(name)}" value="{escape(shown)}">'
        f'<input data-unknown="{escape(name)}" type="checkbox"{checked}> unknown</label>'
    )


def _number_field(name: str, value: Any, unknown: set[str]) -> str:
    shown = "" if value is None or name in unknown else str(value)
    checked = " checked" if name in unknown else ""
    return (
        f'<label>{escape(name)} <input name="{escape(name)}" type="number" step="any" value="{escape(shown)}">'
        f'<input data-unknown="{escape(name)}" type="checkbox"{checked}> unknown</label>'
    )


def _pressure_choice(value: Any) -> Any:
    if value is True:
        return True
    if value is False:
        return False
    return value
