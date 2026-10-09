"""Local review of candidate snaps. Annotations are a separate dataset.

Confirm, correct, reject, and leave-uncertain update that file only.
Original gameplay rows are not opened for writing.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Mapping

ANNOTATION_SCHEMA = "madden.film.annotations.v1"
REVIEW_ACTIONS = frozenset({"confirm", "correct", "reject", "uncertain"})


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
            "review_status": "unreviewed",
            "association_status": "unresolved",
            "snap_id": None,
            "formation": None,
            "play": None,
            "game_state": {},
            "defensive_observation": {},
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
) -> dict[str, Any]:
    """Record a review decision. Does not open the gameplay database."""
    if action not in REVIEW_ACTIONS:
        return {"ok": False, "error": "unknown_review_action", "history_modified": False}
    payload = load_annotations(store, game_id)
    match = next((row for row in payload.get("candidates") or [] if row.get("candidate_id") == candidate_id), None)
    if match is None:
        return {"ok": False, "error": "unknown_candidate", "history_modified": False}
    updates = dict(updates or {})
    if action == "confirm":
        match["review_status"] = "confirmed"
    elif action == "correct":
        match["review_status"] = "corrected"
        for key in (
            "start_s", "end_s", "formation_interval", "presnap_interval", "postsnap_interval",
            "snap_id", "formation", "play", "game_state", "defensive_observation",
            "association_status", "defense_review", "execution_review",
        ):
            if key in updates:
                match[key] = updates[key]
        match["boundary_status"] = "corrected"
    elif action == "reject":
        match["review_status"] = "rejected"
        match["association_status"] = "rejected"
    else:
        match["review_status"] = "uncertain"
        match["association_status"] = "unresolved"
    if action in ("confirm", "correct"):
        if updates.get("snap_id"):
            match["snap_id"] = updates["snap_id"]
            match["association_status"] = "confirmed"
        if updates.get("defense_review"):
            match["defense_review"] = updates["defense_review"]
        if updates.get("execution_review"):
            match["execution_review"] = updates["execution_review"]
        if updates.get("defensive_observation"):
            match["defensive_observation"] = updates["defensive_observation"]
        if updates.get("human_label"):
            match["human_label"] = updates["human_label"]
    match["verified_execution"] = False
    match["missing"] = _missing(match)
    payload["history_modified"] = False
    path = save_annotations(store, payload)
    return {"ok": True, "candidate": match, "path": str(path), "history_modified": False}


def _missing(row: Mapping[str, Any]) -> list[str]:
    missing = []
    if row.get("association_status") != "confirmed" or not row.get("snap_id"):
        missing.append("snap_association")
    if not row.get("formation"):
        missing.append("formation")
    if not row.get("play"):
        missing.append("play")
    defense = row.get("defensive_observation") or {}
    if not defense:
        missing.append("defensive_alignment")
    if defense and not defense.get("coverage_shell"):
        missing.append("coverage_shell")
    return missing


def serve_review(store: str | Path, game_id: str, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Local review server. Bind is localhost. Gameplay history is not opened."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            page = render_review_html(load_annotations(store, game_id)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                body = {}
            result = apply_review_action(
                store, game_id, str(body.get("candidate_id") or ""), str(body.get("action") or ""),
                body.get("updates") if isinstance(body.get("updates"), dict) else None,
            )
            encoded = json.dumps(result).encode("utf-8")
            self.send_response(200 if result.get("ok") else 400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(json.dumps({"serving": f"http://{host}:{port}", "game_id": game_id, "history_modified": False}))
    server.serve_forever()


def render_review_html(payload: Mapping[str, Any]) -> str:
    """One page for several snaps. Buttons are the review controls."""
    cards = []
    candidates = list(payload.get("candidates") or [])
    for index, row in enumerate(candidates):
        cards.append(
            f"""
            <article class="snap" id="snap-{index}">
              <h2>Candidate {escape(str(row.get('candidate_id')))}</h2>
              <p>{escape(str(row.get('start_s')))}s – {escape(str(row.get('end_s')))}s
                 · boundary {escape(str(row.get('boundary_status')))}
                 · confidence {escape(str(row.get('confidence')))}</p>
              <p>Association: {escape(str(row.get('association_status')))}
                 · snap {escape(str(row.get('snap_id') or 'unresolved'))}</p>
              <p>Formation {escape(str(row.get('formation') or 'unknown'))}
                 · play {escape(str(row.get('play') or 'unknown'))}</p>
              <p>Game state: {escape(json.dumps(row.get('game_state') or {}))}</p>
              <p>Defense: {escape(json.dumps(row.get('defensive_observation') or {}))}</p>
              <p>Missing: {escape(', '.join(row.get('missing') or []) or 'none listed')}</p>
              <p>Review: {escape(str(row.get('review_status')))}
                 · provenance {escape(str(row.get('provenance')))}</p>
              <div class="controls">
                <button type="button" data-action="confirm" data-id="{escape(str(row.get('candidate_id')))}">Confirm</button>
                <button type="button" data-action="correct" data-id="{escape(str(row.get('candidate_id')))}">Correct</button>
                <button type="button" data-action="reject" data-id="{escape(str(row.get('candidate_id')))}">Reject</button>
                <button type="button" data-action="uncertain" data-id="{escape(str(row.get('candidate_id')))}">Leave uncertain</button>
                <button type="button" class="next" data-next="{index + 1}">Next snap</button>
              </div>
            </article>
            """
        )
    body = "\n".join(cards) or "<p>No candidate snaps in this annotation file.</p>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Film review {escape(str(payload.get('game_id') or ''))}</title>
  <style>
    body {{ font-family: sans-serif; margin: 1.5rem; background: #111; color: #eee; }}
    article {{ border: 1px solid #444; padding: 1rem; margin-bottom: 1rem; }}
    button {{ margin-right: 0.4rem; }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <h1>Film review</h1>
  <p>Game {escape(str(payload.get('game_id') or ''))}. Recording {escape(str(payload.get('recording_id') or ''))}.</p>
  <p>These controls update the annotation dataset when this page is served locally. They do not rewrite gameplay history.</p>
  {body}
  <script>
    const cards = Array.from(document.querySelectorAll('article.snap'));
    function show(index) {{
      cards.forEach((card, i) => card.classList.toggle('hidden', i !== index));
    }}
    if (cards.length) show(0);
    document.body.addEventListener('click', (event) => {{
      const button = event.target.closest('button');
      if (!button) return;
      if (button.dataset.next) {{
        show(Math.min(cards.length - 1, Number(button.dataset.next)));
        return;
      }}
      const action = button.dataset.action;
      const id = button.dataset.id;
      if (!action || !id) return;
      fetch('/review', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{candidate_id: id, action}})
      }}).then(() => {{
        button.insertAdjacentText('afterend', ' saved');
      }}).catch(() => {{
        button.insertAdjacentText('afterend', ' not served — use film-review --serve');
      }});
    }});
  </script>
</body>
</html>
"""
