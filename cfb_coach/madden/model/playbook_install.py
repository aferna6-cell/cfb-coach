"""Generate -> Review -> Install in Madden -> Confirm -> Ready to Play, per side.

Proposals are JSON files under ``<data dir>/game_prep/<opponent>/``. Generating
or reviewing one never touches the database, so the live callable book cannot
change until Aidan confirms a side. Offense and defense have separate
proposal ids, fingerprints, confirmation and history; one side can be ready
while the other still waits.

A confirmation is refused when:
  - the proposal file was edited (its id no longer matches its contents),
  - a newer proposal for that side exists (superseded),
  - the applied book changed since generation (stale),
  - the typed fingerprint does not match,
  - any formation no longer matches its source book's complete play list,
  - the book breaks the editor limits, or
  - the attestation is missing or too short.
Nothing is written in any of those cases. A successful confirmation snapshots
the database first, then writes the new book and its history in one
transaction. Nothing here claims Madden itself changed; the attestation is
Aidan's statement that he built the book.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cfb_coach.madden import catalog_full, playbook
from cfb_coach.madden.model.playbook_architect import BOOK_NAMES, MAX_FORMATIONS, MAX_PLAYS, digest, fingerprint

SIDES = ("offense", "defense")
HISTORY_KEY = "ml_architect_history.v1"
ATTEST_MIN = 30
OPERATOR = "Aidan"


class InstallError(ValueError):
    """A confirmation or rollback was refused; nothing was written."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(name: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", (name or "unknown").lower()).strip("_") or "unknown"


def prep_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    from cfb_coach.games import data_dir

    return data_dir() / "game_prep"


def proposal_dir(root: Path | str | None, opponent_id: str) -> Path:
    return prep_root(root) / _safe(opponent_id)


def applied_digest(record: Mapping[str, Any] | None) -> str:
    return digest(dict(record or {}))


def proposal_identity(p: Mapping[str, Any]) -> str:
    body = {k: p.get(k) for k in ("side", "opponent_id", "book", "fingerprint", "expected_applied_hash",
                                  "expected_applied_rev", "generated_ts")}
    return digest(body)[:16]


def build_proposal(design: Mapping[str, Any], *, opponent_id: str, applied: Mapping[str, Any] | None,
                   generated_ts: str | None = None) -> dict[str, Any]:
    p = {
        "schema": "madden27.architect_proposal.v1",
        "side": design["side"],
        "opponent_id": opponent_id,
        "generated_ts": generated_ts or _now(),
        "book": design["book"],
        "fingerprint": design["fingerprint"],
        "expected_applied_hash": applied_digest(applied),
        "expected_applied_rev": (applied or {}).get("rev"),
        "design": {k: v for k, v in design.items() if k not in ("book",)},
    }
    p["proposal_id"] = proposal_identity(p)
    return p


def save_proposals(root: Path | str | None, opponent_id: str, proposals: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Write each side's proposal and point ``latest.json`` at it. Other sides keep their pointer."""
    folder = proposal_dir(root, opponent_id)
    folder.mkdir(parents=True, exist_ok=True)
    latest_path = folder / "latest.json"
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        latest = {}
    paths = {}
    for side, p in proposals.items():
        path = folder / f"{side}-{p['proposal_id']}.json"
        path.write_text(json.dumps(p, indent=1, sort_keys=True, default=str), encoding="utf-8")
        latest[side] = p["proposal_id"]
        paths[side] = str(path)
    latest_path.write_text(json.dumps(latest, indent=1, sort_keys=True), encoding="utf-8")
    return paths


def latest_ids(root: Path | str | None, opponent_id: str) -> dict[str, str]:
    try:
        return dict(json.loads((proposal_dir(root, opponent_id) / "latest.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


def load_proposal(root: Path | str | None, opponent_id: str, side: str, proposal_id: str) -> dict[str, Any]:
    path = proposal_dir(root, opponent_id) / f"{side}-{_safe(proposal_id)}.json"
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise InstallError(f"No {side} proposal {proposal_id} for {opponent_id}") from exc


def _history(db: Any) -> dict[str, list[dict[str, Any]]]:
    try:
        raw = json.loads(db.get_meta(HISTORY_KEY) or "{}")
    except (TypeError, ValueError):
        raw = {}
    return {side: list(raw.get(side) or []) for side in SIDES}


def history(db: Any, side: str) -> list[dict[str, Any]]:
    return _history(db)[side]


def _write_atomically(db: Any, items: Mapping[str, str]) -> None:
    conn = db.conn
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        for key, value in items.items():
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _snapshot(db: Any, root: Path | str | None, side: str, action: str) -> str | None:
    if not hasattr(db, "backup_to"):
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = prep_root(root) / "snapshots" / f"madden27-before-{action}-{side}-{stamp}.db"
    db.backup_to(dest)
    return str(dest)


def validate_book(side: str, book: Mapping[str, Any]) -> list[str]:
    """Every formation is one source book's complete play list, inside the editor limits."""
    problems = []
    forms = dict(book.get("formations") or {})
    sources = dict(book.get("formation_sources") or {})
    if not forms:
        problems.append("the book has no formations")
    if len(forms) > MAX_FORMATIONS:
        problems.append(f"{len(forms)} formations exceeds the {MAX_FORMATIONS} cap")
    plays = sum(len(p) for p in forms.values())
    if plays > MAX_PLAYS:
        problems.append(f"{plays} plays exceeds the {MAX_PLAYS}-play editor limit")
    families = {str((catalog_full.architect_books(side).get(sources.get(f, "")) or {}).get("formations", {})
                    .get(f, {}).get("family") or f.split()[0]) for f in forms}
    if len(families) > int(catalog_full.EDITOR_LIMITS["max_formations"]):
        problems.append(f"{len(families)} formation families exceeds the editor limit")
    for form, ps in forms.items():
        src = sources.get(form)
        expected = catalog_full.source_plays(side, src or "", form)
        if expected is None:
            problems.append(f"{form} is not in source book {src!r}")
        elif list(ps) != list(dict.fromkeys(expected)):
            problems.append(f"{form} does not match the complete {src} play list")
    return problems


def confirm(db: Any, *, root: Path | str | None, opponent_id: str, side: str, proposal_id: str,
            typed_fingerprint: str, attestation: str, operator: str = OPERATOR) -> dict[str, Any]:
    if side not in SIDES:
        raise InstallError(f"side must be offense or defense, not {side!r}")
    if operator != OPERATOR:
        raise InstallError("Only Aidan can confirm a Madden playbook installation")
    p = load_proposal(root, opponent_id, side, proposal_id)
    if p.get("side") != side or proposal_identity(p) != proposal_id:
        raise InstallError("Proposal contents changed since generation; refusing")
    latest = latest_ids(root, opponent_id).get(side)
    if latest and latest != proposal_id:
        raise InstallError(f"Proposal {proposal_id} was superseded by {latest}; review the newer page")
    book = dict(p["book"])
    real_fp = fingerprint(side, book["formations"], book["formation_sources"])
    if real_fp != p.get("fingerprint") or (typed_fingerprint or "").strip().upper() != real_fp:
        raise InstallError("Fingerprint does not match this proposal; type the code shown on the setup page")
    if len((attestation or "").strip()) < ATTEST_MIN:
        raise InstallError(f"Attestation required (at least {ATTEST_MIN} characters) stating you built this book in Madden")
    state = playbook._load_state(db)
    current = state["applied"].get(side)
    if applied_digest(current) != p.get("expected_applied_hash"):
        raise InstallError("The applied book changed since this proposal was generated; run game-prep again")
    problems = validate_book(side, book)
    if problems:
        raise InstallError("Book failed validation: " + "; ".join(problems))
    snapshot = _snapshot(db, root, side, "confirm")
    ts = _now()
    new = {
        "side": side, "mode": playbook.CUSTOM, "name": BOOK_NAMES[side],
        "formations": {f: list(ps) for f, ps in book["formations"].items()},
        "formation_sources": dict(book["formation_sources"]),
        "primary_source_book": book.get("source_book"),
        "audibles": {}, "core": list(book["formations"]),
        "rev": int((current or {}).get("rev") or 0) + 1,
        "locked_ts": ts,
        "reason": f"Pregame architect proposal {proposal_id} confirmed installed by {operator}",
        "architect_proposal_id": proposal_id,
        "inventory_fingerprint": real_fp,
        "opponent_id": opponent_id,
    }
    hist = _history(db)
    superseded_pending = state["pending"].pop(side, None)
    hist[side].append({
        "action": "confirm", "proposal_id": proposal_id, "fingerprint": real_fp, "ts": ts,
        "operator": operator, "attestation": attestation.strip(), "old_book": current, "new_book": new,
        "superseded_prep_pending": bool(superseded_pending), "snapshot": snapshot,
    })
    hist[side] = hist[side][-30:]
    state["applied"][side] = new
    _write_atomically(db, {playbook.META_KEY: json.dumps(state), HISTORY_KEY: json.dumps(hist, sort_keys=True)})
    _revoke_generated_macros(db, side)
    return {
        "confirmed": True, "side": side, "proposal_id": proposal_id, "fingerprint": real_fp,
        "revision": new["rev"], "formations": len(new["formations"]),
        "plays": sum(len(ps) for ps in new["formations"].values()), "snapshot": snapshot,
        "note": "The coach's callable book now matches your attestation. The coach cannot see Madden itself.",
    }


def _revoke_generated_macros(db: Any, side: str) -> None:
    if side == "offense":
        from cfb_coach.madden.model.offense_designer import _revoke_generated_macro_approvals

        _revoke_generated_macro_approvals(db)
        return
    try:
        keys = db.conn.execute("SELECT key FROM meta WHERE key LIKE 'ml_defense_approved_macros.v1:%'").fetchall()
    except Exception:  # noqa: BLE001
        return
    for item in keys:
        db.set_meta(str(item[0]), "")


def rollback(db: Any, *, root: Path | str | None = None, side: str, attestation: str,
             operator: str = OPERATOR) -> dict[str, Any]:
    """Restore the book that was applied before the latest confirmation of the current book."""
    if side not in SIDES:
        raise InstallError(f"side must be offense or defense, not {side!r}")
    if operator != OPERATOR:
        raise InstallError("Only Aidan can restore a Madden playbook")
    if len((attestation or "").strip()) < ATTEST_MIN:
        raise InstallError(f"Attestation required (at least {ATTEST_MIN} characters) stating you restored the book in Madden")
    state = playbook._load_state(db)
    current = state["applied"].get(side)
    hist = _history(db)
    match = next((h for h in reversed(hist[side])
                  if h.get("action") == "confirm" and applied_digest(h.get("new_book")) == applied_digest(current)), None)
    if match is None:
        raise InstallError(f"The applied {side} book did not come from a confirmed architect proposal; nothing to roll back")
    previous = match.get("old_book")
    snapshot = _snapshot(db, root, side, "rollback")
    ts = _now()
    if previous:
        restored = dict(previous)
        restored["rev"] = int((current or {}).get("rev") or 0) + 1
        restored["locked_ts"] = ts
        restored["reason"] = f"Restored the book in use before proposal {match['proposal_id']}"
        state["applied"][side] = restored
    else:
        restored = None
        state["applied"].pop(side, None)
    hist[side].append({"action": "rollback", "rollback_of": match["proposal_id"], "ts": ts, "operator": operator,
                       "attestation": attestation.strip(), "old_book": current, "new_book": restored,
                       "snapshot": snapshot})
    hist[side] = hist[side][-30:]
    _write_atomically(db, {playbook.META_KEY: json.dumps(state), HISTORY_KEY: json.dumps(hist, sort_keys=True)})
    _revoke_generated_macros(db, side)
    return {"rolled_back": True, "side": side, "rollback_of": match["proposal_id"],
            "restored": (restored or {}).get("name"), "revision": (restored or {}).get("rev"),
            "snapshot": snapshot}


def side_status(db: Any, root: Path | str | None, opponent_id: str, side: str) -> dict[str, Any]:
    applied = playbook.load_books(db).get(side) if db is not None else None
    pid = latest_ids(root, opponent_id).get(side)
    base = {"side": side, "applied_name": (applied or {}).get("name"), "applied_rev": (applied or {}).get("rev"),
            "proposal_id": pid}
    if not pid:
        return {**base, "state": "no_proposal", "label": "No proposal yet - run game-prep"}
    try:
        p = load_proposal(root, opponent_id, side, pid)
    except InstallError:
        return {**base, "state": "no_proposal", "label": "Proposal file missing - run game-prep"}
    base["fingerprint"] = p.get("fingerprint")
    if (applied or {}).get("architect_proposal_id") == pid:
        return {**base, "state": "ready", "label": "READY - installed and confirmed"}
    book = p.get("book") or {}
    if (applied and applied.get("architect_proposal_id")
            and applied.get("formations") == book.get("formations")
            and applied.get("formation_sources") == book.get("formation_sources")):
        return {**base, "state": "ready", "label": "READY - no change needed; the confirmed book is still the plan"}
    if applied_digest(applied) != p.get("expected_applied_hash"):
        return {**base, "state": "stale", "label": "STALE - the applied book changed; run game-prep again"}
    return {**base, "state": "awaiting_install", "label": "WAITING - install in Madden, then confirm"}


__all__ = [
    "ATTEST_MIN", "HISTORY_KEY", "InstallError", "build_proposal", "confirm", "history", "latest_ids",
    "load_proposal", "proposal_dir", "proposal_identity", "rollback", "save_proposals", "side_status",
    "validate_book",
]
