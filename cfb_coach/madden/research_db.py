"""Madden 27 research DB (v1.18) — what the daily research routine found, pulled by every prep.

A scheduled Claude Code routine researches the current Madden 27 meta each day (web + YouTube
transcripts): defense macros with settings, offense / defense pre-snap adjustments, the Xbox
buttons for each, and playbook notes. It writes ``data/madden27/research_db.json`` on the
``madden-research-db`` branch. Every prep pulls the newest copy (``git fetch`` that branch),
falls back to the last pulled copy in the data dir, then to the packaged seed in this repo.

Defense macro settings are research-built: every field of the Custom Adjustments editor
(``editor_fields.json`` = the field list from Aidan's CFB sheets, same editor) gets the value a
cited source names, else ``Default``. Every row carries its source id.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

DB_BRANCH = "madden-research-db"
DB_REL_PATH = "cfb_coach/data/madden27/research_db.json"
CACHE_FILENAME = "madden27_research_db.json"
STALE_HOURS = 48
FAMILIES_D = ("vert", "flood", "cross", "stack", "scram", "run", "rpo", "screen", "pressure", "red_zone", "prevent")
LOOKS_O = ("man", "pressure", "two_high", "cover2", "single_high")
CONTROL_CONFIDENCE = ("confirmed", "single-source", "conflict")

_REPO = Path(__file__).resolve().parents[2]
_STATE: dict[str, Any] = {}


def _packaged_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "madden27" / "research_db.json"


def _cache_path() -> Path:
    from cfb_coach.games import data_dir

    return data_dir() / CACHE_FILENAME


@lru_cache(maxsize=1)
def editor_fields() -> dict[str, Any]:
    p = Path(__file__).resolve().parent.parent / "data" / "madden27" / "editor_fields.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _git_pull_db(timeout: float = 12.0) -> dict[str, Any] | None:
    """The newest DB from origin/madden-research-db, or None (not a git checkout, offline…)."""
    if not (_REPO / ".git").exists():
        return None
    try:
        subprocess.run(["git", "-C", str(_REPO), "fetch", "-q", "origin", DB_BRANCH],
                       check=True, capture_output=True, timeout=timeout)
        out = subprocess.run(["git", "-C", str(_REPO), "show", f"FETCH_HEAD:{DB_REL_PATH}"],
                             check=True, capture_output=True, timeout=timeout)
        return json.loads(out.stdout.decode("utf-8"))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load(*, pull: bool = False) -> dict[str, Any]:
    """The research DB. ``pull=True`` (prep) fetches the routine's newest copy first."""
    if pull or "db" not in _STATE:
        db, origin = None, ""
        if pull:
            db = _git_pull_db()
            if db is not None and not validate(db):
                try:
                    _cache_path().parent.mkdir(parents=True, exist_ok=True)
                    _cache_path().write_text(json.dumps(db, indent=1, ensure_ascii=False), encoding="utf-8")
                except OSError:
                    pass
                origin = f"pulled origin/{DB_BRANCH}"
            else:
                db = None
        if db is None:
            cached = _read(_cache_path())
            if cached is not None and not validate(cached):
                db, origin = cached, "last pulled copy"
        if db is None:
            db, origin = _read(_packaged_path()) or {"defense_macros": []}, "packaged seed (repo)"
        _STATE.update(db=db, origin=origin)
    return _STATE["db"]


def status() -> dict[str, Any]:
    """One line for the prep page: where the DB came from, when it was researched, how stale."""
    db = load()
    upd = str(db.get("updated") or "")
    age = None
    try:
        dt = datetime.fromisoformat(upd.replace("Z", "+00:00"))
        age = round((datetime.now(timezone.utc) - dt).total_seconds() / 3600.0, 1)
    except ValueError:
        pass
    stale = age is None or age > STALE_HOURS
    line = (f"Research DB: {len(db.get('defense_macros') or [])} defense macros, "
            f"{len(db.get('sources') or [])} sources, researched {upd[:16].replace('T', ' ') or '?'} UTC"
            + (f" ({age:.0f}h ago)" if age is not None else "") + f" — {_STATE.get('origin', '')}")
    if stale:
        line += f" — STALE (> {STALE_HOURS}h): check the daily research routine"
    return {"line": line, "age_hours": age, "stale": stale, "origin": _STATE.get("origin", ""),
            "youtube": (db.get("youtube") or {}).get("status", "")}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def sources() -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in load().get("sources") or [] if s.get("id")}


def defense_macros() -> list[dict[str, Any]]:
    return [m for m in load().get("defense_macros") or [] if m.get("id")]


def defense_macro(mid: str | None) -> dict[str, Any] | None:
    key = (mid or "").strip().upper()
    return next((dict(m) for m in defense_macros() if m["id"].upper() == key), None)


def control(side: str, key: str) -> dict[str, Any]:
    return dict(((load().get("controls_xbox") or {}).get(side) or {}).get(key) or {})


def buttons(side: str, key: str) -> str:
    c = control(side, key)
    b = c.get("buttons") or "VERIFY on screen (no source for this button yet)"
    if c.get("confidence") == "conflict":
        return b
    return b + ("" if c.get("confidence") == "confirmed" else "  (single source — verify once)")


def offense_adjustments() -> list[dict[str, Any]]:
    return list(load().get("offense_adjustments") or [])


def defense_adjustments() -> list[dict[str, Any]]:
    return list(load().get("defense_adjustments") or [])


def _canon(section: str, setting: str) -> tuple[str, str]:
    alias = (editor_fields().get("aliases") or {})
    return section, alias.get(setting, setting)


def full_settings(mid: str) -> list[dict[str, Any]]:
    """Every editor field for this macro: the researched value (with its source) where a source
    names one, else Default. Researched fields the editor list doesn't have are kept at the end
    of their section (marked ``new_field``)."""
    m = defense_macro(mid) or {}
    researched: dict[tuple[str, str], dict[str, Any]] = {}
    for r in m.get("settings") or []:
        sec, name = _canon(str(r.get("section") or ""), str(r.get("setting") or ""))
        researched[(sec.lower(), name.lower())] = dict(r, section=sec, setting=name)
    out: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    fields = editor_fields().get("defense") or {}
    for sec, names in fields.items():
        for name in names:
            k = (sec.lower(), name.lower())
            if k in researched:
                out.append(researched[k])
                used.add(k)
            else:
                out.append({"section": sec, "setting": name, "value": "Default", "source": "default"})
        for k, r in researched.items():
            if k[0] == sec.lower() and k not in used:
                out.append(dict(r, new_field=True))
                used.add(k)
    for k, r in researched.items():  # a section the editor list doesn't have
        if k not in used:
            out.append(dict(r, new_field=True))
    return out


def researched_rows(mid: str) -> list[dict[str, Any]]:
    return [r for r in full_settings(mid) if r.get("source") != "default"]


# ---------------------------------------------------------------------------
# Validation (the routine runs this before every push; prep refuses a bad pull)
# ---------------------------------------------------------------------------

def validate(db: dict[str, Any]) -> list[str]:
    """Problems with a DB, empty when it is usable."""
    errs: list[str] = []
    if db.get("schema") != 1:
        errs.append("schema must be 1")
    try:
        datetime.fromisoformat(str(db.get("updated") or "").replace("Z", "+00:00"))
    except ValueError:
        errs.append("updated must be an ISO timestamp")
    src_ids = {s.get("id") for s in db.get("sources") or []}
    for s in db.get("sources") or []:
        if not s.get("id") or not s.get("url") or not s.get("title"):
            errs.append(f"source missing id/url/title: {s}")
    macros = db.get("defense_macros") or []
    if len(macros) < 10:
        errs.append(f"need at least 10 defense_macros (have {len(macros)})")
    ids = [str(m.get("id") or "").upper() for m in macros]
    if len(set(ids)) != len(ids):
        errs.append("defense_macros ids must be unique")
    for m in macros:
        mid = m.get("id") or "?"
        if not m.get("answers") or any(a not in FAMILIES_D for a in m["answers"]):
            errs.append(f"{mid}: answers must be from {FAMILIES_D}")
        if not m.get("settings"):
            errs.append(f"{mid}: no researched settings")
        for r in m.get("settings") or []:
            if not r.get("section") or not r.get("setting") or str(r.get("value") or "").strip() == "":
                errs.append(f"{mid}: setting row needs section/setting/value: {r}")
            if r.get("source") not in src_ids:
                errs.append(f"{mid}: setting {r.get('setting')!r} cites unknown source {r.get('source')!r}")
        for sid in m.get("sources") or []:
            if sid not in src_ids:
                errs.append(f"{mid}: unknown source {sid!r}")
    for side in ("offense", "defense"):
        for key, c in ((db.get("controls_xbox") or {}).get(side) or {}).items():
            if not c.get("buttons") or c.get("confidence") not in CONTROL_CONFIDENCE:
                errs.append(f"controls {side}.{key}: needs buttons + confidence in {CONTROL_CONFIDENCE}")
            if any(sid not in src_ids for sid in c.get("sources") or []) or not c.get("sources"):
                errs.append(f"controls {side}.{key}: every button needs a known source")
    for a in db.get("offense_adjustments") or []:
        if a.get("type") not in ("hot_route", "audible", "pass_protection", "motion", "fake_snap", "flip_run"):
            errs.append(f"offense adjustment {a.get('id')}: bad type")
        if any(v not in LOOKS_O for v in a.get("vs") or []) or not a.get("vs"):
            errs.append(f"offense adjustment {a.get('id')}: vs must be from {LOOKS_O}")
        if any(sid not in src_ids for sid in a.get("sources") or []) or not a.get("sources"):
            errs.append(f"offense adjustment {a.get('id')}: needs known sources")
    for a in db.get("defense_adjustments") or []:
        ctl = (db.get("controls_xbox") or {}).get("defense") or {}
        if a.get("control") not in ctl:
            errs.append(f"defense adjustment {a.get('id')}: control {a.get('control')!r} not in controls_xbox.defense")
        if any(v not in FAMILIES_D for v in a.get("answers") or []) or not a.get("answers"):
            errs.append(f"defense adjustment {a.get('id')}: answers must be from {FAMILIES_D}")
    return errs


def reset() -> None:
    """Tests: forget the loaded DB."""
    _STATE.clear()


__all__ = [
    "DB_BRANCH", "buttons", "control", "defense_adjustments", "defense_macro", "defense_macros", "editor_fields",
    "full_settings", "load", "offense_adjustments", "researched_rows", "reset", "sources", "status", "validate",
]
