"""Daily AI research — prep reads it instead of scraping the web itself.

A scheduled Cursor agent researches the CFB 27 and Madden 27 meta once a day (patch
notes, guides, YouTube, forums), writes ``research/cfb27.json`` and
``research/madden27.json`` in this repo and pushes them. Prep loads the freshest copy
(GitHub raw → last downloaded copy → the file in this checkout) and turns it into the
same ``MetaScoutResult`` the old live scraper produced, so the playbook pick, meta
priors and prep page all keep working. ``prep --live-scout`` still runs the scraper.

Research file (schema 1)::

    {
      "schema": 1,
      "game": "cfb27" | "madden27",
      "researched_at": "2026-10-01T08:00:00-04:00",
      "researched_by": "Cursor daily research",
      "summary": "2-4 sentences on the current meta",
      "patch": {"version": "1.012", "date": "2026-09-22", "url": "...", "notes": ["..."]},
      "findings": [{"claim": "...", "side": "offense|defense|general", "source": "...",
                    "url": "...", "published": "YYYY-MM-DD", "confidence": "high|medium|low"}],
      "meta_offense": ["..."], "meta_defense": ["..."],
      "suggestions": [{"side": "offense|defense", "label": "...", "tip": "...",
                       "macro_hint": "MACRO-ID or null", "source_url": "..."}],
      "opponents": {
        "<opponent id>": {
          "notes": ["..."],
          "defense_counters": [{"vs": "Cross Wheels", "vs_family": "vert|flood|cross|stack|run|rpo|scram",
                                "families": ["two_high|cover2|single_high|man|pressure"],
                                "calls": ["Cover 4 Quarters"], "why": "...", "source_url": "..."}],
          "offense_counters": [{"vs": "Cover 6", "vs_class": "two_high|cover2|single_high|man|pressure",
                                "plays": ["Mesh Spot"], "why": "...", "source_url": "..."}]
        }
      },
      "sources": [{"title": "...", "url": "...", "kind": "patch|guide|video|forum|news",
                   "published": "YYYY-MM-DD", "accessed": "YYYY-MM-DD"}]
    }
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

GAMES = ("cfb27", "madden27")
SCHEMA = 1
REMOTE_BASE = "https://raw.githubusercontent.com/aferna6-cell/cfb-coach/main/research"
REMOTE_TIMEOUT = 4.0
STALE_HOURS = 36.0
CONCEPT_FAMILIES = {"vert", "flood", "cross", "stack", "run", "rpo", "scram"}
COVERAGE_FAMILIES = {"two_high", "cover2", "single_high", "man", "pressure"}
REPO_DIR = Path(__file__).resolve().parent.parent / "research"

Fetch = Callable[[str, float], str]


def remote_url(game: str) -> str:
    base = os.environ.get("CFB_COACH_RESEARCH_URL", REMOTE_BASE).rstrip("/")
    return f"{base}/{game}.json"


def repo_path(game: str) -> Path:
    return REPO_DIR / f"{game}.json"


def cache_path(game: str) -> Path:
    from cfb_coach.games import data_dir

    return data_dir() / f"ai_research_{game}.json"


def validate(doc: Any, game: str | None = None) -> list[str]:
    """Schema problems (empty list = valid). Used by prep and by scripts/validate_ai_research.py."""
    errs: list[str] = []
    if not isinstance(doc, dict):
        return ["not a JSON object"]
    if doc.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA}")
    if doc.get("game") not in GAMES or (game and doc.get("game") != game):
        errs.append(f"game must be {game or ' or '.join(GAMES)}")
    if _parse_ts(doc.get("researched_at")) is None:
        errs.append("researched_at must be an ISO timestamp with timezone")
    findings = doc.get("findings")
    if not isinstance(findings, list) or not findings:
        errs.append("findings must be a non-empty list")
    else:
        for i, f in enumerate(findings):
            if not isinstance(f, dict) or not f.get("claim") or not str(f.get("url") or "").startswith("http"):
                errs.append(f"findings[{i}] needs a claim and an http(s) url")
    for key in ("meta_offense", "meta_defense"):
        if not isinstance(doc.get(key, []), list):
            errs.append(f"{key} must be a list")
    for i, s in enumerate(doc.get("suggestions") or []):
        if not isinstance(s, dict) or s.get("side") not in ("offense", "defense") or not s.get("tip"):
            errs.append(f"suggestions[{i}] needs side offense|defense and a tip")
    opps = doc.get("opponents") or {}
    if not isinstance(opps, dict):
        errs.append("opponents must be an object keyed by opponent id")
        opps = {}
    for oid, o in opps.items():
        for i, c in enumerate((o or {}).get("defense_counters") or []):
            if c.get("vs_family") not in CONCEPT_FAMILIES:
                errs.append(f"opponents.{oid}.defense_counters[{i}].vs_family must be one of {sorted(CONCEPT_FAMILIES)}")
            bad = [x for x in c.get("families") or [] if x not in COVERAGE_FAMILIES]
            if bad:
                errs.append(f"opponents.{oid}.defense_counters[{i}].families has unknown {bad}")
        for i, c in enumerate((o or {}).get("offense_counters") or []):
            if c.get("vs_class") not in COVERAGE_FAMILIES:
                errs.append(f"opponents.{oid}.offense_counters[{i}].vs_class must be one of {sorted(COVERAGE_FAMILIES)}")
    srcs = doc.get("sources")
    if not isinstance(srcs, list) or not srcs:
        errs.append("sources must be a non-empty list")
    return errs


def _parse_ts(ts: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else None


def age_hours(doc: dict[str, Any], now: datetime | None = None) -> float | None:
    dt = _parse_ts(doc.get("researched_at"))
    if dt is None:
        return None
    now = now or datetime.now(timezone.utc)
    return round((now - dt).total_seconds() / 3600.0, 1)


def _http_get(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "cfb-coach prep", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed https URL
        return resp.read().decode("utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_research(
    game: str,
    *,
    offline: bool = False,
    fetch: Fetch | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Freshest valid research doc for ``game`` → (doc, origin) with origin github|cache|repo|''."""
    candidates: list[tuple[dict[str, Any], str]] = []
    if not offline:
        try:
            doc = json.loads((fetch or _http_get)(remote_url(game), REMOTE_TIMEOUT))
            if not validate(doc, game):
                candidates.append((doc, "github"))
                try:
                    cache_path(game).parent.mkdir(parents=True, exist_ok=True)
                    cache_path(game).write_text(json.dumps(doc), encoding="utf-8")
                except OSError:
                    pass
        except Exception:  # noqa: BLE001 — offline / GitHub down / bad JSON: fall back
            pass
    for path, origin in ((cache_path(game), "cache"), (repo_path(game), "repo")):
        doc = _read_json(path)
        if doc is not None and not validate(doc, game):
            candidates.append((doc, origin))
    if not candidates:
        return None, ""
    return max(candidates, key=lambda c: _parse_ts(c[0].get("researched_at")) or datetime.min.replace(tzinfo=timezone.utc))


def _docs(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Research text as scanner documents (named formation/play extraction)."""
    out = []
    for f in doc.get("findings") or []:
        out.append({"label": f.get("source") or "", "kind": "ai", "url": f.get("url") or "",
                    "date": str(f.get("published") or "")[:10], "text": f.get("claim") or ""})
    for s in doc.get("suggestions") or []:
        out.append({"label": s.get("label") or "", "kind": "ai", "url": s.get("source_url") or "",
                    "date": str(doc.get("researched_at") or "")[:10], "text": f"{s.get('label', '')}: {s.get('tip', '')} ({s.get('side')})"})
    for side in ("offense", "defense"):
        for line in doc.get(f"meta_{side}") or []:
            out.append({"label": f"meta {side}", "kind": "ai", "url": "", "date": str(doc.get("researched_at") or "")[:10],
                        "text": f"{line} ({side})"})
    return out


def to_scout_result(doc: dict[str, Any], game: str, *, origin: str = "", now: datetime | None = None) -> Any:
    """The research doc as the ``MetaScoutResult`` prep and the live callers already consume."""
    from cfb_coach.meta_scout import MetaScoutResult, extract_signals, local_ts

    now = now or datetime.now(timezone.utc)
    age = age_hours(doc, now)
    stale = age is None or age > STALE_HOURS
    patch = doc.get("patch") or {}
    r = MetaScoutResult(
        available=True,
        offline=False,
        from_cache=origin != "github",
        confidence="medium" if stale else "high",
        meta_offense=list(doc.get("meta_offense") or []),
        meta_defense=list(doc.get("meta_defense") or []),
        suggestions=[{**s, "badge": "AI research (daily)"} for s in doc.get("suggestions") or []],
        fetched_at=str(doc.get("researched_at") or ""),
    )
    if game == "madden27":
        from cfb_coach.madden.meta_scout import BASELINE_VERSION, named_signals

        r.baseline_fallback = BASELINE_VERSION
        r.named_signals = named_signals(_docs(doc), now=now)
        if patch:
            # Same dict shape as CFB — prep_browser expects version/date/title/bullets.
            r.patch_notes = [
                {
                    "version": patch.get("version") or "?",
                    "date": patch.get("date") or "",
                    "title": patch.get("title") or "Title update",
                    "bullets": list(patch.get("notes") or []),
                    "url": patch.get("url") or "",
                }
            ]
    else:
        from cfb_coach.meta_entities import aggregate

        r.named_signals = aggregate(_docs(doc), now=now)
        if patch:
            r.patch_notes = [{"version": patch.get("version") or "?", "date": patch.get("date") or "",
                              "title": patch.get("title") or "Title update", "bullets": list(patch.get("notes") or []),
                              "url": patch.get("url") or ""}]
    r.concept_signals, r.rz_signals = extract_signals([d["text"] for d in _docs(doc)])
    r.headlines = [{"title": f.get("claim", "")[:160], "link": f.get("url", ""), "date": f.get("published", ""),
                    "source": f.get("source", "")} for f in doc.get("findings") or []][:30]
    r.sources = [{"url": s.get("url", ""), "title": s.get("title", ""), "fetched": True, "status": 200, "error": "",
                  "label": f"{s.get('title', '')} ({s.get('published') or s.get('accessed') or ''})", "kind": s.get("kind", "")}
                 for s in doc.get("sources") or []]
    r.mode = "ai"
    r.research_status = "ai-stale" if stale else "ai"
    r.fallback_age_hours = age
    r.ttl_policy = "daily AI research (scheduled Cursor agent); prep no longer scrapes the web"
    where = {"github": "GitHub", "cache": "last downloaded copy", "repo": "this checkout"}.get(origin, origin)
    r.message = (
        f"AI research from {local_ts(r.fetched_at)}" + (f" ({age:.0f}h old)" if age is not None else "")
        + f" · {len(doc.get('findings') or [])} findings, {len(doc.get('sources') or [])} sources · via {where}"
        + (" · STALE: the daily research hasn't run — check the Cursor automation" if stale else "")
    )
    if doc.get("summary"):
        r.affect_this_prep = [str(doc["summary"])]
    return r


def prep_research(game: str, *, offline: bool = False, live_scout: bool | None = None, fetch: Fetch | None = None) -> Any:
    """Research for this prep: the daily AI file, or the old scraper with ``--live-scout``."""
    if live_scout is None:
        live_scout = os.environ.get("CFB_COACH_LIVE_SCOUT") == "1" or os.environ.get("CFB_COACH_AI_RESEARCH") == "off"
    if game == "madden27":
        from cfb_coach.madden.meta_scout import _save_cache, run_madden_scout as scraper
    else:
        from cfb_coach.meta_scout import _save_cache, run_meta_scout as scraper
    if live_scout:
        return scraper(offline=offline)
    doc, origin = load_research(game, offline=offline, fetch=fetch)
    if doc is None:
        r = scraper(offline=offline)  # until the first daily research file lands
        r.message = "No daily AI research file yet — old scraper: " + (r.message or "")
        return r
    r = to_scout_result(doc, game, origin=origin)
    _save_cache(r)  # live callers read their priors from this cache
    return r


def opponent_research(game: str, opponent_id: str) -> dict[str, Any]:
    """This opponent's block from the freshest research file on disk (no network)."""
    if os.environ.get("CFB_COACH_AI_RESEARCH") == "off":
        return {}
    best: dict[str, Any] | None = None
    for path in (cache_path(game), repo_path(game)):
        doc = _read_json(path)
        if doc and not validate(doc, game):
            if best is None or (_parse_ts(doc.get("researched_at")) or datetime.min.replace(tzinfo=timezone.utc)) > (
                _parse_ts(best.get("researched_at")) or datetime.min.replace(tzinfo=timezone.utc)
            ):
                best = doc
    return dict(((best or {}).get("opponents") or {}).get(opponent_id) or {})


def opponent_counters(game: str, opponent_id: str, side: str) -> list[dict[str, Any]]:
    """Researched counters vs this opponent: side 'defense' (vs his concepts) or 'offense' (vs his coverages)."""
    return list(opponent_research(game, opponent_id).get(f"{side}_counters") or [])


def opponent_lines(game: str, opponent_id: str) -> list[str]:
    """Plain-text research block for this opponent (prep sheet)."""
    o = opponent_research(game, opponent_id)
    lines = [str(n) for n in o.get("notes") or []]
    for c in o.get("defense_counters") or []:
        lines.append(f"D vs {c.get('vs') or c.get('vs_family')}: {', '.join(c.get('calls') or c.get('families') or [])}"
                     + (f" — {c['why']}" if c.get("why") else ""))
    for c in o.get("offense_counters") or []:
        lines.append(f"O vs {c.get('vs') or c.get('vs_class')}: {', '.join(c.get('plays') or [])}"
                     + (f" — {c['why']}" if c.get("why") else ""))
    return lines


__all__ = [
    "GAMES",
    "load_research",
    "opponent_counters",
    "opponent_lines",
    "opponent_research",
    "prep_research",
    "to_scout_result",
    "validate",
]
