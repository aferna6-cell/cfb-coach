"""Madden 27 live meta scout — same urllib fetch/parse primitives as CFB.

Own trusted URLs, own concept map (Madden book language), own cache file
(<data dir>/madden27_meta_cache.json, 6h TTL). Never raises; offline or failed
fetches fall back to cached scout or baseline madden27-2026-09. Never uses
CFB27 meta.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from cfb_coach.meta_scout import (
    CACHE_TTL_SECONDS,
    PER_URL_TIMEOUT,
    TOTAL_FETCH_BUDGET,
    MetaScoutResult,
    MetaSource,
    _fetch_one,
    _now_iso,
    _parse_fetched,
)

BASELINE_VERSION = "madden27-2026-09"
CACHE_FILENAME = "madden27_meta_cache.json"
FALLBACK_MSG = f"Scout unavailable — using cached/baseline {BASELINE_VERSION}"

# Concrete pages first (patch notes + launch meta guides); failures skip gracefully.
TRUSTED_URLS: list[str] = [
    "https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-nfl-27-title-update-september-16",
    "https://www.madden-school.com/madden-27-september-3rd-2026-title-update/",
    "https://timesaver.gg/blog/madden-nfl-27-best-playbooks-offense-defense",
    "https://www.civil.gg/tips/best-offenses-madden-27",
    "https://timesaver.gg/blog/madden-nfl-27-title-update-september-16-patch-notes",
    "https://www.ea.com/games/madden-nfl/madden-nfl-27/news",
    "https://www.madden-school.com/playbooks/buccaneers/",
    "https://huddle.gg/playbooks/buccaneers-off/",
]


def _c(pattern: str, side: str, label: str, tip: str, macro: str | None = None) -> tuple[re.Pattern[str], dict[str, Any]]:
    return re.compile(pattern, re.I), {"side": side, "label": label, "tip": tip, "macro_hint": macro}


CONCEPT_MAP: list[tuple[re.Pattern[str], dict[str, Any]]] = [
    _c(r"clamp\s*stack", "offense", "Gun Doubles Clamp Stack",
       "Clamp Stack still meta — keep it home; Motion Shuffle Vert Smash vs man, Mesh easy"),
    _c(r"\bbunch\b|\bstack\b", "offense", "Bunch / stack sets",
       "Bunch + stack sets beating launch D — stack rubs vs man; STACK macro ready on D", "STACK"),
    _c(r"stick[\s-]*wheel", "offense", "Mtn Stick Wheel",
       "Stick-wheel hot — Gun Trips X Nasty Mtn Stick Wheel; on D, FLAT-CAP after repeats", "FLAT-CAP"),
    _c(r"\bmesh\b|\bcross(?:ers?|ing)\b", "offense", "Mesh / crossers",
       "Mesh/crossers meta — Mesh easy completions; MESH-RAT after repeated crossers on D", "MESH-RAT"),
    _c(r"\bflood\b|\bsail\b", "offense", "Flood / sail",
       "Flood/sail vs single-high — Flood Sail / PA Flood; FLAT-CAP vs repeated floods", "FLAT-CAP"),
    _c(r"inside\s*zone|\bstretch\b|run\s*game", "offense", "Zone run",
       "Run game: Inside Zone + Stretch preferred post-update — run first vs two-high", "RUN-FIT"),
    _c(r"cover\s*4|quarters|\bmatch\b", "defense", "Cover 4 Quarters / match",
       "Quarters/match home — Nickel Mug Cover 4 Quarters; MATCH-4 only after repeated verticals", "MATCH-4"),
    _c(r"\bmug\b|db\s*fire|\bstunt|\bsim\b", "defense", "Schematic mug pressure",
       "Schematic mug / DB Fire 2 pressure > contain-four — selective, 3rd-medium only", "HEAT"),
    _c(r"\bcontain\b|\bedge\b|\btackles?\b", "defense", "Contain / edge",
       "Patch: OTs pick up contain better — don't live on contain-four; SPY vs repeated scrambles", "SPY"),
    _c(r"\bman\b|cover\s*1|\bpress\b", "defense", "Man coverage",
       "Man only when bunch/stack is handled — O-MAN hot routes vs repeated man", "O-MAN"),
    _c(r"\bblitz\b|\bpressure\b", "defense", "Pressure",
       "Pressure looks — O-PROT slide + Quick Slants / Mesh hot", "O-PROT"),
    _c(r"saleh|4-3|nickel", "defense", "Saleh 4-3 / nickel",
       "Saleh 4-3 / flexible nickel is the D home — avoid buggy 3-4 edge drops"),
]

_EXPERIMENTAL = frozenset({"HEAT", "O-RPO"})


def cache_path() -> Path:
    from cfb_coach.games import data_dir

    return data_dir() / CACHE_FILENAME


def _load_cache() -> dict[str, Any] | None:
    try:
        return json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_cache(result: MetaScoutResult) -> None:
    try:
        payload = result.to_dict()
        payload["_cached_at"] = time.time()
        cache_path().write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def _from_cache(cached: dict[str, Any]) -> MetaScoutResult:
    fields = MetaScoutResult.__dataclass_fields__
    r = MetaScoutResult(**{k: v for k, v in cached.items() if k in fields})
    r.from_cache = True
    r.baseline_fallback = BASELINE_VERSION
    return r


def unavailable(*, offline: bool = False) -> MetaScoutResult:
    return MetaScoutResult(
        available=False,
        offline=offline,
        confidence="low",
        baseline_fallback=BASELINE_VERSION,
        fetched_at=_now_iso(),
        message=(f"Offline — using baseline {BASELINE_VERSION}" if offline else FALLBACK_MSG),
    )


def run_madden_scout(
    *,
    offline: bool = False,
    refresh: bool = False,
    urls: list[str] | None = None,
) -> MetaScoutResult:
    """Fetch/parse Madden sources (or reuse cache). Never raises; ~12s budget."""
    cached = _load_cache()
    if offline:
        if cached:
            r = _from_cache(cached)
            r.offline = True
            r.message = "Offline — using cached Madden scout" if r.available else FALLBACK_MSG
            return r
        return unavailable(offline=True)
    if cached and not refresh and time.time() - float(cached.get("_cached_at") or 0) < CACHE_TTL_SECONDS:
        r = _from_cache(cached)
        r.message = "Using cached Madden scout (<6h)"
        return r

    url_list = list(urls or TRUSTED_URLS[:6])
    sources: list[MetaSource] = []
    bodies: dict[str, str] = {}
    t0 = time.monotonic()
    for url in url_list:
        remaining = TOTAL_FETCH_BUDGET - (time.monotonic() - t0)
        if remaining <= 0.4:
            sources.append(MetaSource(url=url, error="skipped: total fetch budget exhausted"))
            continue
        src, body = _fetch_one(url, min(PER_URL_TIMEOUT, max(0.8, remaining)))
        sources.append(src)
        if body:
            bodies[url] = body

    result = _parse_fetched(sources, bodies, concept_map=CONCEPT_MAP)
    result.baseline_fallback = BASELINE_VERSION
    if not result.available:
        if cached and (cached.get("patch_notes") or cached.get("suggestions")):
            stale = _from_cache(cached)
            stale.message = FALLBACK_MSG
            return stale
        result.message = FALLBACK_MSG
        return result
    result.message = "Live Madden scout OK"
    _save_cache(result)
    return result


def apply_scout(
    result: MetaScoutResult,
    *,
    profile: str,
    offense_only: bool,
    active: list[str],
) -> tuple[list[str], list[str]]:
    """Scout hits → (tips, affect-this-prep lines). Soft only — never mutates inventory."""
    from cfb_coach.madden.franchise import LAB

    tips: list[str] = []
    affect: list[str] = []
    for sug in result.suggestions:
        side = sug.get("side")
        mh = (sug.get("macro_hint") or "").upper()
        if offense_only and side == "defense" and not mh.startswith("O-"):
            continue
        line = f"Meta-grounded (live scout): {sug.get('tip')}"
        if mh in _EXPERIMENTAL and profile != LAB:
            line += f" — primary: {mh} stays benched unless you swap it in"
        elif mh and mh not in active and mh not in _EXPERIMENTAL:
            line += f" — {mh} not Active"
        if line not in tips:
            tips.append(line)
            affect.append(f"{sug.get('label')}: {sug.get('tip')}")
    result.affect_this_prep = affect[:10]
    return tips[:8], affect[:10]
