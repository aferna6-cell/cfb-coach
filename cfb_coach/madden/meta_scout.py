"""Madden 27 live meta research — the CFB 27 research pipeline, pointed at Madden.

Same as CFB prep (``cfb_coach.meta_scout.run_meta_scout``): every prep does LIVE
research — web pages + Google News / Reddit RSS + YouTube (search + creator
channel RSS + transcripts, ``yt_research`` with the Madden 27 profile) in
parallel. Named-entity extraction (``meta_entities``) runs per side against the
Madden 27 playbook catalog (Huddle.gg formation/play lists) and counts which
stock *books* are recommended for offense and defense. The cache is only a loud
fallback when offline or every source fails; then the cited seed baseline.
Never raises. Never uses CFB 27 meta (separate cache file, Madden vocab).
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cfb_coach import meta_scout as _ms
from cfb_coach.meta_scout import (
    TOTAL_FETCH_BUDGET,
    YOUTUBE_BUDGET,
    MetaScoutResult,
    MetaSource,
    _now_iso,
    _parse_fetched,
    local_ts,
)

BASELINE_VERSION = "madden27-2026-09"
CACHE_FILENAME = "madden27_meta_cache.json"
FALLBACK_MSG = f"Scout unavailable — using cached/baseline {BASELINE_VERSION}"
LIVE_POLICY = "live research every prep (web + YouTube); cache is only a fallback when offline or every source fails"

_GN = "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q="
_RD = "https://www.reddit.com/r/Madden/search.rss?sort=new&restrict_sr=1&t=month&q="

# kind: patch | guide | rss
SOURCES: list[dict[str, str]] = [
    {"kind": "patch", "label": "EA Madden 27 title update Sep 16",
     "url": "https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-nfl-27-title-update-september-16"},
    {"kind": "patch", "label": "Madden School TU Sep 3",
     "url": "https://www.madden-school.com/madden-27-september-3rd-2026-title-update/"},
    {"kind": "guide", "label": "TimeSaver best Madden 27 playbooks (O + D)",
     "url": "https://timesaver.gg/blog/madden-nfl-27-best-playbooks-offense-defense"},
    {"kind": "guide", "label": "Civil.GG best Madden 27 offenses",
     "url": "https://www.civil.gg/tips/best-offenses-madden-27"},
    {"kind": "guide", "label": "Madden Prodigy best Madden 27 playbooks",
     "url": "https://www.maddenprodigy.com/best-madden-27-playbooks/"},
    {"kind": "guide", "label": "Operation Sports best defensive playbooks",
     "url": "https://www.operationsports.com/best-defensive-playbooks-to-use-in-madden-27/"},
    {"kind": "rss", "label": "r/Madden: meta / playbook (past month)",
     "url": _RD + "meta+OR+playbook+OR+%22best+defense%22+OR+%22best+offense%22"},
    {"kind": "rss", "label": "Google News: Madden 27 meta / playbooks / patches (30d, incl. YouTube)",
     "url": _GN + "%22Madden+27%22+(meta+OR+playbook+OR+defense+OR+offense+OR+%22title+update%22)+when:30d"},
]
TRUSTED_URLS: list[str] = [s_["url"] for s_ in SOURCES]
_SOURCE_BY_URL = {s_["url"]: s_ for s_ in SOURCES}

MADDEN27_RX = re.compile(r"madden\s*(?:nfl\s*)?27\b|\bmadden27\b", re.I)
_OLD_RX = re.compile(r"madden\s*(?:nfl\s*)?2[3-6]\b|college\s*football|\bcfb\b|\bncaa\b", re.I)


def _c(pattern: str, side: str, label: str, tip: str, macro: str | None = None) -> tuple[re.Pattern[str], dict[str, Any]]:
    return re.compile(pattern, re.I), {"side": side, "label": label, "tip": tip, "macro_hint": macro}


CONCEPT_MAP: list[tuple[re.Pattern[str], dict[str, Any]]] = [
    _c(r"clamp\s*stack", "offense", "Gun Doubles Clamp Stack",
       "Clamp Stack still meta — Texas Y-Stutter Wheel / Mtn Shuffle Verts Smash home"),
    _c(r"\bbunch\b|\bstack\b", "offense", "Bunch / stack sets",
       "Bunch + stack sets beating launch D — stack rubs vs man; STACK macro ready on D", "STACK"),
    _c(r"stick[\s-]*wheel", "offense", "Mtn Stick Wheel",
       "Stick-wheel hot — Gun Trips X Nasty Mtn Stick Wheel; on D, FLAT-CAP after repeats", "FLAT-CAP"),
    _c(r"\bmesh\b|\bcross(?:ers?|ing)\b", "offense", "Mesh / crossers",
       "Mesh/crossers meta — Mesh easy completions; MESH-RAT after repeated crossers on D", "MESH-RAT"),
    _c(r"\bflood\b|\bsail\b", "offense", "Flood / sail",
       "Flood/sail vs single-high — flood concepts in the book's trips/bunch sets; FLAT-CAP vs repeated floods", "FLAT-CAP"),
    _c(r"inside\s*zone|\bstretch\b|run\s*game", "offense", "Zone run",
       "Run game: Inside Zone + Stretch preferred post-update — run first vs two-high", "RUN-FIT"),
    _c(r"cover\s*4|quarters|\bmatch\b", "defense", "Cover 4 Quarters / match",
       "Quarters/match home — Nickel Over Cover 4 Quarters / Tampa 2; MATCH-4 only after repeated verticals", "MATCH-4"),
    _c(r"\bmug\b|db\s*fire|\bstunt|\bsim\b", "defense", "Schematic mug pressure",
       "Sim/stunt pressure off the mug (Nickel Sim 2, Field Sim 3) > contain-four — selective", "HEAT"),
    _c(r"\bcontain\b|\bedge\b|\btackles?\b", "defense", "Contain / edge",
       "Patch: OTs pick up contain better — don't live on contain-four; SPY vs repeated scrambles", "SPY"),
    _c(r"\bman\b|cover\s*1|\bpress\b", "defense", "Man coverage",
       "Man only when bunch/stack is handled — O-MAN hot routes vs repeated man", "O-MAN"),
    _c(r"\bblitz\b|\bpressure\b", "defense", "Pressure",
       "Pressure looks — O-PROT Half Slide + Mtn Triple Slants / Texas Y-Stutter Wheel", "O-PROT"),
    _c(r"saleh|4-3|nickel", "defense", "Saleh 4-3 / nickel",
       "Saleh 4-3 / flexible nickel is the D home — avoid buggy 3-4 edge drops"),
]

_EXPERIMENTAL = frozenset({"HEAT", "O-RPO"})

# Book-name aliases people actually write/say (besides the team nickname + city)
_BOOK_ALIASES: dict[str, list[str]] = {
    "Buccaneers": ["bucs", "tampa bay", "tampa"],
    "49ers": ["niners", "san francisco", "sf 49ers", "forty niners"],
    "Shotgun Classic": ["shotgun classic", "gun classic"],
    "Jaguars": ["jags", "jacksonville"],
    "Eagles": ["philly", "philadelphia"],
    "Chargers": ["bolts"],
}
_BOOK_CONTEXT = re.compile(r"playbook|\bbook\b|\bscheme\b|offen[cs]e|defen[cs]e|\bmeta\b", re.I)
_O_CTX = re.compile(r"offen[cs]e|offensive|\bo\b playbook|pass(?:ing)?|run game", re.I)
_D_CTX = re.compile(r"defen[cs]e|defensive|coverage|blitz|\bd\b playbook", re.I)
BOOK_WINDOW = 70  # chars either side of a book mention


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
        cache_path().parent.mkdir(parents=True, exist_ok=True)
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
        mode="seed",
        research_status="offline" if offline else "failed",
    )


# --- Vocabulary / named signals ----------------------------------------------------

def _madden_extra(forms: list[str]) -> dict[str, list[str]]:
    """Spoken aliases: Huddle "Gun X" is "Shotgun X" in game and in most videos."""
    out: dict[str, list[str]] = {}
    for f in forms:
        alts = []
        if f.startswith("Gun "):
            alts.append("Shotgun " + f[4:])
        if " Str " in f:
            alts.append(f.replace(" Str ", " Strong "))
        if alts:
            out[f] = alts
    return out


def side_vocab(side: str) -> Any:
    from cfb_coach.madden import catalog
    from cfb_coach.meta_entities import build_vocab

    cat = catalog.all_formations(side)
    return build_vocab(cat, _madden_extra(list(cat)))


def _book_patterns(side: str) -> list[tuple[str, re.Pattern[str]]]:
    from cfb_coach.madden import catalog

    out = []
    for b in catalog.book_names(side):
        team = catalog.book_team(side, b) or ""
        alts = {b.lower()} | {a.lower() for a in _BOOK_ALIASES.get(b, [])}
        if team:
            alts.add(team.lower())
        rx = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(a) for a in sorted(alts, key=len, reverse=True)) + r")(?![a-z0-9])", re.I)
        out.append((b, rx))
    return out


def book_mentions(text: str) -> dict[str, dict[str, float]]:
    """{side: {book: weight}} — a book mention only counts next to playbook/offense/defense words;
    side comes from nearby offense/defense words (both sides at half weight when ambiguous)."""
    out: dict[str, dict[str, float]] = {"offense": {}, "defense": {}}
    if not text:
        return out
    for side in ("offense", "defense"):
        for b, rx in _book_patterns(side):
            for m in rx.finditer(text):
                lo, hi = max(0, m.start() - BOOK_WINDOW), m.end() + BOOK_WINDOW
                ctx = text[lo:hi]
                if not _BOOK_CONTEXT.search(ctx):
                    continue
                pos = m.start() - lo
                do = min((abs(x.start() - pos) for x in _O_CTX.finditer(ctx)), default=None)
                dd = min((abs(x.start() - pos) for x in _D_CTX.finditer(ctx)), default=None)
                if b == "Shotgun Classic":
                    do, dd = 0, None
                if do is not None and (dd is None or do + 15 < dd):
                    near = "offense"
                elif dd is not None and (do is None or dd + 15 < do):
                    near = "defense"
                else:
                    near = ""
                if not near:
                    w = 0.5
                elif near == side:
                    w = 1.0
                else:
                    continue
                out[side][b] = min(4.0, out[side].get(b, 0.0) + w)
    return out


def aggregate_books(docs: list[dict[str, Any]], *, now: datetime) -> dict[str, dict[str, Any]]:
    from cfb_coach.meta_entities import KIND_WEIGHT, MAX_REPEAT, REPEAT_BONUS, recency_weight

    agg: dict[str, dict[str, Any]] = {"offense": {}, "defense": {}}
    for d in docs:
        hits = book_mentions(d.get("text") or "")
        w = recency_weight(d.get("date"), now) * KIND_WEIGHT.get(str(d.get("kind") or "page"), 0.8)
        for side, books in hits.items():
            for b, n in books.items():
                rec = agg[side].setdefault(b, {"score": 0.0, "docs": 0, "sources": []})
                rec["score"] = round(rec["score"] + w * min(1.0, n) * (1.0 + REPEAT_BONUS * min(MAX_REPEAT, max(0.0, n - 1))), 4)
                rec["docs"] += 1
                if len(rec["sources"]) < 5:
                    rec["sources"].append({"label": str(d.get("label") or "")[:140], "url": d.get("url") or "",
                                           "date": d.get("date") or "", "kind": d.get("kind") or ""})
    return {side: dict(sorted(v.items(), key=lambda kv: -kv[1]["score"])) for side, v in agg.items()}


def named_signals(docs: list[dict[str, Any]], *, now: datetime) -> dict[str, Any]:
    from cfb_coach.meta_entities import aggregate

    out: dict[str, Any] = {side: aggregate(docs, now=now, voc=side_vocab(side)) for side in ("offense", "defense")}
    out["books"] = aggregate_books(docs, now=now)
    out["docs_scanned"] = out["offense"].get("docs_scanned", 0)
    return out


def seed_docs() -> list[dict[str, Any]]:
    """The cited seed baseline as documents (fallback research)."""
    from cfb_coach.madden.data import load_meta_baseline

    bl = load_meta_baseline()
    docs = []
    for side in ("offense", "defense"):
        for src, names in ((bl.get(side) or {}).get("rankings") or {}).items():
            m = re.search(r"(20\d\d-\d\d-\d\d)", src)
            docs.append({"label": f"seed: {src}", "kind": "seed", "url": "", "date": m.group(1) if m else "",
                         "text": f"best {side} playbook: " + ", ".join(f"{n} playbook {side}" for n in names)})
    for line in bl.get("sources") or []:
        m = re.search(r"(20\d\d-\d\d-\d\d)", line)
        docs.append({"label": f"seed: {line[:80]}", "kind": "seed", "url": "", "date": m.group(1) if m else "", "text": line})
    return docs


def research_from_scout(result: MetaScoutResult | dict[str, Any] | None) -> dict[str, Any]:
    """Compact research dict for `playbook.recommend_book` / focus / the live caller."""
    if result is None:
        return {"mode": "", "named": {}, "books": {}}
    d = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    ns = d.get("named_signals") or {}
    return {"mode": d.get("mode") or "", "status": d.get("research_status") or "",
            "named": {s: ns.get(s) or {} for s in ("offense", "defense")},
            "books": ns.get("books") or {}, "message": d.get("message") or ""}


# --- Orchestration -----------------------------------------------------------------

def _age_hours(ts: str, now: datetime) -> float | None:
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return round((now - dt).total_seconds() / 3600.0, 1)


def run_madden_scout(
    *,
    offline: bool = False,
    refresh: bool = True,  # kept for callers; live research is always the default (like CFB)
    urls: list[str] | None = None,
    now: datetime | None = None,
    youtube: bool = True,
    yt_runner: Any | None = None,
) -> MetaScoutResult:
    """Live Madden research every prep (web + YouTube in parallel). Never raises; bounded time.

    Order: live -> last good cache (loud, with age) -> cited seed baseline."""
    del refresh
    now = now or datetime.now(timezone.utc)
    t_start = time.monotonic()
    cached = _load_cache()

    def _fallback(prefix: str, *, offline_flag: bool, attempted: list[dict[str, Any]] | None = None,
                  yt: dict[str, Any] | None = None) -> MetaScoutResult:
        if cached and (cached.get("available") or cached.get("named_signals")):
            r = _from_cache(cached)
            age = _age_hours(r.fetched_at, now)
            r.offline = offline_flag
            r.mode = "cache"
            r.research_status = "offline" if offline_flag else "failed"
            r.fallback_age_hours = age
            r.message = (f"{prefix} — using cached Madden research from {local_ts(r.fetched_at)}"
                         + (f" ({age:.0f}h old)" if age is not None else ""))
        else:
            r = unavailable(offline=offline_flag)
            r.message = f"{prefix} and no cache — using seed research {BASELINE_VERSION}"
            try:
                r.named_signals = named_signals(seed_docs(), now=now)
            except Exception:  # noqa: BLE001
                r.named_signals = {}
        r.ttl_policy = LIVE_POLICY
        if attempted:
            r.sources = attempted + [x for x in r.sources if x.get("kind") == "seed"]
        if yt is not None:
            r.youtube = yt
        r.elapsed_s = round(time.monotonic() - t_start, 2)
        return r

    if offline:
        return _fallback("Offline (--offline)", offline_flag=True)

    url_list = list(urls or TRUSTED_URLS)
    yt_res: dict[str, Any] = {}
    yt_docs: list[dict[str, Any]] = []
    outer = ThreadPoolExecutor(max_workers=2)
    yt_future = None
    if youtube:
        if yt_runner is None:
            from cfb_coach.yt_research import madden_profile, run_youtube_research

            def yt_runner(**kw: Any) -> Any:  # noqa: F811
                return run_youtube_research(fetch=_ms._yt_fetch, profile=madden_profile(), **kw)

        yt_future = outer.submit(yt_runner, now=now, budget_s=YOUTUBE_BUDGET)
    sources, bodies = _ms._fetch_parallel(url_list, TOTAL_FETCH_BUDGET)
    for src in sources:
        info = _SOURCE_BY_URL.get(src.url) or {}
        src.label = src.label or info.get("label", "")
        src.kind = src.kind or info.get("kind", "")
    if yt_future is not None:
        try:
            yr = yt_future.result(timeout=max(1.0, YOUTUBE_BUDGET + 2.0 - (time.monotonic() - t_start)))
            yt_res = yr.to_dict() if hasattr(yr, "to_dict") else dict(yr)
            yt_docs = list(getattr(yr, "docs", None) or yt_res.get("docs") or [])
            yt_res["docs"] = [{k: v for k, v in d.items() if k != "text"} | {"chars": len(d.get("text") or "")} for d in yt_docs]
        except Exception as exc:  # noqa: BLE001
            yt_res = {"ran": True, "found": 0, "transcripts": 0, "notes": [f"YouTube research failed: {type(exc).__name__}: {exc}"]}
    outer.shutdown(wait=False, cancel_futures=True)

    page_sources = [s_ for s_ in sources if s_.kind != "rss"]
    result = _parse_fetched(page_sources, bodies, concept_map=CONCEPT_MAP)
    result.baseline_fallback = BASELINE_VERSION
    result.sources = [asdict(s_) for s_ in sources]
    if yt_res:
        result.sources.append({
            "url": "https://www.youtube.com/results?search_query=madden+27+meta", "title": "YouTube",
            "fetched": bool(yt_res.get("found")), "status": None,
            "error": "" if yt_res.get("found") else "; ".join(yt_res.get("notes") or [])[:160],
            "label": (f"YouTube: {yt_res.get('found', 0)} Madden 27 videos, {yt_res.get('transcripts', 0)} transcripts "
                      f"(search {yt_res.get('search_ok', 0)}/{yt_res.get('search_total', 0)}, channel RSS "
                      f"{yt_res.get('rss_ok', 0)}/{yt_res.get('rss_total', 0)})"),
            "kind": "youtube",
        })
    heads, texts, docs = _ms._collect_docs(sources, bodies, now=now, game_rx=MADDEN27_RX, old_rx=_OLD_RX)
    for d in yt_docs:
        if d.get("kind") == "youtube_transcript":
            heads.append({"title": d.get("title") or d.get("label"), "link": d.get("url"), "date": d.get("date") or "",
                          "source": f"YouTube transcript · {d.get('channel') or ''}"})
    heads.sort(key=lambda h: h.get("date") or "", reverse=True)
    all_docs = docs + yt_docs
    result.headlines = heads[:30]
    result.named_signals = named_signals(all_docs, now=now)
    result.youtube = yt_res
    n_ok = sum(1 for s_ in sources if s_.fetched)
    ns = result.named_signals
    result.available = (n_ok > 0 or bool(yt_res.get("found"))) and bool(
        result.patch_notes or result.suggestions or heads or ns["offense"].get("formations")
        or ns["defense"].get("formations") or any(ns["books"].values())
    )
    result.ttl_policy = LIVE_POLICY
    if not result.available:
        return _fallback("LIVE RESEARCH FAILED (no source reachable)", offline_flag=False,
                         attempted=result.sources, yt=yt_res)
    result.mode = "live"
    result.research_status = "live" if n_ok >= max(1, len(sources) // 2) else "partial"
    result.previous_fetched_at = (cached or {}).get("fetched_at", "")
    result.elapsed_s = round(time.monotonic() - t_start, 2)
    result.message = (
        f"Live Madden research this prep: {n_ok}/{len(sources)} web sources"
        + (f", YouTube {yt_res.get('found', 0)} videos / {yt_res.get('transcripts', 0)} transcripts" if yt_res else "")
        + f" in {result.elapsed_s:.1f}s"
    )
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
