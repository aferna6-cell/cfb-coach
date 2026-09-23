"""Live meta scout — fetch recent CFB 27 patch notes + online competitive meta.

On prep (unless --offline): hit a small set of trusted URLs via urllib, parse
titles/snippets, cache under ~/.cfb-coach/meta_cache.json (6h TTL), and map
hits onto Aidan's book language as soft Meta-grounded suggestions.

Never hangs the prep path: hard per-URL timeouts + total fetch budget ~15s.
Failures degrade gracefully to cached / baseline cfb27-2026-09.
"""

from __future__ import annotations

import html as html_mod
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASELINE_VERSION = "cfb27-2026-09"
CACHE_TTL_SECONDS = 6 * 3600
PER_URL_TIMEOUT = 2.5
TOTAL_FETCH_BUDGET = 12.0
USER_AGENT = (
    "Mozilla/5.0 (compatible; cfb-coach-meta-scout/1.6; "
    "+https://github.com/aferna6-cell/cfb-coach)"
)
MAX_BODY = 180_000

# Trusted concrete URLs (skip failures gracefully). Prefer known pages over search.
TRUSTED_URLS: list[str] = [
    # EA official news / title updates
    "https://www.ea.com/games/ea-sports-college-football/college-football-27/news/title-update-september-3rd-2026",
    "https://www.ea.com/games/ea-sports-college-football/college-football-27/news/cfb-27-title-update-august-6-2026",
    "https://www.ea.com/games/ea-sports-college-football/college-football-27/news",
    "https://www.ea.com/games/ea-sports-college-football",
    # Patch aggregators
    "https://mp1st.com/title-updates-and-patches/college-football-27-update-1-012-september-22-brings-gameplay-changes",
    "https://mp1st.com/",
    "https://updatecrazy.com/ea-college-football-27-cfb-27-update-1-012-patch-notes/",
    # Competitive / tips style (if 200)
    "https://civil.gg/",
    "https://www.maddenturf.com/",
    "https://www.gamespot.com/games/ea-sports-college-football-25/",
]

# Concept → Aidan book language soft mapping
_CONCEPT_MAP: list[tuple[re.Pattern[str], dict[str, Any]]] = [
    (
        re.compile(r"\bbunch\b", re.I),
        {
            "side": "offense",
            "label": "Gun Bunch X Nasty",
            "tip": "Bunch still meta — keep Gun Bunch X Nasty primary; Mesh Spot / IZ home",
            "macro_hint": "BUNCH",
        },
    ),
    (
        re.compile(r"\bmesh\b", re.I),
        {
            "side": "offense",
            "label": "Mesh Spot",
            "tip": "Mesh family hot — Mesh Spot easy completions; avoid Mesh Post spam into C6",
            "macro_hint": None,
        },
    ),
    (
        re.compile(r"\bcover\s*6\b|\bc6\b|\bsplit[-\s]?field\b", re.I),
        {
            "side": "defense",
            "label": "Cover 6 / split-field",
            "tip": "vs Cover 6 / split-field: RUN FIRST (IZ / HB Base); Mesh Spot underneath — not Flood",
            "macro_hint": None,
        },
    ),
    (
        re.compile(r"\bcover\s*3\b|\bc3\s*sky\b|\bflood\b", re.I),
        {
            "side": "offense",
            "label": "Deep Flood vs C3",
            "tip": "Cover 3 / single-high: Deep Flood / crossers OK; FLOOD macro still benched unless swap planned",
            "macro_hint": "FLOOD",
        },
    ),
    (
        re.compile(r"\b3-?3\s*cub\b|\bcub\b|\bnickel\s*over\b", re.I),
        {
            "side": "defense",
            "label": "Nickel Over / 3-3 Cub",
            "tip": "Nickel Over remains D home; 3-3 Cub pressure looks → CROSS / RPO readiness",
            "macro_hint": "CROSS",
        },
    ),
    (
        re.compile(r"\bquarters\b|\bcover\s*4\b|\btampa\b", re.I),
        {
            "side": "defense",
            "label": "Quarters / Tampa 2",
            "tip": "Two-high money: Quarters / Tampa 2 with no macro most snaps; VERT only after repeated 4-verts",
            "macro_hint": "VERT",
        },
    ),
    (
        re.compile(r"\bcontain\b|\bscram(?:ble)?\b|\bqb\s*spin\b", re.I),
        {
            "side": "defense",
            "label": "Contain / scramble",
            "tip": "Patch contain fix + scramble meta — SCRAM ready; CONTAIN-SCRAM only with explicit swap (ohio_state freer)",
            "macro_hint": "SCRAM",
        },
    ),
    (
        re.compile(r"\brpo\b|\bbubble\b", re.I),
        {
            "side": "offense",
            "label": "RPO / bubble",
            "tip": "RPO/bubble meta — elevate RPO macro readiness after repeated tells; quick game vs pressure",
            "macro_hint": "RPO",
        },
    ),
    (
        re.compile(r"\bcluster\b|\bspot\s*shake\b", re.I),
        {
            "side": "offense",
            "label": "Gun Cluster",
            "tip": "Cluster changeup when Bunch overplayed — Z Spot Shake / Outside Zone",
            "macro_hint": None,
        },
    ),
    (
        re.compile(r"\bblitz\b|\bpressure\b|\bman\s*coverage\b|\bcover\s*1\b", re.I),
        {
            "side": "defense",
            "label": "Pressure / man",
            "tip": "Pressure/man meta — Mesh Spot + Return Whip Trail; HEAT selective on 3rd-short only",
            "macro_hint": "HEAT",
        },
    ),
    (
        re.compile(r"\brun[-\s]?action\b|\bblocking\b|\binside\s*zone\b", re.I),
        {
            "side": "offense",
            "label": "Run game / IZ",
            "tip": "Run-action blocking patch — lean IZ / HB Base early vs two-high; set up Mesh Spot",
            "macro_hint": "RUN-IN",
        },
    ),
]

_PATCH_VER_RE = re.compile(
    r"(?:update|patch|title\s*update|version)\s*(?:#?\s*)?(1\.\d{2,3}(?:\.\d+)?)|"
    r"\b(1\.\d{2,3}(?:\.\d+)?)\b",
    re.I,
)
_DATE_RE = re.compile(
    r"\b((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:,\s*\d{4})?)"
    r"|\b(\d{4}-\d{2}-\d{2})\b"
    r"|\b(Sep(?:tember)?\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.I,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_META_DESC_RE = re.compile(
    r'<meta[^>]+(?:name|property)=["\'](?:description|og:description)["\'][^>]+'
    r'content=["\'](.*?)["\']',
    re.I | re.S,
)
_META_DESC_RE2 = re.compile(
    r'<meta[^>]+content=["\'](.*?)["\'][^>]+'
    r'(?:name|property)=["\'](?:description|og:description)["\']',
    re.I | re.S,
)
_H_RE = re.compile(r"<h[1-3][^>]*>(.*?)</h[1-3]>", re.I | re.S)
_LI_RE = re.compile(r"<li[^>]*>(.*?)</li>", re.I | re.S)
_P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.I | re.S)

# Macros that break proven-8 if ADDed without swap — Alabama must not auto-ADD these
_EXPERIMENTAL_MACROS = frozenset(
    {"FLOOD", "SCREEN", "CONTAIN-SCRAM", "GLASS", "SPOT-LOCK", "PROT"}
)


@dataclass
class MetaSource:
    url: str
    title: str = ""
    fetched: bool = False
    status: int | None = None
    error: str = ""


@dataclass
class MetaScoutResult:
    available: bool = False
    offline: bool = False
    from_cache: bool = False
    confidence: str = "low"  # low | medium | high
    baseline_fallback: str = BASELINE_VERSION
    patch_notes: list[dict[str, Any]] = field(default_factory=list)
    meta_offense: list[str] = field(default_factory=list)
    meta_defense: list[str] = field(default_factory=list)
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    fetched_at: str = ""
    message: str = ""
    affect_this_prep: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cache_path() -> Path:
    env = os.environ.get("CFB_COACH_DB")
    if env:
        return Path(env).expanduser().resolve().parent / "meta_cache.json"
    try:
        d = Path.home() / ".cfb-coach"
        d.mkdir(parents=True, exist_ok=True)
        return d / "meta_cache.json"
    except OSError:
        d = Path("/workspace/cfb-coach/data")
        d.mkdir(parents=True, exist_ok=True)
        return d / "meta_cache.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_html(chunk: str) -> str:
    t = html_mod.unescape(_TAG_RE.sub(" ", chunk or ""))
    return _WS_RE.sub(" ", t).strip()


def _fetch_one(url: str, timeout: float) -> tuple[MetaSource, str]:
    src = MetaSource(url=url)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            src.status = getattr(resp, "status", None) or 200
            raw = resp.read(MAX_BODY)
            charset = "utf-8"
            ctype = resp.headers.get("Content-Type", "")
            if "charset=" in ctype.lower():
                charset = (
                    ctype.lower().split("charset=")[-1].split(";")[0].strip()
                    or "utf-8"
                )
            try:
                body = raw.decode(charset, errors="replace")
            except LookupError:
                body = raw.decode("utf-8", errors="replace")
            src.fetched = True
            m = _TITLE_RE.search(body)
            src.title = _strip_html(m.group(1)) if m else url.rsplit("/", 1)[-1]
            return src, body
    except urllib.error.HTTPError as e:
        src.status = e.code
        src.error = f"HTTP {e.code}"
        return src, ""
    except Exception as e:  # noqa: BLE001 — scout must never raise
        src.error = f"{type(e).__name__}: {e}"
        return src, ""


def _extract_snippets(body: str, limit: int = 24) -> list[str]:
    bits: list[str] = []
    for rx in (_META_DESC_RE, _META_DESC_RE2):
        m = rx.search(body)
        if m:
            bits.append(_strip_html(m.group(1)))
    for m in _H_RE.finditer(body):
        bits.append(_strip_html(m.group(1)))
    for m in _LI_RE.finditer(body):
        t = _strip_html(m.group(1))
        if 20 <= len(t) <= 280:
            bits.append(t)
    for m in _P_RE.finditer(body):
        t = _strip_html(m.group(1))
        if 40 <= len(t) <= 320:
            bits.append(t)
    junk = (
        "width=device", "data-next-head", "charset=", "<meta", "viewport",
        "og:image", "twitter:", "application/ld+json", "window.__",
        "function(", "cookie", "subscribe", "newsletter",
    )
    seen: set[str] = set()
    out: list[str] = []
    for b in bits:
        key = b.lower()
        if key in seen or len(b) < 12:
            continue
        if any(j in key for j in junk):
            continue
        # Prefer gameplay-ish text
        if len(b) > 400:
            b = b[:400].rstrip() + "…"
        seen.add(key)
        out.append(b)
        if len(out) >= limit:
            break
    return out


def _looks_patchy(text: str) -> bool:
    return bool(
        re.search(
            r"patch|title\s*update|hotfix|gameplay|update\s*1\.|version\s*1\.",
            text,
            re.I,
        )
    )


def _looks_meta(text: str) -> bool:
    return bool(
        re.search(
            r"meta|bunch|mesh|cover\s*[1-9]|quarters|tampa|nickel|blitz|rpo|"
            r"formation|playbook|flood|cluster|scram|contain|3-3\s*cub",
            text,
            re.I,
        )
    )


def _parse_fetched(
    sources: list[MetaSource],
    bodies: dict[str, str],
    concept_map: list[tuple[re.Pattern[str], dict[str, Any]]] | None = None,
) -> MetaScoutResult:
    """Parse fetched pages. `concept_map` lets other games (Madden 27) reuse this."""
    concept_map = _CONCEPT_MAP if concept_map is None else concept_map
    patch_notes: list[dict[str, Any]] = []
    offense: list[str] = []
    defense: list[str] = []
    suggestions: list[dict[str, Any]] = []
    seen_sug: set[str] = set()

    for src in sources:
        if not src.fetched:
            continue
        body = bodies.get(src.url) or ""
        snippets = _extract_snippets(body)
        blob = " ".join([src.title] + snippets)

        ver_m = _PATCH_VER_RE.search(blob)
        date_m = _DATE_RE.search(blob)
        version = ""
        if ver_m:
            version = next((g for g in ver_m.groups() if g), "") or ""

        if _looks_patchy(blob) or version:
            bullets = [
                s
                for s in snippets
                if _looks_patchy(s)
                or re.search(
                    r"blocking|contain|spin|catch|coverage|run-action|"
                    r"gameplay|fixed|improved|tuned",
                    s,
                    re.I,
                )
            ][:6]
            if not bullets and snippets:
                bullets = snippets[:3]
            if bullets or version:
                patch_notes.append(
                    {
                        "version": version or "unknown",
                        "date": (date_m.group(0) if date_m else ""),
                        "title": src.title,
                        "bullets": bullets,
                        "url": src.url,
                    }
                )

        for pat, info in concept_map:
            if not pat.search(blob):
                continue
            tip = info["tip"]
            if tip in seen_sug:
                continue
            seen_sug.add(tip)
            suggestions.append(
                {
                    "side": info["side"],
                    "label": info["label"],
                    "tip": tip,
                    "macro_hint": info.get("macro_hint"),
                    "badge": "Meta-grounded (live scout)",
                    "source_url": src.url,
                }
            )
            if info["side"] == "offense":
                offense.append(tip)
            else:
                defense.append(tip)

        for s in snippets:
            if not _looks_meta(s):
                continue
            low = s.lower()
            if any(
                k in low
                for k in (
                    "bunch",
                    "mesh",
                    "flood",
                    "rpo",
                    "zone",
                    "run",
                    "cluster",
                    "spot",
                )
            ):
                if s not in offense and len(offense) < 8:
                    offense.append(s[:220])
            if any(
                k in low
                for k in (
                    "cover",
                    "nickel",
                    "blitz",
                    "man",
                    "quarters",
                    "tampa",
                    "contain",
                    "scram",
                    "cub",
                )
            ):
                if s not in defense and len(defense) < 8:
                    defense.append(s[:220])

    n_ok = sum(1 for s in sources if s.fetched)
    conf = "low"
    if n_ok >= 3 and (patch_notes or suggestions):
        conf = "high"
    elif n_ok >= 1 and (patch_notes or suggestions or offense or defense):
        conf = "medium"

    return MetaScoutResult(
        available=n_ok > 0
        and bool(patch_notes or suggestions or offense or defense),
        offline=False,
        from_cache=False,
        confidence=conf,
        patch_notes=patch_notes[:5],
        meta_offense=offense[:8],
        meta_defense=defense[:8],
        suggestions=suggestions[:12],
        sources=[asdict(s) for s in sources],
        fetched_at=_now_iso(),
        message="" if n_ok else "All fetches failed",
    )


def _load_cache() -> dict[str, Any] | None:
    path = cache_path()
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache(result: MetaScoutResult) -> None:
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cached_at": result.fetched_at or _now_iso(),
            "result": result.to_dict(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _result_from_cache(cached: dict[str, Any]) -> MetaScoutResult | None:
    raw = cached.get("result") or {}
    if not raw:
        return None
    try:
        allowed = MetaScoutResult.__dataclass_fields__
        r = MetaScoutResult(**{k: raw[k] for k in allowed if k in raw})
    except TypeError:
        return None
    r.from_cache = True
    r.available = bool(
        r.patch_notes or r.meta_offense or r.meta_defense or r.suggestions
    )
    return r


def _cache_fresh(cached: dict[str, Any]) -> bool:
    ts = cached.get("cached_at") or ""
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = (
            datetime.now(timezone.utc) - dt.astimezone(timezone.utc)
        ).total_seconds()
        return age < CACHE_TTL_SECONDS
    except Exception:  # noqa: BLE001
        return False


def unavailable_result(
    *, offline: bool = False, message: str = ""
) -> MetaScoutResult:
    msg = message or (
        "Scout unavailable — using cached/baseline cfb27-2026-09"
        if not offline
        else "Offline — using baseline cfb27-2026-09"
    )
    return MetaScoutResult(
        available=False,
        offline=offline,
        from_cache=False,
        confidence="low",
        baseline_fallback=BASELINE_VERSION,
        fetched_at=_now_iso(),
        message=msg,
    )


def run_meta_scout(
    *,
    offline: bool = False,
    refresh: bool = False,
    urls: list[str] | None = None,
) -> MetaScoutResult:
    """Fetch/parse trusted URLs (or reuse cache). Never raises; never blocks >~15s."""
    if offline:
        cached = _load_cache()
        if cached:
            r = _result_from_cache(cached)
            if r:
                r.offline = True
                r.message = (
                    "Scout unavailable — using cached/baseline cfb27-2026-09"
                    if not r.available
                    else "Offline — using cached scout (<6h or last good)"
                )
                return r
        return unavailable_result(offline=True)

    cached = _load_cache()
    if cached and not refresh and _cache_fresh(cached):
        r = _result_from_cache(cached)
        if r:
            r.message = r.message or "Using cached scout (<6h)"
            return r

    url_list = list(urls or TRUSTED_URLS)
    # Cap live fetches so prep never hangs >~12s (prefer concrete patch pages first)
    if urls is None:
        url_list = url_list[:6]
    sources: list[MetaSource] = []
    bodies: dict[str, str] = {}
    t0 = time.monotonic()
    for url in url_list:
        elapsed = time.monotonic() - t0
        remaining = TOTAL_FETCH_BUDGET - elapsed
        if remaining <= 0.4:
            sources.append(
                MetaSource(
                    url=url, error="skipped: total fetch budget exhausted"
                )
            )
            continue
        timeout = min(PER_URL_TIMEOUT, max(0.8, remaining))
        src, body = _fetch_one(url, timeout)
        sources.append(src)
        if body:
            bodies[url] = body

    result = _parse_fetched(sources, bodies)
    if not result.available:
        if cached:
            stale = _result_from_cache(cached)
            if stale and (
                stale.patch_notes
                or stale.meta_offense
                or stale.meta_defense
                or stale.suggestions
            ):
                stale.message = (
                    "Scout unavailable — using cached/baseline cfb27-2026-09"
                )
                stale.from_cache = True
                return stale
        result.message = "Scout unavailable — using cached/baseline cfb27-2026-09"
        result.baseline_fallback = BASELINE_VERSION
        _save_cache(result)
        return result

    result.message = "Live scout OK"
    _save_cache(result)
    return result


def map_scout_to_book_suggestions(
    result: MetaScoutResult,
    *,
    dynasty: str | None = None,
) -> list[dict[str, Any]]:
    """Soft suggestions in Aidan book language; dynasty-aware freeness."""
    from cfb_coach.dynasty import (
        DEFAULT_DYNASTY,
        allow_experimental,
        normalize_dynasty,
    )

    mode = normalize_dynasty(dynasty or DEFAULT_DYNASTY)
    freer = allow_experimental(mode)
    out: list[dict[str, Any]] = []
    for sug in result.suggestions:
        item = dict(sug)
        item["badge"] = "Meta-grounded (live scout)"
        mh = (item.get("macro_hint") or "").upper()
        if not freer and mh in _EXPERIMENTAL_MACROS:
            item["alabama_note"] = (
                f"Alabama: do not break proven-8 for {mh} without explicit swap plan"
            )
            item["actionable"] = False
        else:
            item["actionable"] = True
            if freer and mh:
                item["ohio_state_note"] = (
                    f"Ohio State: freer to test meta-grounded {mh} experimentally"
                )
        out.append(item)
    return out


def apply_scout_bias(
    deltas: list[dict[str, Any]],
    result: MetaScoutResult,
    *,
    dynasty: str | None = None,
    tips: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Lightly bias gameplan/macro deltas from scout hits.

    Returns (deltas, tips, affect_lines).
    Rules preserved: deltas only, 8 active macros, CPU O-only (caller),
    alabama vs ohio_state freeness.
    """
    from cfb_coach.dynasty import (
        DEFAULT_DYNASTY,
        allow_experimental,
        normalize_dynasty,
    )
    from cfb_coach.macros import META_GROUNDED

    mode = normalize_dynasty(dynasty or DEFAULT_DYNASTY)
    freer = allow_experimental(mode)
    tips = list(tips or [])
    affect: list[str] = []
    new_deltas = list(deltas)
    existing_keys = {
        (d.get("action", ""), d.get("target", ""), d.get("field", ""))
        for d in new_deltas
    }

    suggestions = map_scout_to_book_suggestions(result, dynasty=mode)
    if not result.available and not result.from_cache:
        affect.append(
            f"Scout unavailable — deltas use baseline {result.baseline_fallback}"
        )
        result.affect_this_prep = affect
        return new_deltas, tips, affect

    patch_blob = " ".join(
        " ".join(p.get("bullets") or []) + " " + (p.get("title") or "")
        for p in result.patch_notes
    ).lower()

    if "contain" in patch_blob or any(
        "contain" in (s.get("tip") or "").lower() for s in suggestions
    ):
        tip = (
            "Live scout/patch: contain adj respected — "
            "SCRAM / contain integrity elevated"
        )
        if tip not in tips:
            tips.append(tip)
        affect.append(
            "Patch contain fix → elevate SCRAM readiness (no loadout break)"
        )
        key = ("EDIT", "SCRAM", "when_to_arm")
        if key not in existing_keys:
            new_deltas.append(
                {
                    "action": "EDIT",
                    "kind": "macro",
                    "target": "SCRAM",
                    "field": "when_to_arm",
                    "before": "",
                    "after": (
                        "After scramble/contain tells — "
                        "prioritize SCRAM (live scout / patch)"
                    ),
                    "detail": (
                        "Meta-grounded (live scout) — SCRAM readiness elevated "
                        "after contain patch"
                    ),
                    "why": "Meta-grounded (live scout) — patch contain fix",
                    "validated_status": META_GROUNDED,
                    "meta_scout": True,
                }
            )
            existing_keys.add(key)

    if "run-action" in patch_blob or "blocking" in patch_blob:
        tip = (
            "Live scout/patch: run-action blocking improved — "
            "lean IZ / HB Base early vs two-high"
        )
        if tip not in tips:
            tips.append(tip)
        affect.append(
            "Run-action blocking patch → slight O run bias in call tips"
        )

    if "spin" in patch_blob:
        tip = (
            "Live scout/patch: QB spin nerfed behind LOS — "
            "less panic contain; keep base zones"
        )
        if tip not in tips:
            tips.append(tip)
        affect.append("QB spin nerf → fewer panic contain sells")

    for sug in suggestions:
        tip = f"{sug.get('badge', 'Meta-grounded (live scout)')}: {sug.get('tip')}"
        if tip not in tips and len(tips) < 14:
            tips.append(tip)
        mh = (sug.get("macro_hint") or "").upper()
        label = sug.get("label") or mh or "meta"
        if sug.get("actionable") and freer and mh in _EXPERIMENTAL_MACROS:
            key = ("ADD", mh, "recipe")
            if key not in existing_keys:
                new_deltas.append(
                    {
                        "action": "ADD",
                        "kind": "macro",
                        "target": mh,
                        "field": "recipe",
                        "before": "",
                        "after": "experimental active (ohio_state lab)",
                        "detail": (
                            f"Meta-grounded (live scout) experimental test — "
                            f"{label}. Ohio State freer; requires swap at 8/8."
                        ),
                        "why": (
                            f"Meta-grounded (live scout) — "
                            f"{(sug.get('tip') or '')[:120]}"
                        ),
                        "validated_status": META_GROUNDED,
                        "meta_scout": True,
                    }
                )
                existing_keys.add(key)
                affect.append(
                    f"Ohio State: propose experimental {mh} "
                    "(meta-grounded live scout)"
                )
        elif not sug.get("actionable") and mh:
            affect.append(
                f"Alabama: live-meta hit on {mh}/{label} noted — "
                "no proven-8 break without swap"
            )
        else:
            affect.append(f"Soft bias: {label}")

    seen_a: set[str] = set()
    affect_u: list[str] = []
    for a in affect:
        if a not in seen_a:
            seen_a.add(a)
            affect_u.append(a)
    affect_u = affect_u[:12]
    result.affect_this_prep = affect_u
    return new_deltas, tips, affect_u


def format_scout_text(result: MetaScoutResult) -> str:
    """Compact terminal dump of scout section."""
    lines = ["## Live meta scout"]
    if not result.available and not (
        result.patch_notes or result.meta_offense or result.meta_defense
    ):
        lines.append(
            f"  {result.message or 'Scout unavailable — using cached/baseline cfb27-2026-09'}"
        )
        return "\n".join(lines)
    tag = []
    if result.offline:
        tag.append("offline")
    if result.from_cache:
        tag.append("cached")
    tag.append(f"confidence={result.confidence}")
    lines.append(f"  ({', '.join(tag)})")
    if result.patch_notes:
        lines.append("  Patch radar:")
        for p in result.patch_notes:
            lines.append(
                f"    - {p.get('version') or '?'} {p.get('date') or ''} — "
                f"{p.get('title') or ''}"
            )
            for b in (p.get("bullets") or [])[:4]:
                lines.append(f"        • {b}")
    if result.meta_offense:
        lines.append("  What's meta (O):")
        for t in result.meta_offense[:5]:
            lines.append(f"    - {t}")
    if result.meta_defense:
        lines.append("  What's meta (D):")
        for t in result.meta_defense[:5]:
            lines.append(f"    - {t}")
    if result.affect_this_prep:
        lines.append("  How it affects THIS prep:")
        for t in result.affect_this_prep:
            lines.append(f"    - {t}")
    srcs = [s for s in result.sources if s.get("fetched")]
    if srcs:
        lines.append("  Sources:")
        for s in srcs[:6]:
            lines.append(
                f"    - {s.get('title') or s.get('url')} <{s.get('url')}>"
            )
    return "\n".join(lines)
