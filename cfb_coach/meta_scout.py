"""Live meta scout — fetch the CURRENT CFB 27 meta on every prep (when online).

On prep (unless --offline) the scout fetches real, free, no-key sources in
parallel with urllib (standard library only): EA title-update pages (plus any
newer title update discovered on the EA news list), MP1st / UpdateCrazy patch
notes, MaddenTurf + MaddenProdigy meta guides, r/NCAAFBseries search RSS and
Google News RSS (which also surfaces YouTube meta videos by title).

Freshness: the cache (~/.cfb-coach/meta_cache.json) is reused only if it was
fetched TODAY (local date) and < CACHE_TTL_SECONDS old; ``--refresh-meta``
forces a fetch. Offline / all fetches failed -> last good cache -> seed
baseline (cfb_coach/data/meta_baseline.json ``meta_research``).

Besides tips, the scout extracts structured concept signals (general + red
zone / goal line) that ``cfb_coach.meta_align`` turns into small, capped priors,
and a ``changes_since_last`` diff vs the previous cache.

Never hangs the prep path: per-URL timeout + total fetch budget.
"""

from __future__ import annotations

import html as html_mod
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASELINE_VERSION = "cfb27-2026-09"
CACHE_TTL_SECONDS = 6 * 3600
PER_URL_TIMEOUT = 4.0
TOTAL_FETCH_BUDGET = 12.0
MAX_PARALLEL = 8
MAX_DISCOVERED = 2  # newer EA title-update pages found on the news list
USER_AGENT = (
    "Mozilla/5.0 (compatible; cfb-coach-meta-scout/1.12; "
    "+https://github.com/aferna6-cell/cfb-coach)"
)
MAX_BODY = 600_000

_GN = "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q="
_RD = "https://www.reddit.com/r/NCAAFBseries/search.rss?sort=new&restrict_sr=1&t=month&q="
EA_NEWS_URL = "https://www.ea.com/games/ea-sports-college-football/college-football-27/news"

# Real, free sources (no API keys). kind: patch | guide | rss | index
SOURCES: list[dict[str, str]] = [
    {"id": "ea_news", "kind": "index", "label": "EA SPORTS CFB 27 news (title updates)", "url": EA_NEWS_URL},
    {"id": "ea_tu_0903", "kind": "patch", "label": "EA Title Update Sep 3 2026",
     "url": EA_NEWS_URL + "/title-update-september-3rd-2026"},
    {"id": "mp1st_1012", "kind": "patch", "label": "MP1st CFB 27 Update 1.012",
     "url": "https://mp1st.com/title-updates-and-patches/college-football-27-update-1-012-september-22-brings-gameplay-changes"},
    {"id": "updatecrazy_1012", "kind": "patch", "label": "UpdateCrazy CFB 27 1.012 notes",
     "url": "https://updatecrazy.com/ea-college-football-27-cfb-27-update-1-012-patch-notes/"},
    {"id": "maddenturf_books", "kind": "guide", "label": "MaddenTurf best CFB 27 playbooks",
     "url": "https://maddenturf.com/cfb-27-best-playbooks/"},
    {"id": "maddenprodigy_offense", "kind": "guide", "label": "MaddenProdigy CFB 27 offense guide",
     "url": "https://www.maddenprodigy.com/college-football-27-offense-guide/"},
    {"id": "reddit_meta", "kind": "rss", "label": "r/NCAAFBseries: meta / playbook (past month)",
     "url": _RD + "meta+OR+playbook+OR+%22best+plays%22"},
    {"id": "reddit_rz", "kind": "rss", "label": "r/NCAAFBseries: red zone / goal line (past month)",
     "url": _RD + "%22red+zone%22+OR+%22goal+line%22+OR+%22inside+the+5%22"},
    {"id": "gnews_meta", "kind": "rss", "label": "Google News: CFB 27 meta / playbooks / patches (30d, incl. YouTube)",
     "url": _GN + "%22College+Football+27%22+(meta+OR+playbook+OR+%22red+zone%22+OR+%22goal+line%22+OR+%22title+update%22)+when:30d"},
]
TRUSTED_URLS: list[str] = [s["url"] for s in SOURCES]
_SOURCE_BY_URL = {s["url"]: s for s in SOURCES}

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
    label: str = ""
    kind: str = ""


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
    # v1.12: structured signals + freshness
    mode: str = ""  # live | cache | seed
    concept_signals: dict[str, int] = field(default_factory=dict)
    rz_signals: dict[str, int] = field(default_factory=dict)
    headlines: list[dict[str, Any]] = field(default_factory=list)
    changes_since_last: list[str] = field(default_factory=list)
    previous_fetched_at: str = ""
    ttl_policy: str = ""

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
    info = _SOURCE_BY_URL.get(url) or {}
    src = MetaSource(url=url, label=info.get("label", ""), kind=info.get("kind", ""))
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/rss+xml,application/atom+xml;q=0.9,*/*;q=0.8",
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


_SCRIPT_RE = re.compile(r"<(script|style|noscript|svg)\b[^>]*>.*?</\1\s*>", re.I | re.S)
_META_TAG_RE = re.compile(r"<meta\b[^>]{0,2000}>", re.I)
_ATTR_RE = re.compile(r'([a-zA-Z:-]+)\s*=\s*["\']([^"\']*)["\']')
_SNIPPET_CACHE: dict[tuple[int, int, int], list[str]] = {}


def _meta_description(body: str) -> str:
    for m in _META_TAG_RE.finditer(body):
        attrs = {k.lower(): v for k, v in _ATTR_RE.findall(m.group(0))}
        if (attrs.get("name") or attrs.get("property") or "").lower() in ("description", "og:description"):
            return _strip_html(attrs.get("content", ""))
    return ""


def _extract_snippets(body: str, limit: int = 24) -> list[str]:
    key = (hash(body), len(body), limit)
    hash_key = key
    if key in _SNIPPET_CACHE:
        return list(_SNIPPET_CACHE[key])
    if len(_SNIPPET_CACHE) > 64:
        _SNIPPET_CACHE.clear()
    desc = _meta_description(body)
    body = _SCRIPT_RE.sub(" ", body)
    bits: list[str] = []
    if desc:
        bits.append(desc)
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
    _SNIPPET_CACHE[(hash_key[0], hash_key[1], limit)] = list(out)
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


def _parse_iso(ts: str) -> datetime | None:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def local_ts(ts: str) -> str:
    """ISO timestamp -> local wall clock ("2026-09-27 13:21 EDT") for messages."""
    dt = _parse_iso(ts or "")
    if dt is None:
        return ts or "?"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _cache_fresh(cached: dict[str, Any], *, now: datetime | None = None) -> bool:
    """Fresh = fetched TODAY (local calendar date) AND younger than the TTL."""
    dt = _parse_iso(cached.get("cached_at") or "")
    if dt is None:
        return False
    now = now or datetime.now(timezone.utc)
    age = (now - dt.astimezone(timezone.utc)).total_seconds()
    same_day = dt.astimezone().date() == now.astimezone().date()
    return 0 <= age < CACHE_TTL_SECONDS and same_day


# ---------------------------------------------------------------------------
# Structured signals (feed cfb_coach.meta_align priors)
# ---------------------------------------------------------------------------

SIGNAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "bunch": re.compile(r"\bbunch\b", re.I),
    "cluster": re.compile(r"\bcluster\b", re.I),
    "deuce": re.compile(r"deuce\s*close|singleback\s*deuce", re.I),
    "run_first": re.compile(r"run[-\s]first|run[-\s]heavy|running\s+game|run[-\s]action|run\s+blocking", re.I),
    "inside_zone": re.compile(r"\b(?i:inside\s+zone|split\s+zone)\b|\bIZ\b"),
    "duo_power": re.compile(r"\bduo\b|\bpower\s+run|\bcounter\b", re.I),
    "dive": re.compile(r"\b(?:hb|single\s*back|singleback)\s+dive\b|\bdive\b", re.I),
    "mesh": re.compile(r"\bmesh\b|\bdrags?\b|\bshallow\b", re.I),
    "whip": re.compile(r"\bwhip\b", re.I),
    "spot_flat": re.compile(r"\bspot\b|\bflats?\b|quick\s+outs?|\bstick\b|\bslants?\b", re.I),
    "rpo": re.compile(r"\brpos?\b", re.I),
    "play_action": re.compile(r"(?i:play[-\s]action)|\bPA\b"),
    "verticals": re.compile(r"\bverticals\b|four\s+verts|4\s+verts|\bhero\s+ball\b", re.I),
    "man_coverage": re.compile(r"\bman\s+coverage\b|\bpress\s+man\b|\bman\s+press\b", re.I),
    "cpu_gl_wall": re.compile(r"force\s+field|goal[-\s]?line\s+(?:wall|stand)|stuffed\s+at\s+the\s+1", re.I),
}
_RZ_CONTEXT = re.compile(r"red[-\s]?zone|goal[-\s]?line|inside\s+the\s+(?:5|five|10|ten|20)|goal\s+to\s+go|short\s+yardage", re.I)
RZ_WINDOW_BEFORE = 120
RZ_WINDOW_AFTER = 450
_CFB27 = re.compile(r"college\s+football\s+27|\bcfb\s*27\b|\bcfb27\b|ncaa\s*27", re.I)


def extract_signals(texts: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """Per-text (≈ per source/post) concept hits; rz = concept in a red-zone sentence.

    Counting documents rather than raw mentions keeps one long page from
    dominating the signal.
    """
    gen: dict[str, int] = {}
    rz: dict[str, int] = {}
    for text in texts:
        if not text:
            continue
        hit_g: set[str] = set()
        hit_rz: set[str] = set()
        for name, rx in SIGNAL_PATTERNS.items():
            if rx.search(text):
                hit_g.add(name)
        for m in _RZ_CONTEXT.finditer(text):
            # window: a little before the mention, a paragraph after it
            window = text[max(0, m.start() - RZ_WINDOW_BEFORE): m.end() + RZ_WINDOW_AFTER]
            for name, rx in SIGNAL_PATTERNS.items():
                if rx.search(window):
                    hit_rz.add(name)
        for n in hit_g:
            gen[n] = gen.get(n, 0) + 1
        for n in hit_rz:
            rz[n] = rz.get(n, 0) + 1
    return dict(sorted(gen.items())), dict(sorted(rz.items()))


def parse_feed(body: str) -> list[dict[str, str]]:
    """RSS 2.0 or Atom -> [{title, link, date, text}] (stdlib xml, regex fallback)."""
    import xml.etree.ElementTree as ET

    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        root = None
    if root is not None:
        atom = "{http://www.w3.org/2005/Atom}"
        for it in root.iter("item"):
            items.append(
                {
                    "title": _strip_html(it.findtext("title") or ""),
                    "link": (it.findtext("link") or "").strip(),
                    "date": (it.findtext("pubDate") or "").strip(),
                    "text": _strip_html(it.findtext("description") or ""),
                }
            )
        for e in root.iter(atom + "entry"):
            link_el = e.find(atom + "link")
            items.append(
                {
                    "title": _strip_html(e.findtext(atom + "title") or ""),
                    "link": (link_el.get("href") if link_el is not None else "") or "",
                    "date": (e.findtext(atom + "updated") or "").strip(),
                    "text": _strip_html(e.findtext(atom + "content") or "")[:1500],
                }
            )
        return items
    for m in re.finditer(r"<item>(.*?)</item>", body, re.S | re.I):
        chunk = m.group(1)
        t = re.search(r"<title>(.*?)</title>", chunk, re.S)
        items.append({"title": _strip_html(t.group(1) if t else ""), "link": "", "date": "", "text": ""})
    return items


def _item_date(raw: str) -> datetime | None:
    from email.utils import parsedate_to_datetime

    if not raw:
        return None
    dt = _parse_iso(raw)
    if dt:
        return dt
    try:
        dt = parsedate_to_datetime(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError):
        return None


def _discover_title_updates(body: str, known: set[str]) -> list[str]:
    """Newer EA title-update pages linked from the EA news list (page order = newest first)."""
    out: list[str] = []
    for m in re.finditer(r'href="(/games/ea-sports-college-football/college-football-27/news/[a-z0-9-]*(?:title-update|patch)[a-z0-9-]*)"', body):
        url = "https://www.ea.com" + m.group(1)
        if url not in known and url not in out:
            out.append(url)
        if len(out) >= MAX_DISCOVERED:
            break
    return out


def _fetch_parallel(urls: list[str], budget: float) -> tuple[list[MetaSource], dict[str, str]]:
    sources: dict[str, MetaSource] = {}
    bodies: dict[str, str] = {}
    if not urls:
        return [], {}
    t0 = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(urls)))
    futs = {pool.submit(_fetch_one, u, PER_URL_TIMEOUT): u for u in urls}
    done, not_done = wait(futs, timeout=max(0.5, budget))
    for f in done:
        u = futs[f]
        try:
            src, body = f.result()
        except Exception as e:  # noqa: BLE001
            src, body = MetaSource(url=u, error=f"{type(e).__name__}: {e}"), ""
        sources[u] = src
        if body:
            bodies[u] = body
    for f in not_done:
        u = futs[f]
        info = _SOURCE_BY_URL.get(u) or {}
        sources[u] = MetaSource(url=u, error=f"skipped: fetch budget {budget:.0f}s exhausted",
                                label=info.get("label", ""), kind=info.get("kind", ""))
    pool.shutdown(wait=False, cancel_futures=True)
    _ = time.monotonic() - t0
    return [sources[u] for u in urls], bodies


def _headlines_and_texts(
    sources: list[MetaSource], bodies: dict[str, str], *, now: datetime, max_age_days: int = 45
) -> tuple[list[dict[str, Any]], list[str]]:
    heads: list[dict[str, Any]] = []
    texts: list[str] = []
    for src in sources:
        body = bodies.get(src.url) or ""
        if not body:
            continue
        if src.kind == "rss" or body.lstrip().startswith("<?xml") or "<rss" in body[:500]:
            reddit = "reddit.com" in src.url
            for it in parse_feed(body):
                title = it.get("title") or ""
                dt = _item_date(it.get("date") or "")
                if dt and (now - dt.astimezone(timezone.utc)).days > max_age_days:
                    continue
                blob = f"{title}. {it.get('text') or ''}"
                if not reddit and not _CFB27.search(blob):
                    continue  # Google News: must be about CFB 27
                heads.append(
                    {
                        "title": title[:200],
                        "link": it.get("link") or "",
                        "date": dt.date().isoformat() if dt else "",
                        "source": src.label or src.url,
                    }
                )
                texts.append(blob)
        elif src.kind != "index":
            full = _strip_html(_SCRIPT_RE.sub(" ", body))
            texts.append(f"{src.title}. {full[:60000]}")
    heads.sort(key=lambda h: h.get("date") or "", reverse=True)
    return heads[:30], texts


def _diff_results(prev: MetaScoutResult | None, cur: MetaScoutResult) -> list[str]:
    if prev is None:
        return ["First meta fetch — no previous cache to compare."]
    out: list[str] = []
    old_v = {p.get("version") for p in prev.patch_notes}
    for p in cur.patch_notes:
        v = p.get("version")
        if v and v != "unknown" and v not in old_v:
            out.append(f"New patch notes: {v} — {p.get('title') or ''}".strip())
    old_titles = {h.get("title") for h in prev.headlines}
    new_heads = [h for h in cur.headlines if h.get("title") not in old_titles]
    for h in new_heads[:6]:
        out.append(f"New: {h.get('title')} ({h.get('source')}, {h.get('date') or 'undated'})")
    for label, a, b in (("", prev.concept_signals, cur.concept_signals), ("red-zone ", prev.rz_signals, cur.rz_signals)):
        for k in sorted(set(a) | set(b)):
            d = b.get(k, 0) - a.get(k, 0)
            if abs(d) >= 2:
                out.append(f"{label}signal '{k}' {'up' if d > 0 else 'down'} {a.get(k, 0)} → {b.get(k, 0)}")
    old_ok = {s.get("url") for s in prev.sources if s.get("fetched")}
    new_ok = {s.get("url") for s in cur.sources if s.get("fetched")}
    for u in sorted(new_ok - old_ok):
        out.append(f"Source now reachable: {u}")
    for u in sorted(old_ok - new_ok):
        out.append(f"Source unreachable this time: {u}")
    return out[:14] or ["No material change since the last fetch."]


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
        mode="seed",
    )


def seed_result(*, offline: bool = False, message: str = "") -> MetaScoutResult:
    """Fallback built from the seed baseline's cited ``meta_research`` section."""
    try:
        from cfb_coach.gameplan import load_baseline

        research = load_baseline().get("meta_research") or {}
    except Exception:  # noqa: BLE001
        research = {}
    findings = research.get("findings") or []
    texts = [f.get("claim", "") for f in findings]
    gen, rz = extract_signals(texts)
    r = unavailable_result(offline=offline, message=message)
    r.mode = "seed"
    r.concept_signals = gen
    r.rz_signals = rz
    r.fetched_at = research.get("updated", "")
    r.sources = [
        {"url": f.get("url", ""), "title": f.get("source", ""), "fetched": False, "status": None,
         "error": "", "label": f"seed research ({f.get('published', '')})", "kind": "seed"}
        for f in findings
    ]
    return r


def run_meta_scout(
    *,
    offline: bool = False,
    refresh: bool = False,
    urls: list[str] | None = None,
    now: datetime | None = None,
) -> MetaScoutResult:
    """Fetch/parse live sources (or reuse today's cache). Never raises; bounded time.

    Order: fresh-today cache (unless ``refresh``) -> live fetch -> last good cache
    -> seed research. ``offline`` skips the network entirely.
    """
    now = now or datetime.now(timezone.utc)
    ttl = f"reuse only if fetched today and < {CACHE_TTL_SECONDS // 3600}h old; --refresh-meta forces"
    cached = _load_cache()
    prev = _result_from_cache(cached) if cached else None

    if offline:
        if prev and (prev.available or prev.concept_signals):
            prev.offline = True
            prev.mode = "cache"
            prev.ttl_policy = ttl
            prev.message = f"Offline — using cached meta from {local_ts(prev.fetched_at or cached.get('cached_at', ''))}"
            return prev
        r = seed_result(offline=True, message="Offline — no cache; using seed research " + BASELINE_VERSION)
        r.ttl_policy = ttl
        return r

    if cached and prev and not refresh and _cache_fresh(cached, now=now):
        prev.mode = "cache"
        prev.ttl_policy = ttl
        prev.message = f"Using today's cached meta (fetched {local_ts(prev.fetched_at)}); --refresh-meta to force"
        return prev

    url_list = list(urls or TRUSTED_URLS)
    t0 = time.monotonic()
    sources, bodies = _fetch_parallel(url_list, TOTAL_FETCH_BUDGET)
    # Phase 2: follow newer EA title updates linked from the news list
    if urls is None and EA_NEWS_URL in bodies:
        extra = _discover_title_updates(bodies[EA_NEWS_URL], set(url_list))
        left = TOTAL_FETCH_BUDGET - (time.monotonic() - t0)
        if extra and left > 1.0:
            s2, b2 = _fetch_parallel(extra, left)
            for s_ in s2:
                s_.kind = s_.kind or "patch"
                s_.label = s_.label or "EA title update (discovered)"
            sources += s2
            bodies.update(b2)

    page_sources = [s_ for s_ in sources if s_.kind not in ("rss", "index")]
    result = _parse_fetched(page_sources, bodies)
    result.sources = [asdict(s_) for s_ in sources]
    heads, texts = _headlines_and_texts(sources, bodies, now=now)
    result.headlines = heads
    result.concept_signals, result.rz_signals = extract_signals(texts)
    n_ok = sum(1 for s_ in sources if s_.fetched)
    result.available = n_ok > 0 and bool(
        result.patch_notes or result.suggestions or result.meta_offense or heads or result.concept_signals
    )
    result.ttl_policy = ttl
    if not result.available:
        if prev and (prev.available or prev.concept_signals):
            prev.from_cache = True
            prev.mode = "cache"
            prev.ttl_policy = ttl
            prev.message = f"Live fetch failed — using last good cached meta from {local_ts(prev.fetched_at)}"
            return prev
        r = seed_result(message="Live fetch failed and no cache — using seed research " + BASELINE_VERSION)
        r.ttl_policy = ttl
        r.sources = result.sources + r.sources
        return r

    result.mode = "live"
    result.previous_fetched_at = prev.fetched_at if prev else ""
    result.changes_since_last = _diff_results(prev, result)
    result.message = f"Live meta refreshed ({n_ok}/{len(sources)} sources)"
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
    lines.append(
        f"  mode={result.mode or '?'} fetched={result.fetched_at or '?'}"
        + (f" (previous {result.previous_fetched_at})" if result.previous_fetched_at else "")
        + (f" — {result.message}" if result.message else "")
    )
    if result.changes_since_last:
        lines.append("  Changed since last cache:")
        lines += [f"    - {c}" for c in result.changes_since_last[:8]]
    if result.headlines:
        lines.append("  Recent headlines:")
        lines += [f"    - {h.get('date') or ''} {h.get('title')} [{h.get('source')}]" for h in result.headlines[:6]]
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
