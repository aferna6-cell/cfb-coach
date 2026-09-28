"""YouTube research for the CFB 27 meta — no API key, stdlib first.

Discovery (all free, no key):
  * YouTube search results pages (``/results?search_query=...``) — the
    ``ytInitialData`` JSON embedded in the HTML carries video id, title,
    channel, relative publish time and a description snippet.
  * Channel RSS feeds (``/feeds/videos.xml?channel_id=...``) for CFB 27 content
    creators identified by research (see ``CREATOR_CHANNELS``) — exact dates.

Transcripts (free; tiers tried in order, each degrades gracefully):
  1. stdlib InnerTube: POST ``/youtubei/v1/player`` with the iOS client context
     -> caption track list -> ``timedtext`` JSON3 (English, auto captions OK).
  2. ``youtube-transcript-api`` (optional dependency) when importable.
  3. ``yt-dlp`` auto-captions (optional; only when on PATH and time remains).

YouTube often blocks cloud / datacenter IPs ("Sign in to confirm you're not a
bot", HTTP 429). Every tier has a circuit breaker so one block doesn't burn the
prep budget; transcripts are cached per video id forever (they don't change) and
"no English captions" results are cached for a day. Nothing here raises.
"""

from __future__ import annotations

import html as html_mod
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

# --- Constants ---------------------------------------------------------------
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
IOS_CLIENT = {
    "clientName": "IOS",
    "clientVersion": "20.10.4",
    "deviceModel": "iPhone16,2",
    "hl": "en",
    "gl": "US",
}
IOS_UA = "com.google.ios.youtube/20.10.4 (iPhone16,2; U; CPU iOS 18_3_2 like Mac OS X;)"
PLAYER_URL = "https://www.youtube.com/youtubei/v1/player?prettyPrint=false"
SEARCH_URL = "https://www.youtube.com/results?search_query={q}&sp={sp}"
FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
SP_THIS_MONTH = "EgIIBA%253D%253D"  # upload date: this month
SP_RELEVANCE = ""

SEARCH_QUERIES = [
    ("college football 27 meta offense playbook", SP_THIS_MONTH),
    ("college football 27 red zone goal line plays", SP_THIS_MONTH),
    ("cfb 27 best formation plays", SP_RELEVANCE),
]

# CFB 27 creators found by research on 2026-09-27 (YouTube search for
# "college football 27 meta playbook" + MaddenTurf's creator roundup). Channel
# ids come from the search results' browse endpoints. Mixed CFB/Madden
# channels are fine: every video is filtered by title/description/date.
CREATOR_CHANNELS: dict[str, str] = {
    "Civil": "UC2XvsnNKSXGJI5-3o37fHQA",
    "WinCollegeFootball": "UCesd34L4eWHdxUyoSSKJQlg",
    "Swolosimo CFB": "UCkFi3zksDGtwMzTT-TClwJQ",
    "Goated Millz": "UCq0E2VxkuCLW030arg74_jA",
    "HuddleGG": "UCIYBt_UqUqiXs-RZDYLC0sw",
    "CSwee123": "UCRUxNru13ZjOunaqF08vmqA",
    "Ace Madden": "UCHgq1lY94IJSyT3vs6Ao0nA",
    "VENM Fire": "UC5IxYfT8P4YPhCq4q0-6H_w",
    "9to5erz": "UC5V0Qfz48QlzQ9zgVluIbAw",
    "C3Gaming": "UCcQPyX9zYMfjSAPbncRlMuw",
}

CFB27_RELEASE = datetime(2026, 7, 1, tzinfo=timezone.utc)
MAX_VIDEO_AGE_DAYS = 75
MAX_TRANSCRIPTS = 8
PER_REQUEST_TIMEOUT = 5.0
NO_CAPTION_TTL = 24 * 3600
TRANSCRIPT_MAX_CHARS = 60_000

_CFB27_RX = re.compile(r"college\s*football\s*27|\bcfb\s*-?\s*27\b|\bcfb27\b|\bncaa\s*27\b", re.I)
_OLD_TITLE_RX = re.compile(
    r"college\s*football\s*2[3-6]\b|\bcfb\s*-?\s*2[3-6]\b|\bcfb2[3-6]\b|\bncaa\s*1[0-4]\b|\bmadden\b", re.I
)
_CFB_GENERIC_RX = re.compile(r"college\s*football|\bcfb\b", re.I)
_RELEVANT_RX = re.compile(
    r"offen[cs]e|playbook|formation|\bplays?\b|meta|red\s*zone|goal\s*line|scheme|run\s*game|"
    r"passing|money\s*play|audible|bunch|mesh|beat\s+any|unstoppable|best", re.I
)
_RZ_TITLE_RX = re.compile(r"red\s*zone|goal\s*line|inside\s+the\s+(?:5|10|20)|short\s+yardage", re.I)
_DEFENSE_ONLY_RX = re.compile(r"\bdefen[cs]e\b|\bblitz\b|run\s+defen", re.I)


@dataclass(frozen=True)
class YTProfile:
    """Which game a YouTube research pass is for (CFB 27 default; Madden 27 reuses every code path)."""

    game: str
    queries: tuple[tuple[str, str], ...]
    channels: tuple[tuple[str, str], ...]
    title_rx: re.Pattern[str]
    old_rx: re.Pattern[str]
    release: datetime
    relevant_rx: re.Pattern[str]
    defense_penalty: bool = True
    desc_old_rx: re.Pattern[str] | None = None
    old_reason: str = "older title / Madden"


def cfb_profile() -> YTProfile:
    """Built at call time from the module constants (tests monkeypatch them)."""
    return YTProfile(
        game="CFB 27",
        queries=tuple(SEARCH_QUERIES),
        channels=tuple(CREATOR_CHANNELS.items()),
        title_rx=_CFB27_RX,
        old_rx=_OLD_TITLE_RX,
        release=CFB27_RELEASE,
        relevant_rx=_RELEVANT_RX,
        defense_penalty=True,
        desc_old_rx=re.compile(r"madden\s*2[5-7]", re.I),
    )


MADDEN27_RELEASE = datetime(2026, 8, 1, tzinfo=timezone.utc)
MADDEN_SEARCH_QUERIES = [
    ("madden 27 best offense playbook meta", SP_THIS_MONTH),
    ("madden 27 best defense playbook meta", SP_THIS_MONTH),
    ("madden 27 red zone plays", SP_THIS_MONTH),
    ("madden 27 best formations offense defense", SP_RELEVANCE),
]
# Madden-first creators (subset of the mixed CFB/Madden channels above — every
# video is still filtered by title/date, so CFB uploads are dropped).
MADDEN_CREATOR_CHANNELS: dict[str, str] = {
    k: CREATOR_CHANNELS[k] for k in ("HuddleGG", "Ace Madden", "VENM Fire", "Civil", "CSwee123", "9to5erz", "C3Gaming")
}
_MADDEN27_RX = re.compile(r"madden\s*(?:nfl\s*)?27\b|\bm27\b|\bmadden27\b", re.I)
_MADDEN_OLD_RX = re.compile(r"madden\s*(?:nfl\s*)?2[3-6]\b|\bmadden2[3-6]\b|college\s*football|\bcfb\b|\bncaa\b", re.I)
_MADDEN_RELEVANT_RX = re.compile(
    r"offen[cs]e|defen[cs]e|playbook|formation|\bplays?\b|meta|red\s*zone|goal\s*line|scheme|run\s*game|"
    r"passing|money\s*play|audible|blitz|coverage|adjust|bunch|mesh|beat\s+any|unstoppable|best|stop", re.I
)


def madden_profile() -> YTProfile:
    return YTProfile(
        game="Madden 27",
        queries=tuple(MADDEN_SEARCH_QUERIES),
        channels=tuple(MADDEN_CREATOR_CHANNELS.items()),
        title_rx=_MADDEN27_RX,
        old_rx=_MADDEN_OLD_RX,
        release=MADDEN27_RELEASE,
        relevant_rx=_MADDEN_RELEVANT_RX,
        defense_penalty=False,  # Madden prep researches BOTH sides (offense + defense book)
        desc_old_rx=re.compile(r"college\s*football\s*2[5-7]|\bcfb\s*2[5-7]", re.I),
        old_reason="older Madden title / CFB",
    )


@dataclass
class Video:
    video_id: str
    title: str
    channel: str = ""
    channel_id: str = ""
    published: str = ""  # ISO date (YYYY-MM-DD) — exact (RSS) or approximated (search)
    published_exact: bool = False
    description: str = ""
    found_via: str = ""  # search:<query> | rss:<channel>
    relevance: float = 0.0
    transcript_status: str = ""  # ok | cached | no_captions | blocked | error | skipped
    transcript_method: str = ""
    transcript_chars: int = 0
    error: str = ""

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["url"] = self.url
        return d


@dataclass
class YTResult:
    ran: bool = False
    offline: bool = False
    found: int = 0  # unique CFB 27 videos after filtering
    raw_found: int = 0  # before the CFB 27 / recency filter
    rejected_old_title: int = 0
    considered: int = 0  # transcript attempts
    transcripts: int = 0  # transcripts available (fetched now or cached)
    fetched_now: int = 0
    from_cache: int = 0
    methods: dict[str, int] = field(default_factory=dict)
    blocked: list[str] = field(default_factory=list)
    search_ok: int = 0
    search_total: int = 0
    rss_ok: int = 0
    rss_total: int = 0
    elapsed_s: float = 0.0
    videos: list[dict[str, Any]] = field(default_factory=list)
    docs: list[dict[str, Any]] = field(default_factory=list)  # {label, kind, url, date, text, title, channel}
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # docs carry full transcript text — keep the persisted/scout copy small
        d["docs"] = [{k: v for k, v in doc.items() if k != "text"} | {"chars": len(doc.get("text") or "")} for doc in self.docs]
        return d


# --- Helpers -------------------------------------------------------------------

def transcript_cache_dir() -> Path:
    env = os.environ.get("CFB_COACH_DB")
    base = Path(env).expanduser().resolve().parent if env else Path.home() / ".cfb-coach"
    d = base / "yt_transcripts"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        d = Path(tempfile.gettempdir()) / "cfb-coach-yt"
        d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, *, timeout: float, headers: dict[str, str] | None = None, data: bytes | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(getattr(resp, "status", 200) or 200), resp.read(3_000_000).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return int(e.code), ""


def parse_relative_date(text: str, now: datetime) -> str:
    """'10d ago' / '2 weeks ago' / 'Streamed 1 month ago' / '21h ago' -> ISO date (approx)."""
    t = (text or "").lower()
    m = re.search(r"(\d+)\s*(second|sec|s|minute|min|m|hour|hr|h|day|d|week|wk|w|month|mo|year|yr|y)s?\b", t)
    if not m:
        return ""
    n = int(m.group(1))
    unit = m.group(2)
    if unit in ("second", "sec", "s", "minute", "min", "m", "hour", "hr", "h"):
        delta = timedelta(hours=n if unit in ("hour", "hr", "h") else 0)
    elif unit in ("day", "d"):
        delta = timedelta(days=n)
    elif unit in ("week", "wk", "w"):
        delta = timedelta(days=7 * n)
    elif unit in ("month", "mo"):
        delta = timedelta(days=30 * n)
    else:
        delta = timedelta(days=365 * n)
    return (now - delta).date().isoformat()


def _runs_text(o: Any) -> str:
    if not isinstance(o, dict):
        return ""
    if "simpleText" in o:
        return str(o["simpleText"])
    return "".join(str(r.get("text", "")) for r in o.get("runs") or [])


def parse_search_html(body: str, *, now: datetime, query: str = "") -> list[Video]:
    """Videos from a YouTube results page (ytInitialData)."""
    m = re.search(r"var ytInitialData\s*=\s*(\{.*?\});\s*</script>", body, re.S)
    if not m:
        m = re.search(r'window\["ytInitialData"\]\s*=\s*(\{.*?\});', body, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except ValueError:
        return []
    out: list[Video] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            vr = o.get("videoRenderer")
            if isinstance(vr, dict) and vr.get("videoId"):
                owner = (vr.get("ownerText") or vr.get("longBylineText") or {})
                runs = owner.get("runs") or [{}]
                cid = (((runs[0] or {}).get("navigationEndpoint") or {}).get("browseEndpoint") or {}).get("browseId", "")
                snip = " ".join(_runs_text(s.get("snippetText") or {}) for s in vr.get("detailedMetadataSnippets") or [])
                snip = snip or _runs_text(vr.get("descriptionSnippet") or {})
                pub_raw = _runs_text(vr.get("publishedTimeText") or {})
                out.append(
                    Video(
                        video_id=vr["videoId"],
                        title=_runs_text(vr.get("title") or {}),
                        channel=_runs_text(owner),
                        channel_id=cid or "",
                        published=parse_relative_date(pub_raw, now),
                        published_exact=False,
                        description=snip,
                        found_via=f"search:{query}",
                    )
                )
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    return out


def parse_channel_feed(body: str, *, channel: str = "") -> list[Video]:
    """Atom feed from /feeds/videos.xml -> videos with exact publish dates."""
    ns = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015",
          "media": "http://search.yahoo.com/mrss/"}
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    author = root.findtext("a:title", default=channel, namespaces=ns) or channel
    out = []
    for e in root.findall("a:entry", ns):
        vid = e.findtext("yt:videoId", default="", namespaces=ns)
        if not vid:
            continue
        pub = (e.findtext("a:published", default="", namespaces=ns) or "")[:10]
        desc = e.findtext("media:group/media:description", default="", namespaces=ns) or ""
        out.append(
            Video(
                video_id=vid,
                title=html_mod.unescape(e.findtext("a:title", default="", namespaces=ns) or ""),
                channel=e.findtext("a:author/a:name", default=author, namespaces=ns) or author,
                channel_id=e.findtext("yt:channelId", default="", namespaces=ns) or "",
                published=pub,
                published_exact=True,
                description=desc[:1500],
                found_via=f"rss:{author}",
            )
        )
    return out


def _date(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s[:10]).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def classify_video(v: Video, *, now: datetime, profile: YTProfile | None = None) -> tuple[bool, str]:
    """Keep only current videos for the profile's game (CFB 27 default). Returns (keep, reason)."""
    pr = profile or cfb_profile()
    blob = f"{v.title} {v.description}"
    is27 = bool(pr.title_rx.search(v.title) or pr.title_rx.search(v.description[:400]))
    old = bool(pr.old_rx.search(v.title))
    if old and not pr.title_rx.search(v.title):
        return False, pr.old_reason
    dt = _date(v.published)
    if dt is not None and dt < pr.release:
        return False, f"published before {pr.game}"
    if dt is not None and (now - dt).days > MAX_VIDEO_AGE_DAYS:
        return False, f"older than {MAX_VIDEO_AGE_DAYS} days"
    if not is27:
        return False, f"not {pr.game}"
    if pr.desc_old_rx is not None and pr.desc_old_rx.search(v.description[:600]) and not pr.title_rx.search(v.title):
        return False, pr.old_reason
    if not pr.relevant_rx.search(v.title) and not _RZ_TITLE_RX.search(blob):
        return False, "not offense / playbook / meta"
    return True, "ok"


def relevance(v: Video, *, now: datetime, profile: YTProfile | None = None) -> float:
    pr = profile or cfb_profile()
    t = v.title
    score = 1.0
    if _RZ_TITLE_RX.search(t):
        score += 1.5
    if re.search(r"playbook|formation|offen[cs]e|meta|scheme|plays?\b", t, re.I):
        score += 1.0
    if not pr.defense_penalty and re.search(r"defen[cs]e|coverage|blitz", t, re.I):
        score += 0.6
    if pr.defense_penalty and _DEFENSE_ONLY_RX.search(t) and not re.search(r"offen[cs]e", t, re.I):
        score -= 0.8
    dt = _date(v.published)
    if dt is not None:
        age = max(0, (now - dt).days)
        score *= 0.5 ** (age / 30.0)
    else:
        score *= 0.5
    return round(score, 3)


# --- Transcript parsing --------------------------------------------------------

def parse_json3(body: str) -> str:
    try:
        j = json.loads(body)
    except ValueError:
        return ""
    parts = []
    for ev in j.get("events") or []:
        segs = ev.get("segs")
        if not segs:
            continue
        parts.append("".join(str(s.get("utf8", "")) for s in segs))
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def parse_timedtext_xml(body: str) -> str:
    """Legacy <transcript><text start=..>..</text></transcript> or srv3 <p> format."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return ""
    bits = [html_mod.unescape("".join(el.itertext())) for el in root.iter() if el.tag in ("text", "p")]
    return re.sub(r"\s+", " ", " ".join(bits)).strip()


def pick_caption_track(tracks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """English manual > English ASR (original language) > anything English."""
    en = [t for t in tracks if str(t.get("languageCode", "")).lower().startswith("en")]
    if not en:
        return None
    manual = [t for t in en if t.get("kind") != "asr"]
    return (manual or en)[0]


class _Breaker:
    def __init__(self, limit: int = 2) -> None:
        self.limit = limit
        self.fails = 0
        self.lock = threading.Lock()

    @property
    def open(self) -> bool:
        return self.fails >= self.limit

    def hit(self) -> None:
        with self.lock:
            self.fails += 1

    def ok(self) -> None:
        with self.lock:
            self.fails = 0


class Blocked(RuntimeError):
    pass


class NoCaptions(RuntimeError):
    pass


def fetch_transcript_stdlib(video_id: str, *, timeout: float = PER_REQUEST_TIMEOUT) -> str:
    body = json.dumps({"context": {"client": IOS_CLIENT}, "videoId": video_id}).encode()
    status, txt = _get(PLAYER_URL, timeout=timeout, data=body,
                       headers={"Content-Type": "application/json", "User-Agent": IOS_UA})
    if status == 429:
        raise Blocked("player HTTP 429")
    if status != 200 or not txt:
        raise RuntimeError(f"player HTTP {status}")
    d = json.loads(txt)
    ps = d.get("playabilityStatus") or {}
    if ps.get("status") in ("LOGIN_REQUIRED",) or "bot" in str(ps.get("reason", "")).lower():
        raise Blocked(f"player {ps.get('status')}: {ps.get('reason', '')}")
    tracks = ((d.get("captions") or {}).get("playerCaptionsTracklistRenderer") or {}).get("captionTracks") or []
    tr = pick_caption_track(tracks)
    if tr is None:
        raise NoCaptions("no English caption track")
    url = str(tr.get("baseUrl") or "")
    if "fmt=" not in url:
        url += "&fmt=json3"
    status, cap = _get(url, timeout=timeout, headers={"User-Agent": IOS_UA})
    if status == 429:
        raise Blocked("timedtext HTTP 429")
    if status != 200 or not cap:
        raise RuntimeError(f"timedtext HTTP {status}")
    text = parse_json3(cap) if cap.lstrip().startswith("{") else parse_timedtext_xml(cap)
    if not text:
        raise RuntimeError("empty caption body")
    return text


def fetch_transcript_ytapi(video_id: str) -> str:
    """Optional youtube-transcript-api (>=1.0 instance API, falls back to the 0.x API)."""
    import youtube_transcript_api as yta  # noqa: PLC0415 — optional dependency

    try:
        api = yta.YouTubeTranscriptApi()
        if hasattr(api, "fetch"):
            t = api.fetch(video_id, languages=["en", "en-US"])
            return re.sub(r"\s+", " ", " ".join(s.text for s in t.snippets)).strip()
        rows = yta.YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US"])  # type: ignore[attr-defined]
        return re.sub(r"\s+", " ", " ".join(r["text"] for r in rows)).strip()
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        if name in ("RequestBlocked", "IpBlocked", "TooManyRequests"):
            raise Blocked(f"youtube-transcript-api {name}") from e
        if name in ("NoTranscriptFound", "TranscriptsDisabled", "NoTranscriptAvailable"):
            raise NoCaptions(name) from e
        raise


def fetch_transcript_ytdlp(video_id: str, *, timeout: float) -> str:
    exe = shutil.which("yt-dlp")
    if not exe:
        raise RuntimeError("yt-dlp not installed")
    with tempfile.TemporaryDirectory() as td:
        cmd = [exe, "--skip-download", "--write-auto-subs", "--write-subs", "--sub-langs", "en.*,en",
               "--sub-format", "json3", "--no-warnings", "-q", "-o", str(Path(td) / "%(id)s"),
               f"https://www.youtube.com/watch?v={video_id}"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("yt-dlp timeout") from e
        files = sorted(Path(td).glob("*.json3"))
        if not files:
            err = (r.stderr or "").strip().splitlines()[-1:] or ["no subtitle file"]
            if "429" in err[0] or "bot" in err[0].lower():
                raise Blocked(f"yt-dlp: {err[0][:120]}")
            raise NoCaptions(f"yt-dlp: {err[0][:120]}")
        return parse_json3(files[0].read_text(encoding="utf-8", errors="replace"))


def load_cached_transcript(video_id: str, *, now: float | None = None) -> dict[str, Any] | None:
    p = transcript_cache_dir() / f"{re.sub(r'[^A-Za-z0-9_-]', '', video_id)}.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if d.get("status") == "no_captions" and (now or time.time()) - float(d.get("ts", 0)) > NO_CAPTION_TTL:
        return None
    return d


def save_cached_transcript(video_id: str, rec: dict[str, Any]) -> None:
    p = transcript_cache_dir() / f"{re.sub(r'[^A-Za-z0-9_-]', '', video_id)}.json"
    try:
        p.write_text(json.dumps({**rec, "ts": time.time()}), encoding="utf-8")
    except OSError:
        pass


# --- Orchestration -------------------------------------------------------------

def discover(now: datetime, *, deadline: float, fetch: Callable[..., tuple[int, str]] = _get,
             res: YTResult | None = None, profile: YTProfile | None = None) -> list[Video]:
    pr = profile or cfb_profile()
    res = res if res is not None else YTResult()
    jobs: list[tuple[str, str, str]] = []  # (kind, url, label)
    for q, sp in pr.queries:
        jobs.append(("search", SEARCH_URL.format(q=urllib.parse.quote_plus(q), sp=sp), q))
    for name, cid in pr.channels:
        jobs.append(("rss", FEED_URL.format(cid=cid), name))
    res.search_total = sum(1 for j in jobs if j[0] == "search")
    res.rss_total = sum(1 for j in jobs if j[0] == "rss")
    found: dict[str, Video] = {}

    def run(job: tuple[str, str, str]) -> tuple[tuple[str, str, str], list[Video]]:
        kind, url, label = job
        status, body = fetch(url, timeout=min(PER_REQUEST_TIMEOUT, max(1.0, deadline - time.monotonic())))
        if status != 200 or not body:
            return job, []
        vids = parse_search_html(body, now=now, query=label) if kind == "search" else parse_channel_feed(body, channel=label)
        return job, vids

    pool = ThreadPoolExecutor(max_workers=8)
    futs = [pool.submit(run, j) for j in jobs]
    done, _ = wait(futs, timeout=max(0.5, deadline - time.monotonic()))
    for f in done:
        try:
            job, vids = f.result()
        except Exception:  # noqa: BLE001
            continue
        if vids:
            if job[0] == "search":
                res.search_ok += 1
            else:
                res.rss_ok += 1
        for v in vids:
            res.raw_found += 1
            prev = found.get(v.video_id)
            if prev is None or (v.published_exact and not prev.published_exact):
                found[v.video_id] = v
    pool.shutdown(wait=False, cancel_futures=True)
    keep = []
    for v in found.values():
        ok, why = classify_video(v, now=now, profile=pr)
        if not ok:
            if why == pr.old_reason:
                res.rejected_old_title += 1
            continue
        v.relevance = relevance(v, now=now, profile=pr)
        keep.append(v)
    keep.sort(key=lambda v: -v.relevance)
    res.found = len(keep)
    return keep


def run_youtube_research(
    *,
    now: datetime | None = None,
    budget_s: float = 16.0,
    offline: bool = False,
    max_transcripts: int = MAX_TRANSCRIPTS,
    fetch: Callable[..., tuple[int, str]] | None = None,
    transcript_fetchers: list[tuple[str, Callable[[str, float], str]]] | None = None,
    profile: YTProfile | None = None,
) -> YTResult:
    """Find current CFB 27 (or ``profile``) videos and pull transcripts. Bounded by ``budget_s``; never raises."""
    t0 = time.monotonic()
    deadline = t0 + budget_s
    now = now or datetime.now(timezone.utc)
    res = YTResult(ran=True, offline=offline)
    try:
        if offline:
            res.notes.append("offline: YouTube skipped (cached transcripts are still used by id only when online)")
            return res
        vids = discover(now, deadline=t0 + min(6.0, budget_s * 0.4), fetch=fetch or _get, res=res, profile=profile)
        # Transcripts never change: any found video with a cached transcript is used for free;
        # network attempts go to the most relevant uncached videos.
        texts: dict[str, str] = {}
        uncached: list[Video] = []
        for v in vids:
            c = load_cached_transcript(v.video_id)
            if c and c.get("status") == "ok" and c.get("text"):
                v.transcript_status, v.transcript_method = "cached", str(c.get("method") or "")
                texts[v.video_id] = str(c["text"])
            elif c and c.get("status") == "no_captions":
                v.transcript_status = "no_captions"
            else:
                uncached.append(v)
        chosen = uncached[: max(0, max_transcripts)]
        if transcript_fetchers is None:
            tiers: list[tuple[str, Callable[[str, float], str]]] = [("innertube", lambda vid, to: fetch_transcript_stdlib(vid, timeout=to))]
            try:
                import youtube_transcript_api  # noqa: F401,PLC0415

                tiers.append(("youtube-transcript-api", lambda vid, to: fetch_transcript_ytapi(vid)))
            except ImportError:
                res.notes.append("youtube-transcript-api not installed (optional)")
            if shutil.which("yt-dlp"):
                tiers.append(("yt-dlp", lambda vid, to: fetch_transcript_ytdlp(vid, timeout=to)))
            else:
                res.notes.append("yt-dlp not on PATH (optional)")
        else:
            tiers = list(transcript_fetchers)
        breakers = {name: _Breaker(2) for name, _ in tiers}

        def get_one(v: Video) -> tuple[Video, str]:
            cached = load_cached_transcript(v.video_id)
            if cached and cached.get("status") == "ok" and cached.get("text"):
                v.transcript_status, v.transcript_method = "cached", str(cached.get("method") or "")
                return v, str(cached["text"])
            if cached and cached.get("status") == "no_captions":
                v.transcript_status = "no_captions"
                return v, ""
            last_err = ""
            saw_no_caps = False
            for name, fn in tiers:
                left = deadline - time.monotonic()
                if left < 1.5:
                    last_err = last_err or "budget exhausted"
                    break
                if breakers[name].open:
                    last_err = last_err or f"{name}: blocked earlier this prep"
                    continue
                if name == "yt-dlp" and left < 7:
                    continue
                try:
                    text = fn(v.video_id, min(PER_REQUEST_TIMEOUT if name != "yt-dlp" else left - 0.5, left))
                    if text:
                        breakers[name].ok()
                        v.transcript_status, v.transcript_method = "ok", name
                        text = text[:TRANSCRIPT_MAX_CHARS]
                        save_cached_transcript(v.video_id, {"status": "ok", "method": name, "text": text,
                                                            "title": v.title, "channel": v.channel})
                        return v, text
                except Blocked as e:
                    breakers[name].hit()
                    last_err = f"{name}: {e}"
                    continue
                except NoCaptions as e:
                    saw_no_caps = True
                    last_err = f"{name}: {e}"
                    continue
                except Exception as e:  # noqa: BLE001
                    last_err = f"{name}: {type(e).__name__}: {str(e)[:100]}"
                    continue
            if saw_no_caps and "blocked" not in last_err.lower() and "429" not in last_err:
                v.transcript_status = "no_captions"
                save_cached_transcript(v.video_id, {"status": "no_captions", "error": last_err})
            else:
                v.transcript_status = "blocked" if ("Blocked" in last_err or "429" in last_err or "bot" in last_err.lower() or "blocked" in last_err) else "error"
            v.error = last_err
            return v, ""

        if chosen:
            pool = ThreadPoolExecutor(max_workers=4)
            futs = [pool.submit(get_one, v) for v in chosen]
            done, not_done = wait(futs, timeout=max(0.5, deadline - time.monotonic()))
            for f in done:
                try:
                    v, text = f.result()
                except Exception:  # noqa: BLE001
                    continue
                if text:
                    texts[v.video_id] = text
            for f in not_done:
                pass
            pool.shutdown(wait=False, cancel_futures=True)
            for v in chosen:
                if not v.transcript_status:
                    v.transcript_status = "skipped"
                    v.error = v.error or "prep time budget"
        res.considered = len(chosen)
        for v in vids:
            if v.video_id in texts:
                v.transcript_chars = len(texts[v.video_id])
                res.transcripts += 1
                res.methods[v.transcript_method] = res.methods.get(v.transcript_method, 0) + 1
                if v.transcript_status == "cached":
                    res.from_cache += 1
                else:
                    res.fetched_now += 1
            if v.transcript_status == "blocked":
                res.blocked.append(v.video_id)
        for v in vids:
            text = texts.get(v.video_id, "")
            res.docs.append(
                {
                    "label": f"YouTube: {v.channel} — {v.title}"[:160],
                    "kind": "youtube_transcript" if text else "youtube_meta",
                    "url": v.url,
                    "date": v.published,
                    "title": v.title,
                    "channel": v.channel,
                    "text": f"{v.title}. {v.description} {text}".strip(),
                }
            )
        res.videos = [v.to_dict() for v in vids[:25]]
        if res.blocked:
            res.notes.append(
                f"YouTube blocked {len(res.blocked)} transcript request(s) from this network "
                "(common on cloud IPs; usually works from a home connection)"
            )
    except Exception as e:  # noqa: BLE001 — never break prep
        res.notes.append(f"YouTube research error: {type(e).__name__}: {e}")
    finally:
        res.elapsed_s = round(time.monotonic() - t0, 2)
    return res


__all__ = [
    "CREATOR_CHANNELS",
    "YTProfile",
    "cfb_profile",
    "madden_profile",
    "Video",
    "YTResult",
    "classify_video",
    "parse_channel_feed",
    "parse_json3",
    "parse_relative_date",
    "parse_search_html",
    "parse_timedtext_xml",
    "pick_caption_track",
    "run_youtube_research",
]
