"""Build cfb_coach/data/madden27/playbooks_full.json: every Huddle.gg Madden 27 stock book.

Usage:
    python3 scripts/build_madden27_full_catalog.py [cache_dir]

The core ``playbooks.json`` (written by ``build_madden27_playbooks.py``) is left
untouched; prep and the established coordinators keep reading it. This file is
the pregame architect's full catalog: every team and alternate book Huddle lists,
each formation with the plays it has *in that book*, plus Huddle's all-formations
index so the completeness audit can report formations no parsed book contains.

Families are read from each formation URL slug (``/<book>/<family>-<set>/``), not
from a fixed heading list, so new families (Strong I, Split Backs, ...) parse.
A book or formation that fails to parse is recorded under ``parse_failures``;
it is never silently dropped from the audit.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_madden27_playbooks import FAMILY_INGAME, UA, canon_play, lines  # noqa: E402

SITE = "https://huddle.gg"
INDEX = f"{SITE}/playbooks/"
FORMATIONS_INDEX = f"{SITE}/playbooks/formations/"
TEAM_NAMES = {
    "49ers": "San Francisco 49ers", "Bears": "Chicago Bears", "Bengals": "Cincinnati Bengals",
    "Bills": "Buffalo Bills", "Broncos": "Denver Broncos", "Browns": "Cleveland Browns",
    "Buccaneers": "Tampa Bay Buccaneers", "Cardinals": "Arizona Cardinals", "Chargers": "Los Angeles Chargers",
    "Chiefs": "Kansas City Chiefs", "Colts": "Indianapolis Colts", "Commanders": "Washington Commanders",
    "Cowboys": "Dallas Cowboys", "Dolphins": "Miami Dolphins", "Eagles": "Philadelphia Eagles",
    "Falcons": "Atlanta Falcons", "Giants": "New York Giants", "Jaguars": "Jacksonville Jaguars",
    "Jets": "New York Jets", "Lions": "Detroit Lions", "Packers": "Green Bay Packers",
    "Panthers": "Carolina Panthers", "Patriots": "New England Patriots", "Raiders": "Las Vegas Raiders",
    "Rams": "Los Angeles Rams", "Ravens": "Baltimore Ravens", "Saints": "New Orleans Saints",
    "Seahawks": "Seattle Seahawks", "Steelers": "Pittsburgh Steelers", "Texans": "Houston Texans",
    "Titans": "Tennessee Titans", "Vikings": "Minnesota Vikings",
}
ALT_NAMES = {
    "air-raid-classic": "Air Raid Classic", "air-raid": "Air Raid", "balanced": "Balanced",
    "benkerts-dimes": "Benkert's Dimes", "pistol": "Pistol", "run-n-gun": "Run 'N' Gun",
    "run-and-shoot": "Run and Shoot", "run-balanced": "Run Balanced", "run-heavy": "Run Heavy",
    "shotgun": "Shotgun", "shotgun-classic": "Shotgun Classic", "shotgun-mix": "Shotgun Mix",
    "singleback": "Singleback", "spread-classic": "Spread Classic", "spread": "Spread",
    "two-back": "Two Back", "west-coast": "West Coast", "3-4": "3-4", "4-3": "4-3", "46": "46",
    "cover-2": "Cover 2", "multiple-d": "Multiple D",
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _key(text: str) -> str:
    """Huddle slugs double hyphens ("Y-Flex" -> "y--flex"); compare alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def fetch(url: str, cache: Path) -> str:
    f = cache / (re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_") + ".html")
    if f.is_file() and f.stat().st_size > 2000:
        return f.read_text(encoding="utf-8", errors="replace")
    body = subprocess.run(["curl", "-sL", "--max-time", "30", "-A", UA, url],
                          capture_output=True, text=True).stdout
    if len(body) > 2000:
        f.write_text(body, encoding="utf-8")
    return body


def discover_books(cache: Path) -> list[tuple[str, str, str, str | None, str]]:
    """-> [(side, display name, slug, team, kind)] for every book on the index page."""
    body = fetch(INDEX, cache)
    slugs = sorted(set(re.findall(r'href="(?:https://huddle\.gg)?/(?:27/)?playbooks/([a-z0-9-]+-(?:off|def))/"', body)))
    out = []
    for s in slugs:
        base, kind = s.rsplit("-", 1)
        side = "offense" if kind == "off" else "defense"
        team_key = next((k for k in TEAM_NAMES if slug(k) == base), None)
        if team_key:
            out.append((side, team_key, s, TEAM_NAMES[team_key], "team"))
        else:
            out.append((side, ALT_NAMES.get(base, base.replace("-", " ").title()), s, None, "alternate"))
    return out


def parse_book(body: str, book_slug: str) -> list[tuple[str, str, str]]:
    """-> [(family, set, url)] in book order; the family is taken from the URL slug."""
    L, links = lines(body)
    urls = []
    for u in links:
        m = re.search(rf"/playbooks/{re.escape(book_slug)}/([a-z0-9-]+)/?$", u)
        if m and u not in urls:
            urls.append(u)
    i = L.index("Compare") + 1
    j = next(k for k, x in enumerate(L) if x.startswith("Back to the Playbook"))
    out: list[tuple[str, str, str]] = []
    fam = ""
    k = 0
    for x in L[i:j]:
        if k >= len(urls):
            break
        tail = _key(urls[k].rstrip("/").rsplit("/", 1)[-1])
        if fam and tail == _key(fam) + _key(x):
            out.append((fam, x, urls[k]))
            k += 1
        else:
            fam = x
    return out


def parse_formation(body: str) -> tuple[str, list[str]]:
    L, _ = lines(body)
    i = L.index("Compare")
    name = L[i + 1]
    j = next(k for k, x in enumerate(L)
             if x.startswith(("Navigate", "All Playbooks with", "Back to")) and k > i + 1)
    return name, [canon_play(p) for p in L[i + 2: j]]


def parse_formation_index(body: str) -> dict[str, dict[str, list[str]]]:
    """Huddle's all-formations page -> {side: {family: [set names]}}."""
    out: dict[str, dict[str, list[str]]] = {"offense": {}, "defense": {}}
    side = None
    fam = None
    for m in re.finditer(r"<(h2|h3|li)[^>]*>(.*?)</\1>", body, flags=re.S | re.I):
        tag = m.group(1).lower()
        text = html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
        if not text:
            continue
        if tag == "h2":
            low = text.lower()
            side = "offense" if low == "offense" else "defense" if low == "defense" else None
            fam = None
        elif tag == "h3" and side:
            fam = text
            out[side].setdefault(fam, [])
        elif tag == "li" and side and fam:
            if text not in out[side][fam]:
                out[side][fam].append(text)
    return out


def main() -> None:
    cache = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / ".cache" / "madden27_huddle")
    cache.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "game": "Madden 27",
        "source": "Huddle.gg Madden 27 playbook database (per-book formation pages)",
        "source_url": INDEX,
        "accessed": date.today().isoformat(),
        "family_ingame": FAMILY_INGAME,
        "books": {"offense": {}, "defense": {}},
        "formation_index": {},
        "parse_failures": [],
    }
    out["formation_index"] = parse_formation_index(fetch(FORMATIONS_INDEX, cache))
    pool = ThreadPoolExecutor(max_workers=6)
    for side, name, book_slug, team, kind in discover_books(cache):
        url = f"{SITE}/playbooks/{book_slug}/"
        try:
            forms = parse_book(fetch(url, cache), book_slug)
        except (ValueError, StopIteration):
            out["parse_failures"].append({"side": side, "book": name, "url": url, "stage": "book"})
            print(f"skip {name} ({side}): no formation list", file=sys.stderr)
            continue
        bodies = list(pool.map(lambda t: fetch(SITE + t[2] if t[2].startswith("/") else t[2], cache), forms))
        fdict: dict[str, dict] = {}
        for (fam, set_name, furl), fb in zip(forms, bodies):
            try:
                fname, plays = parse_formation(fb)
            except (ValueError, StopIteration):
                out["parse_failures"].append({"side": side, "book": name, "formation": f"{fam} {set_name}",
                                              "url": furl, "stage": "formation"})
                continue
            if fname == f"{set_name} {set_name}":
                fname = set_name
            if not plays:
                out["parse_failures"].append({"side": side, "book": name, "formation": fname,
                                              "url": furl, "stage": "empty_play_list"})
                continue
            fdict[fname] = {"family": FAMILY_INGAME.get(fam, fam), "family_huddle": fam, "set": set_name,
                            "plays": plays, "url": SITE + furl if furl.startswith("/") else furl}
        out["books"][side][name] = {"team": team, "kind": kind, "url": url, "formations": fdict}
        print(f"{side:8} {name:18} {len(fdict):3} formations, {sum(len(v['plays']) for v in fdict.values())} plays")
    dest = Path(__file__).resolve().parent.parent / "cfb_coach" / "data" / "madden27" / "playbooks_full.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
