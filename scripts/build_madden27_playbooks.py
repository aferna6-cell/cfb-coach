"""Rebuild cfb_coach/data/madden27/playbooks.json from Huddle.gg Madden 27 playbook pages.

Usage:
    python3 scripts/build_madden27_playbooks.py [scratch_dir]

For every candidate stock book (offense + defense) it fetches the book page
(formation list, grouped by formation family) and each formation page (every
play in that formation *in that book*), then writes one JSON with
book -> formation -> plays. Pages are cached in the scratch dir so a rebuild
can run offline. Only books named by current meta sources (plus Aidan's
Franchise team, the Lions) are included; add more to BOOKS as needed.

Play names are canonicalised the same way as the CFB catalog builder
(Huddle prints them in upper case): "TEXAS Y-STUTTER WHEEL" -> "Texas Y-Stutter Wheel".
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
from build_cfb27_catalog import canon_play as _canon  # noqa: E402

MADDEN_UPPER = {"Wr1": "WR1", "Wr2": "WR2", "Wr3": "WR3", "Db": "DB", "Lb": "LB", "Dt": "DT", "Ss": "SS", "Fs": "FS",
                "Cb": "CB", "Olb": "OLB", "Mlb": "MLB", "De": "DE", "Nt": "NT", "Rb": "RB"}


def canon_play(raw: str) -> str:
    return " ".join(MADDEN_UPPER.get(w, w) for w in _canon(raw).split())

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128 Safari/537.36"
BASE = "https://huddle.gg/27/playbooks/"
BOOKS = {
    "offense": {
        "Buccaneers": ("buccaneers-off", "Tampa Bay Buccaneers"),
        "Shotgun Classic": ("shotgun-classic-off", None),
        "Lions": ("lions-off", "Detroit Lions"),
        "Texans": ("texans-off", "Houston Texans"),
        "Cardinals": ("cardinals-off", "Arizona Cardinals"),
        "Falcons": ("falcons-off", "Atlanta Falcons"),
        "Dolphins": ("dolphins-off", "Miami Dolphins"),
        "Eagles": ("eagles-off", "Philadelphia Eagles"),
        "Saints": ("saints-off", "New Orleans Saints"),
        "Bengals": ("bengals-off", "Cincinnati Bengals"),
        "Chargers": ("chargers-off", "Los Angeles Chargers"),
    },
    "defense": {
        "49ers": ("49ers-def", "San Francisco 49ers"),
        "Titans": ("titans-def", "Tennessee Titans"),
        "Raiders": ("raiders-def", "Las Vegas Raiders"),
        "Jaguars": ("jaguars-def", "Jacksonville Jaguars"),
        "Texans": ("texans-def", "Houston Texans"),
        "Vikings": ("vikings-def", "Minnesota Vikings"),
        "Lions": ("lions-def", "Detroit Lions"),
    },
}
# Huddle groups formations under these family headings; the in-game family name differs for Gun.
FAMILY_INGAME = {"Gun": "Shotgun"}
FAMILY_HEADINGS = {"Singleback", "I Form", "Pistol", "Gun", "Goal Line", "Hail Mary", "Strong", "Weak", "Full House",
                   "Wildcat", "Maryland I", "Split Back", "Nickel", "Dime", "4-3", "3-4", "4-4", "5-2", "3-3-5",
                   "4-2-5", "Dollar", "Quarter", "Prevent", "46", "3-3-5 Wide", "Big Dime", "Quarter Normal"}


def fetch(url: str, cache: Path) -> str:
    f = cache / (re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_") + ".html")
    if f.is_file() and f.stat().st_size > 2000:
        return f.read_text(encoding="utf-8", errors="replace")
    body = subprocess.run(["curl", "-sL", "--max-time", "25", "-A", UA, url], capture_output=True, text=True).stdout
    f.write_text(body, encoding="utf-8")
    return body


def lines(body: str) -> tuple[list[str], list[str]]:
    links = re.findall(r'href="([^"]+)"', body)
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", body, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", "\n", t))
    return [ln.strip() for ln in t.splitlines() if ln.strip()], links


def parse_book(body: str, slug: str) -> list[tuple[str, str, str]]:
    """-> [(family, formation full name, url)] in book order."""
    L, links = lines(body)
    urls = [u for u in links if f"/27/playbooks/{slug}/" in u and u.rstrip("/") != f"/27/playbooks/{slug}"]
    i = L.index("Compare") + 1
    j = next(k for k, x in enumerate(L) if x.startswith("Back to the Playbook"))
    out: list[tuple[str, str, str]] = []
    fam = ""
    items = L[i:j]
    k = 0
    for x in items:
        if k < len(urls) and x in FAMILY_HEADINGS and not _is_set(x, fam, urls[k]):
            fam = x
            continue
        if k >= len(urls):
            break
        out.append((fam, x, urls[k]))
        k += 1
    return out


def _is_set(x: str, fam: str, url: str) -> bool:
    # "Hail Mary" is both a family and its only set; "Normal" etc. never collide.
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    fam_slug = re.sub(r"[^a-z0-9]+", "-", fam.lower()).strip("-")
    set_slug = re.sub(r"[^a-z0-9]+", "-", x.lower()).strip("-")
    return bool(fam) and tail == f"{fam_slug}-{set_slug}"


def parse_formation(body: str) -> tuple[str, list[str]]:
    L, _ = lines(body)
    i = L.index("Compare")
    name = L[i + 1]
    j = next(k for k, x in enumerate(L) if x.startswith(("Navigate", "All Playbooks with", "Back to")) and k > i + 1)
    return name, [canon_play(p) for p in L[i + 2: j]]


def main() -> None:
    cache = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/madden27_huddle")
    cache.mkdir(parents=True, exist_ok=True)
    out: dict = {"game": "Madden 27", "source": "Huddle.gg Madden 27 playbook database (per-book formation pages)",
                 "accessed": date.today().isoformat(), "family_ingame": FAMILY_INGAME, "books": {"offense": {}, "defense": {}}}
    pool = ThreadPoolExecutor(max_workers=6)
    for side, books in BOOKS.items():
        for book, (slug, team) in books.items():
            body = fetch(f"{BASE}{slug}/", cache)
            try:
                forms = parse_book(body, slug)
            except (ValueError, StopIteration):
                print(f"skip {book} ({side}): no formation list", file=sys.stderr)
                continue
            bodies = list(pool.map(lambda t: fetch("https://huddle.gg" + t[2], cache), forms))
            fdict: dict[str, dict] = {}
            for (fam, set_name, url), fb in zip(forms, bodies):
                try:
                    name, plays = parse_formation(fb)
                except (ValueError, StopIteration):
                    print(f"  skip {book}/{set_name}", file=sys.stderr)
                    continue
                if name == f"{set_name} {set_name}":  # "Hail Mary Hail Mary"
                    name = set_name
                fdict[name] = {"family": FAMILY_INGAME.get(fam, fam), "family_huddle": fam, "set": set_name,
                               "plays": plays}
            out["books"][side][book] = {"team": team, "url": f"{BASE}{slug}/", "formations": fdict}
            print(f"{side:8} {book:16} {len(fdict):3} formations, {sum(len(v['plays']) for v in fdict.values())} plays")
    dest = Path(__file__).resolve().parent.parent / "cfb_coach" / "data" / "madden27" / "playbooks.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
