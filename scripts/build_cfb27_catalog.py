"""Rebuild cfb_coach/data/cfb27_formations.json from saved CFB.FAN formation pages.

Usage (pages fetched with curl into a scratch dir first):
    for s in gun-bunch-x-nasty gun-cluster ...; do
      curl -sL -o form_$s.html "https://cfb.fan/playbooks/formations/$s/"; done
    # per-book formation pages (what you actually get when you add the formation
    # from that stock book in the custom playbook editor):
    mkdir perbook; curl -sL -o "perbook/ohio-state-off__gun-bunch-x-nasty.html" \
        "https://cfb.fan/27/playbooks/ohio-state-off/gun-bunch-x-nasty/"   # ... one per book/formation
    python3 scripts/build_cfb27_catalog.py /path/to/scratch/dir

Each CFB.FAN *formation* page lists every play that formation has across ALL
CFB 27 playbooks (the union), plus which playbooks carry it. The custom
playbook editor lets you pull a formation from any book, so this union is the
vocabulary the autonomous playbook may choose from.
"""

from __future__ import annotations

import glob
import html
import json
import re
import sys
from datetime import date
from pathlib import Path

UPPER = {"RZ", "PA", "HB", "QB", "RPO", "TE", "WR", "DBL", "WK", "X", "Y", "Z", "H", "FB", "TD",
         "RB", "OL", "DIY", "5WR", "4WR", "U", "FK", "I", "II"}
SPECIAL = {"GOALLINE": "GoalLine", "MTN": "Mtn", "SHIFT": "Shift"}


def canon_word(w: str) -> str:
    if "-" in w:
        return "-".join(canon_word(p) for p in w.split("-"))
    if w in UPPER or w.isdigit() or re.fullmatch(r"\d+[A-Z]*", w):
        return w
    if w in SPECIAL:
        return SPECIAL[w]
    return w[:1] + w[1:].lower()


def canon_play(raw: str) -> str:
    return " ".join(canon_word(w) for w in raw.strip().split())


def _lines(h: str) -> list[str]:
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", h, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", "\n", t))
    return [ln.strip() for ln in t.split("\n") if ln.strip()]


def parse(path: str) -> tuple[str, dict]:
    h = Path(path).read_text(encoding="utf-8", errors="replace")
    L = _lines(h)
    slug = Path(path).stem.replace("form_", "")
    i = L.index("Compare")
    name = L[i + 1]
    j = next(k for k, ln in enumerate(L) if ln.startswith("All Playbooks with"))
    plays = [canon_play(p) for p in L[i + 2: j]]
    books = []
    for ln in L[j + 1:]:
        if ln.startswith("©"):
            break
        books.append(ln)
    return name, {
        "slug": slug,
        "plays": plays,
        "books": books,
        "source": f"https://cfb.fan/playbooks/formations/{slug}/",
    }


def book_names(path: str, slug: str) -> dict[str, str]:
    """book slug (e.g. ohio-state-off) -> display name, from a formation page's 'All Playbooks with' links."""
    h = Path(path).read_text(encoding="utf-8", errors="replace")
    out = {}
    for b, name in re.findall(r'href="/27/playbooks/([a-z0-9-]+-off)/' + re.escape(slug) + r'/"[^>]*>\s*([^<]+?)\s*<', h):
        out[b] = html.unescape(name)
    return out


def parse_book_page(path: str) -> list[str]:
    L = _lines(Path(path).read_text(encoding="utf-8", errors="replace"))
    i = L.index("Compare")
    j = next(k for k, ln in enumerate(L) if ln.startswith("Navigate "))
    return [canon_play(p) for p in L[i + 1: j]]


# Plays Aidan has logged in a formation that no stock list carries under that exact
# name. His logs are ground truth for what his in-game book contains.
LOGGED_ADDITIONS = {
    "Gun Bunch X Nasty": {
        "Mesh Spot": "Logged by Aidan (30 snaps, Ohio State dynasty). Not in the stock Ohio State list (which has "
                     "'Return Mesh Spot'); 'Mesh Spot' is in the Washington State / other Bunch X Nasty lists.",
    },
    "Singleback Deuce Close": {
        "HB Dive": "Logged by Aidan (3 snaps, Ohio State dynasty). Not in the stock Ohio State list; present in other "
                   "books' Singleback Deuce Close lists.",
    },
}


def main(src_dir: str) -> None:
    out: dict = {
        "game": "CFB 27",
        "source": "CFB.FAN playbook database: formation pages (union of plays across all CFB 27 playbooks) and "
                  "per-book formation pages (by_book = the exact plays you get when you add that formation from that "
                  "stock book in the custom playbook editor). logged_additions = plays Aidan has logged that no stock "
                  "list carries under that name (his logs are ground truth).",
        "accessed": date.today().isoformat(),
        "citations": [
            {"what": "formation play lists (union + per stock book)", "source": "CFB.FAN CFB 27 playbook database",
             "url": "https://cfb.fan/27/playbooks/", "note": "e.g. https://cfb.fan/27/playbooks/ohio-state-off/gun-bunch-x-nasty/"},
            {"what": "which books/formations the meta uses (Ohio State Bunch X Nasty + Cluster; Washington State Bunch X "
                     "Nasty, Pistol Trips / Pistol U Off Trips run game, Singleback Deuce Close red zone; West Virginia "
                     "Gun Power I Tight run book)",
             "source": "MaddenTurf - The Best Playbooks for College Football 27 (updated 2026-07-21)",
             "url": "https://maddenturf.com/cfb-27-best-playbooks/"},
            {"what": "CFB 27 custom playbooks: formations from any stock book, 4 audibles per formation",
             "source": "ClutchPoints - How to Create Custom Playbooks in College Football 27",
             "url": "https://clutchpoints.com/gaming/how-to-create-custom-playbooks-in-college-football-27"},
            {"what": "adding a formation in the CFB 27 editor brings every play in it (so the book unit is the formation)",
             "source": "Aidan (in-game observation, 2026-09-27)", "url": ""},
        ],
        "formations": {},
    }
    for f in sorted(glob.glob(str(Path(src_dir) / "form_*.html"))):
        name, rec = parse(f)
        names = book_names(f, rec["slug"])
        by_book = {}
        for bp in sorted(glob.glob(str(Path(src_dir) / "perbook" / f"*__{rec['slug']}.html"))):
            bslug = Path(bp).stem.split("__")[0]
            try:
                plays = parse_book_page(bp)
            except (ValueError, StopIteration):
                continue
            bname = names.get(bslug) or bslug.replace("-off", "").replace("-", " ").title()
            by_book[bname] = {"plays": plays, "source": f"https://cfb.fan/27/playbooks/{bslug}/{rec['slug']}/"}
        rec["by_book"] = by_book
        adds = LOGGED_ADDITIONS.get(name) or {}
        if adds:
            rec["logged_additions"] = adds
            rec["plays"] = rec["plays"] + [p for p in adds if p not in rec["plays"]]
        out["formations"][name] = rec
    dest = Path(__file__).resolve().parents[1] / "cfb_coach" / "data" / "cfb27_formations.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {dest} ({len(out['formations'])} formations)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
