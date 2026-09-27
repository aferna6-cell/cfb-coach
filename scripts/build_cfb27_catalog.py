"""Rebuild cfb_coach/data/cfb27_formations.json from saved CFB.FAN formation pages.

Usage (pages fetched with curl into a scratch dir first):
    for s in gun-bunch-x-nasty gun-cluster ...; do
      curl -sL -o form_$s.html "https://cfb.fan/playbooks/formations/$s/"; done
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


def main(src_dir: str) -> None:
    out: dict = {
        "game": "CFB 27",
        "source": "CFB.FAN playbook database (formation pages = union of plays across all CFB 27 playbooks)",
        "accessed": date.today().isoformat(),
        "formations": {},
    }
    for f in sorted(glob.glob(str(Path(src_dir) / "form_*.html"))):
        name, rec = parse(f)
        out["formations"][name] = rec
    dest = Path(__file__).resolve().parents[1] / "cfb_coach" / "data" / "cfb27_formations.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {dest} ({len(out['formations'])} formations)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
