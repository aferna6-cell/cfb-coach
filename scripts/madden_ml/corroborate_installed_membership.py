"""Cross-check installed offensive formations against a second, independent public listing (huddle.gg formation pages, private cache).

Writes only DERIVED metadata (formation/play names + match booleans; names may be public per the owner). The cached HTML is never committed.
huddle.gg rights are NOT established for anything beyond reading names: this is a membership cross-check, not a training source.
Usage: python scripts/madden_ml/corroborate_installed_membership.py --cache /workspace/vod_data/research/huddle_cache
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cfb_coach.madden.model import catalog_semantics as cs



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--book", default="buccaneers")
    a = ap.parse_args()
    snap = cs.installed_snapshot()["offense"]
    out = {"schema": "madden27.installed_membership_corroboration.v1", "side": "offense", "second_source": "huddle.gg formation page (private cache)",
           "second_source_rights": "names read for cross-check only; not a training source; HTML not committed", "formations": {}}
    for formation, plays in sorted(snap["formations"].items()):
        f = a.cache / f"https_huddle_gg_27_playbooks_{a.book}_off_{cs.slug(formation)}.html"
        if not f.is_file():
            out["formations"][formation] = {"status": "second_source_missing", "installed": sorted(plays)}
            continue
        raw = f.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))
        m = re.search(r"Compare " + re.escape(formation) + r" (.*?) Navigate ", text, re.I)
        listed_blob = m.group(1).upper() if m else ""
        found = sorted(p for p in plays if re.search(r"(^| )" + re.escape(p.upper()) + r"( |$)", listed_blob))
        books = re.search(r"All Playbooks with " + re.escape(formation) + r" (.*?) Back to ", text, re.I)
        out["formations"][formation] = {"status": "compared", "installed": sorted(plays), "corroborated": found,
                                        "not_found_in_second_source": sorted(set(plays) - set(found)),
                                        "books_listing_formation": books.group(1).split() if books else [],
                                        "cache_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    cs.MEMBERSHIP_PATH.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: (v["status"], len(v.get("corroborated", [])), len(v["installed"])) for k, v in out["formations"].items()}))


if __name__ == "__main__":
    main()
