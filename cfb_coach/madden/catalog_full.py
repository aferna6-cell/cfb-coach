"""Full Madden 27 formation catalog for the pregame playbook architect.

Two files feed it:
  - ``playbooks.json``: the core books prep and the established coordinators use.
  - ``playbooks_full.json``: every Huddle.gg team and alternate book, written by
    ``scripts/build_madden27_full_catalog.py``.

Each record is one formation *from one source book* with that book's complete
play list. The same formation name in two books is two records, because the
play lists differ; a cross-book union is never treated as installable.

Personnel is inferred from the formation name (offense) or the defensive family
(DB-count convention). It is labeled as inferred, never as verified.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from cfb_coach.madden import catalog
from cfb_coach.madden.data import _data_path

SIDES = ("offense", "defense")

# Per-side cap the user requested; fewer is allowed.
REQUESTED_MAX_FORMATIONS = 15
EDITOR_LIMITS: dict[str, Any] = {
    "max_formations": 15,
    "max_sets": 56,
    "max_plays": 500,
    "terminology": (
        "Madden calls a family (Gun, Singleback, Nickel) a formation and a named alignment "
        "(Gun Trips TE, Nickel Double Mug) a set. This app calls the named alignment a formation, "
        "so the architect's 15-formation cap is a 15-set cap and the 15-family editor limit holds automatically."
    ),
    "sources": [
        {"url": "https://madden.tools/articles/how-to/build-a-playbook",
         "claim": "Madden 27 custom playbook limits: 15 formations, 56 sets, 500 plays"},
        {"url": "https://forums.operationsports.com/forums/madden-nfl-football/1022420-madden-25-custom-playbooks-play-limit.html",
         "claim": "Madden 25 custom playbooks capped at 500 plays and 51 formations"},
    ],
    "verification": "third_party_reported_not_verified_in_game",
    "mixed_source": (
        "Sets can come from different stock books. A set's play list depends on the source book, "
        "so every set is installed with the complete play list of exactly one source book."
    ),
}

_TWO_BACK = {"i form", "strong", "weak", "strong i", "weak i", "maryland i", "power i", "split backs",
             "split back", "split t", "full house", "wildcat"}
_DEF_PACKAGE = {
    "4-3": "base", "3-4": "base", "46": "base_heavy", "5-2": "base_heavy", "4-4": "base_heavy",
    "nickel": "nickel", "4-2-5": "nickel", "3-3-5": "nickel", "3-3-5 wide": "nickel", "2-4-5": "nickel",
    "dime": "dime", "big dime": "dime", "quarter": "quarter", "dollar": "quarter", "quarter normal": "quarter",
    "prevent": "prevent", "goal line": "goal_line",
}
_DEF_DBS = {"base": 4, "base_heavy": 4, "nickel": 5, "dime": 6, "quarter": 7, "prevent": 7, "goal_line": 3}
_PA = re.compile(r"\bpa\b|play[\s-]*action|boot", re.I)
_RPO = re.compile(r"\brpo\b|alert|read\b", re.I)
_SCREEN = re.compile(r"screen|swing|bubble", re.I)
_QUICK = re.compile(r"slant|stick|spot|snag|mesh|drag|quick|hitch|curl|out\b|smash|shallow|cross", re.I)
_MATCH = re.compile(r"match|palms|sky\s*match|quarters?\s*match|mod", re.I)


def _norm_family(family: str) -> str:
    return " ".join((family or "").lower().split())


@lru_cache(maxsize=1)
def load_full() -> dict[str, Any]:
    try:
        return json.loads(_data_path("playbooks_full.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def full_available() -> bool:
    return bool((load_full().get("books") or {}).get("offense"))


def architect_books(side: str) -> dict[str, dict[str, dict[str, Any]]]:
    """{book: {"team", "kind", "url", "formations": {name: record}}}, full catalog first, core fallback."""
    out: dict[str, dict[str, Any]] = {}
    full = (load_full().get("books") or {}).get(side) or {}
    core = catalog.books(side)
    for name, rec in full.items():
        out[name] = {**rec, "catalog_origin": "playbooks_full.json"}
    for name, rec in core.items():
        if name not in out:
            out[name] = {**rec, "kind": "team" if rec.get("team") else "alternate",
                         "catalog_origin": "playbooks.json"}
    return out


def book_plays(side: str) -> dict[str, dict[str, list[str]]]:
    """{book: {formation: plays}} for every catalogued book on ``side``."""
    return {
        book: {f: list(r.get("plays") or []) for f, r in (rec.get("formations") or {}).items()}
        for book, rec in architect_books(side).items()
    }


def source_plays(side: str, book: str, formation: str) -> list[str] | None:
    rec = ((architect_books(side).get(book) or {}).get("formations") or {}).get(formation)
    return list(rec.get("plays") or []) if rec else None


def offense_personnel(family: str, set_name: str) -> str:
    fam = _norm_family(family)
    s = (set_name or "").lower()
    if fam == "goal line":
        return "heavy (13/22/23)"
    if fam == "hail mary":
        return "spread (00/10)"
    if fam in _TWO_BACK or "full house" in s or s.startswith(("weak i", "strong")) and fam == "pistol":
        if "full house" in s or fam == "full house":
            return "two back (21/22/30)"
        return "two back (21/22)"
    if "5wr" in s or ("empty" in s and not re.search(r"\bte\b|tight|y off|nasty", s)):
        return "empty (00/10)"
    tes = 0
    for pat in (r"\bte\b", r"tight", r"\bwing\b", r"deuce", r"nasty", r"y off", r"\bu off\b", r"pair", r"double wing",
                r"dbl wing", r"heavy", r"\bace\b"):
        if re.search(pat, s):
            tes += 1
    if re.search(r"wing pair|double tight|dbl tight|tight doubles|deuce|ace double wing|heavy|wing tight", s):
        return "12/13 (two+ TE)"
    if re.search(r"4\s*wr|quads|spread|5 wide|empty", s) and tes == 0:
        return "10 (four WR)"
    return "11 (one back, one TE)"


def defense_package(family: str) -> str:
    return _DEF_PACKAGE.get(_norm_family(family), "other")


def offense_profile(plays: list[str]) -> dict[str, Any]:
    n = len(plays) or 1
    runs = [p for p in plays if catalog.is_run(p)]
    return {
        "n_plays": len(plays),
        "runs": len(runs),
        "passes": len(plays) - len(runs),
        "run_share": round(len(runs) / n, 3),
        "play_action": sum(bool(_PA.search(p)) for p in plays),
        "rpo": sum(bool(_RPO.search(p)) and not catalog.is_run(p) for p in plays),
        "screens": sum(bool(_SCREEN.search(p)) for p in plays),
        "quick_game": sum(bool(_QUICK.search(p)) and not catalog.is_run(p) for p in plays),
        "shots": sum(catalog.is_deep(p) for p in plays),
    }


def defense_profile(plays: list[str]) -> dict[str, Any]:
    from cfb_coach.madden.defense_select import call_family

    fams: dict[str, int] = {}
    for p in plays:
        key = call_family(p) or "unclassified"
        fams[key] = fams.get(key, 0) + 1
    match = sum(bool(_MATCH.search(p)) for p in plays)
    man = fams.get("man", 0) + sum(bool(re.search(r"cover\s*0|zero", p, re.I)) for p in plays)
    single = fams.get("single_high", 0) + fams.get("man", 0)
    two = fams.get("two_high", 0) + fams.get("cover2", 0)
    return {
        "n_plays": len(plays),
        "coverage_families": fams,
        "man_calls": man,
        "match_calls": match,
        "zone_calls": max(0, len(plays) - man - fams.get("pressure", 0)),
        "pressure_calls": fams.get("pressure", 0),
        "single_high_calls": single,
        "two_high_calls": two,
    }


def offense_situations(family: str, set_name: str, prof: dict[str, Any]) -> list[str]:
    out = []
    pers = offense_personnel(family, set_name)
    if prof["runs"] >= 3 and ("heavy" in pers or "two back" in pers or "12/13" in pers):
        out += ["short_yardage", "goal_line"]
    elif prof["runs"] >= 3:
        out.append("short_yardage")
    if prof["quick_game"] >= 3:
        out += ["third_and_medium", "two_minute"]
    if prof["shots"] >= 2:
        out.append("explosive_shot")
    if prof["passes"] >= 4 and prof["quick_game"] >= 2:
        out.append("third_and_long")
    if prof["play_action"] >= 2:
        out.append("play_action_early_down")
    if prof["screens"] >= 2:
        out.append("pressure_answer")
    if _norm_family(family) == "hail mary":
        out = ["end_of_half_only"]
    return sorted(dict.fromkeys(out)) or ["normal"]


def defense_situations(family: str, prof: dict[str, Any]) -> list[str]:
    pkg = defense_package(family)
    out = []
    if pkg in ("goal_line", "base_heavy"):
        out += ["goal_line", "short_yardage"]
    if pkg == "base":
        out += ["early_down", "short_yardage", "run_defense"]
    if pkg == "nickel":
        out += ["early_down", "third_and_medium"]
    if pkg in ("dime", "quarter"):
        out += ["third_and_long", "two_minute"]
    if pkg == "prevent":
        out = ["end_of_half_only"]
    if prof["pressure_calls"] >= 3:
        out.append("pressure_package")
    if prof["two_high_calls"] >= 3:
        out.append("explosive_prevention")
    return sorted(dict.fromkeys(out)) or ["early_down"]


def formation_records(side: str) -> list[dict[str, Any]]:
    """Every catalogued (source book, formation) with its complete play list and characteristics."""
    accessed = load_full().get("accessed") or catalog.load_playbooks().get("accessed")
    out: list[dict[str, Any]] = []
    for book, rec in architect_books(side).items():
        for name, frec in (rec.get("formations") or {}).items():
            plays = list(dict.fromkeys(frec.get("plays") or []))
            family = str(frec.get("family") or name.split()[0])
            set_name = str(frec.get("set") or name)
            row: dict[str, Any] = {
                "side": side, "formation": name, "family": family, "set": set_name,
                "source_book": book, "book_kind": rec.get("kind") or ("team" if rec.get("team") else "alternate"),
                "team": rec.get("team"), "plays": plays,
                "source_url": frec.get("url") or rec.get("url"),
                "verification": {
                    "status": "third_party_catalog",
                    "source": "Huddle.gg Madden 27 per-book formation page",
                    "catalog_file": rec.get("catalog_origin"),
                    "accessed": accessed,
                    "in_game_verified": False,
                },
            }
            if side == "offense":
                prof = offense_profile(plays)
                row["personnel"] = offense_personnel(family, set_name)
                row["personnel_source"] = "name_inference"
                row["profile"] = prof
                row["situations"] = offense_situations(family, set_name, prof)
            else:
                prof = defense_profile(plays)
                pkg = defense_package(family)
                row["personnel"] = f"{pkg} ({_DEF_DBS.get(pkg, '?')} DB)" if pkg in _DEF_DBS else pkg
                row["package"] = pkg
                row["personnel_source"] = "family_db_count_convention"
                row["profile"] = prof
                row["situations"] = defense_situations(family, prof)
            out.append(row)
    out.sort(key=lambda r: (r["formation"], r["source_book"]))
    return out


def catalog_drift(side: str) -> list[dict[str, Any]]:
    """Formations whose core play list differs from the newer full catalog for the same book."""
    full = (load_full().get("books") or {}).get(side) or {}
    out = []
    for book, rec in catalog.books(side).items():
        if book not in full:
            continue
        new_forms = full[book].get("formations") or {}
        for name, frec in (rec.get("formations") or {}).items():
            old = list(frec.get("plays") or [])
            new = list((new_forms.get(name) or {}).get("plays") or [])
            if not new:
                out.append({"book": book, "formation": name, "change": "missing_in_full_catalog"})
            elif set(old) != set(new):
                out.append({"book": book, "formation": name, "change": "play_list_changed",
                            "added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new))})
    return out


def _match_key(name: str) -> str:
    """Huddle's index says "Shotgun X" and "4 WR"; its book pages say "Gun X" and "4WR"."""
    raw = (name or "").lower().strip()
    if raw.startswith("shotgun "):
        raw = "gun " + raw[len("shotgun "):]
    return re.sub(r"[^a-z0-9]", "", raw)


def completeness_audit() -> dict[str, Any]:
    """Covered vs missing books and formations. Never claims coverage it did not parse."""
    full = load_full()
    index = full.get("formation_index") or {}
    sides: dict[str, Any] = {}
    for side in SIDES:
        books = architect_books(side)
        names = {f for rec in books.values() for f in (rec.get("formations") or {})}
        pairs = sum(len(rec.get("formations") or {}) for rec in books.values())
        plays = sum(len(f.get("plays") or []) for rec in books.values() for f in (rec.get("formations") or {}).values())
        listed: list[str] = []
        for fam, sets in (index.get(side) or {}).items():
            fam_ingame = (full.get("family_ingame") or {}).get(fam, fam)
            for s in sets:
                listed.append(f"{fam_ingame} {s}" if s != fam else s)
        name_keys = {_match_key(n) for n in names}
        listed_keys = {_match_key(n) for n in listed}
        missing = sorted({n for n in listed if _match_key(n) not in name_keys})
        unlisted = sorted(n for n in names if listed and _match_key(n) not in listed_keys)
        sides[side] = {
            "books_catalogued": len(books),
            "book_names": sorted(books),
            "book_formation_pairs": pairs,
            "plays_listed": plays,
            "unique_formations": len(names),
            "formation_index_size": len(listed),
            "formation_index_covered": len(listed) - len(missing) if listed else None,
            "formations_missing_from_parsed_books": missing,
            "catalogued_but_not_in_index": unlisted,
            "catalog_drift_vs_core": catalog_drift(side),
        }
    fails = list(full.get("parse_failures") or [])
    complete = bool(full) and not fails and all(
        not s["formations_missing_from_parsed_books"] and s["formation_index_size"] for s in sides.values()
    )
    return {
        "source": full.get("source") or catalog.load_playbooks().get("source"),
        "accessed": full.get("accessed") or catalog.load_playbooks().get("accessed"),
        "full_catalog_present": bool(full),
        "sides": sides,
        "parse_failures": fails,
        "complete": complete,
        "claim": ("Every formation in Huddle.gg's Madden 27 formation index appears in at least one parsed book."
                  if complete else
                  "Coverage is NOT complete; see missing formations and parse failures."),
        "not_verified": [
            "In-game play lists are not verified by this tool; Huddle.gg is a third-party source.",
            "Title updates after the access date may add or remove plays.",
            "Formations that exist only in Madden Ultimate Team or other modes are out of scope.",
        ],
        "editor_limits": EDITOR_LIMITS,
    }


__all__ = [
    "EDITOR_LIMITS", "REQUESTED_MAX_FORMATIONS", "architect_books", "book_plays", "catalog_drift",
    "completeness_audit", "defense_package", "formation_records", "full_available", "load_full",
    "offense_personnel", "source_plays",
]
