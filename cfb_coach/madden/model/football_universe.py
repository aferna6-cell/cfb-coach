"""Versioned, searchable football-universe repository (issue #121): every formation/play we can LAWFULLY identify, both sides.

Four evidence units are kept apart and counted separately, never merged into one "coverage" number:

  metadata_indexed   the (side, source book, formation, play) listing exists in a catalogue we may use
  name_decomposed    the NAME parses into football structure (hypothesis-level, soft feature only; not an assignment)
  image_geometry     a first-party/licensed play-art record exists (offense: user screenshots; defense: none yet)
  structure_known    some part of the football structure is known (offense: >=1 receiver/back assignment; defense: a decomposed call)
  roles_verified     per-player assignments verified from first-party evidence (diagram_verified / in_game_verified)

A name is not a diagram, a diagram is not an assignment, an assignment is not a read progression. Nothing here is a hard-coded count: every
number is derived from the data at build time. Read-only at runtime; the live callers get a small cached feature row per (side, formation, play).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from cfb_coach.madden import catalog
from cfb_coach.madden.model import defense_semantics as ds
from cfb_coach.madden.model import offense_assignments as oa

SCHEMA = "madden27.football_universe.v1"
DATA = Path(__file__).resolve().parents[2] / "data" / "madden27"
UNIVERSE_PATH = DATA / "football_universe.json"
GAME = "madden27"
VERIFIED_TIERS = ("diagram_verified", "in_game_verified")
KNOWN_TIERS = ("diagram_verified", "in_game_verified", "source_documented")

# Rights are per SOURCE and are recorded, not assumed. train_rights False => metadata only, never model input beyond the name key.
SOURCES = {
    "repo_playbook_catalog": {"what": "playbooks.json in this repository (formation/play membership by book)", "train_rights": "metadata_only",
                              "note": "provenance of the original capture is recorded in the repository data; names/membership only"},
    "owner_screenshots": {"what": "owner-supplied screenshots of their own Madden 27 / AceMadden listings, private analysis", "train_rights": "owner_private_analysis",
                          "note": "images are not reproduced in the repository; only derived 16-dimension occupancy features"},
    "owner_installed_book": {"what": "owner-attested installed playbook snapshot", "train_rights": "owner_first_party", "note": "defines what is LEGAL to call"},
    "acemadden_site": {"what": "AceMadden public listings", "train_rights": "NOT_ESTABLISHED",
                       "note": "terms (rev. 2026-06-03) assert copyright over play art and restrict redistribution of purchased content; they say nothing "
                               "about automated collection or ML training; robots.txt allows crawling. Bulk ingestion needs written permission "
                               "(support@acemadden.com). Not ingested by this change."},
}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def play_key(side: str, formation: str, play: str) -> str:
    return f"{GAME}|{side}|{slug(formation)}|{slug(play)}"


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def _installed() -> dict[str, set[tuple[str, str]]]:
    snap = json.loads((DATA / "installed_playbook_snapshot.json").read_text(encoding="utf-8"))["sides"]
    return {side: {(f, p) for f, plays in v["formations"].items() for p in plays} for side, v in snap.items()}


def _offense_evidence() -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    art = json.loads((DATA / "offense_play_art_features.json").read_text(encoding="utf-8"))
    art_by = {(r["formation"], r["play"]): r for r in art["plays"]}
    asg = json.loads((DATA / "offense_assignments.json").read_text(encoding="utf-8"))
    asg_by = {(r["formation"], r["play"]): r for r in asg["plays"]}
    return art_by, asg_by


def _slot_counts(rec: Mapping[str, Any]) -> dict[str, int]:
    c = Counter(str(s.get("tier") or "unknown") for s in (rec.get("slots") or {}).values())
    return dict(c)


def _defense_structure() -> dict[str, dict]:
    path = DATA / "defense_play_catalog.json"
    if not path.is_file():
        return {}
    cat = json.loads(path.read_text(encoding="utf-8"))
    return {e["raw_name"]: e for e in cat["entries"]}


def _features(e: Mapping[str, Any]) -> dict[str, float]:
    """Small, cached, NON-ZERO-ONLY feature row per play (computed offline; live code only looks it up).

    Soft name-level structure plus evidence flags. Never an assignment: the evidence flags tell a model how much to trust the rest."""
    f: dict[str, float] = {f"ev_{k}": 1.0 for k, v in e["tiers"].items() if v}
    if e["side"] == "offense":
        h = e["detail"].get("name_hint") or {}
        if h.get("play_type"):
            f[f"off_type_{h['play_type']}"] = 1.0
        if h.get("run_scheme"):
            f[f"off_run_{h['run_scheme']}"] = 1.0
    else:
        key = e["detail"].get("dk_key")
        if key:
            f.update({k: round(v, 4) for k, v in ds.key_axes(key).items() if v})
    return f


def build() -> dict[str, Any]:
    installed = _installed()
    art_by, asg_by = _offense_evidence()
    dstruct = _defense_structure()
    plays: dict[str, dict[str, Any]] = {}
    memberships = 0
    raw_by_slug: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for side in ("offense", "defense"):
        for book in catalog.book_names(side):
            for formation, names in catalog.book_formations(side, book).items():
                for play in names:
                    raw_by_slug[play_key(side, formation, play)].add((side, formation, play))
    ambiguous = {k for k, v in raw_by_slug.items() if len(v) > 1}

    def key_for(side: str, formation: str, play: str) -> str:
        k = play_key(side, formation, play)
        return k + "#" + hashlib.sha1(f"{formation}\x00{play}".encode()).hexdigest()[:6] if k in ambiguous else k

    for side in ("offense", "defense"):
        for book in catalog.book_names(side):
            for formation, names in catalog.book_formations(side, book).items():
                for play in names:
                    memberships += 1
                    k = key_for(side, formation, play)
                    e = plays.get(k)
                    if e is None:
                        e = plays[k] = {"key": k, "side": side, "formation": formation, "play": play, "books": [], "sources": ["repo_playbook_catalog"]}
                        if play_key(side, formation, play) in ambiguous:
                            e["alias_group"] = play_key(side, formation, play)      # spelling variants are kept apart, never merged
                    e["books"].append(book)
    # installed-but-not-in-the-catalogue plays are still real: add them (the owner's own book is the strongest source)
    for side, pairs in installed.items():
        for formation, play in pairs:
            k = key_for(side, formation, play)
            if k not in plays:
                plays[k] = {"key": k, "side": side, "formation": formation, "play": play, "books": [], "sources": []}
            if "owner_installed_book" not in plays[k]["sources"]:
                plays[k]["sources"].append("owner_installed_book")
    for e in plays.values():
        e["books"] = sorted(set(e["books"]))
        pair = (e["formation"], e["play"])
        e["installed"] = pair in installed.get(e["side"], set())
        tiers = {"metadata_indexed": True, "name_decomposed": False, "image_geometry": False, "structure_known": False, "roles_verified": False}
        detail: dict[str, Any] = {}
        if e["side"] == "offense":
            art = art_by.get(pair)
            if art and art.get("model_eligible"):
                tiers["image_geometry"] = True
                detail["art_sha256"] = art.get("image_sha256")
                if "owner_screenshots" not in e["sources"]:
                    e["sources"].append("owner_screenshots")
            hint = oa.name_hint(e["play"])
            tiers["name_decomposed"] = hint.get("play_type", "unknown") != "unknown"
            detail["name_hint"] = {"play_type": hint.get("play_type"), "run_scheme": hint.get("run_scheme")}
            asg = asg_by.get(pair)
            if asg:
                sc = _slot_counts(asg)
                detail["slot_tiers"] = sc
                detail["read_order_documented"] = bool((asg.get("read_order") or {}).get("positions"))
                tiers["structure_known"] = any(sc.get(t, 0) for t in KNOWN_TIERS)
                tiers["roles_verified"] = any(sc.get(t, 0) for t in VERIFIED_TIERS)
        else:
            res = ds.resolve(e["play"], e["formation"])
            tiers["name_decomposed"] = bool(res.key)
            detail["dk_key"] = res.key
            d = dstruct.get(e["play"])
            if d:
                level = (d.get("verification") or {}).get("level")
                led = d.get("ledger") or {}
                detail["decomposition"] = level
                detail["ledger_known_slots"] = int(led.get("known_count", 0))
                detail["ledger_verified_slots"] = int(led.get("verified_count", 0))
                tiers["structure_known"] = detail["ledger_known_slots"] > 0         # per-defender knowledge only; a decomposed NAME is not that
                tiers["roles_verified"] = level == "role_verified" and detail["ledger_verified_slots"] > 0
        e["tiers"], e["detail"] = tiers, detail
        e["features"] = _features(e)
    out_plays = [plays[k] for k in sorted(plays)]
    body = {"schema": SCHEMA, "game": GAME, "sources": SOURCES, "plays": out_plays, "listing_memberships": memberships,
            "books": {s: catalog.book_names(s) for s in ("offense", "defense")}}
    body["digest"] = _digest({"plays": out_plays, "memberships": memberships})
    return body


def write(path: Path | None = None) -> Path:
    path = Path(path or UNIVERSE_PATH)
    path.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


_CACHE: dict[str, Any] = {}


def load(path: Path | None = None) -> dict[str, Any]:
    p = Path(path or UNIVERSE_PATH)
    key = str(p)
    if key not in _CACHE:
        _CACHE[key] = json.loads(p.read_text(encoding="utf-8"))
        _CACHE[key]["_index"] = {e["key"]: e for e in _CACHE[key]["plays"]}
    return _CACHE[key]


def validate(u: Mapping[str, Any]) -> list[str]:
    errs = []
    seen = set()
    for e in u["plays"]:
        if e["key"] in seen:
            errs.append(f"duplicate key {e['key']}")
        seen.add(e["key"])
        if e["key"].split("#")[0] != play_key(e["side"], e["formation"], e["play"]):
            errs.append(f"key mismatch {e['key']}")
        t = e["tiers"]
        if t["roles_verified"] and not t["structure_known"]:
            errs.append(f"{e['key']}: roles_verified without structure_known")
        if t["roles_verified"] and e["side"] == "defense" and e["detail"].get("ledger_known_slots", 0) == 0:
            errs.append(f"{e['key']}: defensive roles_verified without any verified ledger slot")
        if t["image_geometry"] and e["side"] == "defense":
            errs.append(f"{e['key']}: no defensive play art has been captured")
        if not e["tiers"]["metadata_indexed"]:
            errs.append(f"{e['key']}: not indexed")
    return errs


# ---------------------------------------------------------------------------------------------------------------------- report
def coverage_report(u: Mapping[str, Any] | None = None) -> dict[str, Any]:
    u = u or load()
    out: dict[str, Any] = {"schema": SCHEMA, "digest": u["digest"], "listing_memberships_total": u["listing_memberships"], "sides": {}}
    for side in ("offense", "defense"):
        ps = [e for e in u["plays"] if e["side"] == side]
        forms = {e["formation"] for e in ps}
        r = {"books": len(u["books"][side]), "formation_families": len(forms),
             "formation_play_listings_across_books": sum(len(e["books"]) for e in ps),
             "unique_formation_play_pairs": len(ps),
             "spelling_variant_groups_kept_separate": len({e["alias_group"] for e in ps if "alias_group" in e}),
             "installed_pairs": sum(e["installed"] for e in ps)}
        for t in ("metadata_indexed", "name_decomposed", "image_geometry", "structure_known", "roles_verified"):
            n = sum(bool(e["tiers"][t]) for e in ps)
            r[t] = {"plays": n, "pct": round(100.0 * n / len(ps), 2) if ps else 0.0}
        if side == "offense":
            slots = Counter()
            for e in ps:
                slots.update(e["detail"].get("slot_tiers") or {})
            r["slot_level"] = {"slots_total": sum(slots.values()), "by_tier": dict(slots),
                               "plays_with_documented_read_order": sum(bool(e["detail"].get("read_order_documented")) for e in ps)}
        else:
            r["defender_slots_known"] = sum(e["detail"].get("ledger_known_slots", 0) for e in ps)
            r["defender_slots_verified"] = sum(e["detail"].get("ledger_verified_slots", 0) for e in ps)
        out["sides"][side] = r
    out["not_in_this_library"] = [
        "AceMadden bulk listings (permission not established; see sources.acemadden_site)",
        "any defensive play art (none captured)",
        "receiver-level routes for all but the plays with an assignments record",
        "eleven-player assignments for any play on either side",
    ]
    return out


# ---------------------------------------------------------------------------------------------------------------------- search
def _tokens(s: str) -> set[str]:
    return set(slug(s).split("_")) - {""}


def search(query: str, *, side: str | None = None, installed: bool | None = None, limit: int = 20, u: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Token search over formation + play names (and book names). Non-installed plays are first-class results."""
    u = u or load()
    q = _tokens(query)
    scored = []
    for e in u["plays"]:
        if side and e["side"] != side:
            continue
        if installed is not None and e["installed"] != installed:
            continue
        hay = _tokens(e["formation"]) | _tokens(e["play"]) | set().union(*(_tokens(b) for b in e["books"])) if e["books"] else _tokens(e["formation"]) | _tokens(e["play"])
        hit = len(q & hay)
        if hit:
            scored.append((-hit, -int(e["tiers"]["roles_verified"]) - int(e["tiers"]["structure_known"]), e["key"], e))
    scored.sort(key=lambda t: t[:3])
    return [t[3] for t in scored[:limit]]


def get(side: str, formation: str, play: str, u: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    u = u or load()
    return u["_index"].get(play_key(side, formation, play)) if "_index" in u else next((e for e in u["plays"] if e["key"] == play_key(side, formation, play)), None)


def feature_row(side: str, formation: str, play: str, u: Mapping[str, Any] | None = None) -> dict[str, float]:
    """O(1) cached lookup for live code. Unknown plays return {} (zero information), never a guess."""
    e = get(side, formation, play, u)
    return dict(e["features"]) if e else {}


def formation_evidence(side: str, formation: str, u: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evidence summary of one formation family across all books (for pregame proposals; informational, not a score)."""
    u = u or load()
    ps = [e for e in u["plays"] if e["side"] == side and e["formation"] == formation]
    n = len(ps)
    return {"plays": n, **{t: sum(bool(e["tiers"][t]) for e in ps) for t in ("name_decomposed", "image_geometry", "structure_known", "roles_verified")}}
