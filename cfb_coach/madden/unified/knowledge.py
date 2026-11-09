"""Shared football knowledge layer: one representation consumed by both coordinators.

Builds ``FootballPlay`` objects from the repo's existing, already-sourced catalogs
(defense_play_catalog, offense_assignments, football_knowledge concepts). It adds no
new football claims: coverage structure comes from the catalog's own axes and
name-decomposed variants, offense concepts from the name-hint layer
(``confidence`` stays low; a name is not route proof), and assignments only from
verified/documented tiers. Everything else stays UNKNOWN.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from cfb_coach.madden.unified.types import (DEF, OFF, UNKNOWN, Assignment, FootballPlay)

DATA = Path(__file__).resolve().parents[2] / "data" / "madden27"

# Exact defensive variants that must never be merged (spec §4).
VARIANT_BUCKETS = ("cover_2_man", "tampa_2", "cover_3_match", "cover3", "cover_4_quarters", "cover_4_palms",
                   "cover_1_hole", "cover_1_robber", "cover_6", "cover_9", "cover_0", "cover_2_invert",
                   "cover_1_contain", "cover_4_drop", "trap_2")
_NAME_FAMILY = (
    (r"cover\s*0|\b0\s+blitz", "cover0", "man"), (r"cover\s*9", "cover9", "match"),
    (r"cover\s*6", "cover6", "match"), (r"tampa\s*2", "cover2", "zone"),
    (r"cover\s*2\s*man|2\s*man", "cover2", "man"), (r"cover\s*2|\b2\s+(?:hard|invert|zone)", "cover2", "zone"),
    (r"palms|quarters|cover\s*4", "cover4", "match"), (r"cover\s*3.*match|3\s+match", "cover3", "match"),
    (r"cover\s*3|\b3\s+(?:sky|cloud|buzz)|fz\s*3", "cover3", "zone"), (r"cover\s*1|\b1\s+(?:hole|robber|press)",
                                                                       "cover1", "man"),
)
_TOKENS = ("sky", "cloud", "buzz", "hole", "robber", "press", "palms", "quarters", "tampa", "match", "man", "invert",
           "hard flat", "contain", "mable", "blitz")


def _load(name: str) -> Any:
    return json.loads((DATA / name).read_text())


@lru_cache(maxsize=1)
def defense_catalog() -> dict[str, Mapping[str, Any]]:
    cat = _load("defense_play_catalog.json")
    return {e["id"]: e for e in cat["entries"]}


@lru_cache(maxsize=1)
def offense_assignment_index() -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for p in _load("offense_assignments.json")["plays"]:
        out.setdefault((p["formation"], p["play"]), p)
    return out


def parse_defense_name(play: str) -> tuple[str, str, tuple[str, ...]]:
    """(family, type, variant tokens) from the label alone — fallback only."""
    t = play.lower()
    fam, typ = UNKNOWN.lower(), "unknown"
    for pat, f, ty in _NAME_FAMILY:
        if re.search(pat, t):
            fam, typ = f, ty
            break
    toks = tuple(x.replace(" ", "_") for x in _TOKENS if x in t)
    if "blitz" in t and typ == "unknown":
        typ = "pressure"
    return fam, typ, toks


_BLITZ = frozenset({"sim", "zone_blitz", "man_blitz", "stunt", "blitz"})
_TYPE_NORM = {"spot_zone": "zone", "zone": "zone", "man": "man", "match": "match"}


def _pressure(entry: Mapping[str, Any] | None, play: str) -> str:
    if entry:
        v = ((entry.get("pressure") or {}).get("concept") or {}).get("value")
        if v and v != "unspecified":
            return str(v)
    return "blitz" if "blitz" in play.lower() else "standard"


def defense_play(formation: str, play: str, book: str | None = None) -> FootballPlay:
    entry = defense_catalog().get(f"inst:{formation}|{play}")
    unknown = ["defender_assignments"]
    n_fam, n_typ, n_tok = parse_defense_name(play)
    if entry:
        cov = entry["coverage"]
        fam = cov.get("family") or n_fam
        raw_t = (cov.get("type") or {}).get("value")
        typ = _TYPE_NORM.get(raw_t) or (n_typ if n_typ != "unknown" else "unknown")
        shell = (cov.get("deep_shell") or {}).get("value")
        bucket = entry.get("legacy_bucket", "unknown")
        variant = tuple(v["token"] for v in cov.get("variants", [])) or n_tok
        conf = 0.6 if bucket != "unknown" and raw_t else 0.45 if fam else 0.3
        prov = ("defense_play_catalog", f"verification:{entry['verification']['level']}")
        assign = tuple(Assignment(s["slot"], s.get("assignment") or UNKNOWN, s.get("alignment"),
                                  True, "defense_play_catalog")
                       for s in entry["ledger"]["slots"] if s.get("status") == "verified")
        if not assign:
            unknown.append("all_11_alignments")
    else:
        fam, typ, variant = n_fam, n_typ, n_tok
        shell, bucket, conf, prov, assign = None, "unknown", 0.25, ("play_name_only",), ()
        unknown.append("catalog_entry")
    press = _pressure(entry, play)
    blitz = press in _BLITZ
    if blitz and typ in ("unknown", "man", "zone"):
        typ = "pressure" if typ == "unknown" else typ
    if bucket not in ("unknown", fam, "cover3", "cover2", "cover1", "cover4") and bucket:
        vkey = bucket
    else:
        vkey = fam + (":" + "+".join(variant) if variant else "")
    if fam == UNKNOWN.lower() and bucket in ("unknown", ""):
        vkey = "unknown"
    sig = (typ, fam, vkey)
    feats = {"coverage_type": typ, "coverage_family": fam, "deep_shell": shell, "pressure": press, "bucket": bucket,
             "signature": sig, "man": 1.0 if typ == "man" else 0.5 if typ == "match" else 0.0, "blitz": blitz}
    return FootballPlay(side=DEF, book=book, formation=formation, play=play, concept_path=(typ, fam, vkey),
                        signature=sig, variant=tuple(variant), features=feats, assignments=assign, confidence=conf,
                        provenance=prov, unknown_fields=tuple(unknown))


def offense_play(formation: str, play: str, book: str | None = None) -> FootballPlay:
    from cfb_coach.madden.model.football_knowledge import profile_for_play
    from cfb_coach.madden.model.offense_assignments import name_hint

    hint = name_hint(play)
    prof = profile_for_play(play)
    ptype = hint.get("play_type", "unknown")
    concept = prof.get("concept_id") or (hint.get("concepts") or [None])[0] or hint.get("run_scheme") or ptype
    principle = prof.get("principle") or {}
    rec = offense_assignment_index().get((formation, play))
    assigns: list[Assignment] = []
    unknown = ["routes", "blocking"]
    prov = ["play_name_hint"]
    conf = 0.35
    if rec:
        from cfb_coach.madden.model.offense_assignments import routes_of
        from cfb_coach.madden.model.offense_taxonomy import VERIFIED_TIERS
        for slot, s in rec["slots"].items():
            if s["tier"] != "unknown" and s["category"] == "route":
                r = s["route"]
                assigns.append(Assignment(slot, r["family"], r.get("band"), s["tier"] in VERIFIED_TIERS, s.get("source")))
        if assigns:
            unknown = [u for u in unknown if u != "routes"] + (["partial_routes"] if len(assigns) < 5 else [])
            prov.append("offense_assignments")
            conf = 0.55 if any(a.verified for a in assigns) else 0.45
        _ = routes_of
    short = principle.get("route_depth") in ("short", "quick") or concept in ("mesh", "stick", "spacing", "slant_flat", "screen")
    deep = principle.get("route_depth") == "deep" or concept in ("four_verts", "wheel", "dagger") or "deep" in play.lower()
    feats = {"play_type": ptype, "concept": concept, "route_depth": principle.get("route_depth"),
             "quick": principle.get("time_to_develop") == "quick" or short, "deep": deep,
             "may_exploit": tuple(principle.get("may_exploit", ())), "reduced_by": tuple(principle.get("reduced_by", ())),
             "horizontal_conflict": bool(principle.get("horizontal_conflict")),
             "vertical_conflict": bool(principle.get("vertical_conflict")), "motion": bool(hint.get("motion"))}
    sig = (ptype, str(concept))
    return FootballPlay(side=OFF, book=book, formation=formation, play=play, concept_path=(ptype, str(concept)),
                        signature=sig, features=feats, assignments=tuple(assigns), confidence=conf,
                        provenance=tuple(prov), unknown_fields=tuple(unknown))


def plays_from_inventory(side: str, inventory: Mapping[str, Iterable[str]],
                         sources: Mapping[str, str] | None = None) -> list[FootballPlay]:
    mk = offense_play if side == OFF else defense_play
    out = []
    for fm, plays in inventory.items():
        for p in dict.fromkeys(plays):
            out.append(mk(fm, p, (sources or {}).get(fm)))
    return out


def knowledge_coverage_report() -> dict[str, Any]:
    """Honest counts: how much of the catalog has reliable assignments (spec §2.7)."""
    dc = defense_catalog()
    ver = sum(1 for e in dc.values() if e["ledger"].get("known_count", 0) > 0)
    full = sum(1 for e in dc.values() if e["ledger"].get("known_count", 0) >= 11)
    unk_bucket = sum(1 for e in dc.values() if e.get("legacy_bucket") == "unknown")
    oa = offense_assignment_index()
    off_partial = sum(1 for p in oa.values() if any(s["tier"] != "unknown" and s["category"] == "route" for s in p["slots"].values()))
    off_ver = sum(1 for p in oa.values() if any(s["tier"] in ("in_game_verified", "diagram_verified") for s in p["slots"].values()))
    return {"defense_plays_in_catalog": len(dc), "defense_with_any_verified_defender": ver, "defense_fully_verified_11": full,
            "defense_variant_unclassified": unk_bucket, "offense_plays_with_assignment_rows": len(oa),
            "offense_with_any_route": off_partial, "offense_with_verified_route": off_ver}


def offense_features_from_signature(sig: tuple[str, ...]) -> dict[str, Any]:
    """Feature dict for an opponent offensive call known only at signature level (type, concept)."""
    from cfb_coach.madden.model.football_knowledge import CONCEPTS

    ptype = sig[0] if sig else "unknown"
    concept = sig[1] if len(sig) > 1 else "unknown"
    pr = CONCEPTS.get(concept, {})
    return {"play_type": ptype, "concept": concept, "route_depth": pr.get("route_depth"),
            "quick": pr.get("time_to_develop") == "quick" or pr.get("route_depth") in ("short", "shallow"),
            "deep": pr.get("route_depth") == "deep" or concept in ("four_verts", "wheel", "dagger"),
            "may_exploit": tuple(pr.get("may_exploit", ())), "reduced_by": tuple(pr.get("reduced_by", ()))}


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


@lru_cache(maxsize=1)
def _defense_name_index() -> dict[str, set[tuple[str, ...]]]:
    idx: dict[str, set[tuple[str, ...]]] = {}
    for id_ in defense_catalog():
        if id_.startswith("inst:"):
            f, p = id_[5:].split("|", 1)
            idx.setdefault(_norm_name(p), set()).add(defense_play(f, p).signature)
    return idx


def defense_signature_by_name(name: str) -> tuple[str, ...] | None:
    """Catalog signature for a coverage NAME when it maps to exactly one signature; otherwise None (never guessed)."""
    sigs = _defense_name_index().get(_norm_name(name), set())
    return next(iter(sigs)) if len(sigs) == 1 else None
