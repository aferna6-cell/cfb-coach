"""Eleven-slot offensive assignment database for the installed Madden 27 offense.

Truth rules (enforced by ``validate``):
* every installed play has exactly 11 slot records (QB + 5 OL + 5 skill);
* a slot with no evidence is ``UNKNOWN``; unknown is never promoted to verified;
* only ``diagram_verified`` / ``in_game_verified`` count as verified, and only an
  annotation that carries a traceable source may use any tier above ``unknown``;
* play-NAME hints are ``football_inference`` at the *play* level and never create
  a route or blocking assignment for any slot.

Stdlib only. Never touches the live DB, saved models or the installed playbook.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import offense_taxonomy as tx

SCHEMA = "madden27.offense_assignments.v1"
OVERLAY_SCHEMA = "madden27.offense_assignments.overlay.v1"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "madden27"
DB_PATH = DATA_DIR / "offense_assignments.json"
SNAPSHOT_PATH = DATA_DIR / "installed_playbook_snapshot.json"
ANNOTATIONS_PATH = DATA_DIR / "installed_play_annotations.json"
FINGERPRINT = "OFF-3C6D-E543"
EXPECTED_PLAYS = 93
SLOT_CATEGORIES = frozenset({"route", "block", "run", "option", "play_action", "decoy", "motion",
                             "pass_block", "unknown"})
UNKNOWN_SLOT = {"category": "unknown", "task": "UNKNOWN", "params": {}, "tier": "unknown", "source": ""}


class AssignmentError(ValueError):
    """Invalid assignment record; never silently repaired."""


def play_id(formation: str, play: str, book: str = "Buccaneers") -> str:
    from cfb_coach.madden.model.play_knowledge import play_key
    return play_key("offense", book, formation, play)


# ----------------------------------------------------------------- name-level hints

_RUN = (
    (r"\bqb draw\b", "run", "qb_draw"), (r"\blead read option\b", "option", "lead_read_option"),
    (r"\bread option\b", "option", "read_option"), (r"\brpo\b", "rpo", None),
    (r"slip screen", "screen", None), (r"\bdraw\b", "run", "draw"), (r"off tackle", "run", "off_tackle"),
    (r"zone split|inside zone split", "run", "split_zone"), (r"inside zone", "run", "inside_zone"),
    (r"\bmid zone\b", "run", "mid_zone"), (r"\bstretch\b", "run", "outside_zone"),
    (r"\bdive\b", "run", "dive"), (r"\bcounter\b", "run", "counter"), (r"\bpower\b", "run", "power"),
    (r"\bsweep\b", "run", "sweep"), (r"\bzone\b", "run", "zone_unspecified"),
)
_PA = re.compile(r"\bpa\b|\bboot\b")
_PASS_TOKENS = re.compile(
    r"mesh|flood|smash|dagger|mills|spacing|stick|vert|\bgos\b|hi lo|slant|drag|unders|shallow|cross|drive|"
    r"post|dig|curl|hitch|wheel|texas|spot|rub|fork|zig|corner|out\b|fade|choice|bench|deep|cluster|bunch|"
    r"ohio|portland|salem|return|sluggo|swing|stutter|clear|shake|over")


def name_hint(play: str) -> dict[str, Any]:
    """Play-level hint from the NAME only. Never a player assignment."""
    n = re.sub(r"\s+", " ", play.lower()).strip()
    hint: dict[str, Any] = {"play_type": "unknown", "run_scheme": None, "concepts": [],
                            "motion": bool(re.match(r"mtn\b", n)), "tier": "football_inference", "confidence": 0.35,
                            "basis": "play-name tokens (Madden naming convention); not a diagram"}
    for pat, ptype, scheme in _RUN:
        if re.search(pat, n):
            hint["play_type"], hint["run_scheme"] = ptype, scheme
            if ptype == "run":
                hint["concepts"].append("run_zone" if (scheme or "").endswith("zone") or scheme in
                                        ("split_zone", "zone_unspecified") else "run_gap")
            elif ptype in ("rpo", "option"):
                hint["concepts"].append(ptype)
            elif ptype == "screen":
                hint["concepts"].append("screen")
            return hint
    if _PA.search(n):
        hint["play_type"] = "play_action"
        hint["concepts"].append("play_action")
    elif _PASS_TOKENS.search(n):
        hint["play_type"] = "pass"
    for pat, concept in tx.NAME_CONCEPT_TOKENS:
        if re.search(pat, n) and concept not in hint["concepts"]:
            hint["concepts"].append(concept)
    return hint


# ----------------------------------------------------------------- build


def _skill_slots(formation: str) -> tuple[tuple[str, ...], str]:
    doc = tx.SKILL_SLOTS_DOCUMENTED.get(formation)
    if doc:
        return doc, "source_documented"
    return tuple(f"SKILL_{i}" for i in range(1, 6)), "unknown"


def _slot_from_job(job: Mapping[str, Any], source: str, tier: str) -> dict[str, Any]:
    kind, task = job["kind"], job["task"]
    params = dict(job.get("params") or {})
    rec: dict[str, Any] = {"category": kind, "task": task, "params": params, "tier": tier, "source": source,
                           "alignment_side": params.get("side", "unknown"), "pre_snap_motion": "unknown",
                           "conditional": None, "concept_relation": None}
    if kind == "route":
        spec = tx.route_from_job(task, params)
        rec["route"] = {"family": spec.family, "side": spec.side, "band": spec.band,
                        "published": spec.published, "depth_basis": tx.DEPTH_BASIS}
    return rec


def _blank(slot: str) -> dict[str, Any]:
    return {**copy.deepcopy(UNKNOWN_SLOT), "slot": slot, "alignment_side": "unknown",
            "pre_snap_motion": "unknown", "conditional": None, "concept_relation": None}


def build_database(snapshot: Mapping[str, Any], annotations: Mapping[str, Any]) -> dict[str, Any]:
    side = snapshot["sides"]["offense"]
    if side.get("inventory_fingerprint") != FINGERPRINT:
        raise AssignmentError(f"snapshot fingerprint {side.get('inventory_fingerprint')!r} != {FINGERPRINT}")
    ann = {(a["formation"], a["play"]): a for a in annotations.get("annotations", []) if a.get("side") == "offense"}
    plays: list[dict[str, Any]] = []
    for formation, names in side["formations"].items():
        book = side["formation_sources"][formation]
        skill, personnel_tier = _skill_slots(formation)
        for order, name in enumerate(names):
            slots = {s: _blank(s) for s in (*tx.FIXED_SLOTS, *skill)}
            a = ann.get((formation, name))
            reads, concepts, source = [], [], ""
            if a:
                source = str(a.get("source") or "")
                tier = str(a.get("evidence") or "source_documented")
                for job in a["jobs"]:
                    pos = job["position"]
                    if pos not in slots:
                        raise AssignmentError(f"{formation}/{name}: published slot {pos!r} not in formation slots")
                    slots[pos] = {"slot": pos, **_slot_from_job(job, source, tier)}
                reads = list(a.get("read_order") or [])
                concepts = [{"name": c, "tier": tier, "source": source} for c in a.get("concepts", [])]
            hint = name_hint(name)
            plays.append({
                "play_id": play_id(formation, name, book), "formation": formation, "play": name, "book": book,
                "slot_order": order,
                "personnel": {"skill_slots": list(skill), "tier": personnel_tier,
                              "note": "slot names as published for annotated plays in this formation"
                              if personnel_tier != "unknown" else "no sourced play in this formation; skill slots unnamed"},
                "slots": slots, "name_hint": hint,
                "concepts": concepts,
                "read_order": {"positions": reads, "tier": "source_documented" if reads else "unknown",
                               "source": source if reads else ""},
                "provenance": [source] if source else [],
                "disputes": [],
            })
    db = {"schema": SCHEMA, "inventory_fingerprint": FINGERPRINT, "count": len(plays),
          "tiers": list(tx.EVIDENCE_TIERS), "taxonomy_version": tx.TAXONOMY_VERSION, "plays": plays}
    validate(db)
    return db


# ----------------------------------------------------------------- validate


def validate(db: Mapping[str, Any]) -> None:
    if db.get("schema") != SCHEMA:
        raise AssignmentError("unknown assignment schema")
    plays = db.get("plays") or []
    if len(plays) != EXPECTED_PLAYS:
        raise AssignmentError(f"{len(plays)} plays; expected {EXPECTED_PLAYS}")
    seen: set[str] = set()
    for p in plays:
        pid = p["play_id"]
        if pid in seen:
            raise AssignmentError(f"duplicate play id {p['formation']}/{p['play']}")
        seen.add(pid)
        if pid != play_id(p["formation"], p["play"], p["book"]):
            raise AssignmentError(f"{p['formation']}/{p['play']}: play id does not match identity")
        slots = p["slots"]
        want = set(tx.FIXED_SLOTS) | set(p["personnel"]["skill_slots"])
        if len(slots) != 11 or set(slots) != want:
            raise AssignmentError(f"{p['formation']}/{p['play']}: needs exactly 11 slots, got {sorted(slots)}")
        for name, s in slots.items():
            if s["tier"] not in tx.EVIDENCE_TIERS:
                raise AssignmentError(f"{p['play']}/{name}: bad tier {s['tier']!r}")
            if s["category"] not in SLOT_CATEGORIES:
                raise AssignmentError(f"{p['play']}/{name}: bad category {s['category']!r}")
            if s["tier"] == "unknown":
                if s["category"] != "unknown" or s["task"] != "UNKNOWN":
                    raise AssignmentError(f"{p['play']}/{name}: an unknown slot cannot carry a task")
            else:
                if s["category"] == "unknown" or s["task"] == "UNKNOWN":
                    raise AssignmentError(f"{p['play']}/{name}: a non-unknown tier needs a real task")
                if not s.get("source"):
                    raise AssignmentError(f"{p['play']}/{name}: tier {s['tier']} requires a source")
            if s["category"] == "route" and s["task"] not in tx.ROUTE_FAMILIES:
                raise AssignmentError(f"{p['play']}/{name}: route {s['task']!r} not in taxonomy")
        if p["name_hint"]["tier"] != "football_inference":
            raise AssignmentError(f"{p['play']}: name hint must stay football_inference")
        ro = p["read_order"]
        if ro["positions"]:
            routes = {n for n, s in slots.items() if s["category"] == "route"}
            if not set(ro["positions"]) <= routes or not ro["source"]:
                raise AssignmentError(f"{p['play']}: read order must reference sourced routes")


def load_database(path: str | Path | None = None) -> dict[str, Any]:
    db = json.loads(Path(path or DB_PATH).read_text(encoding="utf-8"))
    validate(db)
    return db


OVERLAY_PATH = DATA_DIR / "offense_user_annotations.json"
# Verified tiers must cite the USER's own evidence, never third-party text.
VERIFIED_SOURCE_PREFIXES = ("user_screenshot:", "user_recording:", "user_ingame:")


def load_effective(overlay_path: str | Path | None = None) -> dict[str, Any]:
    """Base database + the user's annotation overlay when one exists. All consumers use this."""
    base = load_database()
    path = Path(overlay_path) if overlay_path else OVERLAY_PATH
    if not path.exists():
        return base
    merged, _ = merge_overlay(base, json.loads(path.read_text(encoding="utf-8")))
    return merged


def index(db: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(p["formation"], p["play"]): p for p in db["plays"]}


# ----------------------------------------------------------------- overlay (user annotation import)


def merge_overlay(db: Mapping[str, Any], overlay: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Merge user annotations. Stronger tier wins; equal tier with a different task is a recorded dispute."""
    if overlay.get("schema") != OVERLAY_SCHEMA:
        raise AssignmentError("unknown overlay schema")
    out = copy.deepcopy(dict(db))
    idx = index(out)
    log: list[str] = []
    for entry in overlay.get("annotations", []):
        key = (entry["formation"], entry["play"])
        if key not in idx:
            raise AssignmentError(f"overlay names a play that is not installed: {key}")
        play = idx[key]
        for slot, rec in entry["slots"].items():
            if slot not in play["slots"]:
                raise AssignmentError(f"{key}: slot {slot!r} does not exist in this formation")
            tier = rec["tier"]
            if tier not in tx.EVIDENCE_TIERS:
                raise AssignmentError(f"{key}/{slot}: bad tier {tier!r}")
            if tier == "unknown":
                continue
            if not rec.get("source"):
                raise AssignmentError(f"{key}/{slot}: annotation needs a source")
            if tier in tx.VERIFIED_TIERS and not str(rec["source"]).startswith(VERIFIED_SOURCE_PREFIXES):
                raise AssignmentError(f"{key}/{slot}: {tier} must cite the user's own evidence "
                                      f"({', '.join(VERIFIED_SOURCE_PREFIXES)}), not {rec['source']!r}")
            new = {"slot": slot, **_slot_from_job({"kind": rec["category"], "task": rec["task"],
                                                   "params": rec.get("params") or {}}, rec["source"], tier)}
            for extra in ("pre_snap_motion", "conditional", "concept_relation"):
                if rec.get(extra) is not None:
                    new[extra] = rec[extra]
            old = play["slots"][slot]
            if tx.TIER_RANK[tier] < tx.TIER_RANK[old["tier"]]:
                play["slots"][slot] = new
                log.append(f"{key[1]}/{slot}: {old['tier']} -> {tier}")
                if old["tier"] != "unknown" and (old["task"], old["params"].get("side")) != (new["task"], new["params"].get("side")):
                    play["disputes"].append({"slot": slot, "replaced": old, "by": new["tier"]})
            elif tx.TIER_RANK[tier] == tx.TIER_RANK[old["tier"]] and (old["task"], old["params"].get("side")) != (new["task"], new["params"].get("side")):
                play["disputes"].append({"slot": slot, "existing": old, "conflicting": new})
                log.append(f"{key[1]}/{slot}: DISPUTE at tier {tier}")
            else:
                log.append(f"{key[1]}/{slot}: kept stronger {old['tier']}")
        if entry.get("read_order"):
            play["read_order"] = {"positions": list(entry["read_order"]["positions"]), "tier": entry["read_order"]["tier"],
                                  "source": entry["read_order"]["source"]}
        for c in entry.get("concepts", []):
            play["concepts"].append(c)
    validate(out)
    return out, log


# ----------------------------------------------------------------- statistics / audit


def play_status(p: Mapping[str, Any]) -> str:
    tiers = [s["tier"] for s in p["slots"].values()]
    if all(t in tx.VERIFIED_TIERS for t in tiers):
        return "fully_verified"
    if any(t != "unknown" for t in tiers):
        return "partial"
    return "unknown"


def routes_of(p: Mapping[str, Any], min_tier: str = "source_documented") -> list[tuple[str, tx.RouteSpec]]:
    """Known routes (slot, spec) at or above ``min_tier``."""
    out = []
    for name, s in p["slots"].items():
        if s["category"] == "route" and tx.TIER_RANK[s["tier"]] <= tx.TIER_RANK[min_tier]:
            r = s["route"]
            out.append((name, tx.RouteSpec(r["family"], r["side"], r["band"], r.get("published", ""))))
    return out


def coverage_stats(db: Mapping[str, Any]) -> dict[str, Any]:
    plays = db["plays"]
    status = Counter(play_status(p) for p in plays)
    tier_by_pos: dict[str, Counter] = {}
    per_play = []
    for p in plays:
        c = Counter(s["tier"] for s in p["slots"].values())
        per_play.append({"formation": p["formation"], "play": p["play"], "status": play_status(p),
                         "slots_by_tier": {t: c.get(t, 0) for t in tx.EVIDENCE_TIERS},
                         "known_slots": 11 - c.get("unknown", 0),
                         "verified_slots": sum(c.get(t, 0) for t in tx.VERIFIED_TIERS),
                         "route_combo_known": len(routes_of(p)) >= 2,
                         "name_hint_type": p["name_hint"]["play_type"],
                         "name_hint_scheme": p["name_hint"]["run_scheme"]})
        for name, s in p["slots"].items():
            key = name if name in tx.FIXED_SLOTS else "SKILL"
            tier_by_pos.setdefault(key, Counter())[s["tier"]] += 1
    hint = Counter(p["name_hint"]["play_type"] for p in plays)
    return {
        "plays": len(plays), "fingerprint": db["inventory_fingerprint"],
        "fully_verified_plays": status.get("fully_verified", 0),
        "partial_plays": status.get("partial", 0), "unknown_plays": status.get("unknown", 0),
        "plays_with_sourced_route_combination": sum(1 for r in per_play if r["route_combo_known"]),
        "plays_with_name_level_type_hint": sum(1 for p in plays if p["name_hint"]["play_type"] != "unknown"),
        "name_hint_types": dict(hint),
        "slot_records": 11 * len(plays),
        "slots_by_tier": dict(Counter(s["tier"] for p in plays for s in p["slots"].values())),
        "tier_by_position": {k: dict(v) for k, v in sorted(tier_by_pos.items())},
        "per_play": per_play,
    }


def audit_gaps(db: Mapping[str, Any]) -> dict[str, Any]:
    """Exact remaining gaps, never rounded into completion."""
    stats = coverage_stats(db)
    unknown_ol = sum(1 for p in db["plays"] for n in tx.OL_SLOTS if p["slots"][n]["tier"] == "unknown")
    unknown_qb = sum(1 for p in db["plays"] if p["slots"]["QB"]["tier"] == "unknown")
    unknown_skill = sum(1 for p in db["plays"] for n, s in p["slots"].items()
                        if n not in tx.FIXED_SLOTS and s["tier"] == "unknown")
    return {
        "unknown_ol_slots": unknown_ol, "ol_slots_total": 5 * len(db["plays"]),
        "unknown_qb_slots": unknown_qb, "unknown_skill_slots": unknown_skill,
        "skill_slots_total": 5 * len(db["plays"]),
        "plays_without_any_sourced_slot": [f"{p['formation']} / {p['play']}" for p in db["plays"] if play_status(p) == "unknown"],
        "formations_with_unnamed_skill_slots": sorted({p["formation"] for p in db["plays"] if p["personnel"]["tier"] == "unknown"}),
        "disputes": [{"play": p["play"], **d} for p in db["plays"] for d in p["disputes"]],
        "name_hint_unclassified": [p["play"] for p in db["plays"] if p["name_hint"]["play_type"] == "unknown"],
        "verified_slot_total": sum(r["verified_slots"] for r in stats["per_play"]),
    }


def database_digest(db: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(db, sort_keys=True).encode()).hexdigest()[:16]
