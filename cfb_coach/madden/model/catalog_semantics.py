"""Staged catalogue semantics (issue #127): from a listed play name to model-ready football knowledge, one evidence stage at a time.

Stages (a play/slot is at the HIGHEST stage it has actually earned; nothing skips a stage):
  1 metadata_only            listed in a catalogue we may use (name, formation, book membership)
  2 source_image             an authorized/first-party image is referenced by id + sha256 (the image itself stays private)
  3 geometry_extracted       an extractor proposed markers/routes/zones/arrows with confidence (a PROPOSAL, never a fact)
  4 geometry_reviewed        a person confirmed the overlay matches the picture (still not a player responsibility)
  5 assignment_reviewed      a person mapped slot -> assignment / defender -> responsibility from first-party evidence
  6 concept_verified         a route combination / coverage structure follows from reviewed assignments by an explicit rule
  7 model_feature_eligible   passes the quality gates below; only these rows carry football features to a model

Third-party documentation (e.g. madden.tools text) is recorded as ``third_party_documented``: useful to PROPOSE, never to certify.
Unknown stays the literal string "unknown". Nothing here touches live scoring; consumers read ``export()`` offline.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "madden27.catalog_semantics.v1"
DATA = Path(__file__).resolve().parents[2] / "data" / "madden27"
MEMBERSHIP_PATH = DATA / "installed_membership_corroboration.json"
REVIEWS_PATH = DATA / "assignment_reviews.jsonl"
STAGES = ("metadata_only", "source_image", "geometry_extracted", "geometry_reviewed", "assignment_reviewed", "concept_verified",
          "model_feature_eligible")
FUNNEL = ("indexed", "art_available", "downloaded", "geometry_extracted", "human_reviewed", "assignment_verified", "concept_verified",
          "model_feature_eligible")
# per-slot evidence, weakest first. Only the last two count as verified.
SLOT_TIERS = ("unknown", "name_inference", "third_party_documented", "geometry_unreviewed", "geometry_reviewed", "human_reviewed",
              "in_game_verified")
VERIFIED_SLOT = ("human_reviewed", "in_game_verified")
REVIEW_EVIDENCE_OK = ("user_screenshot:", "user_recording:", "licensed_export:", "in_game:")
DECISIONS = ("confirm", "correct", "reject", "unknown")
RIGHTS = {
    "owner_installed_book": "owner_first_party",
    "repo_playbook_catalog": "metadata_only",
    "owner_screenshots": "owner_private_analysis (derived features only; images private)",
    "madden_tools_text": "third_party_documented: proposal source only, no training on raw content",
    "huddle_gg_listing": "names cross-check only; rights not established",
    "acemadden_site": "metadata_bulk, images_download, images_local_storage per owner-reported grant (PR #126); ml_training and redistribution NOT granted",
}
MIN_VERIFIED_SKILL_SLOTS = 3          # gate: a pass play needs >=3 reviewed eligible receivers before it may feed route features
OFF_SKILL = re.compile(r"^(WR\d|SLWR|TE\d|HB\d|FB\d?|QB)$")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _load(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def installed_snapshot() -> dict[str, Any]:
    return _load("installed_playbook_snapshot.json")["sides"]


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


# ---------------------------------------------------------------- slot evidence

def slot_tier(slot: Mapping[str, Any]) -> str:
    """Map the legacy offense_assignments tier onto the staged vocabulary WITHOUT upgrading anything."""
    t = str(slot.get("tier") or "unknown")
    return {"source_documented": "third_party_documented", "football_inference": "name_inference", "diagram_verified": "geometry_reviewed",
            "diagram_extracted_unreviewed": "geometry_unreviewed", "in_game_verified": "in_game_verified"}.get(t, "unknown" if t not in SLOT_TIERS else t)


def _route(slot: Mapping[str, Any]) -> str:
    r = slot.get("route") or {}
    return str(r.get("family") or slot.get("task") or "unknown") if slot.get("category") == "route" else str(slot.get("category") or "unknown")


def proposal(slot: Mapping[str, Any]) -> dict[str, Any]:
    r = slot.get("route") or {}
    return {"category": slot.get("category") or "unknown", "assignment": _route(slot), "side": r.get("side") or slot.get("alignment_side") or "unknown",
            "published_label": r.get("published") or "", "source": slot.get("source") or "", "tier": slot_tier(slot)}


# ---------------------------------------------------------------- reviews (the only path to "verified")

class ReviewError(ValueError):
    pass


def validate_review(r: Mapping[str, Any]) -> dict[str, Any]:
    need = ("play_key", "slot", "decision", "reviewer", "reviewed_at", "evidence_ref")
    miss = [k for k in need if not str(r.get(k) or "").strip()]
    if miss:
        raise ReviewError(f"review missing {miss}")
    if r["decision"] not in DECISIONS:
        raise ReviewError(f"decision must be one of {DECISIONS}")
    if not str(r["evidence_ref"]).startswith(REVIEW_EVIDENCE_OK):
        raise ReviewError(f"evidence_ref must be first-party/licensed ({REVIEW_EVIDENCE_OK}); names, art URLs and extractor output cannot certify")
    if r["decision"] == "correct" and not str(r.get("assignment") or "").strip():
        raise ReviewError("a correction must state the assignment")
    if str(r.get("reviewer")).lower() in {"extractor", "auto", "model", "bot"}:
        raise ReviewError("automated reviewers cannot certify")
    return dict(r)


def load_reviews(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or REVIEWS_PATH
    if not p.is_file():
        return []
    return [validate_review(json.loads(l)) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def import_review_csv(csv_path: Path, out: Path | None = None) -> int:
    """Append validated rows from a filled review queue CSV (idempotent: identical rows are not duplicated)."""
    p = out or REVIEWS_PATH
    seen = {json.dumps(r, sort_keys=True) for r in load_reviews(p)}
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("decision") or "").strip():
                continue
            r = validate_review({k: (v or "").strip() for k, v in r.items() if k in
                                 ("play_key", "slot", "decision", "assignment", "reviewer", "reviewed_at", "evidence_ref", "note")})
            s = json.dumps(r, sort_keys=True)
            if s not in seen:
                seen.add(s)
                rows.append(s)
    with open(p, "a", encoding="utf-8") as fh:
        fh.writelines(s + "\n" for s in rows)
    return len(rows)


def resolve_slot(prop: Mapping[str, Any], reviews: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine a proposal with its human reviews. Disagreements are kept, not averaged away."""
    rs = list(reviews)
    out = {"proposal": dict(prop), "reviews": rs, "disagreement": False, "tier": prop["tier"], "assignment": "unknown"}
    if not rs:
        return out
    final = {r["decision"] if r["decision"] != "correct" else "correct:" + r["assignment"] for r in rs}
    if len(final) > 1:
        out["disagreement"] = True
        out["tier"] = prop["tier"] if prop["tier"] not in VERIFIED_SLOT else "geometry_reviewed"
        return out
    d = rs[-1]
    if d["decision"] == "confirm":
        out.update(tier="in_game_verified" if d["evidence_ref"].startswith("in_game:") else "human_reviewed", assignment=prop["assignment"])
    elif d["decision"] == "correct":
        out.update(tier="in_game_verified" if d["evidence_ref"].startswith("in_game:") else "human_reviewed", assignment=d["assignment"],
                   corrected_from=prop["assignment"])
    else:
        out.update(tier="unknown", assignment="unknown")
    return out


# ---------------------------------------------------------------- concepts (only from reviewed routes)

def _side_routes(slots: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for s in slots.values():
        if s["tier"] in VERIFIED_SLOT and s["assignment"] != "unknown":
            by.setdefault(s["proposal"]["side"], []).append(s["assignment"])
    return by


CONCEPT_RULES = {
    # name: (predicate over one side's verified route list, what it stresses, its known weakness) -- football principles, not Madden data
    "smash": (lambda r: "corner" in r and any(x in r for x in ("hitch", "curl", "out")), "Cover 2 corner (high-low on the flat defender)",
              "Cover 4/Quarters corner sits on the corner route"),
    "slants": (lambda r: r.count("slant") >= 2, "press man and soft zone inside windows", "inside-leverage man and robber"),
    "stick": (lambda r: any(x in r for x in ("hitch", "stick")) and "flat" in r, "curl/flat defender in 3-deep zone", "pattern-match flat defenders"),
    "verticals": (lambda r: sum(x in ("streak", "seam", "post", "fade") for x in r) >= 2, "single-high safety (two verticals on one safety)",
                  "two-high shells and deep-zone depth"),
    "flood": (lambda r: len({x for x in r if x in ("flat", "out", "corner", "wheel", "streak", "fade")}) >= 3,
              "three-level stretch of a Cover 3 side", "Cover 3 Match / Palms carry rules"),
    "mesh": (lambda r: r.count("drag") >= 2, "man coverage (crossing traffic)", "zone defenders sitting in the middle"),
}


def verified_concepts(slots: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for side, routes in _side_routes(slots).items():
        for name, (pred, stresses, weak) in CONCEPT_RULES.items():
            if pred(routes):
                out.append({"concept": name, "side": side, "challenges": stresses, "weakness": weak, "basis": "rule over human-reviewed routes"})
    return out


# ---------------------------------------------------------------- records

def offense_records(reviews: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    reviews = load_reviews() if reviews is None else reviews
    asg = {(p["formation"], p["play"]): p for p in _load("offense_assignments.json")["plays"]}
    art = {(p["formation"], p["play"]): p for p in _load("offense_play_art_features.json")["plays"]}
    cap = {(p["formation"], p["installed_play_name"]): p for p in _load("ace_offense_capture_manifest.json")["plays"]}
    mem = json.loads(MEMBERSHIP_PATH.read_text(encoding="utf-8"))["formations"] if MEMBERSHIP_PATH.is_file() else {}
    by_slot: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in reviews:
        by_slot.setdefault((r["play_key"], r["slot"]), []).append(r)
    out = []
    for formation, plays in sorted(installed_snapshot()["offense"]["formations"].items()):
        for play in plays:
            key = f"madden27|offense|{slug(formation)}|{slug(play)}"
            a, c = asg.get((formation, play), {}), cap.get((formation, play))
            slots = {name: resolve_slot(proposal(s), by_slot.get((key, name), [])) for name, s in sorted((a.get("slots") or {}).items())}
            m = mem.get(formation, {})
            rec = {"key": key, "side": "offense", "formation": formation, "play": play, "installed": True,
                   "membership": {"owner_installed_book": True, "second_source_corroborated": play in m.get("corroborated", []) if m else "unknown"},
                   "source_image": ({"kind": "owner_screenshot_of_third_party_art", "image_id": c["capture_path"], "sha256": c["screenshot_sha256"],
                                     "stored": "private (not on this host, not in git)"} if c else "unknown"),
                   "geometry": "occupancy_features_only" if (formation, play) in art else "unknown",
                   "slots": slots, "concepts_verified": verified_concepts(slots),
                   "concepts_proposed": [x["name"] for x in a.get("concepts", [])] or (a.get("name_hint") or {}).get("concepts") or [],
                   "provenance": sorted(set(a.get("provenance") or [])) + (["owner_installed_book"])}
            rec["stage"] = _stage(rec)
            out.append(rec)
    return out


def defense_records() -> list[dict[str, Any]]:
    cat = {e["raw_name"]: e for e in _load("defense_play_catalog.json")["entries"]}
    out = []
    for formation, plays in sorted(installed_snapshot()["defense"]["formations"].items()):
        for play in plays:
            e = cat.get(play, {})
            cov = e.get("coverage") or {}
            rec = {"key": f"madden27|defense|{slug(formation)}|{slug(play)}", "side": "defense", "formation": formation, "play": play, "installed": True,
                   "membership": {"owner_installed_book": True, "second_source_corroborated": "unknown"}, "source_image": "unknown", "geometry": "unknown",
                   "defenders": "unknown",   # no defender-level evidence exists yet; never filled from the name
                   "coverage_proposed": {"family": cov.get("family") or "unknown", "type": (cov.get("type") or {}).get("value") or "unknown",
                                         "type_status": (cov.get("type") or {}).get("status") or "unknown", "basis": "play name + documented glossary, not observed"},
                   "coverage_verified": "unknown", "pressure_verified": "unknown", "front_verified": "unknown",
                   "provenance": ["owner_installed_book"] + sorted({s for x in e.get("evidence", []) for s in [x.get("source")] if s})}
            rec["stage"] = _stage(rec)
            out.append(rec)
    return out


def _stage(rec: Mapping[str, Any]) -> str:
    st = "metadata_only"
    if rec["source_image"] != "unknown":
        st = "source_image"
    if rec["geometry"] not in ("unknown", "occupancy_features_only"):
        st = "geometry_extracted"
    slots = rec.get("slots") or {}
    reviewed = [s for n, s in slots.items() if s["tier"] in VERIFIED_SLOT and OFF_SKILL.match(n)]
    if reviewed:
        st = "assignment_reviewed"
    if rec.get("concepts_verified"):
        st = "concept_verified"
    if st == "concept_verified" and len(reviewed) >= MIN_VERIFIED_SKILL_SLOTS and not any(s["disagreement"] for s in slots.values()):
        st = "model_feature_eligible"
    return st


def funnel(recs: list[dict[str, Any]]) -> dict[str, int]:
    rank = {s: i for i, s in enumerate(STAGES)}
    ge = lambda r, s: rank[r["stage"]] >= rank[s]
    return {"indexed": len(recs), "art_available": sum(r["source_image"] != "unknown" for r in recs),
            "downloaded": 0,  # no authorized art store on this host; owner screenshots exist only as hashes here
            "geometry_extracted": sum(r["geometry"] not in ("unknown", "occupancy_features_only") for r in recs),
            "occupancy_features_only": sum(r["geometry"] == "occupancy_features_only" for r in recs),
            "human_reviewed": sum(any(s["tier"] in VERIFIED_SLOT for s in (r.get("slots") or {}).values()) for r in recs),
            "assignment_verified": sum(ge(r, "assignment_reviewed") for r in recs),
            "concept_verified": sum(ge(r, "concept_verified") for r in recs),
            "model_feature_eligible": sum(ge(r, "model_feature_eligible") for r in recs)}


def review_queue(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fastest-win first: confirm third-party-documented skill slots, then annotate unknown skill slots, then defense coverage."""
    q = []
    for r in recs:
        if r["side"] == "offense":
            for n, s in r["slots"].items():
                if s["tier"] in VERIFIED_SLOT or not OFF_SKILL.match(n) or n == "QB":
                    continue
                p = s["proposal"]
                pri = 1 if p["tier"] == "third_party_documented" else 3
                q.append({"priority": pri, "play_key": r["key"], "formation": r["formation"], "play": r["play"], "slot": n,
                          "task": "confirm_or_correct" if pri == 1 else "annotate", "proposed": p["assignment"], "proposed_side": p["side"],
                          "proposed_tier": p["tier"], "evidence_to_use": "user_screenshot:<file> of the in-game play art or user_recording:<file>",
                          "art_ref": r["source_image"]["image_id"] if isinstance(r["source_image"], dict) else "unknown"})
        else:
            q.append({"priority": 2, "play_key": r["key"], "formation": r["formation"], "play": r["play"], "slot": "coverage_shell",
                      "task": "confirm_or_correct", "proposed": f"{r['coverage_proposed']['family']}/{r['coverage_proposed']['type']}",
                      "proposed_side": "n/a", "proposed_tier": "name_inference", "evidence_to_use": "user_screenshot:<file> of the in-game coverage art",
                      "art_ref": "unknown"})
    return sorted(q, key=lambda x: (x["priority"], x["play_key"], x["slot"]))


def export(reviews: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Versioned, model-consumable export. Feature rows are populated ONLY for model_feature_eligible plays."""
    o, d = offense_records(reviews), defense_records()
    rows = []
    for r in o + d:
        row = {"key": r["key"], "stage": r["stage"], "eligible": r["stage"] == "model_feature_eligible", "features": {}}
        if row["eligible"]:
            row["features"] = {f"concept_{c['concept']}": 1.0 for c in r["concepts_verified"]}
            for n, s in r["slots"].items():
                if s["tier"] in VERIFIED_SLOT:
                    row["features"][f"route_{s['assignment']}"] = row["features"].get(f"route_{s['assignment']}", 0.0) + 1.0
        rows.append(row)
    body = {"schema": SCHEMA, "rights": RIGHTS, "stages": STAGES, "slot_tiers": SLOT_TIERS,
            "funnel": {"offense": funnel(o), "defense": funnel(d)}, "feature_rows": rows}
    body["digest"] = _digest(body)
    return body


def name_hint_accuracy() -> dict[str, Any]:
    """Real (tiny) accuracy check: name-token concept guesses vs concepts documented by an independent source for the same plays."""
    hits = tot = 0
    rows = []
    for p in _load("offense_assignments.json")["plays"]:
        doc = {c["name"] for c in p.get("concepts", []) if c.get("tier") == "source_documented"} - {"motion"}
        if not doc:
            continue
        guess = set((p.get("name_hint") or {}).get("concepts") or [])
        tot += 1
        hits += bool(doc & guess)
        rows.append({"play": f"{p['formation']} | {p['play']}", "documented": sorted(doc), "name_guess": sorted(guess)})
    return {"plays": tot, "any_overlap": hits, "rows": rows}
