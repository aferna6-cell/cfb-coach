"""Build / validate / load the Madden 27 defensive play knowledge catalog (``defense_play_catalog.json``).

The catalog is generated from ``defense_lexicon`` plus the owner's recorded defensive book, so it is reproducible and every
claim carries a source id and an evidence status. It records WHAT IS KNOWN, WHAT IS A FOOTBALL HYPOTHESIS and WHAT IS UNKNOWN
for each of the 123 installed formation/play entries and the 27 external variants. It never invents a defender assignment:
the 11-slot ledger of every entry is UNKNOWN because no first-party play-art evidence exists yet.

Catalog coverage (an entry exists) and responsibility verification (the eleven roles are proven) are reported separately.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from cfb_coach.madden.model import defense_lexicon as L

SCHEMA = "madden27.defense_play_catalog.v1"
CATALOG_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = ROOT / "cfb_coach" / "data" / "madden27" / "defense_play_catalog.json"
SNAPSHOT_PATH = ROOT / "cfb_coach" / "data" / "madden27" / "installed_playbook_snapshot.json"
ANNOTATIONS_PATH = ROOT / "cfb_coach" / "data" / "madden27" / "installed_play_annotations.json"
MATRIX_PATH = ROOT / "docs" / "madden_ml" / "DEFENSE_KNOWLEDGE_MATRIX.csv"

# Sources whose text/assignments must never reach model features (terms restrict reuse, or the claim is unverified).
NO_FEATURE_USE = frozenset({"mutgg_cover1", "mutgg_cover3_naming", "civil_zone_drop_hook", "prodigy_multiple_d",
                            "repo_third_party_annotations"})
LEVELS = ("catalog_only", "name_decomposed", "mechanic_documented", "role_partially_documented", "role_verified")
LEDGER_SLOTS = 11
ALIASES = {"Cov 3 Buzz Match": ("Cover 3 Buzz Match",), "Cover 3 Buzz Match": ("Cov 3 Buzz Match",)}


def slug(name: str) -> str:
    s = " ".join(str(name).lower().replace("-", " ").replace("_", " ").split())
    s = " ".join("cover" if w == "cov" else w for w in s.split())
    return s.replace(" ", "_")


def normalize_name(name: str) -> str:
    return slug(name).replace("_", " ")


# ------------------------------------------------------------------------------------------------ axis records

def _eligible(status: str, sources: Iterable[str | None]) -> bool:
    srcs = [s for s in sources if s]
    return status in L.FEATURE_ELIGIBLE_STATUS and all(s not in NO_FEATURE_USE for s in srcs)


def axis(value: Any, status: str, sources: Iterable[str | None], conf: str = "low", note: str = "") -> dict[str, Any]:
    srcs = [s for s in sources if s]
    out = {"value": value, "status": status, "sources": srcs, "confidence": conf, "feature_eligible": _eligible(status, srcs)}
    if note:
        out["note"] = note
    return out


def _unknown_axis(note: str) -> dict[str, Any]:
    return axis("unknown", "unknown", [], "low", note)


# ------------------------------------------------------------------------------------------------ structure of one name

def parse_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Structured coverage + pressure axes for a name spec. Pure; shared by the builder and the runtime fallback."""
    base = spec.get("base")
    variants = list(spec.get("variants") or [])
    cov: dict[str, Any] = {}
    b = L.BASES.get(base) if base else None
    if b:
        cov["deep_shell"] = axis(b["shell"], b["status"], [b["src"]], "medium", b["text"])
    elif spec.get("shell_name") is not None:
        cov["deep_shell"] = axis(spec["shell_name"], "name_derived", [], "low", "the label itself states the deep count")
    else:
        cov["deep_shell"] = _unknown_axis("the label names no coverage shell; a trailing digit is kept only as an unverified hint")
    # coverage type (how underneath defenders are tied to routes)
    if "match" in variants:
        ea = base == "c3"
        cov["type"] = axis("match", "ea_documented" if ea else "name_derived", ["ea_deep_dive"] if ea else [], "medium" if ea else "low",
                           "EA documents Cover 3 Match" if ea else "label says Match; EA does not list this family")
    elif "man" in variants:
        cov["type"] = axis("man", "name_derived", [], "medium", "label says Man")
    elif spec.get("spot"):
        cov["type"] = axis("spot_zone", "name_derived", [], "medium", "label says spot")
    elif "mable" in variants:
        cov["type"] = _unknown_axis("Mable is described as match by one unverified summary and as a zone/man mix by another")
    elif base == "c6" or (base == "c4" and ("quarters" in variants or "palms" in variants)):
        cov["type"] = axis("match", "ea_documented", ["ea_deep_dive"], "medium", "EA lists this family among the match-check coverages")
    elif base == "c4" and ("drop" in variants or "zone" in variants):
        cov["type"] = axis("spot_zone", "football_hypothesis", ["domain_convention"], "low", "Drop/Zone in the label suggests a spot-drop zone")
    elif b and b["type"] != "unknown":
        cov["type"] = axis(b["type"], b["status"], [b["src"]], "medium")
    else:
        cov["type"] = _unknown_axis("coverage type not established by the label or a cited source")
    cov["variants"] = []
    for v in variants:
        rec = L.VARIANTS[v]
        cov["variants"].append({"token": v, "present": axis(True, "name_derived", [], "high", "the token appears in the label"),
                                "meaning": rec["meaning"], "meaning_status": rec["status"], "meaning_sources": [rec["src"]] if rec["src"] else [],
                                "meaning_confidence": rec["conf"],
                                "meaning_feature_eligible": _eligible(rec["status"], [rec["src"]])})
    if spec.get("target"):
        cov["double_team_target"] = axis(spec["target"], "name_derived", [], "medium", "receiver named in the label")
    cov["base"] = base
    cov["family"] = b["family"] if b else None
    # pressure axis, kept separate from coverage
    p = spec.get("pressure") or "standard"
    pr = L.PRESSURE[p]
    pres: dict[str, Any] = {
        "concept": axis(p, pr["status"], [pr["src"]], pr["conf"], pr["meaning"]),
        "rusher_count": _unknown_axis("rusher count is not verified for any play"),
        "dropping_defenders": _unknown_axis("which defender drops is not verified"),
        "origin_tokens": [{"token": t, "status": "name_derived"} for t in spec.get("origin") or []],
        "classification_basis": spec.get("basis") or "name",
    }
    if spec.get("also"):
        pres["secondary_concept"] = axis(spec["also"], "name_derived", [], "low", "a second pressure token appears in the label")
    h = spec.get("hint")
    pres["digit_hint"] = ({"value": int(h), "status": "name_derived", "meaning": "UNVERIFIED: a trailing digit may indicate the deep-shell count behind the pressure",
                           "feature_eligible": True} if h is not None else {"value": None, "status": "unknown", "feature_eligible": False})
    return {"coverage": cov, "pressure": pres}


def legacy_bucket_for(spec: Mapping[str, Any]) -> str:
    """Parent class in the CURRENT offense model (``offense_play_features.BUCKET_TO_SCHEMES`` keys / scheme ids).

    Deliberately conservative: specific schemes only where the label establishes them, the generic parent bucket otherwise,
    and ``unknown`` for pressure-only names. This is the parent-sharing rule for variants the old model never saw.
    """
    base = spec.get("base")
    v = set(spec.get("variants") or [])
    if base == "c0":
        return "cover_0"
    if base == "c1":
        return "cover_1_hole" if "hole" in v else "cover_1_robber" if "robber" in v else "cover_1_contain" if "contain" in v else "cover1"
    if base == "t2":
        return "tampa_2"
    if base == "c2":
        return ("cover_2_man" if "man" in v else "cover_2_invert" if "invert" in v else "trap_2" if "trap" in v
                else "cover_2_spot" if spec.get("spot") else "cover2")
    if base == "c3":
        return "cover_3_match" if "match" in v else "cover_3_spot" if spec.get("spot") else "cover3"
    if base == "c4":
        return "cover_4_quarters" if "quarters" in v else "cover_4_palms" if "palms" in v else "cover_4_drop" if "drop" in v else "cover4"
    if base == "c6":
        return "cover_6"
    if base == "c9":
        return "cover_9"
    return "unknown"


# ------------------------------------------------------------------------------------------------ catalog build

def _installed_rows() -> tuple[list[tuple[str, str]], str, dict[str, str]]:
    snap = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))["sides"]["defense"]
    rows = [(f, p) for f, plays in snap["formations"].items() for p in plays]
    return rows, snap["inventory_fingerprint"], dict(snap["formation_sources"])


def _third_party_jobs() -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for a in json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))["annotations"]:
        if a.get("side") == "defense":
            out[(a["formation"], a["play"])] = [{"position": j["position"], "task": j["task"]} for j in a.get("jobs", [])]
    return out


def _stress(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    traits = [t for t in [spec.get("base"), *(spec.get("variants") or []), spec.get("pressure"), spec.get("also")] if t]
    out = []
    for t in dict.fromkeys(traits):
        key = {"man": "man", "match": "match"}.get(t, t)
        if key in L.STRESS:
            out.append({"trait": t, "stress": L.STRESS[key], "limitation": L.STRESS_LIMITATION, "status": "football_hypothesis",
                        "sources": ["domain_convention"], "model_input": False})
    return out


def _unknowns(spec: Mapping[str, Any], parsed: Mapping[str, Any]) -> list[str]:
    u = ["All 11 defender alignments and assignments: no first-party Xbox play-art evidence (screenshot hash) exists for this play.",
         "Opponent pre-snap adjustments (press, shade, drops, match overrides, user control): unobservable while on offense; NOT assumed absent."]
    for tok in spec.get("unknown_tokens") or []:
        u.append(f"Meaning of the token '{tok}' in this call is unknown.")
    for v in parsed["coverage"]["variants"]:
        if v["meaning_status"] in ("unknown", "third_party_unverified", "football_hypothesis") and v["meaning_confidence"] != "high":
            u.append(f"Madden 27 implementation of '{v['token']}' ({v['meaning_status']}): {v['meaning']}.")
    if parsed["coverage"]["deep_shell"]["value"] == "unknown":
        u.append("Coverage behind the pressure is unknown (a trailing digit is only a hint).")
    u.append("Rusher count and which defender drops are unverified.")
    return u


def _sibling_text(here: set[str], there: set[str]) -> list[str]:
    texts = []
    for a in sorted(here - there):
        for b in sorted(there - here):
            note = L.SIBLING_NOTES.get(frozenset({a, b}))
            if note:
                texts.append(note)
    return texts


def build_catalog() -> dict[str, Any]:
    rows, fp, book = _installed_rows()
    jobs = _third_party_jobs()
    entries: list[dict[str, Any]] = []
    specs: dict[str, Mapping[str, Any]] = {}

    def make(eid: str, raw: str, formation: str | None, status: str) -> dict[str, Any]:
        spec = L.NAME_SPECS[raw]
        parsed = parse_spec(spec)
        specs[eid] = spec
        tp = jobs.get((formation, raw)) if formation else None
        mech = any(a.get("status") == "ea_documented" and a.get("feature_eligible") for a in
                   [parsed["coverage"]["deep_shell"], parsed["coverage"]["type"], parsed["pressure"]["concept"]]
                   ) or any(v["meaning_status"] == "ea_documented" for v in parsed["coverage"]["variants"])
        decomposed = bool(spec.get("base") or spec.get("variants") or spec.get("hint") is not None or spec.get("origin")
                          or (spec.get("pressure") != "standard" and spec.get("basis") != "repo_audit_115"))
        level = ("role_partially_documented" if tp else "mechanic_documented" if mech else "name_decomposed" if decomposed else "catalog_only")
        ev = [{"source": "repo_snapshot", "claim": "installed in the owner's recorded defensive book (DEF-F1CF-1862)"}] if status == "installed" else \
             [{"source": "domain_convention", "claim": "named in the issue's required external-variant list"}]
        if raw in L.EXTERNAL_OBSERVED_FORMATIONS:
            ev.append({"source": "prodigy_multiple_d", "claim": f"name listed under '{L.EXTERNAL_OBSERVED_FORMATIONS[raw]}' (existence only; page read partially)"})
        entry = {
            "id": eid, "raw_name": raw, "raw_formation": formation, "status": status,
            "source_book": book.get(formation) if formation else None,
            "name_key": slug(raw), "aliases": list(ALIASES.get(raw, ())), "possible_duplicate_of": None,
            "formation_facts": None,
            "coverage": parsed["coverage"], "pressure": parsed["pressure"],
            "legacy_bucket": legacy_bucket_for(spec),
            "ledger": {"slots": [{"slot": f"S{i}", "position": None, "alignment": None, "assignment": None, "status": "unknown"}
                                 for i in range(1, LEDGER_SLOTS + 1)], "known_count": 0, "verified_count": 0,
                       "third_party_reference": ({"source": "repo_third_party_annotations", "jobs": tp, "model_input": False,
                                                  "note": "published by a site whose terms restrict ML reuse; research display only"} if tp else None)},
            "offensive_stress_hypotheses": _stress(spec),
            "unknowns": _unknowns(spec, parsed),
            "evidence": ev,
            "verification": {"level": level, "needs_xbox_screenshot": True, "verified_slots": 0, "screenshot_sha256": None},
            "review": {"status": "unreviewed", "reviewer": None},
            "licensed_for_training": {"name_tokens": True, "ea_documented_mechanics": True, "third_party_text": False},
            "catalog_formations_observed": ([L.EXTERNAL_OBSERVED_FORMATIONS[raw]] if raw in L.EXTERNAL_OBSERVED_FORMATIONS else []),
        }
        if formation in L.FORMATIONS:
            f = L.FORMATIONS[formation]
            entry["formation_facts"] = {"book": f["book"], "dl": f["dl"], "lb": f["lb"], "db": f["db"],
                                        "status": "football_hypothesis", "sources": [f["basis"]], "note": f["note"]}
        return entry

    for formation, raw in rows:
        entries.append(make(f"inst:{formation}|{raw}", raw, formation, "installed"))
    for raw in L.EXTERNAL_NAMES:
        entries.append(make(f"ext:{raw}", raw, None, "external_identified"))
    by_slug: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_slug.setdefault(e["name_key"], []).append(e)
    for e in entries:
        twins = [t for t in by_slug[e["name_key"]] if t["raw_name"] != e["raw_name"]]
        if twins:
            e["possible_duplicate_of"] = twins[0]["id"]
    # siblings and explicit differences
    def tokens(e: Mapping[str, Any]) -> set[str]:
        t = {v["token"] for v in e["coverage"]["variants"]}
        if e["pressure"]["concept"]["value"] != "standard":
            t.add("pressure:" + str(e["pressure"]["concept"]["value"]))
        if e["pressure"].get("secondary_concept"):
            t.add("pressure:" + str(e["pressure"]["secondary_concept"]["value"]))
        t |= {"origin:" + o["token"].lower() for o in e["pressure"]["origin_tokens"]}
        if e["pressure"]["digit_hint"]["value"] is not None:
            t.add("hint:" + str(e["pressure"]["digit_hint"]["value"]))
        if e["coverage"].get("double_team_target"):
            t.add("target:" + str(e["coverage"]["double_team_target"]["value"]).lower())
        t |= {"unknown_token:" + u for u in specs[e["id"]].get("unknown_tokens", [])}
        return t

    def group(e: Mapping[str, Any]) -> tuple:
        return (e["coverage"]["family"],) if e["coverage"]["family"] else ("pressure", e["pressure"]["concept"]["value"])

    for e in entries:
        mine = tokens(e)
        cands = []
        for o in entries:
            if o is e or o["name_key"] == e["name_key"] or group(o) != group(e):
                continue
            theirs = tokens(o)
            cands.append((len(mine ^ theirs), o["raw_formation"] != e["raw_formation"], o["id"], o, theirs))
        cands.sort(key=lambda t: t[:3])
        chosen, seen = [], set()
        for dist, _f, _i, o, theirs in cands:                       # nearest structure first (fewest differing tokens)
            if o["name_key"] not in seen and len(chosen) < 4:
                seen.add(o["name_key"])
                chosen.append((o, theirs))
        for dist, _f, _i, o, theirs in cands:                       # plus any sibling with an explicitly documented contrast
            if o["name_key"] not in seen and len(chosen) < 6 and _sibling_text(mine, theirs):
                seen.add(o["name_key"])
                chosen.append((o, theirs))
        e["nearest_siblings"] = [{"entry": o["id"], "raw_name": o["raw_name"], "raw_formation": o["raw_formation"],
                                  "only_here": sorted(mine - theirs), "only_there": sorted(theirs - mine),
                                  "documented_difference": _sibling_text(mine, theirs),
                                  "unverified": "Any further difference in defender roles is unknown without first-party play art."}
                                 for o, theirs in chosen]
        e["same_name_other_formations"] = sorted({t["raw_formation"] for t in by_slug[e["name_key"]]
                                                  if t["raw_formation"] and t["raw_formation"] != e["raw_formation"] and t["raw_name"] == e["raw_name"]})
    summary = _summary(entries)
    catalog = {
        "schema": SCHEMA, "version": CATALOG_VERSION, "game": "Madden NFL 27", "built_for_fingerprint": fp,
        "truth_scope": ("A source-backed STRUCTURAL catalog. It proves which plays exist and what their names, EA-documented mechanics and cited "
                        "football definitions say. It does NOT prove any eleven-defender assignment: ledger_known_count is 0 for every entry."),
        "status_vocabulary": list(L.STATUS_ORDER), "feature_eligible_statuses": list(L.FEATURE_ELIGIBLE_STATUS),
        "feature_policy": ("Model features may use only axis values with feature_eligible=true (name tokens, EA-documented mechanics, licensed "
                           "football definitions). Third-party text and per-position jobs are never model inputs. Stress hypotheses are never inputs."),
        "sources": {k: dict(v, feature_use=(k not in NO_FEATURE_USE),
                            accessed=(v.get("accessed") or (L.ACCESSED if v["fetched"] else None)))
                    for k, v in sorted(L.SOURCES.items())},
        "lexicon": {"bases": L.BASES, "variants": L.VARIANTS, "pressure": L.PRESSURE, "formations": L.FORMATIONS},
        "summary": summary, "entries": entries,
    }
    catalog["digest"] = digest_of(catalog)
    return catalog


def _summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    lv: dict[str, int] = {}
    for e in entries:
        lv[e["verification"]["level"]] = lv.get(e["verification"]["level"], 0) + 1
    inst = [e for e in entries if e["status"] == "installed"]
    return {"entries": len(entries), "installed": len(inst), "external": len(entries) - len(inst),
            "unique_names": len({e["name_key"] for e in entries}), "unique_installed_names": len({e["name_key"] for e in inst}),
            "verification_levels": dict(sorted(lv.items())), "ledger_known_slots": sum(e["ledger"]["known_count"] for e in entries),
            "ledger_verified_slots": sum(e["ledger"]["verified_count"] for e in entries),
            "entries_needing_xbox_screenshot": sum(e["verification"]["needs_xbox_screenshot"] for e in entries),
            "pressure_concepts": dict(sorted(_count(e["pressure"]["concept"]["value"] for e in entries).items())),
            "deep_shell": dict(sorted(_count(str(e["coverage"]["deep_shell"]["value"]) for e in entries).items())),
            "coverage_type": dict(sorted(_count(e["coverage"]["type"]["value"] for e in entries).items()))}


def _count(vals: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return out


def digest_of(catalog: Mapping[str, Any]) -> str:
    body = {k: v for k, v in catalog.items() if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]


def write_catalog(catalog: Mapping[str, Any], path: Path = CATALOG_PATH) -> Path:
    path.write_text(json.dumps(catalog, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_catalog(path: Path | str = CATALOG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ validation

def validate_catalog(cat: Mapping[str, Any]) -> list[str]:
    """Every reason this catalog must not be trusted (empty = valid). Enforces coverage, provenance and the no-fabrication gates."""
    errs: list[str] = []
    if cat.get("schema") != SCHEMA:
        errs.append("wrong schema")
    if cat.get("digest") != digest_of(cat):
        errs.append("digest does not match content (edited by hand?)")
    rows, fp, _book = _installed_rows()
    if cat.get("built_for_fingerprint") != fp:
        errs.append(f"built for {cat.get('built_for_fingerprint')} but the recorded book is {fp}")
    entries = cat.get("entries") or []
    ids = [e.get("id") for e in entries]
    if len(set(ids)) != len(ids):
        errs.append("duplicate entry ids")
    inst = {(e["raw_formation"], e["raw_name"]) for e in entries if e.get("status") == "installed"}
    if inst != set(rows):
        errs.append(f"installed entries differ from the recorded book: missing {sorted(set(rows) - inst)[:3]} extra {sorted(inst - set(rows))[:3]}")
    if sum(1 for e in entries if e.get("status") == "installed") != len(rows):
        errs.append("installed entry count differs from the recorded book")
    ext = {e["raw_name"] for e in entries if e.get("status") == "external_identified"}
    if ext != set(L.EXTERNAL_NAMES):
        errs.append(f"external variants differ from the required list: {sorted(set(L.EXTERNAL_NAMES) ^ ext)}")
    srcs = cat.get("sources") or {}
    for e in entries:
        eid = e.get("id")
        for need in ("raw_name", "name_key", "coverage", "pressure", "ledger", "unknowns", "evidence", "verification", "review",
                     "nearest_siblings", "offensive_stress_hypotheses", "licensed_for_training"):
            if need not in e:
                errs.append(f"{eid}: missing {need}")
        if not e.get("evidence"):
            errs.append(f"{eid}: no evidence recorded")
        axes = [e["coverage"]["deep_shell"], e["coverage"]["type"], e["pressure"]["concept"], e["pressure"]["rusher_count"],
                e["pressure"]["dropping_defenders"]]
        for ax in axes:
            if ax.get("status") not in L.STATUS_ORDER:
                errs.append(f"{eid}: bad status {ax.get('status')!r}")
            for s in ax.get("sources", []):
                if s not in srcs:
                    errs.append(f"{eid}: unknown source {s!r}")
            if ax.get("status") == "xbox_verified" and not e["verification"].get("screenshot_sha256"):
                errs.append(f"{eid}: xbox_verified claim without a screenshot hash")
            if ax.get("feature_eligible") and (ax.get("status") not in L.FEATURE_ELIGIBLE_STATUS or any(s in NO_FEATURE_USE for s in ax.get("sources", []))):
                errs.append(f"{eid}: feature-eligible claim rests on a non-trainable source or status")
        led = e["ledger"]
        if len(led["slots"]) != LEDGER_SLOTS:
            errs.append(f"{eid}: ledger must have {LEDGER_SLOTS} slots")
        known = sum(1 for s in led["slots"] if s["status"] != "unknown")
        if known != led["known_count"] or led["verified_count"] != sum(1 for s in led["slots"] if s["status"] == "xbox_verified"):
            errs.append(f"{eid}: ledger counts inconsistent")
        if led["verified_count"] and not e["verification"].get("screenshot_sha256"):
            errs.append(f"{eid}: verified slots without a screenshot hash")
        if led.get("third_party_reference") and led["third_party_reference"].get("model_input"):
            errs.append(f"{eid}: third-party jobs flagged as a model input")
        if e["verification"]["level"] not in LEVELS:
            errs.append(f"{eid}: bad verification level")
        if e["verification"]["level"] == "role_verified" and led["verified_count"] != LEDGER_SLOTS:
            errs.append(f"{eid}: role_verified requires all {LEDGER_SLOTS} slots verified")
        if any(h.get("model_input") for h in e["offensive_stress_hypotheses"]):
            errs.append(f"{eid}: a stress hypothesis is marked as a model input")
    return errs


# ------------------------------------------------------------------------------------------------ review matrix

MATRIX_COLUMNS = ("id", "status", "formation", "source_book", "raw_name", "name_key", "base", "deep_shell", "deep_shell_status", "type", "type_status",
                  "variants", "pressure_concept", "digit_hint", "origin_tokens", "verification_level", "nearest_sibling", "verified_difference",
                  "unknown_count", "needs_xbox_screenshot", "ledger_known_slots", "evidence_sources", "review_status", "legacy_bucket")


def matrix_rows(cat: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for e in cat["entries"]:
        sibs = e["nearest_siblings"] or [{}]
        s = next((x for x in sibs if x.get("documented_difference")), sibs[0])
        diff = ("; ".join(s.get("documented_difference", []))
                or ("name-level only: here %s vs there %s (no cited difference in defender roles)" % (s.get("only_here") or "-", s.get("only_there") or "-") if s else ""))
        out.append({
            "id": e["id"], "status": e["status"], "formation": e["raw_formation"] or "", "source_book": e["source_book"] or "", "raw_name": e["raw_name"],
            "name_key": e["name_key"], "base": e["coverage"]["base"] or "", "deep_shell": e["coverage"]["deep_shell"]["value"],
            "deep_shell_status": e["coverage"]["deep_shell"]["status"], "type": e["coverage"]["type"]["value"], "type_status": e["coverage"]["type"]["status"],
            "variants": ",".join(v["token"] for v in e["coverage"]["variants"]), "pressure_concept": e["pressure"]["concept"]["value"],
            "digit_hint": "" if e["pressure"]["digit_hint"]["value"] is None else e["pressure"]["digit_hint"]["value"],
            "origin_tokens": ",".join(t["token"] for t in e["pressure"]["origin_tokens"]), "verification_level": e["verification"]["level"],
            "nearest_sibling": f"{s.get('raw_name', '')} [{s.get('raw_formation') or 'external'}]" if s else "", "verified_difference": diff,
            "unknown_count": len(e["unknowns"]), "needs_xbox_screenshot": e["verification"]["needs_xbox_screenshot"],
            "ledger_known_slots": e["ledger"]["known_count"],
            "evidence_sources": ",".join(sorted({x["source"] for x in e["evidence"]} | {s2 for ax in (e["coverage"]["deep_shell"], e["coverage"]["type"]) for s2 in ax.get("sources", [])})),
            "review_status": e["review"]["status"], "legacy_bucket": e["legacy_bucket"]})
    return out


def write_matrix(cat: Mapping[str, Any], path: Path = MATRIX_PATH) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MATRIX_COLUMNS)
        w.writeheader()
        w.writerows(matrix_rows(cat))
    return path
