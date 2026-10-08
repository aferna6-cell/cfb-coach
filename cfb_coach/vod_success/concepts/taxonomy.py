"""Offense call → concept → family, and defense string → coverage + pressure.

Name key = lowercase, keep only ``[a-z0-9]`` (the same key as ``catalog.norm``).
A name no rule supports stays unmapped. Nothing here guesses a concept, and
nothing here reads a machine-specific data path.
"""

from __future__ import annotations

import collections
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

CONF_MIN = 0.5
SCHEMA = "vod_concepts.v1"

FAMILY_OF = {
    "inside_zone": "inside_run", "duo": "inside_run", "power": "inside_run", "counter": "inside_run",
    "trap": "inside_run", "dive": "inside_run", "iso_lead": "inside_run", "draw": "inside_run",
    "base_run": "inside_run", "option_read": "inside_run",
    "outside_zone": "outside_run", "toss_sweep": "outside_run", "off_tackle": "outside_run",
    "option_pitch": "outside_run",
    "mesh": "quick_pass", "stick": "quick_pass", "curl_flat": "quick_pass", "curls_hitch": "quick_pass",
    "slants": "quick_pass", "outs": "quick_pass", "snag_spacing": "quick_pass", "pivot_whip": "quick_pass",
    "rb_route": "quick_pass", "bench": "quick_pass",
    "smash": "intermediate", "flood_sail": "intermediate", "levels": "intermediate", "drive": "intermediate",
    "dig": "intermediate", "crossers": "intermediate", "corner": "intermediate", "wheel": "intermediate",
    "boot": "intermediate",
    "dagger": "deep", "verts": "deep", "post": "deep", "mills": "deep", "fade": "deep", "scissors": "deep",
    "double_move": "deep", "hail_mary": "deep",
    "screen": "screen", "jet_touch_pass": "screen",
    "rpo_bubble": "rpo", "rpo_screen": "rpo", "rpo_slant_glance": "rpo", "rpo_alert": "rpo",
    "rpo_read": "rpo", "rpo_other": "rpo",
    "kneel_spike": "non_play",
}
LOW_CONF_CONCEPTS = {"bench", "rb_route", "base_run", "wheel", "corner", "snag_spacing", "jet_touch_pass", "boot"}
PASS_RULES = [
    ("hail_mary", r"hail mary"),
    ("double_move", r"\bn go\b|\bn nod\b|\bnod vertical|stick nod|sluggo"),
    ("mesh", r"\bmesh\b"),
    ("smash", r"smash"),
    ("flood_sail", r"flood|\bsail\b|saildig|shallowsail"),
    ("dagger", r"dagger"),
    ("levels", r"\blevels\b"),
    ("mills", r"\bmillsy?\b"),
    ("drive", r"\bdrive\b"),
    ("verts", r"\bverts?\b|verticals|verti cals|\bseams?\b|xseam|wrseam|\ball go\b|\bstreaks?\b"),
    ("stick", r"stick"),
    ("curl_flat", r"curl ?flats?\b|curlflat"),
    ("curls_hitch", r"\bcurls?\b|\bhitch\b|\bstops\b|comebacks?\b"),
    ("post", r"(double|dbl|deep|slot|cheat flat x|y-?) ?post\b|post shot|\bpost (dig|cross)\b|^post\b"),
    ("bench", r"\bbench\b"),
    ("pivot_whip", r"\bpivot\b|\bwhips?\b"),
    ("dig", r"\bdigs?\b|\bdeep in\b|\bins?\b|\bdbl ins\b|z-?dig"),
    ("snag_spacing", r"spacing|\bsnag\b|\bspot\b"),
    ("slants", r"\bslants?\b|slantswing"),
    ("outs", r"\bouts?\b|\bx-?out\b|\btech out\b"),
    ("crossers", r"cross|crossers|shallow|\bdrag\b|\bover\b"),
    ("corner", r"\bcorners?\b"),
    ("fade", r"\bfades?\b"),
    ("scissors", r"scissors"),
    ("wheel", r"wheel"),
    ("rb_route", r"\bangle\b|\btexas\b|\bswing\b|\bunder\b|\bchoice\b|\brail\b|\bdelay\b|\bleak\b|\bflat\b|y-?option|wr option|slot option"),
]
RUN_RULES = [
    ("draw", r"\bdraw\b"),
    ("counter", r"counter|\bctr\b|\bwham\b"),
    ("trap", r"\btrap\b"),
    ("power", r"power"),
    ("duo", r"\bduo|duowrap"),
    ("outside_zone", r"outside zone|wide zone|stretch"),
    ("toss_sweep", r"toss|sweep|pitch|reverse|end around|crack"),
    ("off_tackle", r"off ?tackl"),
    ("inside_zone", r"inside zone|\bzone\b|zone split|split zone|\bhb inside\b"),
    ("iso_lead", r"\biso\b|\blead\b|\bslam\b|\bblast\b|\bpunch\b|plunge"),
    ("dive", r"\bdive\b|\bgut\b|\bsneak\b|\bfb inside\b"),
    ("base_run", r"\bhb base\b|quick base|^base$"),
]
EXACT = {
    "inside zone": "inside_zone", "hb zone": "inside_zone", "zone wk": "inside_zone", "hb zone wk": "inside_zone",
    "zone weak": "inside_zone", "outside zone": "outside_zone", "wide zone": "outside_zone", "hb stretch": "outside_zone",
    "stretch": "outside_zone", "hb power o": "power", "power o": "power", "hb power": "power", "hb power g": "power",
    "counter": "counter", "counter y": "counter", "hb counter": "counter", "hb dive": "dive", "fb dive": "dive",
    "hb draw": "draw", "hb duo": "duo", "hb iso": "iso_lead", "hb slam": "iso_lead", "mesh": "mesh", "smash": "smash",
    "flood": "flood_sail", "dagger": "dagger", "levels": "levels", "drive": "drive", "verticals": "verts",
    "four verticals": "verts", "stick": "stick", "curl flat": "curl_flat", "curl flats": "curl_flat",
    "curls": "curls_hitch", "slants": "slants", "quick slants": "slants", "spacing": "snag_spacing",
    "hb slip screen": "screen", "read option": "option_read", "speed option": "option_pitch",
    "hail mary": "hail_mary", "pa boot": "boot",
}
COVERAGE_FAMILIES = [
    "cover_0", "cover_1", "cover_2", "cover_2_man", "tampa_2", "cover_3", "cover_4", "cover_6", "cover_9",
    "man_other", "zone_other", "prevent",
]
PRESSURE_BLITZ = (
    r"blitz|fire|\bdog\b|sting|crash|\bedge\b|pinch|\bbuck\b|\bzero\b|\bmug\b|shoot|\bgaps?\b|"
    r"overload|storm|thunder|\bstunt\b|casino|boca|smoke|\bgo fire\b|\bdt pop\b"
)
PLAIN_WORDS = set(
    "cover cov tampa quarters palms drop cloud sky skywk buzz hard flat invert press contain hole robber match lock "
    "willie trap zone show str wk field mike lurk roll double wr1 wr2 deep prevent man nickel dime".split()
)
PRESSURE_SIM = r"\bsim\b|sim pressure|creeper|simpressure"


def nkey(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _spaced(name: str) -> str:
    text = re.sub(r"[_]+", " ", name.lower()).strip()
    return re.sub(r"\s+", " ", text)


def offense_concept(name: str) -> tuple[str | None, dict[str, Any]]:
    """Return ``(concept or None, info)`` for one display name."""
    text = _spaced(name)
    info: dict[str, Any] = {"pa": False, "rpo": False, "matches": []}
    if re.search(r"\bkneel\b|\bspike\b", text):
        return "kneel_spike", dict(info, rule="kneel/spike", n_rules=1)
    if text in EXACT:
        concept = EXACT[text]
        return concept, dict(info, pa=text.startswith("pa "), rpo=False, rule=f"exact:{text}", n_rules=1, exact=True)
    if re.search(r"\brpo|rpoalert", text) or re.search(
        r"\balert (bubble|smoke|screen|swing|flat|slant|out|split|snag|cheat|qb)", text
    ):
        info["rpo"] = True
        for concept, pattern in (
            ("rpo_bubble", r"bubble|smoke"),
            ("rpo_screen", r"screen|swing"),
            ("rpo_slant_glance", r"slant|glance"),
            ("rpo_alert", r"alert"),
            ("rpo_read", r"\bread\b"),
        ):
            if re.search(pattern, text):
                return concept, dict(info, rule=f"rpo:{pattern}", n_rules=1)
        return "rpo_other", dict(info, rule="rpo", n_rules=1)
    if re.search(r"screen|jailbreak|tunnel", text):
        return "screen", dict(info, rule="screen", n_rules=1)
    if re.search(r"touch pass|jet pass|shovel", text):
        return "jet_touch_pass", dict(info, rule="touch/jet pass", n_rules=1)
    if re.search(r"(speed|load|triple|shock ?h?) option", text):
        return "option_pitch", dict(info, rule="speed/load/triple/shock option", n_rules=1)
    if re.search(r"(^|hb |qb |mtn |close )(zone )?read option|zone read|midline|\bveer\b", text) and not re.search(
        r"\by[ -]?read", text
    ):
        return "option_read", dict(info, rule="read option", n_rules=1)
    if re.search(r"\bpa\b|play action|\bboot\b|waggle|bootleg", text):
        info["pa"] = True
    hits = [(concept, pattern) for concept, pattern in PASS_RULES if re.search(pattern, text)]
    if hits:
        concept = hits[0][0]
        return concept, dict(
            info, rule=f"pass:{hits[0][1]}", n_rules=len({hit[0] for hit in hits}), matches=[hit[0] for hit in hits]
        )
    if info["pa"]:
        if re.search(r"\bboot\b|waggle|bootleg|\bslide\b", text):
            return "boot", dict(info, rule="pa boot/waggle/slide", n_rules=1)
        return None, dict(info, reason="play-action call whose route concept the name does not give")
    hits = [(concept, pattern) for concept, pattern in RUN_RULES if re.search(pattern, text)]
    if hits:
        concept = hits[0][0]
        return concept, dict(
            info, rule=f"run:{hits[0][1]}", n_rules=len({hit[0] for hit in hits}), matches=[hit[0] for hit in hits]
        )
    return None, dict(info, reason="no rule matches the name")


def defense_map(name: str) -> tuple[str | None, str | None, str | None, dict[str, Any]]:
    """Return ``(coverage_family, variant, pressure, info)`` for one defensive string."""
    raw = name
    cleaned = re.sub(r"\(.*?\)", " ", name)
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9;]", " ", cleaned.lower())).strip()
    key = nkey(name)
    info: dict[str, Any] = {}
    if ";" in raw or re.fullmatch(r"(seamflat|deephalf|curlflat|cloudflat|mancoverage|hookcurl|deepthird|deepquarter)+", key):
        man = "mancoverage" in key
        half = "deephalf" in key
        if half and man:
            return "cover_2_man", "man", "none", dict(info, method="overlay_pattern", confidence="low", rule="DEEPHALF+MANCOVERAGE")
        if half:
            return "cover_2", "zone", "none", dict(info, method="overlay_pattern", confidence="low", rule="DEEPHALF zone drops")
        if man:
            return "man_other", "man", None, dict(info, method="overlay_pattern", confidence="low", rule="MANCOVERAGE only")
        return "zone_other", "zone", None, dict(info, method="overlay_pattern", confidence="low", rule="zone drops only")
    if re.search(r"prevent", text):
        return "prevent", "zone", "none", dict(info, method="exact_rule", confidence="high", rule="prevent")
    merged = " " not in raw.strip() and len(key) > 14
    pressure = None
    if re.search(PRESSURE_BLITZ, text) or re.search(r"blitz|fire", key):
        pressure = "blitz"
    elif re.search(PRESSURE_SIM, text) or (key.split("pressure")[0][-3:] if "pressure" in key else "").find("sim") >= 0:
        pressure = "sim"
    if re.search(PRESSURE_SIM, text) or "simpressure" in key or re.search(r"\bsim\b", text):
        pressure = "sim"
    variant = "match" if "match" in text else None
    family = None
    rule = None
    if "tampa" in key:
        family, rule = "tampa_2", "tampa"
    elif re.search(r"cov(er)?0|zero|blitz0", key) or re.search(r"\b(blitz|buck|pinch|fire|dog) ?[0o]$", text) or (
        re.search(r"\b[0o]$", text) and pressure == "blitz"
    ):
        family, rule = "cover_0", "cover 0 / zero / trailing 0"
    elif re.search(r"cov(er)?2man|\b2 man\b", key + " " + text):
        family, rule = "cover_2_man", "cover 2 man"
    else:
        found = re.search(r"cov(er)?s?(\d)", key)
        if found:
            family, rule = f"cover_{found.group(2)}", "cover N"
        elif re.search(r"quarters|palms", key):
            family, rule = "cover_4", "quarters/palms"
        elif re.search(r"robber|\bhole\b", text):
            family, rule = "cover_1", "robber/hole"
        elif re.search(r"\bman 3 deep\b", text):
            family, rule = "man_other", "man 3 deep"
        elif re.search(r"bracket", text):
            digit = re.search(r"\b(\d)\b", text)
            if digit:
                family, rule = f"cover_{digit.group(1)}", "bracket N"
        else:
            found = re.search(r"(^|\s)(\d)(\s|$)", text)
            if found is None and re.match(r"\d", key):
                found = re.match(r"(\d)", key)
                digit = found.group(1) if found else None
            else:
                digit = found.group(2) if found else None
            if digit in "1234679":
                family, rule = f"cover_{digit}", "trailing/leading coverage number"
    if family is None and re.search(r"\bman\b", text):
        family, rule = "man_other", "man"
    if family and re.fullmatch(r"cover_\d", family) and family[-1] not in "0123469":
        family = None
    if family and variant is None:
        variant = "man" if family in ("cover_0", "cover_1", "cover_2_man", "man_other") else "zone"
    if pressure is None and family:
        tokens = [token for token in re.split(r"[ ;]+", text) if token]
        if merged:
            pressure = "none" if re.fullmatch(r"(cover|cov)\d(man|hardflat|contain|containpress|skydrop|buzzmable|tampadrop|)", key) else None
        elif all(token in PLAIN_WORDS or token.isdigit() for token in tokens):
            pressure = "none"
    if family is None and pressure is None:
        return None, None, None, dict(info, reason="no coverage or pressure rule matches")
    if family and rule in ("cover N", "tampa", "cover 2 man", "quarters/palms") and not merged:
        confidence = "high"
    elif not merged:
        confidence = "med"
    else:
        confidence = "low"
    if family is None:
        confidence = "med" if not merged else "low"
    method = "exact_rule" if confidence == "high" else ("name_pattern_merged_ocr" if merged else "name_pattern")
    return family, variant, pressure, dict(info, method=method, confidence=confidence, rule=rule or "pressure words only")


def _game(value: Any) -> str:
    text = nkey(str(value or ""))
    if "cfb" in text or "college" in text:
        return "cfb27"
    if text in {"", "madden", "madden27", "m27"}:
        return "madden27"
    return text or "madden27"


def _confidence(row: dict[str, Any], field: str) -> float:
    blob = row.get("conf")
    if isinstance(blob, dict) and blob.get(field) is not None:
        try:
            return float(blob[field])
        except (TypeError, ValueError):
            return 0.0
    named = row.get(f"conf_{field}")
    if named is None and field == "play":
        named = row.get("conf_play")
    if named is None and field in {"def_call", "coverage_seen", "def_overlay"}:
        named = row.get("conf_coverage") or row.get("conf_look")
    if named is None:
        return 1.0
    try:
        return float(named)
    except (TypeError, ValueError):
        return 0.0


def _play_name(row: dict[str, Any]) -> str:
    call = row.get("offense_call")
    if isinstance(call, dict) and call.get("play"):
        return str(call["play"])
    for key in ("play", "call", "our_call"):
        if row.get(key):
            return str(row[key])
    return ""


def _defense_strings(row: dict[str, Any]) -> list[tuple[str, str]]:
    look = row.get("defense_look")
    if isinstance(look, dict):
        out = []
        for field in ("def_call", "coverage_seen", "def_overlay"):
            value = look.get(field)
            if value:
                out.append((field, str(value)))
        return out
    out = []
    if row.get("def_call"):
        out.append(("def_call", str(row["def_call"])))
    coverage = row.get("coverage_seen") or row.get("coverage") or row.get("look")
    if coverage:
        out.append(("coverage_seen", str(coverage)))
    if row.get("def_overlay"):
        out.append(("def_overlay", str(row["def_overlay"])))
    return out


def _known_success(row: dict[str, Any]) -> bool:
    if "success" not in row:
        return False
    value = row.get("success")
    if value is None or value == "":
        return False
    return True


def build_taxonomy(
    rows: list[dict[str, Any]],
    *,
    version: str,
    label_version: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Map every offense and defense name in ``rows``. Returns ``(document, unmapped)``."""
    offense_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    offense_known: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    spellings: dict[str, dict[str, collections.Counter]] = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter)
    )
    sides: dict[str, dict[str, collections.Counter]] = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter)
    )
    defense_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    defense_fields: dict[str, dict[str, collections.Counter]] = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter)
    )
    for row in rows:
        if row.get("side") == "special_teams" or row.get("is_kickoff"):
            continue
        game = _game(row.get("game"))
        play = _play_name(row)
        if play and _confidence(row, "play") >= CONF_MIN:
            key = nkey(play)
            if key:
                offense_counts[game][key] += 1
                spellings[game][key][play] += 1
                sides[game][key][str(row.get("side") or "unknown")] += 1
                if _known_success(row):
                    offense_known[game][key] += 1
        for field, value in _defense_strings(row):
            if value and _confidence(row, field) >= CONF_MIN:
                key = nkey(value)
                if key:
                    defense_counts[game][key] += 1
                    defense_fields[game][key][(field, value)] += 1

    offense: dict[str, dict[str, Any]] = collections.defaultdict(dict)
    defense: dict[str, dict[str, Any]] = collections.defaultdict(dict)
    unmapped: list[dict[str, Any]] = []
    for game in sorted(offense_counts):
        for key, count in offense_counts[game].most_common():
            raw = spellings[game][key].most_common(1)[0][0]
            nice = [spelling for spelling, _n in spellings[game][key].most_common() if not spelling.isupper()]
            raw = nice[0] if nice else raw
            concept, info = offense_concept(raw)
            record: dict[str, Any] = {
                "raw_spellings": dict(spellings[game][key]),
                "canonical_name": raw,
                "canonical_source": "raw label text",
                "n_rows": count,
                "n_rows_known_success": offense_known[game][key],
                "by_offense_side": dict(sides[game][key]),
                "concept": concept,
                "family": FAMILY_OF.get(concept) if concept else None,
                "pa": bool(info.get("pa")),
                "rpo": bool(info.get("rpo")),
            }
            if concept:
                if info.get("exact"):
                    method, confidence = "exact_rule", "high"
                else:
                    method = "name_pattern"
                    if info.get("n_rules", 1) == 1 and concept not in LOW_CONF_CONCEPTS:
                        confidence = "high"
                    elif concept not in LOW_CONF_CONCEPTS:
                        confidence = "med"
                    else:
                        confidence = "low"
                record.update(
                    method=method,
                    confidence=confidence,
                    rule=info.get("rule"),
                    other_rule_matches=[match for match in info.get("matches", [])[1:]],
                )
            else:
                record.update(method=None, confidence=None, reason=info.get("reason"))
                unmapped.append({
                    "game": game, "side": "offense", "name_key": key, "raw_name": raw, "canonical_name": raw,
                    "n_rows": count, "n_rows_known_success": offense_known[game][key], "pa": bool(info.get("pa")),
                    "reason": info.get("reason"), "by_offense_side": dict(sides[game][key]),
                })
            offense[game][key] = record
    for game in sorted(defense_counts):
        for key, count in defense_counts[game].most_common():
            fields = defense_fields[game][key]
            raw = fields.most_common(1)[0][0][1]
            family, variant, pressure, info = defense_map(raw)
            record = {
                "raw_spellings": {value: n for (_field, value), n in fields.items()},
                "fields": dict(collections.Counter(field for (field, _value) in fields.elements())),
                "n_rows": count,
                "coverage_family": family,
                "coverage_variant": variant,
                "pressure": pressure,
            }
            if family or pressure:
                record.update(method=info["method"], confidence=info["confidence"], rule=info.get("rule"))
            else:
                record.update(method=None, confidence=None, reason=info.get("reason"))
                unmapped.append({
                    "game": game, "side": "defense", "name_key": key, "raw_name": raw, "n_rows": count,
                    "reason": info.get("reason"), "fields": record["fields"],
                })
            defense[game][key] = record
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    concepts = sorted(FAMILY_OF)
    document = {
        "schema": SCHEMA,
        "version": version,
        "created_at": now,
        "label_version": label_version,
        "name_key": "lowercase, keep only [a-z0-9] (cfb-coach catalog.norm)",
        "conf_min_names": CONF_MIN,
        "offense": {
            "families": sorted(set(FAMILY_OF.values())),
            "concepts": {concept: {"family": FAMILY_OF[concept]} for concept in concepts},
            "pa_rule": "pa=true when the name has 'PA', 'play action', 'boot', 'waggle'; the route concept is kept",
            "rpo_rule": "rpo=true for 'RPO' or Madden 'Alert <bubble/screen/...>' tags; concept = rpo_<type>",
            "names": offense,
        },
        "defense": {
            "coverage_families": list(COVERAGE_FAMILIES),
            "coverage_variants": ["zone", "match", "man"],
            "pressure_types": ["none", "sim", "blitz"],
            "pressure_rule": "blitz words -> blitz; 'sim' -> sim; a plain coverage name -> none; otherwise unknown",
            "names": defense,
        },
        "unmapped_file": f"unmapped_{version}.json",
    }
    unmapped.sort(key=lambda item: (item["game"], item["side"], -item["n_rows"]))
    return document, unmapped


def write_taxonomy(out_dir: str | Path, document: dict[str, Any], unmapped: list[dict[str, Any]]) -> Path:
    """Write ``<version>.json``, the markdown summary, the unmapped list, and ``LATEST.json``."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    version = str(document["version"])
    (root / f"{version}.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    (root / f"unmapped_{version}.json").write_text(
        json.dumps(
            {
                "concepts_version": version,
                "label_version": document.get("label_version"),
                "created_at": document.get("created_at"),
                "note": "Names no rule supports. They stay unmapped rather than guessed.",
                "items": unmapped,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    (root / "LATEST.json").write_text(
        json.dumps(
            {
                "latest": version,
                "path": f"{version}.json",
                "label_version": document.get("label_version"),
                "created_at": document.get("created_at"),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    (root / f"{version}.md").write_text(_markdown(document, unmapped), encoding="utf-8")
    return root / f"{version}.json"


def _markdown(document: dict[str, Any], unmapped: list[dict[str, Any]]) -> str:
    def pct(part: int, whole: int) -> str:
        return f"{100 * part / whole:.1f}%" if whole else "-"

    lines = [
        f"# VOD concept taxonomy {document['version']} (labels {document['label_version']})",
        "",
        f"Created {document['created_at']}. Machine-readable: `{document['version']}.json`; unmapped names: `{document['unmapped_file']}`.",
        "",
        "Name key = lowercase, keep only [a-z0-9]. Only names read at conf >= 0.5 are counted. Special teams excluded.",
        "",
        "## Offense: call name -> concept -> family",
        "",
        "| game | distinct names | distinct mapped | rows | rows mapped |",
        "|---|---|---|---|---|",
    ]
    for game, names in sorted(document["offense"]["names"].items()):
        mapped = [key for key, rec in names.items() if rec.get("concept")]
        rows = sum(rec["n_rows"] for rec in names.values())
        mapped_rows = sum(names[key]["n_rows"] for key in mapped)
        lines.append(
            f"| {game} | {len(names)} | {len(mapped)} ({pct(len(mapped), len(names))}) | {rows} | {mapped_rows} ({pct(mapped_rows, rows)}) |"
        )
    lines += ["", "## Defense: call / look -> coverage family + pressure", "",
              "| game | strings | mapped | distinct | distinct mapped |", "|---|---|---|---|---|"]
    for game, names in sorted(document["defense"]["names"].items()):
        total = sum(rec["n_rows"] for rec in names.values())
        mapped_recs = [rec for rec in names.values() if rec.get("coverage_family") or rec.get("pressure")]
        mapped = sum(rec["n_rows"] for rec in mapped_recs)
        lines.append(
            f"| {game} | {total} | {mapped} ({pct(mapped, total)}) | {len(names)} | {len(mapped_recs)} ({pct(len(mapped_recs), len(names))}) |"
        )
    lines += ["", "## Top unmapped names", "", "| game | side | name | rows | reason |", "|---|---|---|---|---|"]
    for item in sorted(unmapped, key=lambda row: -row["n_rows"])[:40]:
        lines.append(f"| {item['game']} | {item['side']} | {item['raw_name']} | {item['n_rows']} | {item.get('reason')} |")
    lines += [
        "",
        "## Rules",
        "",
        "- Offense order: RPO tags, screens, option, PA flag, pass route rules, PA boot fallback, run rules. "
        "Whole-name exact rules are method exact_rule, confidence high. One pattern match is high except the weak concepts "
        + ", ".join(sorted(LOW_CONF_CONCEPTS))
        + "; several matches are med (first rule wins). A name with no supporting rule stays unmapped.",
        "- Defense: coverage family from Cover N, Tampa, Quarters/Palms (cover_4), Robber/Hole (cover_1), Cover 2 Man, "
        "prevent, bracket N, a lone coverage digit, or 'man'. Pressure is blitz, sim, none for a plain coverage name, "
        "or unknown. Overlay strings map with low confidence.",
        "",
    ]
    return "\n".join(lines)
