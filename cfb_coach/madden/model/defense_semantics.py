"""Runtime for the Madden 27 defensive knowledge catalog: resolve a revealed defense label into structured axes, mix it with
same-game history, and expose it as ADDITIVE model features.

What this is
    * ``resolve(raw)``: a raw post-play label (the existing ``coverage_seen`` text, untouched) -> a catalog key such as
      ``dk:cover_3_sky``, a generic-pressure key, a parsed generic Cover N key, or ``None`` when nothing is recognized.
    * ``expected_axes(mix)``: the probability-weighted structural description of an opponent defense mix (deep shell, coverage
      type, variant tokens, pressure concept, digit hints, evidence mix).
    * ``dk_features(route_stats, expected)``: the ``dk_*`` feature block, including interactions with the play's route structure.
      Interactions are inputs to a LEARNED model; no weight, ranking or beater table lives here.

What this is not
    * No pre-snap adjustment inference. Opponent press/shade/drop/override changes are unobservable on offense and are never
      assumed absent; the evidence features carry the uncertainty instead.
    * No claim of defender-level knowledge: every ledger slot in the catalog is UNKNOWN, so no feature depends on one.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

from cfb_coach.madden.model import defense_catalog as dc
from cfb_coach.madden.model import defense_lexicon as L

KEY_PREFIX = "dk:"
GENERIC_PRESSURE_KEY = KEY_PREFIX + "_generic_pressure"
FEATURE_VERSION = "defense_semantics.v1"

SHELLS = ("0", "1", "2", "3", "4", "split", "unk")
TYPES = ("man", "spot_zone", "match", "unk")
PRESS = ("standard", "zone_blitz", "man_blitz", "sim", "stunt", "unspecified")
HINTS = (0, 1, 2, 3)
# Variant presence features (the TOKEN is in the label: name_derived). Meaning-bearing tokens are folded where the meaning is shared.
VAR_GROUPS: dict[str, tuple[str, ...]] = {
    "hole": ("hole",), "robber": ("robber",), "contain_spy": ("contain", "spy"), "press": ("press",),
    "double_bracket": ("double", "bracket", "switch"), "invert": ("invert",), "hard_flat": ("hard_flat",), "roll": ("roll",),
    "trap": ("trap",), "drop": ("drop",), "cloud": ("cloud",), "sky": ("sky",), "buzz": ("buzz",), "mable": ("mable",),
    "lock": ("lock",), "seam": ("seam",), "show2": ("show2",), "quarters": ("quarters",), "palms": ("palms",),
    "side_tag": ("str", "wk", "field"), "unmapped": ("willie", "sammie", "sting", "thief", "coffee_house", "zone"),
}

AXIS_NAMES: tuple[str, ...] = (
    tuple(f"dk_shell_{s}" for s in SHELLS) + tuple(f"dk_type_{t}" for t in TYPES) + tuple(f"dk_var_{v}" for v in VAR_GROUPS)
    + tuple(f"dk_press_{p}" for p in PRESS) + ("dk_press_any",) + tuple(f"dk_hint_{h}" for h in HINTS)
    + ("dk_ev_info", "dk_ev_exact", "dk_ev_ea", "dk_ev_unknown_axes", "dk_ev_ledger_known"))
INTERACTION_NAMES: tuple[str, ...] = (
    "dk_xa_press_shallow", "dk_xa_press_deep", "dk_xa_press_horizontal", "dk_xa_zblitz_shallow", "dk_xa_zblitz_horizontal",
    "dk_xa_mblitz_shallow", "dk_xa_mblitz_deep", "dk_xa_sim_shallow", "dk_xa_sim_deep", "dk_xa_stunt_shallow",
    "dk_xa_man_horizontal", "dk_xa_man_shallow", "dk_xa_match_horizontal", "dk_xa_match_intermediate",
    "dk_xa_spot_shallow", "dk_xa_spot_horizontal", "dk_xa_shell2_deepmid", "dk_xa_shell3_deepwide", "dk_xa_shell4_deep",
    "dk_xa_split_asym", "dk_xa_hole_horizontal", "dk_xa_robber_horizontal", "dk_xa_bracket_shallow")
DK_NAMES: tuple[str, ...] = AXIS_NAMES + INTERACTION_NAMES


# ------------------------------------------------------------------------------------------------ catalog access

@lru_cache(maxsize=1)
def catalog() -> dict[str, Any]:
    cat = dc.load_catalog()
    errs = dc.validate_catalog(cat)
    if errs:
        raise ValueError("defense knowledge catalog invalid: " + "; ".join(errs[:3]))
    return cat


def data_digest() -> str:
    return str(catalog()["digest"])


@lru_cache(maxsize=1)
def _index() -> dict[str, Any]:
    cat = catalog()
    by_key: dict[str, list[dict[str, Any]]] = {}
    exact: dict[tuple[str, str], dict[str, Any]] = {}
    for e in cat["entries"]:
        by_key.setdefault(e["name_key"], []).append(e)
        if e["raw_formation"]:
            exact[(e["raw_formation"], e["raw_name"])] = e
    return {"by_key": by_key, "exact": exact}


def is_key(k: Any) -> bool:
    return isinstance(k, str) and k.startswith(KEY_PREFIX)


# ------------------------------------------------------------------------------------------------ resolution

@dataclass(frozen=True)
class Resolution:
    raw: str                      # EXACTLY as entered (never altered)
    key: str | None               # dk:<name_key> | dk:_generic_pressure | None
    resolution: str               # exact_installed | exact_external | exact_by_formation | parsed_generic | generic_pressure | unknown
    legacy_bucket: str            # parent class in the current offense model ('unknown' for pressure-only)
    entry_ids: tuple[str, ...] = ()
    n_formations: int = 0         # installed formations where this exact name exists (assignments may differ between them)
    note: str = ""


_GENERIC_PRESSURE = {"pressure", "blitz", "heavy pressure", "pass rush"}
_COVER_RE = re.compile(r"^(?:cover|cov)\s*(\d)\b\s*(.*)$")
_TOKEN_TO_VARIANT = {"hole": "hole", "robber": "robber", "contain": "contain", "spy": "spy", "press": "press", "invert": "invert",
                     "man": "man", "match": "match", "quarters": "quarters", "palms": "palms", "drop": "drop", "sky": "sky",
                     "cloud": "cloud", "buzz": "buzz", "trap": "trap", "zone": "zone"}
_DIGIT_BASE = {"0": "c0", "1": "c1", "2": "c2", "3": "c3", "4": "c4", "6": "c6", "9": "c9"}


def _norm(raw: Any) -> str:
    return dc.normalize_name(str(raw or ""))


def _parsed_spec(norm: str) -> dict[str, Any] | None:
    """Conservative fallback for a generic or uncatalogued 'Cover N ...' label (the labels the old recognizer already accepted)."""
    if norm == "tampa 2":
        return {"base": "t2", "variants": [], "pressure": "standard"}
    if norm == "trap 2":
        return {"base": "c2", "variants": ["trap"], "pressure": "standard"}
    if norm == "cover 0 blitz":
        return {"base": "c0", "variants": [], "pressure": "man_blitz"}
    m = _COVER_RE.match(norm)
    if not m or m.group(1) not in _DIGIT_BASE:
        return None
    toks = m.group(2).split()
    variants, unknown, spot = [], [], False
    for t in toks:
        if t in _TOKEN_TO_VARIANT:
            variants.append(_TOKEN_TO_VARIANT[t])
        elif t == "spot":
            spot = True                               # the old recognizer accepted 'Cover 2 spot' as an explicit spot-zone scheme
        else:
            unknown.append(t)
    if unknown:
        return None                                   # never guess a meaning for an unrecognized token
    return {"base": _DIGIT_BASE[m.group(1)], "variants": variants, "pressure": "standard", **({"spot": True} if spot else {})}


def resolve(raw: Any, formation: str | None = None) -> Resolution:
    text = "" if raw is None else str(raw)
    norm = _norm(text)
    if not norm:
        return Resolution(text, None, "unknown", "unknown", note="empty label")
    if norm in _GENERIC_PRESSURE:
        return Resolution(text, GENERIC_PRESSURE_KEY, "generic_pressure", "unknown",
                          note="a pressure was reported but not its type, count or the coverage behind it")
    idx = _index()
    key = dc.slug(text)
    if formation:
        e = idx["exact"].get((formation, text.strip()))
        if e:
            return Resolution(text, KEY_PREFIX + e["name_key"], "exact_by_formation", e["legacy_bucket"], (e["id"],), 1)
    ents = idx["by_key"].get(key)
    if ents:
        inst = [e for e in ents if e["status"] == "installed"]
        res = "exact_installed" if inst else "exact_external"
        return Resolution(text, KEY_PREFIX + key, res, ents[0]["legacy_bucket"], tuple(e["id"] for e in ents),
                          len({e["raw_formation"] for e in inst}))
    spec = _parsed_spec(norm)
    if spec is not None:
        return Resolution(text, KEY_PREFIX + "parsed:" + key, "parsed_generic", dc.legacy_bucket_for(spec),
                          note="generic or uncatalogued Cover N label; family-level features only")
    return Resolution(text, None, "unknown", "unknown", note="not in the catalog and not a generic Cover N label")


def _record(key: str) -> dict[str, Any]:
    """Axis record for a mix key (a catalog entry, a parsed generic label, or generic pressure)."""
    if key == GENERIC_PRESSURE_KEY:
        spec = {"base": None, "variants": [], "pressure": "unspecified", "basis": "generic pressure label"}
        return {"parsed": dc.parse_spec(spec), "legacy": "unknown", "exact": False}
    body = key[len(KEY_PREFIX):]
    if body.startswith("parsed:"):
        slug = body[len("parsed:"):]
        spec = _parsed_spec(slug.replace("_", " ")) or {"base": None, "variants": [], "pressure": "standard"}
        return {"parsed": dc.parse_spec(spec), "legacy": dc.legacy_bucket_for(spec), "exact": False}
    e = _index()["by_key"][body][0]
    return {"parsed": {"coverage": e["coverage"], "pressure": e["pressure"]}, "legacy": e["legacy_bucket"], "exact": True}


def legacy_bucket(key: str) -> str:
    return _record_cached(key)["legacy"]


@lru_cache(maxsize=1024)
def _record_cached(key: str) -> dict[str, Any]:
    return _record(key)


# ------------------------------------------------------------------------------------------------ per-key vector

def _eligible(ax: Mapping[str, Any]) -> bool:
    return bool(ax.get("feature_eligible"))


@lru_cache(maxsize=1024)
def key_axes(key: str) -> dict[str, float]:
    """Axis features of one key (a point mass on that label). Only ``feature_eligible`` claims contribute."""
    r = _record_cached(key)
    cov, pres = r["parsed"]["coverage"], r["parsed"]["pressure"]
    f = {n: 0.0 for n in AXIS_NAMES}
    sh = cov["deep_shell"]
    sv = str(sh["value"]) if _eligible(sh) and sh["value"] != "unknown" else "unk"
    f["dk_shell_" + (sv if sv in SHELLS else "unk")] = 1.0
    ty = cov["type"]
    tv = ty["value"] if _eligible(ty) and ty["value"] in TYPES else "unk"
    f["dk_type_" + tv] = 1.0
    toks = {v["token"] for v in cov["variants"]}
    for g, members in VAR_GROUPS.items():
        if toks & set(members):
            f["dk_var_" + g] = 1.0
    pv = pres["concept"]["value"]
    f["dk_press_" + (pv if pv in PRESS else "unspecified")] = 1.0
    if pres.get("secondary_concept") and pres["secondary_concept"]["value"] in PRESS:
        f["dk_press_" + pres["secondary_concept"]["value"]] = 1.0
    f["dk_press_any"] = 0.0 if pv == "standard" else 1.0
    h = pres["digit_hint"]["value"]
    if h in HINTS:
        f[f"dk_hint_{h}"] = 1.0
    f["dk_ev_exact"] = 1.0 if r["exact"] else 0.0
    ea = any(ax.get("status") == "ea_documented" and _eligible(ax) for ax in (sh, ty, pres["concept"])) or \
        any(v["meaning_status"] == "ea_documented" for v in cov["variants"])
    f["dk_ev_ea"] = 1.0 if ea else 0.0
    f["dk_ev_unknown_axes"] = 1.0 if (f["dk_shell_unk"] and f["dk_type_unk"]) else 0.0
    f["dk_ev_ledger_known"] = 0.0                   # no entry has a verified slot; kept so the evidence axis is explicit
    return f


def expected_axes(family_mix: Mapping[str, float] | None) -> dict[str, float]:
    """Mass-weighted axis features. Mass is NOT renormalized: a low-information mix yields small values and a small ``dk_ev_info``."""
    out = {n: 0.0 for n in AXIS_NAMES}
    if not family_mix:
        return out
    info = 0.0
    for key, p in family_mix.items():
        if p <= 0:
            continue
        info += p
        for n, v in key_axes(key).items():
            if v:
                out[n] += p * v
    out["dk_ev_info"] = info
    return out


# ------------------------------------------------------------------------------------------------ interactions with the play

def dk_features(art: Mapping[str, float] | None, expected: Mapping[str, float]) -> dict[str, float]:
    """Axis expectation + interactions with the play's stock route-art geometry.

    ``art`` is ``offense_art_features.art_features(formation, play)`` (screenshot-derived share of path ink by depth/side). It is the only
    play-structure signal available for all 93 installed plays (none has first-party player route assignments yet), and it is stock
    geometry, not a per-player assignment. The products are INPUTS to a learned model; no weight or ranking lives here.
    """
    f = dict(expected)
    for n in INTERACTION_NAMES:
        f[n] = 0.0
    if not art or not art.get("art_present"):
        return f
    e = expected
    sh, de, hz, im = art["art_shallow_share"], art["art_deep_share"], art["art_horizontal_share"], art["art_intermediate_share"]
    f["dk_xa_press_shallow"] = e["dk_press_any"] * sh
    f["dk_xa_press_deep"] = e["dk_press_any"] * de
    f["dk_xa_press_horizontal"] = e["dk_press_any"] * hz
    f["dk_xa_zblitz_shallow"] = e["dk_press_zone_blitz"] * sh
    f["dk_xa_zblitz_horizontal"] = e["dk_press_zone_blitz"] * hz
    f["dk_xa_mblitz_shallow"] = e["dk_press_man_blitz"] * sh
    f["dk_xa_mblitz_deep"] = e["dk_press_man_blitz"] * de
    f["dk_xa_sim_shallow"] = e["dk_press_sim"] * sh
    f["dk_xa_sim_deep"] = e["dk_press_sim"] * de
    f["dk_xa_stunt_shallow"] = e["dk_press_stunt"] * sh
    f["dk_xa_man_horizontal"] = e["dk_type_man"] * hz
    f["dk_xa_man_shallow"] = e["dk_type_man"] * sh
    f["dk_xa_match_horizontal"] = e["dk_type_match"] * hz
    f["dk_xa_match_intermediate"] = e["dk_type_match"] * im
    f["dk_xa_spot_shallow"] = e["dk_type_spot_zone"] * sh
    f["dk_xa_spot_horizontal"] = e["dk_type_spot_zone"] * hz
    f["dk_xa_shell2_deepmid"] = e["dk_shell_2"] * art["art_deep_middle"]
    f["dk_xa_shell3_deepwide"] = e["dk_shell_3"] * (art["art_deep_left"] + art["art_deep_right"])
    f["dk_xa_shell4_deep"] = e["dk_shell_4"] * de
    f["dk_xa_split_asym"] = e["dk_shell_split"] * abs(art["art_left_share"] - art["art_right_share"])
    f["dk_xa_hole_horizontal"] = e["dk_var_hole"] * hz
    f["dk_xa_robber_horizontal"] = e["dk_var_robber"] * hz
    f["dk_xa_bracket_shallow"] = e["dk_var_double_bracket"] * sh
    return f


# ------------------------------------------------------------------------------------------------ the post-play form is a LOSSY channel

def form_store(typed: Any) -> str | None:
    """What the EXISTING post-play form stores in ``coverage_seen`` for text typed in "they ran" (offense, previous snap).

    Calls the real parser (``parse_madden_situation``): keyword rules turn the typed text into a coarse canonical label, or nothing.
    Example: every "Cover 3 ..." call becomes "Cover 3 Sky"; every Cover 1 variant becomes "Cover 1"; "Fire Zone 3" becomes "pressure".
    """
    from cfb_coach.madden.situation import parse_madden_situation
    sit = parse_madden_situation(str(typed or ""), default_side="offense")
    return sit.coverage_hint if getattr(sit, "coverage_source", None) == "last" and sit.coverage_hint else None


@lru_cache(maxsize=1)
def _channel() -> dict[str, Any]:
    """Preimage of the form: stored label (normalized) -> the catalog name keys that the real parser maps to it."""
    pre: dict[str, set[str]] = {}
    typed = {e["raw_name"] for e in catalog()["entries"]} | set(_LEGACY_TYPED)
    outcome: dict[str, str | None] = {}
    for t in sorted(typed):
        stored = form_store(t)
        outcome[t] = stored
        if stored:
            r = resolve(t)
            if r.key:
                pre.setdefault(_norm(stored), set()).add(r.key)
    return {"pre": {k: sorted(v) for k, v in pre.items()}, "outcome": outcome}


# typed spellings the old recognizer accepted (they are plausible user input even though they are not catalog play names)
_LEGACY_TYPED = ("cover 1", "cover 2", "cover 3", "cover 4", "cover 0", "cover 0 blitz", "cover 2 spot", "trap 2", "cover 9")


def stored_label_weights(label: Any, typed: Any = None) -> dict[str, float]:
    """Mix keys (summing to 1) for a label READ FROM THE DATABASE, or {} when nothing usable.

    * ``pressure`` / ``blitz`` -> the generic-pressure key (a pressure was reported; its type is unknown).
    * A label the form can produce (e.g. "Cover 3 Sky") could have come from any catalog name the parser maps to it, so it is a uniform
      mixture over those names plus its own exact reading. That is the maximum-entropy assumption; it is revisited the day capture is
      lossless. It keeps family-level structure and blurs variant detail instead of pretending the variant was observed.
    * A label the form can never produce (e.g. "Cover 3 Cloud") was captured losslessly and is read exactly.
    * ``typed`` is the exact text the user entered (``snaps.coverage_typed``, NULL on every row logged before it existed). When it
      resolves to a catalog name or to untyped pressure it is read EXACTLY and the lossy stored label is ignored; when it is empty or
      unrecognized the stored label is read as above, so older rows stay ambiguous and nothing is invented.
    """
    t = _norm(typed)
    if t:
        if t in _GENERIC_PRESSURE:
            return {GENERIC_PRESSURE_KEY: 1.0}
        k = resolve(typed).key
        if k:
            return {k: 1.0}
    norm = _norm(label)
    if not norm:
        return {}
    if norm in _GENERIC_PRESSURE:
        return {GENERIC_PRESSURE_KEY: 1.0}
    cands = _channel()["pre"].get(norm)
    own = resolve(label).key
    if not cands:
        return {own: 1.0} if own else {}
    keys = sorted(set(cands) | ({own} if own else set()))
    return {k: 1.0 / len(keys) for k in keys}


def channel_audit() -> dict[str, Any]:
    """How much of each catalog name survives the form: exactly / rewritten to a coarser label / dropped. Read-only, deterministic."""
    ch = _channel()
    names = sorted({e["raw_name"] for e in catalog()["entries"]})
    exact = [n for n in names if ch["outcome"][n] and _norm(ch["outcome"][n]) == _norm(n)]
    dropped = [n for n in names if not ch["outcome"][n]]
    rewritten = [(n, ch["outcome"][n]) for n in names if ch["outcome"][n] and _norm(ch["outcome"][n]) != _norm(n)]
    groups: dict[str, list[str]] = {}
    for n, lab in rewritten + [(n, ch["outcome"][n]) for n in exact]:
        groups.setdefault(lab, []).append(n)
    return {"unique_names": len(names), "stored_exactly_as_typed": len(exact), "rewritten_to_coarser_label": len(rewritten),
            "nothing_stored": len(dropped), "exact_names": exact, "dropped_names": dropped,
            "stored_label_to_possible_typed_names": {k: sorted(v) for k, v in sorted(groups.items())},
            "largest_collapse": max(((k, len(v)) for k, v in groups.items()), key=lambda kv: kv[1])}


# ------------------------------------------------------------------------------------------------ labels -> counts / mixes

def family_counts(labels: Iterable[Any], *, stored: bool = True) -> dict[str, float]:
    """Mass per mix key for closed, earlier labels. Unrecognized labels add no mass (they are reported separately).

    ``stored=True`` (default) reads labels as they come out of the database, i.e. through the form channel (see
    ``stored_label_weights``). ``stored=False`` reads each label as exact typed text.
    """
    out: dict[str, float] = {}
    for lab in labels:
        if isinstance(lab, tuple):                      # (stored label, exact typed text)
            lab, typed = lab
        else:
            typed = None
        w = stored_label_weights(lab, typed) if stored else ({resolve(lab).key: 1.0} if resolve(lab).key else {})
        for k, v in w.items():
            out[k] = out.get(k, 0.0) + v
    return out


def project_legacy(mix: Mapping[str, float]) -> dict[str, float]:
    """Legacy-keyed view of a (possibly mixed) mix: dk keys are projected to their parent class, legacy keys pass through."""
    out: dict[str, float] = {}
    for k, v in mix.items():
        b = legacy_bucket(k) if is_key(k) else k
        out[b] = out.get(b, 0.0) + float(v)
    return out


def split_mix(mix: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """(legacy part, dk part) of a mix."""
    legacy = {k: v for k, v in mix.items() if not is_key(k)}
    fam = {k: v for k, v in mix.items() if is_key(k)}
    return legacy, fam


# ------------------------------------------------------------------------------------------------ defensive candidate features (offline)

PERSONNEL_NAMES = ("dk_pers_base_db4", "dk_pers_nickel_db5", "dk_pers_dime_db6", "dk_pers_46", "dk_pers_mug_single", "dk_pers_mug_double",
                   "dk_pers_odd_front", "dk_pers_over_front")
CANDIDATE_NAMES: tuple[str, ...] = tuple(n for n in AXIS_NAMES if n not in ("dk_ev_info",)) + PERSONNEL_NAMES + ("dk_cand_in_catalog",)


def candidate_features(formation: str, play: str) -> dict[str, float]:
    """Structural description of ONE of our defensive calls (a point mass), for OFFLINE candidate scoring.

    Formation personnel are football_hypothesis facts (Wikipedia nickel/46 definitions and naming convention); they are labelled as such
    in the catalog. The caller decides whether to use them; nothing here selects a call.
    """
    r = resolve(play, formation)
    f = {n: 0.0 for n in CANDIDATE_NAMES}
    if r.key is None:
        return f
    f.update({k: v for k, v in key_axes(r.key).items() if k in f})
    f["dk_cand_in_catalog"] = 1.0 if r.resolution.startswith("exact") else 0.0
    fm = L.FORMATIONS.get(formation)
    if fm:
        db = fm["db"]
        f["dk_pers_base_db4"] = 1.0 if db == 4 else 0.0
        f["dk_pers_nickel_db5"] = 1.0 if db == 5 else 0.0
        f["dk_pers_dime_db6"] = 1.0 if db == 6 else 0.0
        f["dk_pers_46"] = 1.0 if formation.startswith("46") else 0.0
        f["dk_pers_mug_single"] = 1.0 if "Single Mug" in formation else 0.0
        f["dk_pers_mug_double"] = 1.0 if "Double Mug" in formation else 0.0
        f["dk_pers_odd_front"] = 1.0 if formation.endswith("Odd") else 0.0
        f["dk_pers_over_front"] = 1.0 if formation.endswith("Over") else 0.0
    return f


def candidate_vector(formation: str, play: str) -> list[float]:
    f = candidate_features(formation, play)
    return [f[n] for n in CANDIDATE_NAMES]


# ------------------------------------------------------------------------------------------------ human-readable explanation

def explain(raw: Any, formation: str | None = None) -> dict[str, Any]:
    """What the catalog KNOWS, what is only a football hypothesis, and what is UNKNOWN, for one label. Never a play recommendation."""
    r = resolve(raw, formation)
    out: dict[str, Any] = {"raw": r.raw, "resolution": r.resolution, "key": r.key, "legacy_bucket": r.legacy_bucket, "note": r.note}
    if r.key is None or r.key == GENERIC_PRESSURE_KEY or r.key.startswith(KEY_PREFIX + "parsed:"):
        out["known"], out["hypotheses"] = [], []
        out["unknown"] = ["every defender role", "rusher count", "coverage behind any pressure"]
        return out
    e = _index()["by_key"][r.key[len(KEY_PREFIX):]][0]
    cov, pres = e["coverage"], e["pressure"]
    axes = [("deep shell", cov["deep_shell"]), ("coverage type", cov["type"]), ("pressure", pres["concept"])]
    out["known"] = [f"{n}: {a['value']} [{a['status']}]" for n, a in axes if a["status"] in ("ea_documented", "xbox_verified")]
    out["hypotheses"] = [f"{n}: {a['value']} [{a['status']}]" for n, a in axes if a["status"] in ("football_hypothesis", "name_derived", "third_party_unverified")]
    out["variants"] = [{"token": v["token"], "meaning": v["meaning"], "status": v["meaning_status"]} for v in cov["variants"]]
    out["unknown"] = list(e["unknowns"])
    out["verification_level"] = e["verification"]["level"]
    out["nearest_siblings"] = [{"name": s["raw_name"], "documented_difference": s["documented_difference"]} for s in e["nearest_siblings"][:3]]
    out["same_name_in_formations"] = e.get("same_name_other_formations", [])
    return out
