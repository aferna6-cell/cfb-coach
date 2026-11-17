"""The restricted Ravens two-formation book: 12 call IDs -> 14 complete candidate decisions.

A candidate is the WHOLE decision the scorer evaluates:
    formation + base play + package/call ID + required macro + optional macro (Zone It)
Mandatory macros are attached here, before scoring, and can never be removed afterwards.
The older ``civil_match_nickel_combined.json`` is reference-only and is never read here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[3]
DATASET = REPO / "datasets/madden27/lions_ravens_two_formation_defense_v1.json"
STOCK_BOOKS = REPO / "cfb_coach/data/madden27/playbooks_full.json"
LIONS_NSM_DATASET = REPO / "datasets/madden27/lions_franchise_nickel_single_mug_v1.json"
CATALOG_ID = "lions_ravens_match_and_nickel_stock_v1"
LIONS_FRANCHISE_CATALOG_ID = "lions_franchise_ravens_plus_nickel_single_mug_v1"

NICKEL = "Nickel 2-4 Wide"
DIME = "Dime 2-3 Odd"
NSM = "Nickel Single Mug"
FORMATIONS = (NICKEL, DIME)
LIONS_FRANCHISE_FORMATIONS = (NICKEL, DIME, NSM)
PACKAGES = {"match_coverage": NICKEL, "nickel_stock": NICKEL, "spot_drop_fallback": NICKEL, "dime_blitz": DIME,
            "nickel_single_mug": NSM}
REQUIRED_MACRO_BY_PACKAGE = {"nickel_stock": "24_BASE", "dime_blitz": "DB_FIRE",
                             "match_coverage": None, "spot_drop_fallback": None,
                             "nickel_single_mug": None}
STEELERS = "Steelers"
RAVENS = "Ravens"
ZONE_IT = "ZONE_IT"
APPROVED_CALL_IDS = (
    "match_quarters", "match_palms", "match_cover9_show2", "match_cover3", "match_cover2", "match_cover6",
    "nickel_cloud", "nickel_cover2", "nickel_tampa2", "nickel_cover9", "stock_cover4_zone", "dime_dbfire2",
)

# Coverage structure (football_priors descriptor vocabulary). Keys come from the repo's
# defense_semantics knowledge base (``dk:`` keys); Zone It turns match rules off (spot drops),
# which is a structural statement from the dataset's macro_catalog, not an outcome claim.
_DK_KEY = {
    "match_quarters": "dk:cover_4_quarters", "match_palms": "dk:cover_4_palms",
    "match_cover9_show2": "dk:cover_9_show_2", "match_cover3": "dk:cover_3_match",
    "match_cover2": "dk:cover_2_match", "match_cover6": "dk:cover_6",
    "nickel_cloud": "dk:cover_3_cloud", "nickel_cover2": "dk:cover_2_match",
    "nickel_tampa2": "dk:tampa_2", "nickel_cover9": "dk:cover_9_show_2",
    "stock_cover4_zone": "dk:cover_4_zone", "dime_dbfire2": "dk:db_fire_2",
    "nsm_quarters_left_tex": "dk:cover_4_quarters", "nsm_quarters_texas_4": "dk:cover_4_quarters",
    "nsm_quarters_manup_short": "dk:cover_4_quarters", "nsm_quarters_manup_long": "dk:cover_4_quarters",
    "nsm_quarters_4man_rush": "dk:cover_4_quarters", "nsm_quarters_run_support": "dk:cover_4_quarters",
    "nsm_tampa2_shade": "dk:tampa_2",
}
NSM_CALL_IDS = (
    "nsm_quarters_left_tex", "nsm_quarters_texas_4", "nsm_quarters_manup_short",
    "nsm_quarters_manup_long", "nsm_quarters_4man_rush", "nsm_quarters_run_support",
    "nsm_tampa2_shade",
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    call_id: str
    formation: str
    base_play: str
    package: str
    required_macro: str | None
    optional_macro: str | None
    is_blitz: bool
    structure: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=False)
    raw: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def macros(self) -> tuple[str, ...]:
        return tuple(m for m in (self.required_macro, self.optional_macro) if m)

    def macro_display(self, catalog: "RavensCatalog") -> str | None:
        names = [catalog.macro_name(m) for m in self.macros]
        return " + ".join(names) if names else None

    def as_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "call_id": self.call_id, "formation": self.formation,
                "base_play": self.base_play, "package": self.package, "required_macro": self.required_macro,
                "optional_macro": self.optional_macro, "is_blitz": self.is_blitz}


@dataclass(frozen=True)
class RavensCatalog:
    raw: Mapping[str, Any]
    sha256: str
    candidates: tuple[Candidate, ...]
    approved_call_ids: tuple[str, ...] = APPROVED_CALL_IDS
    allowed_formations: tuple[str, ...] = FORMATIONS
    stock_team: str = RAVENS

    @property
    def calls(self) -> dict[str, Mapping[str, Any]]:
        return {c["call_id"]: c for c in self.raw["stock_calls"]}

    def macro_name(self, macro_id: str) -> str:
        return str(self.raw["macro_catalog"][macro_id]["display_name"])

    def by_id(self) -> dict[str, Candidate]:
        return {c.candidate_id: c for c in self.candidates}

    def version(self) -> str:
        return f"{self.raw.get('catalog_id')}@{self.sha256[:12]}"

    @property
    def is_lions_franchise(self) -> bool:
        return self.stock_team == STEELERS or self.raw.get("catalog_id") == LIONS_FRANCHISE_CATALOG_ID

    def stock_book(self) -> dict[str, list[str]]:
        """In-repo Huddle.gg inventory for this catalog's stock team and formations."""
        if self.stock_team == STEELERS:
            return steelers_stock_book()
        return ravens_stock_book()


def _structure(call_id: str, zone_it: bool) -> dict[str, Any]:
    from cfb_coach.madden.model import football_priors as fp
    desc = fp.descriptor(_DK_KEY[call_id]) or {"shell": "unk", "type": "unk", "variants": set(), "press": "standard"}
    desc = {"shell": desc["shell"], "type": desc["type"], "variants": set(desc["variants"]), "press": desc["press"]}
    if call_id == "nickel_tampa2":
        desc["variants"].add("tampa")
    if zone_it:  # match rules off, defenders play spot zones from the same alignment/check
        desc["type"] = "spot_zone"
        desc["variants"] -= {"quarters", "palms"}
    desc["dk_key"] = _DK_KEY[call_id]
    desc["zone_it"] = bool(zone_it)
    return desc


def contract_errors(raw: Mapping[str, Any]) -> list[str]:
    """Every scheme rule the owner stated, checked against the authoritative dataset."""
    errs: list[str] = []
    if raw.get("catalog_id") != CATALOG_ID:
        errs.append("unexpected catalog_id")
    if raw.get("source_defensive_playbook") != "Baltimore Ravens":
        errs.append("defensive playbook is not Baltimore Ravens")
    if [f.get("name") for f in raw.get("formations") or []] != list(FORMATIONS):
        errs.append("formations are not exactly Nickel 2-4 Wide + Dime 2-3 Odd")
    calls = raw.get("stock_calls") or []
    ids = [c.get("call_id") for c in calls]
    if len(ids) != 12 or len(set(ids)) != 12 or tuple(ids) != APPROVED_CALL_IDS:
        errs.append("call IDs are not exactly the 12 approved IDs")
    macros = raw.get("macro_catalog") or {}
    if set(macros) != {"24_BASE", "DB_FIRE", ZONE_IT}:
        errs.append("macro catalog is not exactly 24 Base, DB Fire, Zone It")
    counts: dict[str, int] = {}
    for c in calls:
        pkg = c.get("package")
        counts[pkg] = counts.get(pkg, 0) + 1
        if pkg not in PACKAGES:
            errs.append(f"{c.get('call_id')}: unknown package {pkg}")
            continue
        if c.get("formation") != PACKAGES[pkg]:
            errs.append(f"{c.get('call_id')}: package {pkg} must be in {PACKAGES[pkg]}")
        if c.get("required_macro_id") != REQUIRED_MACRO_BY_PACKAGE[pkg]:
            errs.append(f"{c.get('call_id')}: required macro must be {REQUIRED_MACRO_BY_PACKAGE[pkg]}")
        for m in c.get("optional_macros") or []:
            if m != ZONE_IT or pkg != "match_coverage":
                errs.append(f"{c.get('call_id')}: optional macro {m} only allowed as Zone It on Match calls")
        if c.get("manual_setup_changes_allowed") is not False:
            errs.append(f"{c.get('call_id')}: manual setup changes must be disallowed (stock-only)")
        if c.get("formation") == DIME and (c.get("base_play") != "DB Fire 2" or
                                           c.get("eligible_only_if_model_selects_blitz") is not True):
            errs.append("Dime 2-3 Odd may only hold DB Fire 2 as a model-selected blitz")
    if counts != {"match_coverage": 6, "nickel_stock": 4, "spot_drop_fallback": 1, "dime_blitz": 1}:
        errs.append(f"package counts differ from 6/4/1/1: {counts}")
    rules = raw.get("user_assignment_rules") or []
    if len(rules) != 18 or len({r.get("assignment_id") for r in rules}) != 18:
        errs.append("expected exactly 18 unique Match-guide USER assignments")
    for r in rules:
        if r.get("call_id") not in ids:
            errs.append(f"USER rule {r.get('assignment_id')} points at unknown call")
    return errs


@lru_cache(maxsize=4)
def _load(path: str, mtime_ns: int) -> RavensCatalog:
    blob = Path(path).read_bytes()
    raw = json.loads(blob)
    errs = contract_errors(raw)
    if errs:
        raise ValueError("Ravens defensive scheme contract violated: " + "; ".join(errs))
    cands: list[Candidate] = []
    for c in raw["stock_calls"]:
        pkg = c["package"]
        base = Candidate(c["call_id"], c["call_id"], c["formation"], c["base_play"], pkg,
                         c.get("required_macro_id"), None, pkg == "dime_blitz",
                         _structure(c["call_id"], False), c)
        cands.append(base)
        for m in c.get("optional_macros") or []:
            cands.append(Candidate(f"{c['call_id']}+zone_it", c["call_id"], c["formation"], c["base_play"], pkg,
                                   c.get("required_macro_id"), m, False, _structure(c["call_id"], True), c))
    return RavensCatalog(raw, hashlib.sha256(blob).hexdigest(), tuple(cands))


def load(path: str | Path | None = None) -> RavensCatalog:
    p = Path(path or DATASET)
    return _load(str(p.resolve()), p.stat().st_mtime_ns)


@lru_cache(maxsize=2)
def ravens_stock_book() -> dict[str, list[str]]:
    """The in-repo third-party Ravens defensive inventory (huddle.gg), not the installed Xbox book."""
    data = json.loads(STOCK_BOOKS.read_text(encoding="utf-8"))
    forms = data["books"]["defense"]["Ravens"]["formations"]
    return {f: list(forms[f]["plays"]) for f in FORMATIONS if f in forms}


def stock_catalog_check(cat: RavensCatalog) -> dict[str, bool]:
    stock = cat.stock_book()
    return {c.candidate_id: c.base_play in stock.get(c.formation, ()) for c in cat.candidates}


def installed_two_formation_book(defense_book: Mapping[str, Any] | None) -> dict[str, list[str]]:
    forms = (defense_book or {}).get("formations", defense_book or {})
    if not isinstance(forms, Mapping):
        return {}
    return {f: sorted(str(p) for p in forms[f]) for f in FORMATIONS if f in forms}


def scheme_book(cat: RavensCatalog | None = None) -> dict[str, list[str]]:
    """The two-formation dataset itself as a book ({formation: [base plays]}), only plays the 14
    candidates use. The online civil profile (t351u) treats this as the legal book: Aidan installs the
    same civil package in the Chiefs league. Never adds a play that is not a dataset candidate."""
    cat = cat or load()
    out: dict[str, set[str]] = {}
    for c in cat.candidates:
        out.setdefault(c.formation, set()).add(c.base_play)
    forms = getattr(cat, "allowed_formations", FORMATIONS)
    return {f: sorted(out[f]) for f in forms if f in out}


def installed_fingerprint(two_formation_book: Mapping[str, Sequence[str]]) -> str | None:
    """Hash of the exact installed Nickel 2-4 Wide + Dime 2-3 Odd inventory (None when neither exists)."""
    if not two_formation_book:
        return None
    body = [[f, sorted(set(two_formation_book[f]))] for f in FORMATIONS if f in two_formation_book]
    return "RAV-DEF-" + hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()[:12].upper()


def candidate_for_execution(cat: RavensCatalog, formation: str | None, play: str | None,
                            macros: Sequence[str] | None) -> tuple[Candidate | None, str]:
    """Map an executed (formation, play, macro set) to exactly one candidate, or the reason it cannot be.

    ``macros=None`` means NO macro evidence (not "no macro"): then a play that has a macro'd sibling
    (Cover 2 Match Match/Nickel, Quarters/Palms +/- Zone It) or a required macro is ambiguous.
    """
    if formation not in getattr(cat, "allowed_formations", FORMATIONS):
        return None, "not_two_formation"
    hits = [c for c in cat.candidates if c.formation == formation and c.base_play == play]
    if not hits:
        return None, "play_not_in_scheme"
    if macros is None:
        if len(hits) == 1 and not hits[0].macros:
            return hits[0], "ok"
        return None, "macro_unconfirmed"
    names = {cat.macro_name(m).lower(): m for m in cat.raw["macro_catalog"]}
    ids = {names.get(str(x).strip().lower(), str(x).strip().upper().replace(" ", "_")) for x in macros if x}
    exact = [c for c in hits if set(c.macros) == ids]
    if len(exact) == 1:
        return exact[0], "ok"
    return None, "macro_mismatch"


def steelers_stock_book() -> dict[str, list[str]]:
    """The in-repo third-party Steelers defensive inventory (huddle.gg).

    Steelers is the only Madden 27 stock book that contains Nickel Single Mug
    together with Nickel 2-4 Wide and Dime 2-3 Odd. All 12 current civil-defense
    base plays also exist in those two Steelers formations.
    """
    data = json.loads(STOCK_BOOKS.read_text(encoding="utf-8"))
    forms = data["books"]["defense"][STEELERS]["formations"]
    return {f: list(forms[f]["plays"]) for f in LIONS_FRANCHISE_FORMATIONS if f in forms}


def books_with_current_formations_and_nsm() -> list[str]:
    """Stock defensive playbooks that contain all three exact formation names."""
    data = json.loads(STOCK_BOOKS.read_text(encoding="utf-8"))
    hits = []
    for team, book in data["books"]["defense"].items():
        forms = set((book.get("formations") or {}))
        if set(LIONS_FRANCHISE_FORMATIONS) <= forms:
            hits.append(team)
    return hits


@lru_cache(maxsize=4)
def _load_lions_franchise(path: str, mtime_ns: int, base_sha: str) -> RavensCatalog:
    base = load()
    blob = Path(path).read_bytes()
    owner = json.loads(blob)
    if owner.get("formation") != NSM or owner.get("package") != "nickel_single_mug":
        raise ValueError("Lions Nickel Single Mug file must name formation Nickel Single Mug / package nickel_single_mug")
    steel = steelers_stock_book()
    nsm_plays = steel.get(NSM, ())
    extra_calls = list(owner.get("stock_calls") or [])
    ids = [c.get("call_id") for c in extra_calls]
    if tuple(ids) != NSM_CALL_IDS or len(set(ids)) != len(NSM_CALL_IDS):
        raise ValueError("Nickel Single Mug call IDs are not exactly the 7 PDF stock-play setups")
    for c in extra_calls:
        if c.get("formation") != NSM or c.get("package") != "nickel_single_mug":
            raise ValueError(f"{c.get('call_id')}: must be Nickel Single Mug / nickel_single_mug")
        if c.get("base_play") not in nsm_plays:
            raise ValueError(f"{c.get('call_id')}: '{c.get('base_play')}' is not in Steelers Nickel Single Mug")
        if c.get("required_macro_id") != REQUIRED_MACRO_BY_PACKAGE["nickel_single_mug"] and c.get("required_macro_id") not in (owner.get("macro_catalog") or {}):
            # allowed: None, or a macro defined in this file
            if c.get("required_macro_id") is not None:
                raise ValueError(f"{c.get('call_id')}: unknown required macro {c.get('required_macro_id')}")
        if not c.get("setup_steps"):
            raise ValueError(f"{c.get('call_id')}: setup_steps must be the PDF steps verbatim")
    merged = dict(base.raw)
    macros = dict(base.raw.get("macro_catalog") or {})
    macros.update(owner.get("macro_catalog") or {})
    merged.update({
        "catalog_id": LIONS_FRANCHISE_CATALOG_ID,
        "source_defensive_playbook": "Pittsburgh Steelers (context for Nickel 2-4 Wide + Dime 2-3 Odd + Nickel Single Mug); the original 12 calls also exist in the Baltimore Ravens book",
        "formations": list(base.raw.get("formations") or []) + [
            {"name": NSM, "role": "lions_franchise_nickel_single_mug_package",
             "allowed_packages": ["nickel_single_mug"],
             "label": owner.get("label")},
        ],
        "macro_catalog": macros,
        "stock_calls": list(base.raw.get("stock_calls") or []) + extra_calls,
        "lions_nsm_owner": owner.get("owner_instruction") or {},
        "omitted_pdf_pages": owner.get("omitted_pdf_pages") or [],
        "confirmation_status": owner.get("confirmation_status"),
    })
    cands = list(base.candidates)
    for c in extra_calls:
        cands.append(Candidate(c["call_id"], c["call_id"], c["formation"], c["base_play"], c["package"],
                               c.get("required_macro_id"), None, False,
                               _structure(c["call_id"], False), c))
    sha = hashlib.sha256(base.sha256.encode() + b"+" + blob).hexdigest()
    return RavensCatalog(merged, sha, tuple(cands),
                         approved_call_ids=APPROVED_CALL_IDS + NSM_CALL_IDS,
                         allowed_formations=LIONS_FRANCHISE_FORMATIONS,
                         stock_team=STEELERS)


def load_lions_franchise(path: str | Path | None = None) -> RavensCatalog:
    """Current Ravens two-formation civil defense plus the t449u Nickel Single Mug package.

    Used only for Lions human franchise games. CPU, Chiefs, and Titans keep ``load()``.
    """
    p = Path(path or LIONS_NSM_DATASET)
    return _load_lions_franchise(str(p.resolve()), p.stat().st_mtime_ns, load().sha256)


def is_lions_human_defense_session(db: Any, session_id: str | None, extras: Mapping[str, Any] | None,
                                   opponent_id: str | None) -> bool:
    """True only for a Lions (primary) franchise game against a human opponent.

    Chiefs online, Titans, CPU opponents, and untagged sessions stay on the two-formation
    Ravens catalog.
    """
    from cfb_coach.madden.scheme import play_cooldown as pcd
    from cfb_coach.opponents import is_cpu_opponent
    extras = extras or {}
    if pcd.is_titans_session(db, session_id, extras) or pcd.is_online_session(db, session_id, extras):
        return False
    if is_cpu_opponent(opponent_id):
        return False
    league = extras.get("league") or extras.get("franchise")
    dynasty = extras.get("dynasty")
    user_team = extras.get("user_team") or extras.get("user")
    sess_opp = None
    sess_opp_type = extras.get("opponent_type")
    if db is not None and session_id:
        try:
            row = db.conn.execute(
                "SELECT league, dynasty, user_team, opponent_id, opponent_type "
                "FROM game_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        except Exception:  # noqa: BLE001
            row = None
        if row:
            def _col(name, idx):
                try:
                    return row[name]
                except Exception:  # noqa: BLE001
                    return row[idx] if len(row) > idx else None
            league = league or _col("league", 0)
            dynasty = dynasty or _col("dynasty", 1)
            user_team = user_team or _col("user_team", 2)
            sess_opp = _col("opponent_id", 3)
            sess_opp_type = sess_opp_type or _col("opponent_type", 4)
    if is_cpu_opponent(sess_opp) or str(sess_opp_type or "").strip().lower() in {"cpu", "cpu_dynasty"}:
        return False
    team = " ".join(str(user_team or "").strip().split()).lower()
    if team in {"detroit lions", "lions", "det"} or team.endswith(" lions"):
        return True
    if str(dynasty or "").strip().lower() in {"primary", "franchise_primary"}:
        return True
    return False


def catalog_for_defense_session(db: Any, session_id: str | None, extras: Mapping[str, Any] | None,
                                opponent_id: str | None) -> RavensCatalog:
    if is_lions_human_defense_session(db, session_id, extras, opponent_id):
        return load_lions_franchise()
    return load()
