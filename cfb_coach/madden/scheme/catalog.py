"""Load and validate the user-curated 26-variant Lions offensive scheme catalog.

The dataset (``datasets/madden27/lions_offensive_scheme_catalog.json``, draft PR #152) is
expert guidance: names, setups and situation/coverage *tags*. It contains no measured
outcomes. Callable: the 26 ``plays[].variant_id`` entries everywhere, plus (owner instruction
2026-11-10, t335u, ``datasets/madden27/civil_redzone_owner_t335u.json``) the plays of the two
catalog red-zone packages, mapped to their stock formations, only inside the opponent 10
(``field_zone == "inside_10"``). The pending unnamed 5WR setup stays blocked.

Owner instruction 2026-11-11 (t349u, ``datasets/madden27/civil_designed_runs_owner_t349u.json``): the designed
runs already in the Buccaneers book in the four scheme formations (12 stock plays, no screens, no new RPOs) are
regular candidates (``field_zone == "any"``: outside the opponent 10), each its own ``variant_id``, called as the
stock play. They live in a separate owner file so the 26-variant catalog bytes, and every confirmation and
trained artifact pinned to its version, stay unchanged.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

DATASET_RELPATH = Path("datasets") / "madden27" / "lions_offensive_scheme_catalog.json"
REDZONE_RELPATH = Path("datasets") / "madden27" / "civil_redzone_owner_t335u.json"
RUNS_RELPATH = Path("datasets") / "madden27" / "civil_designed_runs_owner_t349u.json"
STOCK_PLAYBOOKS_RELPATH = Path("cfb_coach") / "data" / "madden27" / "playbooks_full.json"
STOCK_BOOK = "Buccaneers"
EXPECTED_VARIANTS = 26          # core variants (any field position)
EXPECTED_REDZONE_VARIANTS = 7   # red-zone plays: 5 catalog package plays (t335u) + 2 Close Flex plays (t338u)
INSIDE_10 = "inside_10"         # red-zone play callable only inside the opponent 10
RED_ZONE = "red_zone"           # red-zone play callable inside the 20 (inside the 10 and 10-20)
EXPECTED_REDZONE_FORMATIONS = {"Pistol Bunch TE": 3, "Gun Empty Quads": 2, "I Form Close Flex": 2}
# owner t349u: designed runs from the Buccaneers book in the scheme formations (no screens, no new RPOs)
EXPECTED_RUN_VARIANTS = 12
EXPECTED_RUN_FORMATIONS = {"Gun Doubles Clamp Stack": 4, "Gun Bunch TE": 5, "Gun Tight Doubles": 2, "Gun 5WR Tight": 1}
RUN, PASS = "run", "pass"       # play_family: the run-pass mix (coordinator.pick_mixed) gates on it
CORE, REDZONE_GROUP, DESIGNED_RUN = "core", "red_zone", "designed_run"
FORBIDDEN_RUN_WORDS = ("screen", "rpo")   # t349u: "no need for screens and rpos besides the goal to go one"


def field_band(yardline: Any) -> str | None:
    """inside_10 (>= 90) | ten_to_twenty (80-89) | outside (< 80) | None when unknown."""
    try:
        y = int(yardline)
    except (TypeError, ValueError):
        return None
    if y >= 90:
        return "inside_10"
    if y >= 80:
        return "ten_to_twenty"
    return "outside"
EXPECTED_PAIRS = 12
EXPECTED_FORMATIONS = {
    "Gun Doubles Clamp Stack": 6,
    "Gun Bunch TE": 7,
    "Gun Tight Doubles": 4,
    "Gun 5WR Tight": 9,
}


class CatalogError(ValueError):
    """The scheme catalog is missing, malformed or does not match its contract."""


@dataclass(frozen=True)
class Adjustment:
    target: str
    action: str
    macro_marked: bool
    index: int
    guide_text: str = ""     # the guide's own wording when it is not "<target>: <action>" (e.g. "Crosser Slot WR")

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "target": self.target, "action": self.action,
                "macro_marked": self.macro_marked, **({"guide_text": self.guide_text} if self.guide_text else {})}

    def line(self) -> str:
        return f"{self.target}: {self.action}" + (f' (guide: "{self.guide_text}")' if self.guide_text else "")


@dataclass(frozen=True)
class SchemeVariant:
    variant_id: str
    variant_name: str
    playbook: str
    formation: str
    base_play: str
    role: str
    scheme_category: str
    adjustments: tuple[Adjustment, ...]
    expert_situation_tags: tuple[str, ...]
    expert_coverage_tags: tuple[str, ...]
    coverage_tags_known: bool
    name_verified: bool
    present_in_book_as_shown: bool
    verified_in_user_game: bool
    adjustments_verified_in_user_game: bool
    unadjusted_allowed: bool
    verification_notes: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)
    field_zone: str = "any"        # "any" | "inside_10" | "red_zone"  (owner instructions t335u, t338u)
    package_id: str = ""
    play_family: str = PASS        # "pass" | "run" (explicit in the data files, never guessed from a name)
    group: str = CORE              # "core" (26) | "red_zone" | "designed_run" (owner t349u)

    @property
    def is_run(self) -> bool:
        return self.play_family == RUN

    def allowed_at(self, yardline: Any) -> bool:
        """Field-position gate (owner t338u):
        inside the opponent 10 -> red-zone plays only; 10-20 -> red_zone-tagged plays + regular variants;
        outside the 20 -> regular variants only. Unknown spot -> regular variants only."""
        band = field_band(yardline)
        if self.field_zone == "any":
            return band != "inside_10"
        if self.field_zone == INSIDE_10:
            return band == "inside_10"
        if self.field_zone == RED_ZONE:
            return band in ("inside_10", "ten_to_twenty")
        return False

    @property
    def pair(self) -> tuple[str, str]:
        return (self.formation, self.base_play)

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id, "variant_name": self.variant_name, "playbook": self.playbook,
            "formation": self.formation, "base_play": self.base_play, "role": self.role,
            "scheme_category": self.scheme_category,
            "adjustments": [a.as_dict() for a in self.adjustments],
            "expert_situation_tags": list(self.expert_situation_tags),
            "expert_coverage_tags": list(self.expert_coverage_tags),
            "coverage_tags_known": self.coverage_tags_known, "name_verified": self.name_verified,
            "verified_in_user_game": self.verified_in_user_game,
            "adjustments_verified_in_user_game": self.adjustments_verified_in_user_game,
            "unadjusted_allowed": self.unadjusted_allowed, "field_zone": self.field_zone,
            "package_id": self.package_id or None, "play_family": self.play_family, "group": self.group,
        }


@dataclass(frozen=True)
class BlockedEntry:
    entry_id: str
    kind: str          # red_zone_package | pending_5wr_setup
    reason: str
    formation: str | None
    base_plays: tuple[str, ...]


@dataclass(frozen=True)
class SchemeCatalog:
    catalog_id: str
    schema_version: str
    playbook: str
    digest: str                 # sha256 of the dataset bytes = dataset version
    path: str
    variants: tuple[SchemeVariant, ...]
    blocked: tuple[BlockedEntry, ...]
    data_quality_flags: tuple[str, ...]
    redzone_digest: str = ""
    redzone_owner: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)
    runs_digest: str = ""
    runs_owner: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def core_variants(self) -> tuple[SchemeVariant, ...]:
        """The 26 catalog pass variants."""
        return tuple(v for v in self.variants if v.group == CORE)

    @property
    def run_variants(self) -> tuple[SchemeVariant, ...]:
        """The 12 owner-confirmed designed runs (t349u)."""
        return tuple(v for v in self.variants if v.group == DESIGNED_RUN)

    @property
    def regular_variants(self) -> tuple[SchemeVariant, ...]:
        """Everything callable outside the opponent 10: the 26 pass variants + the designed runs."""
        return tuple(v for v in self.variants if v.field_zone == "any")

    @property
    def redzone_variants(self) -> tuple[SchemeVariant, ...]:
        return tuple(v for v in self.variants if v.field_zone != "any")

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(v.variant_id for v in self.variants)

    def get(self, variant_id: str) -> SchemeVariant | None:
        return next((v for v in self.variants if v.variant_id == variant_id), None)

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        seen: list[tuple[str, str]] = []
        for v in self.variants:
            if v.pair not in seen:
                seen.append(v.pair)
        return tuple(seen)

    @property
    def blocked_ids(self) -> frozenset[str]:
        return frozenset(b.entry_id for b in self.blocked)

    def version(self) -> str:
        return f"{self.catalog_id}@{self.digest[:12]}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_path() -> Path:
    return repo_root() / DATASET_RELPATH


def _name_verified(p: Mapping[str, Any]) -> bool:
    name = str(p.get("variant_name") or "")
    if "unverified" in name.lower() or "provisional" in str(p.get("variant_id") or ""):
        return False
    if p.get("variant_name_verified_in_screenshot") is False:
        return False
    return True


def parse(raw: Mapping[str, Any], *, digest: str, path: str = "") -> SchemeCatalog:
    plays = raw.get("plays")
    if not isinstance(plays, list) or not plays:
        raise CatalogError("catalog has no plays[]")
    variants: list[SchemeVariant] = []
    for p in plays:
        vid = str(p.get("variant_id") or "").strip()
        if not vid:
            raise CatalogError("a play has no variant_id")
        for key in ("formation", "base_play", "variant_name"):
            if not str(p.get(key) or "").strip():
                raise CatalogError(f"{vid}: missing {key}")
        adj = p.get("adjustments")
        if not isinstance(adj, list):
            raise CatalogError(f"{vid}: adjustments must be a list")
        adjustments = []
        for i, a in enumerate(adj):
            if not str(a.get("target") or "").strip() or not str(a.get("action") or "").strip():
                raise CatalogError(f"{vid}: adjustment {i} missing target/action")
            adjustments.append(Adjustment(str(a["target"]).strip(), str(a["action"]).strip(),
                                          bool(a.get("macro_marked")), i))
        avail = p.get("availability") or {}
        cov = tuple(str(t) for t in (p.get("expert_coverage_tags") or ()))
        cov_known = bool(cov) and p.get("coverage_guidance_status") != "not_provided_in_screenshots"
        variants.append(SchemeVariant(
            variant_id=vid, variant_name=str(p["variant_name"]), playbook=str(p.get("playbook") or ""),
            formation=str(p["formation"]), base_play=str(p["base_play"]), role=str(p.get("role") or ""),
            scheme_category=str(p.get("scheme_category") or ""), adjustments=tuple(adjustments),
            expert_situation_tags=tuple(str(t) for t in (p.get("expert_situation_tags") or ())),
            expert_coverage_tags=cov, coverage_tags_known=cov_known, name_verified=_name_verified(p),
            present_in_book_as_shown=bool(avail.get("present_in_buccaneers_playbook_as_shown")),
            verified_in_user_game=bool(avail.get("verified_in_user_game")),
            adjustments_verified_in_user_game=bool(avail.get("adjustments_verified_in_user_game")),
            # The dataset never marks an unadjusted base play as an approved variant.
            unadjusted_allowed=bool(p.get("unadjusted_allowed", False)),
            verification_notes=str(p.get("verification_notes") or ""), raw=p))
    ids = [v.variant_id for v in variants]
    if len(set(ids)) != len(ids):
        raise CatalogError("duplicate variant_id")
    blocked: list[BlockedEntry] = []
    for pk in raw.get("red_zone_packages") or ():
        blocked.append(BlockedEntry(str(pk.get("package_id")), "red_zone_package",
                                    "unverified red-zone package (lesson heading/formation unconfirmed); never callable",
                                    pk.get("formation"), tuple(pk.get("base_plays") or ())))
    for pk in raw.get("pending_5wr_variants") or ():
        blocked.append(BlockedEntry(str(pk.get("pending_id")), "pending_5wr_setup",
                                    str(pk.get("reason_not_callable") or "unnamed pending setup; never callable"),
                                    pk.get("formation"), tuple(x for x in [pk.get("base_play")] if x)))
    if set(ids) & {b.entry_id for b in blocked}:
        raise CatalogError("a blocked entry shares an id with a callable variant")
    return SchemeCatalog(
        catalog_id=str(raw.get("catalog_id") or ""), schema_version=str(raw.get("schema_version") or ""),
        playbook=str(raw.get("installed_offensive_playbook") or ""), digest=digest, path=path,
        variants=tuple(variants), blocked=tuple(blocked),
        data_quality_flags=tuple(str(x) for x in (raw.get("data_quality_flags") or ())))


def load(path: str | Path | None = None) -> SchemeCatalog:
    p = Path(path) if path else default_path()
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise CatalogError(f"scheme catalog not found at {p}: {exc}") from exc
    try:
        raw = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise CatalogError(f"scheme catalog is not valid JSON: {exc}") from exc
    cat = parse(raw, digest=hashlib.sha256(data).hexdigest(), path=str(p))
    rz = p.parent / REDZONE_RELPATH.name
    if rz.is_file():
        rb = rz.read_bytes()
        cat = with_redzone(cat, raw, json.loads(rb.decode("utf-8")), digest=hashlib.sha256(rb).hexdigest())
    runs = p.parent / RUNS_RELPATH.name
    if runs.is_file():
        rb = runs.read_bytes()
        cat = with_designed_runs(cat, json.loads(rb.decode("utf-8")), digest=hashlib.sha256(rb).hexdigest())
    return cat


@lru_cache(maxsize=1)
def stock_offense_plays(book: str = STOCK_BOOK) -> dict[str, tuple[str, ...]]:
    """formation -> play names in the in-repo stock offense book (exact names)."""
    data = json.loads((repo_root() / STOCK_PLAYBOOKS_RELPATH).read_text())
    forms = data["books"]["offense"][book]["formations"]
    return {f: tuple(str(x) for x in (d.get("plays") or ())) for f, d in forms.items()}


def with_designed_runs(cat: SchemeCatalog, owner: Mapping[str, Any], *, digest: str) -> SchemeCatalog:
    """Owner instruction t349u: the designed runs already in the Buccaneers book, in the scheme's formations.

    Fails closed (CatalogError) unless every entry is an exact stock play name in that formation of the in-repo
    Buccaneers book, the formation is one of the scheme's 4 formations, the play is a designed run (play_family
    "run"), it is not a screen or an RPO, and it has no adjustments (stock play; variation = the play name)."""
    if owner.get("catalog_version") != cat.version():
        raise CatalogError(f"designed-run owner file is for {owner.get('catalog_version')}, catalog is {cat.version()}")
    stock = stock_offense_plays(STOCK_BOOK)
    scheme_forms = {v.formation for v in cat.core_variants}
    extra: list[SchemeVariant] = []
    for m in owner.get("variants") or ():
        vid, form, play = str(m.get("variant_id") or ""), str(m.get("formation") or ""), str(m.get("base_play") or "")
        if not vid or not form or not play:
            raise CatalogError(f"designed run {m!r}: missing variant_id/formation/base_play")
        if form not in scheme_forms:
            raise CatalogError(f"{vid}: '{form}' is not one of the scheme formations")
        if play not in stock.get(form, ()):
            raise CatalogError(f"{vid}: '{play}' is not a stock play in {STOCK_BOOK} '{form}' (exact name)")
        if any(w in play.lower().split() for w in FORBIDDEN_RUN_WORDS):
            raise CatalogError(f"{vid}: screens and RPOs are not designed runs (owner t349u)")
        if m.get("play_family") != RUN:
            raise CatalogError(f"{vid}: designed run must have play_family 'run'")
        if m.get("adjustments"):
            raise CatalogError(f"{vid}: designed runs are stock plays; no adjustments may be invented")
        if str(m.get("variant_name") or "") != play:
            raise CatalogError(f"{vid}: the variation of a stock run is the stock play name itself")
        extra.append(SchemeVariant(
            variant_id=vid, variant_name=play, playbook=cat.playbook, formation=form, base_play=play,
            role="owner_designed_run", scheme_category="Run", adjustments=(),
            expert_situation_tags=(), expert_coverage_tags=(), coverage_tags_known=False, name_verified=True,
            present_in_book_as_shown=True, verified_in_user_game=False, adjustments_verified_in_user_game=False,
            unadjusted_allowed=True, verification_notes=str(m.get("note") or ""), raw=m, field_zone="any",
            package_id="", play_family=RUN, group=DESIGNED_RUN))
    ids = [v.variant_id for v in cat.variants] + [v.variant_id for v in extra]
    if len(set(ids)) != len(ids):
        raise CatalogError("duplicate variant_id in the designed-run owner file")
    if len({v.pair for v in extra}) != len(extra):
        raise CatalogError("duplicate formation/play in the designed-run owner file")
    from dataclasses import replace
    return replace(cat, variants=cat.variants + tuple(extra), runs_digest=digest,
                   runs_owner=dict(owner.get("owner_instruction") or {}))


def with_redzone(cat: SchemeCatalog, raw: Mapping[str, Any], owner: Mapping[str, Any], *, digest: str) -> SchemeCatalog:
    """Owner instruction t335u: the two catalog red-zone packages become inside-10 candidates.

    Every field comes from the catalog package (base-play list, adjustments verbatim, tags) except the
    formation, which the owner file maps from the sheet name + the stock-book co-occurrence. Nothing is
    invented: a play not listed in its package, or adjustments that differ from the package, is an error."""
    if owner.get("catalog_version") != cat.version():
        raise CatalogError(f"red-zone owner mapping is for {owner.get('catalog_version')}, catalog is {cat.version()}")
    packages = {str(pk.get("package_id")): pk for pk in raw.get("red_zone_packages") or ()}
    extra: list[SchemeVariant] = []
    used: set[str] = set()
    for m in owner.get("variants") or ():
        zone = str(m.get("field_zone") or INSIDE_10)
        if zone not in (INSIDE_10, RED_ZONE):
            raise CatalogError(f"{m.get('variant_id')}: bad field_zone {zone}")
        if m.get("package_id") is None:
            # owner-supplied play (t338u screenshots): adjustments are the guide's setup, given verbatim in the file
            pk = {"package_id": "", "role": "expert_red_zone_play", "expert_coverage_tags": [],
                  "expert_situation_tags": m.get("expert_situation_tags") or []}
            adj = list(m.get("adjustments") or [])
        else:
            pk = packages.get(str(m.get("package_id")))
            if pk is None:
                raise CatalogError(f"{m.get('variant_id')}: unknown package {m.get('package_id')}")
            if m.get("package_base_play_label") not in (pk.get("base_plays") or ()):
                raise CatalogError(f"{m.get('variant_id')}: '{m.get('package_base_play_label')}' is not listed in {pk['package_id']}")
            adj = pk.get("adjustments") or [] if m.get("adjustments_from_package") else []
        adjustments = tuple(Adjustment(str(a["target"]).strip(), str(a["action"]).strip(), bool(a.get("macro_marked")), i,
                                       str(a.get("guide_text") or ""))
                            for i, a in enumerate(adj))
        cov = tuple(str(t) for t in (pk.get("expert_coverage_tags") or ()))
        extra.append(SchemeVariant(
            variant_id=str(m["variant_id"]), variant_name=str(m["variant_name"]), playbook=cat.playbook,
            formation=str(m["formation"]), base_play=str(m["base_play"]), role=str(pk.get("role") or ""),
            scheme_category="Red Zone", adjustments=adjustments,
            expert_situation_tags=tuple(str(t) for t in (pk.get("expert_situation_tags") or ())),
            expert_coverage_tags=cov, coverage_tags_known=bool(cov), name_verified=False,
            present_in_book_as_shown=True, verified_in_user_game=False, adjustments_verified_in_user_game=False,
            unadjusted_allowed=not adjustments,      # stock play: the package lists nothing to adjust
            verification_notes=str(m.get("note") or ""), raw=m, field_zone=zone,
            package_id=str(pk["package_id"]), play_family=_family(m), group=REDZONE_GROUP))
        if pk["package_id"]:
            used.add(str(pk["package_id"]))
    ids = [v.variant_id for v in cat.variants] + [v.variant_id for v in extra]
    if len(set(ids)) != len(ids):
        raise CatalogError("duplicate variant_id in the red-zone mapping")
    blocked = tuple(b for b in cat.blocked if b.entry_id not in used)
    from dataclasses import replace
    return replace(cat, variants=cat.variants + tuple(extra), blocked=blocked, redzone_digest=digest,
                   redzone_owner=dict(owner.get("owner_instruction") or {}))


def _family(m: Mapping[str, Any]) -> str:
    fam = m.get("play_family")
    if fam not in (RUN, PASS):
        raise CatalogError(f"{m.get('variant_id')}: play_family must be 'run' or 'pass'")
    return str(fam)


@lru_cache(maxsize=4)
def load_cached(path: str | None = None) -> SchemeCatalog:
    return load(path)


def contract_errors(cat: SchemeCatalog) -> list[str]:
    """Every way the loaded catalog differs from the approved 26 / 12 / 4 contract (+ 7 red-zone, + 12 runs)."""
    errs: list[str] = []
    core = cat.core_variants
    if len(core) != EXPECTED_VARIANTS:
        errs.append(f"expected {EXPECTED_VARIANTS} variants, found {len(core)}")
    pairs = {v.pair for v in core}
    if len(pairs) != EXPECTED_PAIRS:
        errs.append(f"expected {EXPECTED_PAIRS} formation/base-play pairs, found {len(pairs)}")
    rz = cat.redzone_variants
    if rz:
        rc: dict[str, int] = {}
        for v in rz:
            rc[v.formation] = rc.get(v.formation, 0) + 1
            if v.field_zone not in (INSIDE_10, RED_ZONE):
                errs.append(f"{v.variant_id}: red-zone play without a red-zone gate")
        if len(rz) != EXPECTED_REDZONE_VARIANTS or rc != EXPECTED_REDZONE_FORMATIONS:
            errs.append(f"red-zone plays {rc} != {EXPECTED_REDZONE_FORMATIONS}")
    if any(v.is_run for v in core):
        errs.append("a core catalog variant is marked as a run (the 26 are passes)")
    runs = cat.run_variants
    if runs:
        rc2: dict[str, int] = {}
        for v in runs:
            rc2[v.formation] = rc2.get(v.formation, 0) + 1
            if v.field_zone != "any" or not v.is_run or v.adjustments:
                errs.append(f"{v.variant_id}: designed run must be a regular-field stock run")
        if len(runs) != EXPECTED_RUN_VARIANTS or rc2 != EXPECTED_RUN_FORMATIONS:
            errs.append(f"designed runs {rc2} != {EXPECTED_RUN_FORMATIONS}")
    counts: dict[str, int] = {}
    for v in core:
        counts[v.formation] = counts.get(v.formation, 0) + 1
    if counts != EXPECTED_FORMATIONS:
        errs.append(f"formation counts {counts} != {EXPECTED_FORMATIONS}")
    for v in cat.variants:
        if not v.adjustments and not v.unadjusted_allowed:
            errs.append(f"{v.variant_id}: no adjustments and unadjusted version not allowed")
        if v.playbook and v.playbook != cat.playbook:
            errs.append(f"{v.variant_id}: playbook {v.playbook!r} != catalog playbook {cat.playbook!r}")
    return errs
