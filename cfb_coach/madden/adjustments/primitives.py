"""Adjustment primitives that already exist in the research files.

A primitive is accepted only when ``editor_fields.json``, ``research_db.json``,
or the pre-snap adjustment kinds name it. Anything else is rejected. Motion is
registered and blocked: its Xbox control is a source conflict, so it is not
usable. Editor rows a source names that are not in ``editor_fields.json`` are
rejected the same way. Fields with no researched value stay available as slots
and are filled with the literal ``Default`` by the defense generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cfb_coach.madden import research_db as rdb
from cfb_coach.madden.adjustments.candidates import MacroSide


class PrimitiveKind(str, Enum):
    """Designer label for a sourced row. It is not a wire-contract enum."""

    HOT_ROUTE = "hot_route"
    PROTECTION = "protection"
    MOTION = "motion"
    AUDIBLE = "audible"
    SHADING = "shading"
    SHELL = "shell"
    ZONE_DROP = "zone_drop"
    ASSIGNMENT = "assignment"
    DL_ALIGNMENT = "dl_alignment"
    STUNT = "stunt"
    LB_ADJUSTMENT = "lb_adjustment"
    CONTAIN = "contain"
    PRESSURE = "pressure"
    COVERAGE_MOD = "coverage_mod"
    UNKNOWN = "UNKNOWN"

# Kinds the existing pre-snap module already dispatches. A research row whose
# type is outside this set is not an adjustment this coach can describe.
_OFFENSE_KINDS = {
    "hot_route": PrimitiveKind.HOT_ROUTE,
    "pass_protection": PrimitiveKind.PROTECTION,
    "audible": PrimitiveKind.AUDIBLE,
    "motion": PrimitiveKind.MOTION,
    "fake_snap": PrimitiveKind.UNKNOWN,
    "flip_run": PrimitiveKind.UNKNOWN,
}

_FIELD_KIND: dict[tuple[str, str], PrimitiveKind] = {}

# Sourced from the 2026-10-10 pre-snap verification, not from research_db.
# EA: "Any route can be assigned a chip."
# 818 Madden: the HB hot-route menu includes Chip and Release.
CHIP_AND_RELEASE_ID = "hot_hb_chip_and_release"
CHIP_AND_RELEASE_SOURCES = (
    "https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay",
    "https://www.818madden.com/tools/position-adjustments",
)
CHIP_AND_RELEASE_WHY = (
    "HB hot-route menu includes Chip and Release. Any route can be assigned a chip."
)


def _kind_for_field(section: str, setting: str) -> PrimitiveKind:
    if setting == "Coverage Shading":
        return PrimitiveKind.SHADING
    if setting == "Coverage Shell":
        return PrimitiveKind.SHELL
    if section == "Zone Drops":
        return PrimitiveKind.ZONE_DROP
    if section == "Individuals":
        return PrimitiveKind.ASSIGNMENT
    if setting in ("DL Technique", "DL Point of Attack"):
        return PrimitiveKind.DL_ALIGNMENT
    if setting == "DL Stunts":
        return PrimitiveKind.STUNT
    if setting in ("LB Shift", "LB Assignments"):
        return PrimitiveKind.LB_ADJUSTMENT
    if setting == "QB Contain":
        return PrimitiveKind.CONTAIN
    if setting == "Show Blitz":
        return PrimitiveKind.PRESSURE
    if section in ("General", "Secondary", "Strategy", "Coverage Checks"):
        return PrimitiveKind.COVERAGE_MOD
    return PrimitiveKind.UNKNOWN


@dataclass(frozen=True)
class RegisteredPrimitive:
    primitive_id: str
    side: MacroSide
    kind: PrimitiveKind
    label: str
    usable: bool
    block_reason: str | None = None
    source_ids: tuple[str, ...] = ()
    confidence: str | None = None
    target: str | None = None
    route: str | None = None
    vs: tuple[str, ...] = ()
    answers: tuple[str, ...] = ()
    section: str | None = None
    setting: str | None = None
    value: str | None = None
    control: str | None = None
    origin: str = ""


def editor_field_keys() -> list[tuple[str, str]]:
    """Defense editor rows in file order. There is no offense section."""
    fields = rdb.editor_fields().get("defense") or {}
    return [(section, name) for section, names in fields.items() for name in names]


def field_key(section: str, setting: str) -> str:
    return f"{section}|{setting}"


def canonical_field(section: str, setting: str) -> tuple[str, str] | None:
    """Map a research row onto an editor field, or None when it is not one."""
    aliases = rdb.editor_fields().get("aliases") or {}
    setting = str(aliases.get(setting, setting))
    section_l = str(section).casefold()
    setting_l = setting.casefold()
    for sec, name in editor_field_keys():
        if sec.casefold() == section_l and name.casefold() == setting_l:
            return sec, name
    return None


def value_id(section: str, setting: str, value: str) -> str:
    return f"field:{section}|{setting}={value}"


def _index() -> dict[str, RegisteredPrimitive]:
    known_sources = set(rdb.sources())
    out: dict[str, RegisteredPrimitive] = {}

    for section, setting in editor_field_keys():
        pid = f"slot:{field_key(section, setting)}"
        out[pid] = RegisteredPrimitive(
            primitive_id=pid,
            side=MacroSide.DEF,
            kind=_kind_for_field(section, setting),
            label=f"{section} / {setting}",
            usable=True,
            section=section,
            setting=setting,
            origin="editor_fields",
        )

    controls = (rdb.load().get("controls_xbox") or {}).get("offense") or {}
    motion = controls.get("motion") or {}
    motion_confidence = motion.get("confidence")
    motion_sources = tuple(s for s in (motion.get("sources") or []) if s in known_sources)
    out["control:offense:motion"] = RegisteredPrimitive(
        primitive_id="control:offense:motion",
        side=MacroSide.OFF,
        kind=PrimitiveKind.MOTION,
        label="Motion",
        usable=motion_confidence == "confirmed" and bool(motion_sources),
        block_reason=(
            None
            if motion_confidence == "confirmed" and motion_sources
            else (
                "motion stays off: no Xbox control is verified yet. "
                "EA's official preplay page has no motion entry. "
                "Realsport101, Civil.GG, and ClutchPoints conflict. "
                "Best estimate is select the player, then left stick left/right, "
                "medium confidence, pending a Practice-mode test."
            )
        ),
        source_ids=motion_sources,
        confidence=str(motion_confidence) if motion_confidence else None,
        control="motion",
        origin="controls_xbox",
    )

    for row in rdb.offense_adjustments():
        kind_name = str(row.get("type") or "")
        if kind_name not in _OFFENSE_KINDS:
            continue
        pid = str(row.get("id") or "")
        sources = tuple(s for s in (row.get("sources") or []) if s in known_sources)
        if not pid or not sources:
            continue
        control = {
            "hot_route": "hot_route",
            "audible": "audible",
            "pass_protection": "pass_protection",
            "motion": "motion",
        }.get(kind_name)
        confidence = None
        if control:
            confidence = ((controls.get(control) or {}).get("confidence")) or None
        blocked = kind_name == "motion" or confidence == "conflict"
        out[pid] = RegisteredPrimitive(
            primitive_id=pid,
            side=MacroSide.OFF,
            kind=_OFFENSE_KINDS[kind_name],
            label=pid,
            usable=not blocked and bool(row.get("target") or row.get("route") or kind_name == "audible"),
            block_reason="motion is not a verified control" if blocked else None,
            source_ids=sources,
            confidence=str(confidence) if confidence else None,
            target=str(row.get("target") or "") or None,
            route=str(row.get("route") or "") or None,
            vs=tuple(str(v) for v in (row.get("vs") or [])),
            control=control,
            origin="offense_adjustments",
        )

    # Meta Researcher 2026-10-10. Not a research_db row. Only the details
    # the write-up sources: HB hot-route menu item, and EA's chip sentence.
    out[CHIP_AND_RELEASE_ID] = RegisteredPrimitive(
        primitive_id=CHIP_AND_RELEASE_ID,
        side=MacroSide.OFF,
        kind=PrimitiveKind.HOT_ROUTE,
        label="HB Chip and Release",
        usable=True,
        source_ids=CHIP_AND_RELEASE_SOURCES,
        confidence="verified-m27",
        target="HB",
        route="Chip and Release",
        vs=(),
        control="hot_route",
        origin="writeup",
    )

    for row in rdb.defense_adjustments():
        pid = str(row.get("id") or "")
        control = str(row.get("control") or "")
        sources = tuple(s for s in (row.get("sources") or []) if s in known_sources)
        def_controls = (rdb.load().get("controls_xbox") or {}).get("defense") or {}
        if not pid or control not in def_controls or not sources:
            continue
        confidence = (def_controls.get(control) or {}).get("confidence")
        out[f"presnap:{pid}"] = RegisteredPrimitive(
            primitive_id=f"presnap:{pid}",
            side=MacroSide.DEF,
            kind=PrimitiveKind.COVERAGE_MOD,
            label=str(row.get("label") or pid),
            usable=False,
            block_reason=(
                "pre-snap button adjustment; it does not name a Custom Adjustments editor value"
            ),
            source_ids=sources,
            confidence=str(confidence) if confidence else None,
            answers=tuple(str(a) for a in (row.get("answers") or [])),
            control=control,
            origin="defense_adjustments",
        )

    for macro in rdb.defense_macros():
        for row in macro.get("settings") or []:
            canon = canonical_field(str(row.get("section") or ""), str(row.get("setting") or ""))
            source = str(row.get("source") or "")
            value = str(row.get("value") or "").strip()
            if canon is None or source not in known_sources or not value or value == "Default":
                continue
            section, setting = canon
            pid = value_id(section, setting, value)
            current = out.get(pid)
            sources = (source,) if current is None else tuple(dict.fromkeys((*current.source_ids, source)))
            out[pid] = RegisteredPrimitive(
                primitive_id=pid,
                side=MacroSide.DEF,
                kind=_kind_for_field(section, setting),
                label=f"{setting} {value}",
                usable=True,
                source_ids=sources,
                confidence=None,
                section=section,
                setting=setting,
                value=value,
                answers=tuple(str(a) for a in (macro.get("answers") or [])),
                origin="defense_macros",
            )
    return out


def registry() -> dict[str, RegisteredPrimitive]:
    return _index()


def require_primitive(primitive_id: str) -> RegisteredPrimitive:
    found = registry().get(primitive_id)
    if found is None:
        raise KeyError(
            f"{primitive_id!r} is not in editor_fields.json, research_db.json, "
            "or the existing adjustment kinds"
        )
    if not found.usable:
        raise ValueError(found.block_reason or f"{primitive_id} is not a usable primitive")
    return found


def offense_primitives() -> list[RegisteredPrimitive]:
    return [
        row for row in registry().values()
        if row.origin in ("offense_adjustments", "writeup")
    ]


def usable_offense() -> list[RegisteredPrimitive]:
    return [row for row in offense_primitives() if row.usable]


def unverified() -> list[RegisteredPrimitive]:
    """Registered primitives the generator will not place on a draft."""
    return [row for row in registry().values() if not row.usable]


def fields_without_researched_value() -> list[tuple[str, str]]:
    valued = {
        (row.section, row.setting)
        for row in registry().values()
        if row.origin == "defense_macros" and row.section and row.setting
    }
    return [pair for pair in editor_field_keys() if pair not in valued]


def rejected_non_editor_settings() -> list[dict[str, str]]:
    """Source rows that name a cell ``editor_fields.json`` does not have."""
    dropped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for macro in rdb.defense_macros():
        for row in macro.get("settings") or []:
            section = str(row.get("section") or "")
            setting = str(row.get("setting") or "")
            value = str(row.get("value") or "")
            if canonical_field(section, setting) is not None:
                continue
            key = (section, setting, value)
            if key in seen:
                continue
            seen.add(key)
            dropped.append({
                "macro_id": str(macro.get("id") or ""),
                "section": section,
                "setting": setting,
                "value": value,
                "reason": "not a field in editor_fields.json",
            })
    return dropped


def lab_dict(primitive: RegisteredPrimitive) -> dict[str, object]:
    """Shape ``offense_macro_lab.conflicts`` already understands."""
    kind = {
        PrimitiveKind.HOT_ROUTE: "hot_route",
        PrimitiveKind.PROTECTION: "pass_protection",
        PrimitiveKind.AUDIBLE: "audible",
        PrimitiveKind.MOTION: "motion",
    }.get(primitive.kind, primitive.kind.value)
    return {
        "id": primitive.primitive_id,
        "type": kind,
        "kind": kind,
        "target": primitive.target,
        "route": primitive.route,
        "vs": list(primitive.vs),
        "label": primitive.label,
        "sources": list(primitive.source_ids),
    }


__all__ = [
    "CHIP_AND_RELEASE_ID",
    "CHIP_AND_RELEASE_SOURCES",
    "CHIP_AND_RELEASE_WHY",
    "PrimitiveKind",
    "RegisteredPrimitive",
    "canonical_field",
    "editor_field_keys",
    "field_key",
    "fields_without_researched_value",
    "lab_dict",
    "offense_primitives",
    "registry",
    "rejected_non_editor_settings",
    "require_primitive",
    "unverified",
    "usable_offense",
    "value_id",
]
