"""Original Custom Adjustment drafts for the installed Madden book.

Offense drafts are researched hot routes, protections, and audibles on plays
in the applied book, plus a route combo when two primitives pass the existing
conflict check. Motion is omitted: its control is a source conflict.

Defense drafts fill every defense editor field with a researched value or the
literal ``Default``. Two-recipe sheets come from ``defense_macro_lab``. A
Dime Fire 2 sheet is added so shell, show-blitz, and the other sourced cells
on that recipe are present; rows that are not in ``editor_fields.json`` are
dropped. Linebacker shift and assignment have no researched value, so they
stay ``Default``.

Every generated adjustment starts at DRAFT. Moving it requires user-confirmation
evidence at each step. Nothing here creates a macro on Xbox. The no-adjustment
baseline is the unmodified installed play and is always in the set.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden import catalog, playbook, research_db as rdb
from cfb_coach.madden.adjustments.candidates import (
    WIRE_VERSION,
    AdjustmentPrimitive,
    EditorFieldAssignment,
    EditorFieldSource,
    MacroCandidate,
    MacroJointFeatures,
    MacroSide,
    MacroState,
    candidate_to_json,
    is_live_selectable,
)
from cfb_coach.madden.adjustments.primitives import (
    PrimitiveKind,
    RegisteredPrimitive,
    canonical_field,
    lab_dict,
    require_primitive,
    usable_offense,
    value_id,
)
from cfb_coach.madden.model.schema import defensive_editor_field_names
from cfb_coach.madden.model.defense_macro_lab import compose_defensive_macros
from cfb_coach.madden.model.offense_macro_lab import conflicts as _offense_conflicts

DRAFT_META_KEY = "macro_designer_drafts.v1"
DESIGN_VERSION = "1"
XBOX_DISCLAIMER = "User attestation only. This program did not create the macro on Xbox."

OFFENSE_FORMATIONS = (
    "Gun Doubles Clamp Stack",
    "Gun 5WR Tight",
    "Gun Trips X Nasty",
    "Pistol Deuce Close",
    "Pistol Trips",
)
DEFENSE_FORMATIONS = (
    "Dime 3-2 Odd",
    "Nickel Double Mug",
    "Nickel Single Mug",
    "4-3 Over",
    "Nickel Over",
)

_ORDER = (
    MacroState.DRAFT,
    MacroState.EDITOR_VERIFIED,
    MacroState.INSTALLED,
    MacroState.ARMED,
    MacroState.ELIGIBLE,
    MacroState.EVALUATED,
)
_REL = {"confirmed": 1.0, "single-source": 0.5, "conflict": 0.0}


class NotLiveSelectable(ValueError):
    """The macro's state is not ARMED, ELIGIBLE, or EVALUATED."""


@dataclass(frozen=True)
class CatalogComparison:
    catalog_id: str
    source: str
    relation: str
    shared_settings: tuple[str, ...] = ()
    distinct_settings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DesignRecord:
    candidate: MacroCandidate
    downside: str | None
    uncertainty_note: str | None
    catalog_comparison: tuple[CatalogComparison, ...]
    install_instructions: tuple[str, ...]
    confirmation_evidence: tuple[str, ...] = ()
    xbox_created: bool = False

    def __post_init__(self) -> None:
        if self.xbox_created:
            raise ValueError("this program does not create macros on Xbox")


def trimmed_formations() -> dict[str, dict[str, list[str]]]:
    """Aidan's trimmed book: five Bucs offense formations and five 49ers defenses."""
    offense = catalog.book_formations("offense", "Buccaneers")
    defense = catalog.book_formations("defense", "49ers")
    return {
        "offense": {name: list(offense[name]) for name in OFFENSE_FORMATIONS},
        "defense": {name: list(defense[name]) for name in DEFENSE_FORMATIONS},
    }


def install_trimmed_book(db: Any) -> None:
    """Write the trimmed book into ``active_playbook_json`` on a fixture DB."""
    forms = trimmed_formations()
    state = {"applied": {}, "pending": {}}
    for side, book in (("offense", "Buccaneers"), ("defense", "49ers")):
        names = list(forms[side])
        state["applied"][side] = {
            "side": side,
            "mode": "custom",
            "name": playbook.CUSTOM_NAME[side],
            "source_book": book,
            "formations": forms[side],
            "core": names,
            "rev": 1,
            "locked_ts": "2026-10-08T00:00:00+00:00",
            "reason": "fixture trimmed custom book",
            "trimmed": True,
        }
    db.set_meta(playbook.META_KEY, json.dumps(state))


def installed_formations(db: Any) -> dict[str, dict[str, list[str]]]:
    books = playbook.load_books(db)
    out: dict[str, dict[str, list[str]]] = {}
    for side in ("offense", "defense"):
        out[side] = {
            str(formation): list(plays)
            for formation, plays in ((books.get(side) or {}).get("formations") or {}).items()
        }
    return out


def require_installed(
    formations: Mapping[str, Sequence[str]] | None,
    formation: str,
    play: str,
) -> None:
    plays = None if not formations else formations.get(formation)
    if plays is None or play not in list(plays):
        raise ValueError(f"{formation!r} / {play!r} is not in the installed book")


def offense_conflict(parts: Sequence[Mapping[str, Any]]) -> str | None:
    """Existing offense conflict rules. A reason means the combo is rejected."""
    return _offense_conflicts([dict(part) for part in parts])


def defense_value_conflict(sheets: Sequence[Mapping[tuple[str, str], str]]) -> str | None:
    """Two researched values for one editor field cannot share a sheet."""
    merged: dict[tuple[str, str], str] = {}
    for sheet in sheets:
        for key, value in sheet.items():
            previous = merged.get(key)
            if previous is not None and previous != value:
                section, setting = key
                return f"{section} / {setting} is {previous!r} and {value!r}"
            merged[key] = value
    return None


def rejected_offense_pairs() -> list[tuple[str, str, str]]:
    rows = [row for row in usable_offense() if row.kind is not PrimitiveKind.MOTION]
    rejected: list[tuple[str, str, str]] = []
    for left, right in combinations(rows, 2):
        reason = offense_conflict([lab_dict(left), lab_dict(right)])
        if reason:
            rejected.append((left.primitive_id, right.primitive_id, reason))
    return rejected


def backup_database(path: Path) -> Path:
    """Copy a DB file before a designer write. Raises if the file is missing."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"no database to back up at {source}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = source.with_name(f"{source.name}.bak-{stamp}")
    shutil.copy2(source, dest)
    return dest


def generate(
    formations: Mapping[str, Mapping[str, Sequence[str]]],
) -> list[DesignRecord]:
    """Drafts plus a no-adjustment baseline for every installed play."""
    records = [
        *_baselines(formations),
        *_offense(formations.get("offense") or {}),
        *_defense(formations.get("defense") or {}),
    ]
    ids = [row.candidate.macro_id for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("generated macro ids are not unique")
    return records


def generate_from_db(db: Any) -> list[DesignRecord]:
    formations = installed_formations(db)
    if not formations.get("offense") and not formations.get("defense"):
        raise ValueError("active_playbook_json has no installed formations")
    return generate(formations)


def stage_drafts(db: Any, *, backup: bool = True) -> dict[str, Any]:
    """Store drafts on a designer-only meta key.

    Snap history and ``active_macros:*`` loadouts are not written. When
    ``backup`` is set, the DB file is copied before the meta write.
    """
    if backup:
        if hasattr(db, "conn"):
            db.conn.commit()
        backup_database(Path(db.path))
    records = generate_from_db(db)
    payload = {
        "__schema__": WIRE_VERSION,
        "xbox_created": False,
        "candidates": [candidate_to_json(row.candidate) for row in records],
    }
    db.set_meta(DRAFT_META_KEY, json.dumps(payload))
    return {"stored": len(records), "meta_key": DRAFT_META_KEY, "xbox_created": False}


def transition(record: DesignRecord, to_state: MacroState, evidence: str) -> DesignRecord:
    """Move one step. Evidence is required for every state past DRAFT.

    The stored note says the user attested. It is not an Xbox creation claim.
    New drafts are not moved by ``generate``.
    """
    if record.candidate.is_no_adjustment:
        raise ValueError("the no-adjustment baseline is not a custom adjustment")
    if to_state is MacroState.UNKNOWN:
        raise ValueError("UNKNOWN is not a transition target")
    text = (evidence or "").strip()
    if not text:
        raise ValueError("transition past DRAFT requires user-confirmation evidence")
    if XBOX_DISCLAIMER not in text:
        text = f"{text} {XBOX_DISCLAIMER}"
    current = record.candidate.state
    if to_state is MacroState.RETIRED:
        if current is MacroState.RETIRED:
            raise ValueError("the macro is already retired")
    else:
        if current not in _ORDER or to_state not in _ORDER:
            raise ValueError("unknown state transition")
        current_index = _ORDER.index(current)
        target_index = _ORDER.index(to_state)
        if target_index != current_index + 1:
            nxt = _ORDER[current_index + 1].value if current_index + 1 < len(_ORDER) else "RETIRED"
            raise ValueError(f"{current.value} moves to {nxt} before {to_state.value}")
    candidate = replace(record.candidate, state=to_state)
    return replace(
        record,
        candidate=candidate,
        confirmation_evidence=(*record.confirmation_evidence, text),
        xbox_created=False,
    )


def require_live(candidate: MacroCandidate) -> None:
    if candidate.conflicts:
        raise NotLiveSelectable(f"{candidate.macro_id} has conflicting adjustments")
    if not is_live_selectable(candidate.state):
        raise NotLiveSelectable(
            f"{candidate.macro_id} is {candidate.state.value} and is not live-selectable"
        )


def select_live(
    records: Sequence[DesignRecord],
    *,
    formation: str,
    play: str,
) -> MacroCandidate:
    """Return an armed-or-later macro for this play, or the no-adjustment baseline.

    Drafts, editor-verified sheets, and installed-but-unarmed sheets are refused.
    This is a legality gate. The Madden ML Lead owns ranking.
    """
    pool = [
        row.candidate
        for row in records
        if row.candidate.formation == formation and row.candidate.play == play
    ]
    if not pool:
        raise ValueError(f"{formation!r} / {play!r} is not in this candidate set")
    armed = [
        row for row in pool
        if not row.is_no_adjustment and not row.conflicts and is_live_selectable(row.state)
    ]
    if armed:
        order = {MacroState.EVALUATED: 0, MacroState.ELIGIBLE: 1, MacroState.ARMED: 2}
        armed.sort(key=lambda row: (
            order.get(row.state, 9),
            row.joint.uncertainty if row.joint.uncertainty is not None else 1.0,
            row.macro_id,
        ))
        return armed[0]
    baselines = [
        row for row in pool
        if row.is_no_adjustment and not row.conflicts and is_live_selectable(row.state)
    ]
    if baselines:
        return baselines[0]
    raise NotLiveSelectable(f"no live-selectable candidate for {formation} / {play}")


def _baselines(formations: Mapping[str, Mapping[str, Sequence[str]]]) -> list[DesignRecord]:
    records: list[DesignRecord] = []
    for side_name, side in (("offense", MacroSide.OFF), ("defense", MacroSide.DEF)):
        for formation, plays in (formations.get(side_name) or {}).items():
            for play in plays:
                require_installed(formations[side_name], formation, play)
                candidate = MacroCandidate(
                    macro_id=_digest("BASE", side, formation, play, ()),
                    version=DESIGN_VERSION,
                    side=side,
                    state=MacroState.ELIGIBLE,
                    formation=formation,
                    play=play,
                    adjustment_primitives=(),
                    editor_fields=None,
                    tactical_purpose="Call the installed play with no pre-snap adjustment.",
                    situations=(),
                    conflicts=(),
                    joint=MacroJointFeatures(
                        compatibility=1.0,
                        situation=None,
                        action_count=0,
                        control_reliability=None,
                        empirical_results=None,
                        uncertainty=None,
                    ),
                    is_no_adjustment=True,
                )
                records.append(DesignRecord(
                    candidate=candidate,
                    downside="Calling the play unchanged leaves the researched adjustment unused.",
                    uncertainty_note="No adjustment is added, so there is no macro outcome to verify.",
                    catalog_comparison=(),
                    install_instructions=("No Custom Adjustment. Call the installed play.",),
                ))
    return records


def _offense(formations: Mapping[str, Sequence[str]]) -> list[DesignRecord]:
    primitives = [
        row for row in usable_offense()
        if row.kind in (PrimitiveKind.HOT_ROUTE, PrimitiveKind.PROTECTION, PrimitiveKind.AUDIBLE)
    ]
    composable = [row for row in primitives if row.kind is not PrimitiveKind.AUDIBLE]
    records: list[DesignRecord] = []
    for formation, plays in formations.items():
        play_list = list(plays)
        runs = [play for play in play_list if catalog.is_run(play) and "rpo" not in play.casefold()]
        passes = [play for play in play_list if _is_pass(play)]
        for play in passes:
            require_installed(formations, formation, play)
            for primitive in primitives:
                if primitive.kind is PrimitiveKind.AUDIBLE and not runs:
                    continue
                audible_to = runs[0] if primitive.kind is PrimitiveKind.AUDIBLE else None
                if audible_to:
                    require_installed(formations, formation, audible_to)
                records.append(_offense_record(formation, play, (primitive,), audible_to=audible_to))
            for left, right in combinations(composable, 2):
                reason = offense_conflict([lab_dict(left), lab_dict(right)])
                if reason:
                    continue
                records.append(_offense_record(formation, play, (left, right)))
    return records


def _defense(formations: Mapping[str, Sequence[str]]) -> list[DesignRecord]:
    if not formations:
        return []
    records: list[DesignRecord] = []
    for draft in compose_defensive_macros(formations, max_drafts=12):
        sheet = _sheet_from_rows(draft.get("settings") or [])
        parents = tuple(str(pid) for pid in (draft.get("parent_recipes") or []))
        situations = tuple(str(item) for item in (draft.get("answers") or []))
        purpose = " / ".join(part for part in (_research_purpose(pid) for pid in parents) if part)
        for pair in draft.get("base_pairs") or []:
            formation = str(pair.get("formation") or "")
            play = str(pair.get("play") or "")
            require_installed(formations, formation, play)
            records.append(_defense_record(
                formation, play, sheet,
                situations=situations,
                purpose=purpose or "Researched defensive adjustments combined on an installed play.",
                composed=True,
                parents=parents,
                compatibility_note=str(draft.get("compatibility") or ""),
                source_formation=_research_base(parents[0])[0] if parents else "",
            ))
    records.extend(_dime_fire_sheet(formations))
    return records


def _dime_fire_sheet(formations: Mapping[str, Sequence[str]]) -> list[DesignRecord]:
    """Shell and show-blitz exist only on the researched DIME FIRE 2 recipe."""
    formation, play = "Dime 3-2 Odd", "DB Fire 2"
    if play not in list(formations.get(formation) or []):
        return []
    macro = rdb.defense_macro("DIME FIRE 2")
    if not macro:
        return []
    sheet = _sheet_from_rows(rdb.full_settings("DIME FIRE 2"))
    return [_defense_record(
        formation, play, sheet,
        situations=tuple(str(item) for item in (macro.get("answers") or ())),
        purpose=str(macro.get("purpose") or ""),
        composed=False,
        parents=("DIME FIRE 2",),
        compatibility_note=(
            "Canonical editor cells match DIME FIRE 2. Source rows that are not in "
            "editor_fields.json were dropped."
        ),
        source_formation=str((macro.get("base") or {}).get("formation") or ""),
    )]


def _offense_record(
    formation: str,
    play: str,
    primitives: tuple[RegisteredPrimitive, ...],
    *,
    audible_to: str | None = None,
) -> DesignRecord:
    for primitive in primitives:
        require_primitive(primitive.primitive_id)
    wire = tuple(
        _wire_offense(primitive, audible_to=audible_to if primitive.kind is PrimitiveKind.AUDIBLE else None)
        for primitive in primitives
    )
    situations = tuple(dict.fromkeys(look for primitive in primitives for look in primitive.vs))
    confidences = [primitive.confidence for primitive in primitives]
    composed = len(primitives) > 1
    purpose = " ".join(_offense_why(primitive.primitive_id) for primitive in primitives).strip()
    candidate = MacroCandidate(
        macro_id=_digest("ADJ", MacroSide.OFF, formation, play, [f"{item.name}={item.value}" for item in wire]),
        version=DESIGN_VERSION,
        side=MacroSide.OFF,
        state=MacroState.DRAFT,
        formation=formation,
        play=play,
        adjustment_primitives=wire,
        editor_fields=None,
        tactical_purpose=purpose or None,
        situations=situations,
        conflicts=(),
        joint=_joint(
            compatibility=1.0,
            action_count=len(wire),
            reliability=_reliability(confidences),
            uncertainty=_uncertainty(confidences, composed=composed),
        ),
        is_no_adjustment=False,
    )
    downside = (
        "These primitives were researched separately. The combination has no verified outcome."
        if composed
        else "This is one researched adjustment. It has no verified outcome until you confirm it in the editor."
    )
    return DesignRecord(
        candidate=candidate,
        downside=downside,
        uncertainty_note="Researched primitive only. The custom adjustment has not been editor-verified.",
        catalog_comparison=_compare_offense(situations),
        install_instructions=_offense_install(formation, play, primitives, audible_to),
    )


def _defense_record(
    formation: str,
    play: str,
    sheet: dict[str, EditorFieldAssignment],
    *,
    situations: tuple[str, ...],
    purpose: str,
    composed: bool,
    parents: tuple[str, ...],
    compatibility_note: str,
    source_formation: str,
) -> DesignRecord:
    _assert_complete(sheet)
    wire = _prims_from_sheet(sheet)
    match = _formation_compatibility(source_formation, formation)
    candidate = MacroCandidate(
        macro_id=_digest("ADJ", MacroSide.DEF, formation, play, [f"{item.name}={item.value}" for item in wire]),
        version=DESIGN_VERSION,
        side=MacroSide.DEF,
        state=MacroState.DRAFT,
        formation=formation,
        play=play,
        adjustment_primitives=wire,
        editor_fields=sheet,
        tactical_purpose=purpose or None,
        situations=situations,
        conflicts=(),
        joint=_joint(
            compatibility=match,
            action_count=1,
            reliability=_reliability(["confirmed"]),
            uncertainty=_uncertainty([None], composed=composed),
        ),
        is_no_adjustment=False,
    )
    if composed:
        downside = (
            "The parent recipes were researched separately. Shared field values do not "
            "prove the combination plays the way either source describes."
        )
        note = compatibility_note or "Combined researched settings still need an in-game check."
    else:
        downside = (
            "This sheet uses one researched macro's canonical editor values on an installed play. "
            "It is not a new combination and has no verified outcome."
        )
        note = compatibility_note
    if "No verified outcome" not in note:
        note = f"{note.rstrip('.')}. No verified outcome is on file."
    return DesignRecord(
        candidate=candidate,
        downside=downside,
        uncertainty_note=note.strip(),
        catalog_comparison=_compare_defense(sheet, situations, parents, composed),
        install_instructions=_defense_install(candidate.macro_id, formation, play),
    )


def _sheet_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, EditorFieldAssignment]:
    researched: dict[str, EditorFieldAssignment] = {}
    for row in rows:
        if row.get("new_field"):
            continue
        canon = canonical_field(str(row.get("section") or ""), str(row.get("setting") or ""))
        if canon is None:
            continue
        _section, setting = canon
        value = str(row.get("value") or "").strip()
        source = str(row.get("source") or "")
        if not value or value == "Default" or source == "default":
            continue
        require_primitive(value_id(canon[0], canon[1], value))
        previous = researched.get(setting)
        if previous is not None and previous.value != value:
            reason = defense_value_conflict([{canon: previous.value or ""}, {canon: value}])
            raise ValueError(reason or "conflicting editor values")
        researched[setting] = EditorFieldAssignment(
            value=value,
            source=EditorFieldSource.RESEARCHED,
            provenance=source,
            confidence=None,
        )
    sheet = {
        name: researched.get(name) or EditorFieldAssignment(
            value="Default",
            source=EditorFieldSource.DEFAULT,
            provenance=None,
            confidence=None,
        )
        for name in defensive_editor_field_names()
    }
    _assert_complete(sheet)
    return sheet


def _assert_complete(sheet: Mapping[str, EditorFieldAssignment]) -> None:
    expected = defensive_editor_field_names()
    missing = [name for name in expected if name not in sheet]
    extra = [name for name in sheet if name not in set(expected)]
    if missing or extra or len(sheet) != len(expected):
        raise ValueError(f"defense editor sheet is missing {missing[:3]} extra {extra[:3]}")
    for key, entry in sheet.items():
        if entry.value in (None, ""):
            raise ValueError(f"{key} has no value")
        if entry.source is EditorFieldSource.DEFAULT and entry.value != "Default":
            raise ValueError(f"{key} is default-sourced but the value is {entry.value!r}")
        if entry.source is EditorFieldSource.RESEARCHED and entry.value == "Default":
            raise ValueError(f"{key} is marked researched with value Default")
        if entry.source is EditorFieldSource.UNKNOWN:
            raise ValueError(f"{key} has an unknown source")


def _prims_from_sheet(sheet: Mapping[str, EditorFieldAssignment]) -> tuple[AdjustmentPrimitive, ...]:
    prims: list[AdjustmentPrimitive] = []
    for name, entry in sheet.items():
        if entry.source is not EditorFieldSource.RESEARCHED or not entry.value:
            continue
        prims.append(AdjustmentPrimitive(
            name=name,
            value=entry.value,
            source=EditorFieldSource.RESEARCHED,
        ))
    return tuple(sorted(prims, key=lambda item: (item.name or "", item.value or "")))


def _wire_offense(primitive: RegisteredPrimitive, *, audible_to: str | None) -> AdjustmentPrimitive:
    value = audible_to if audible_to else (primitive.route or primitive.value)
    return AdjustmentPrimitive(
        name=primitive.primitive_id,
        value=value,
        source=EditorFieldSource.RESEARCHED,
    )


def _joint(
    *,
    compatibility: float | None,
    action_count: int,
    reliability: float | None,
    uncertainty: float | None,
) -> MacroJointFeatures:
    """Ranking inputs. ``situation`` stays unknown: its scale is not defined yet."""
    return MacroJointFeatures(
        compatibility=compatibility,
        situation=None,
        action_count=action_count,
        control_reliability=reliability,
        empirical_results=None,
        uncertainty=uncertainty,
    )


def _reliability(confidences: Sequence[str | None]) -> float | None:
    known = [_REL[item] for item in confidences if item in _REL]
    if not known:
        return None
    return min(known)


def _uncertainty(confidences: Sequence[str | None], *, composed: bool) -> float:
    score = 0.35
    for item in confidences:
        if item == "conflict":
            score = max(score, 1.0)
        elif item == "single-source":
            score = max(score, 0.55)
        elif item in (None, ""):
            score = max(score, 0.5)
        elif item == "confirmed":
            score = max(score, 0.35)
    if composed:
        score = min(1.0, score + 0.2)
    return round(score, 2)


def _formation_compatibility(source_formation: str, installed: str) -> float:
    src = (source_formation or "").casefold()
    inst = installed.casefold()
    if not src or "any" in src:
        return 0.5
    if src == inst or src in inst or inst in src:
        return 1.0
    return 0.5


def _digest(
    prefix: str,
    side: MacroSide,
    formation: str,
    play: str,
    primitive_ids: Sequence[str],
) -> str:
    tag = "OFF" if side is MacroSide.OFF else "DEF"
    raw = "|".join([prefix, tag, formation, play, ",".join(sorted(primitive_ids))])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}-{tag}-{digest}"


def _is_pass(play: str) -> bool:
    return not catalog.is_run(play) and "rpo" not in play.casefold()


def _offense_why(primitive_id: str) -> str:
    for row in rdb.offense_adjustments():
        if row.get("id") == primitive_id:
            return str(row.get("why") or "")
    return ""


def _research_purpose(macro_id: str) -> str:
    return str((rdb.defense_macro(macro_id) or {}).get("purpose") or "")


def _research_base(macro_id: str) -> tuple[str, str]:
    base = (rdb.defense_macro(macro_id) or {}).get("base") or {}
    return str(base.get("formation") or ""), str(base.get("play") or "")


def _offense_install(
    formation: str,
    play: str,
    primitives: Sequence[RegisteredPrimitive],
    audible_to: str | None,
) -> tuple[str, ...]:
    from cfb_coach.madden.offense_macros import EDITOR_PATH

    steps = []
    for primitive in primitives:
        target = audible_to if primitive.kind is PrimitiveKind.AUDIBLE else None
        steps.append(_offense_buttons(primitive, target))
    path = " → ".join(EDITOR_PATH)
    return (
        f"Open {path}.",
        "Enter only the researched rows. Leave every other editor row at Default.",
        *steps,
        f"The installed play is {play} ({formation}).",
        "This program did not create the macro on Xbox. Leave it a draft until you confirm the editor rows.",
    )


def _offense_buttons(primitive: RegisteredPrimitive, audible_to: str | None) -> str:
    if primitive.kind is PrimitiveKind.HOT_ROUTE:
        return (
            rdb.buttons("offense", "hot_route")
            .replace("the receiver's icon button", f"{primitive.target}'s icon button")
            .replace("pick the route", f"pick {primitive.route}")
        )
    if primitive.kind is PrimitiveKind.PROTECTION:
        return rdb.buttons("offense", "pass_protection").replace(
            "pick the protection", f"pick {primitive.route}"
        )
    if primitive.kind is PrimitiveKind.AUDIBLE:
        return rdb.buttons("offense", "audible").replace("that audible", audible_to or "the run audible")
    raise ValueError(f"{primitive.primitive_id} has no verified offense control")


def _defense_install(macro_id: str, formation: str, play: str) -> tuple[str, ...]:
    from cfb_coach.madden.macros import EDITOR_PATH, LOADOUT_N

    fire = rdb.buttons("defense", "custom_adjustments").replace("the macro", macro_id)
    return (
        f"Open {' → '.join(EDITOR_PATH)}.",
        "Set every editor row. Use each researched value in this sheet. Set every other row to Default.",
        f"Test it on the installed play {play} ({formation}).",
        f"Arm it in one of the {LOADOUT_N} defense slots only after the rows match the screen.",
        f"At the line, after you have armed it: {fire}",
        "This program did not create the macro on Xbox. A draft stays out of the live call until you confirm editor verification, install, and arming.",
    )


def _compare_offense(situations: Sequence[str]) -> tuple[CatalogComparison, ...]:
    from cfb_coach.madden.data import load_macro_catalog

    catalog_rows = (load_macro_catalog().get("macros") or {})
    wanted: list[tuple[str, str]] = []
    if "man" in situations:
        wanted.append(("O-MAN", "nearest man-beater in the catalog; its rows were not copied"))
    if "pressure" in situations:
        wanted.append(("O-PROT", "nearest pressure protection in the catalog; its rows were not copied"))
    if "two_high" in situations:
        wanted.append(("O-RPO", "catalog O-RPO is an RPO tag, not this audible"))
    notes: list[CatalogComparison] = []
    for cid, relation in wanted:
        row = catalog_rows.get(cid) or {}
        if row.get("side") != "offense":
            continue
        notes.append(CatalogComparison(cid, "macro_catalog", relation))
    return tuple(notes)


def _compare_defense(
    sheet: Mapping[str, EditorFieldAssignment],
    situations: Sequence[str],
    parents: Sequence[str],
    composed: bool,
) -> tuple[CatalogComparison, ...]:
    from cfb_coach.madden.data import load_macro_catalog

    researched = {
        key: entry.value
        for key, entry in sheet.items()
        if entry.source is EditorFieldSource.RESEARCHED and entry.value
    }
    best: tuple[int, str, tuple[str, ...], tuple[str, ...]] | None = None
    for macro in rdb.defense_macros():
        theirs: dict[str, str] = {}
        for row in macro.get("settings") or []:
            canon = canonical_field(str(row.get("section") or ""), str(row.get("setting") or ""))
            value = str(row.get("value") or "").strip()
            if canon and value and value != "Default":
                theirs[canon[1]] = value
        shared = tuple(sorted(
            f"{setting}={value}"
            for setting, value in researched.items()
            if theirs.get(setting) == value
        ))
        distinct = tuple(sorted(
            f"{setting}={value}"
            for setting, value in researched.items()
            if theirs.get(setting) != value
        ))
        if best is None or len(shared) > best[0]:
            best = (len(shared), str(macro.get("id") or ""), shared, distinct)
    notes: list[CatalogComparison] = []
    if best is not None:
        _, mid, shared, distinct = best
        if composed:
            relation = (
                f"combined {', '.join(parents)}; closest researched macro is {mid}; "
                "the sheet is not that macro"
            )
        elif not distinct and parents:
            relation = f"sourced specialization of {parents[0]}; canonical editor values match {mid}"
        else:
            relation = f"closest researched macro is {mid}; the sheet is not that macro"
        notes.append(CatalogComparison(mid, "research_db", relation, shared, distinct))
    catalog_rows = (load_macro_catalog().get("macros") or {})
    for cid, row in catalog_rows.items():
        if row.get("side") != "defense":
            continue
        family = str(row.get("concept_family") or "")
        if family and family in situations:
            notes.append(CatalogComparison(
                cid,
                "macro_catalog",
                "same concept family; catalog rows are approx and were not copied into this sheet",
            ))
    return tuple(notes)


__all__ = [
    "DEFENSE_FORMATIONS",
    "DESIGN_VERSION",
    "DRAFT_META_KEY",
    "OFFENSE_FORMATIONS",
    "XBOX_DISCLAIMER",
    "CatalogComparison",
    "DesignRecord",
    "NotLiveSelectable",
    "backup_database",
    "defense_value_conflict",
    "generate",
    "generate_from_db",
    "install_trimmed_book",
    "installed_formations",
    "offense_conflict",
    "rejected_offense_pairs",
    "require_installed",
    "require_live",
    "select_live",
    "stage_drafts",
    "transition",
    "trimmed_formations",
]
