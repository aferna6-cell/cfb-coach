"""Original Custom Adjustment drafts for the installed Madden book.

Offense drafts are researched hot routes, protections, and audibles on plays
in the applied book. Two-action drafts are emitted only for pairs Meta
Researcher marked legal, plus HB Chip and Release with max protect as the
sanctioned alternative to the conditional HB-flat pair. Those drafts stay
DRAFT. Motion stays off: no Xbox control is verified yet.

Defense drafts fill every defense editor field with a researched value or the
literal ``Default``. Two-recipe sheets come from ``defense_macro_lab``. A
Dime Fire 2 sheet is added so shell, show-blitz, and the other sourced cells
on that recipe are present; rows that are not in ``editor_fields.json`` are
dropped. Linebacker shift and assignment have no researched value, so they
stay ``Default``.

Every generated adjustment starts at DRAFT. Moving it requires user-confirmation
evidence at each step. Nothing here creates a macro on Xbox. The no-adjustment
baseline is the unmodified installed play and is always in the set. Its live
legality is ``is_baseline_live_legal``: ``is_no_adjustment`` on a legal installed
play, with no read of lifecycle state and no call to ``is_live_selectable``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
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
    CHIP_AND_RELEASE_ID,
    CHIP_AND_RELEASE_WHY,
    PrimitiveKind,
    RegisteredPrimitive,
    canonical_field,
    require_primitive,
    usable_offense,
    value_id,
)
from cfb_coach.madden.macros import LOADOUT_N
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


class ArmCapExceeded(ValueError):
    """A side already has LOADOUT_N armed or active custom macros."""


_ACTIVE_STATES = (MacroState.ARMED, MacroState.ELIGIBLE, MacroState.EVALUATED)


@dataclass(frozen=True)
class CatalogComparison:
    catalog_id: str
    source: str
    relation: str
    shared_settings: tuple[str, ...] = ()
    distinct_settings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DesignRecord:
    """Designer-owned record. ``candidate`` is the wire object.

    ``design_record_id`` is the contract field ``DSN-<macro_id>`` on
    ``MacroCandidate``. Downside, catalog comparison, and install steps stay
    here.
    """

    design_record_id: str
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
        if self.design_record_id != self.candidate.design_record_id:
            raise ValueError("design_record_id must match DSN-<macro_id> on the candidate")


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


@dataclass(frozen=True)
class OffensePairVerdict:
    """One evaluated pair of researched offense primitives."""

    left_id: str
    right_id: str
    emit: bool
    conflict: str | None
    label: str
    confidence: str
    sources: tuple[str, ...]
    situational_tags: tuple[str, ...] = ()
    note: str = ""


_EA_GAMEPLAY = "https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-gameplay"
_EA_PREPLAY = "https://www.ea.com/games/madden-nfl/madden-nfl-27/controls-hub/m27-xb-preplay-offense"
_AUDIBLE_CONFLICT = "audible replaces the play"


def offense_pair_verdicts() -> tuple[OffensePairVerdict, ...]:
    """The six researched offense pairs, in evaluation order.

    Labels come from the 2026-10-10 Meta Researcher verification. A pair with
    ``emit`` false is not generated.
    """
    return (
        OffensePairVerdict(
            "hot_slant_vs_man",
            "hot_hb_flat_vs_pressure",
            emit=True,
            conflict=None,
            label="VERIFIED-M27",
            confidence="high",
            sources=(_EA_GAMEPLAY,),
            situational_tags=("man_blitz",),
            note=(
                "Hot routes on different players stack. Same defensive look is a "
                "tactics note, not a game rule. Both fit a man blitz."
            ),
        ),
        OffensePairVerdict(
            "hot_slant_vs_man",
            "audible_run_vs_two_high",
            emit=False,
            conflict=_AUDIBLE_CONFLICT,
            label="HYPOTHESIS",
            confidence="medium",
            sources=(_EA_GAMEPLAY,),
            note="Audible replaces the play. No M27 source documents that an audible clears the slant.",
        ),
        OffensePairVerdict(
            "hot_slant_vs_man",
            "max_protect_vs_pressure",
            emit=True,
            conflict=None,
            label="VERIFIED-M27",
            confidence="high",
            sources=(_EA_PREPLAY, _EA_GAMEPLAY),
            note=(
                "Protection (LT) and hot routes (Y) are separate menus. "
                "EA says both can be saved in one Custom Adjustment."
            ),
        ),
        OffensePairVerdict(
            "hot_hb_flat_vs_pressure",
            "audible_run_vs_two_high",
            emit=False,
            conflict=_AUDIBLE_CONFLICT,
            label="HYPOTHESIS",
            confidence="medium",
            sources=(_EA_GAMEPLAY,),
            note="Audible replaces the play. The HB would be the ball carrier on the run check.",
        ),
        OffensePairVerdict(
            "hot_hb_flat_vs_pressure",
            "max_protect_vs_pressure",
            emit=False,
            conflict="conditional: unknown precedence; max protect and an HB flat are not emitted together",
            label="CONDITIONAL",
            confidence="low to medium",
            sources=(
                "https://fenixbazaar.com/2025/08/10/madden-26-leverage-system/",
                "https://madden.tools/playbooks/book/madden-franchise-offense-maxprotect",
                "https://www.818madden.com/tools/position-adjustments",
                _EA_GAMEPLAY,
            ),
            note=(
                "CARRIED-FORWARD/HYPOTHESIS. Which assignment the HB obeys is not sourced. "
                "The sanctioned alternative is HB Chip and Release (VERIFIED-M27)."
            ),
        ),
        OffensePairVerdict(
            "audible_run_vs_two_high",
            "max_protect_vs_pressure",
            emit=False,
            conflict=_AUDIBLE_CONFLICT,
            label="HYPOTHESIS",
            confidence="medium",
            sources=(_EA_GAMEPLAY,),
            note=(
                "Audible replaces the play. Max protect is pass-only. "
                "ID Mike is the run-side equivalent (VERIFIED-M27)."
            ),
        ),
    )


def sanctioned_offense_pairs() -> tuple[OffensePairVerdict, ...]:
    """HB Chip and Release with max protect, instead of the conditional flat pair."""
    return (
        OffensePairVerdict(
            "max_protect_vs_pressure",
            CHIP_AND_RELEASE_ID,
            emit=True,
            conflict=None,
            label="VERIFIED-M27",
            confidence="high",
            sources=(
                _EA_GAMEPLAY,
                "https://www.818madden.com/tools/position-adjustments",
            ),
            note=(
                "Sanctioned alternative to HB flat plus max protect. "
                "The primitive is the HB hot-route menu item Chip and Release. "
                "No other chip details are claimed."
            ),
        ),
    )


def _pair_index() -> dict[frozenset[str], OffensePairVerdict]:
    rows = (*offense_pair_verdicts(), *sanctioned_offense_pairs())
    return {frozenset((row.left_id, row.right_id)): row for row in rows}


def offense_conflict(parts: Sequence[Mapping[str, Any]]) -> str | None:
    """Reason when a combo must not be emitted. None means it may be drafted.

    Researched pairs use the Meta Researcher verdicts. Other shapes still use
    ``offense_macro_lab.conflicts``.
    """
    ids = [str(part.get("id") or "") for part in parts]
    if len(ids) == 2 and all(ids):
        verdict = _pair_index().get(frozenset(ids))
        if verdict is not None:
            return verdict.conflict
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
    """Researched pairs that are not emitted, with the verdict's conflict text."""
    rejected: list[tuple[str, str, str]] = []
    for verdict in offense_pair_verdicts():
        if verdict.conflict:
            rejected.append((verdict.left_id, verdict.right_id, verdict.conflict))
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


def active_custom_macros(
    records: Sequence[DesignRecord],
    side: MacroSide,
) -> list[DesignRecord]:
    """Armed, eligible, or evaluated custom macros on one side. Baselines are omitted."""
    rows = [
        row for row in records
        if row.candidate.side is side
        and row.candidate.is_no_adjustment is False
        and row.candidate.state in _ACTIVE_STATES
    ]
    rows.sort(key=lambda row: row.candidate.macro_id or "")
    return rows


def require_arm_cap(records: Sequence[DesignRecord], side: MacroSide) -> None:
    """EA's screen allows 10. This coach arms at most LOADOUT_N per side."""
    active = active_custom_macros(records, side)
    if len(active) > LOADOUT_N:
        raise ArmCapExceeded(
            f"{side.value} has {len(active)} armed or active macros; the cap is {LOADOUT_N}"
        )


def transition(
    record: DesignRecord,
    to_state: MacroState,
    evidence: str,
    *,
    roster: Sequence[DesignRecord] = (),
) -> DesignRecord:
    """Move one step. Evidence is required for every state past DRAFT.

    The stored note says the user attested. It is not an Xbox creation claim.
    New drafts are not moved by ``generate``. Entering ARMED, ELIGIBLE, or
    EVALUATED fails when that side already has ``LOADOUT_N`` active macros.
    Drafts are not capped.
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
    if to_state in _ACTIVE_STATES and current not in _ACTIVE_STATES:
        others = [
            row for row in roster
            if row.candidate.macro_id != record.candidate.macro_id
        ]
        if len(active_custom_macros(others, record.candidate.side)) >= LOADOUT_N:
            raise ArmCapExceeded(
                f"{record.candidate.side.value} already has {LOADOUT_N} armed or active macros"
            )
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


def is_baseline_live_legal(
    candidate: MacroCandidate,
    installed: Mapping[str, Mapping[str, Sequence[str]]],
) -> bool:
    """True when this is the unmodified play in the installed book.

    ``is_no_adjustment=True`` on a legal installed play is always live-legal.
    This function does not call ``is_live_selectable`` and does not read
    ``candidate.state``. DRAFT, RETIRED, UNKNOWN, and ELIGIBLE are the same
    result. The baseline must not go through the custom-adjustment lifecycle.
    """
    if candidate.is_no_adjustment is not True:
        return False
    if candidate.conflicts or candidate.editor_fields or candidate.adjustment_primitives:
        return False
    if candidate.side is MacroSide.OFF:
        book_side = "offense"
    elif candidate.side is MacroSide.DEF:
        book_side = "defense"
    else:
        return False
    formation = candidate.formation
    play = candidate.play
    if not formation or not play:
        return False
    plays = (installed.get(book_side) or {}).get(formation)
    return plays is not None and play in list(plays)


def require_live(
    candidate: MacroCandidate,
    *,
    installed: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> None:
    """Refuse a custom adjustment that is not armed-or-later.

    A no-adjustment baseline does not use that rule. It is live-legal only
    through :func:`is_baseline_live_legal`.
    """
    if candidate.is_no_adjustment:
        if installed is not None and is_baseline_live_legal(candidate, installed):
            return
        if installed is None:
            raise NotLiveSelectable(
                f"{candidate.macro_id} is a no-adjustment baseline; "
                "live legality needs the installed book and does not use lifecycle state"
            )
        raise NotLiveSelectable(
            f"{candidate.macro_id} is not a legal no-adjustment play in the installed book"
        )
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
    installed: Mapping[str, Mapping[str, Sequence[str]]],
) -> MacroCandidate:
    """Return an armed-or-later macro for this play, or the no-adjustment baseline.

    Drafts, editor-verified sheets, and installed-but-unarmed sheets are refused.
    The baseline is accepted by :func:`is_baseline_live_legal` for any lifecycle
    state, including a state other than ELIGIBLE. This is a legality gate.
    The Madden ML Lead owns ranking. A side with more than ``LOADOUT_N``
    armed or active custom macros is refused.
    """
    for side in (MacroSide.OFF, MacroSide.DEF):
        require_arm_cap(records, side)
    pool = [
        row.candidate
        for row in records
        if row.candidate.formation == formation and row.candidate.play == play
    ]
    if not pool:
        raise ValueError(f"{formation!r} / {play!r} is not in this candidate set")
    armed = [
        row for row in pool
        if row.is_no_adjustment is False and not row.conflicts and is_live_selectable(row.state)
    ]
    if armed:
        order = {MacroState.EVALUATED: 0, MacroState.ELIGIBLE: 1, MacroState.ARMED: 2}
        armed.sort(key=lambda row: (
            order.get(row.state, 9),
            row.joint.uncertainty if row.joint.uncertainty is not None else 1.0,
            row.macro_id,
        ))
        return armed[0]
    baselines = [row for row in pool if is_baseline_live_legal(row, installed)]
    if baselines:
        return baselines[0]
    raise NotLiveSelectable(f"no live-selectable candidate for {formation} / {play}")


def _baselines(formations: Mapping[str, Mapping[str, Sequence[str]]]) -> list[DesignRecord]:
    records: list[DesignRecord] = []
    for side_name, side in (("offense", MacroSide.OFF), ("defense", MacroSide.DEF)):
        for formation, plays in (formations.get(side_name) or {}).items():
            for play in plays:
                require_installed(formations[side_name], formation, play)
                macro_id = _digest("BASE", side, formation, play, ())
                candidate = MacroCandidate(
                    macro_id=macro_id,
                    design_record_id=design_record_id_for(macro_id),
                    version=DESIGN_VERSION,
                    side=side,
                    # Stored label only. is_baseline_live_legal does not read it.
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
                    design_record_id=design_record_id_for(macro_id),
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
    by_id = {row.primitive_id: row for row in primitives}
    legal_pairs = [
        verdict for verdict in (*offense_pair_verdicts(), *sanctioned_offense_pairs())
        if verdict.emit
    ]
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
            for verdict in legal_pairs:
                left = by_id.get(verdict.left_id)
                right = by_id.get(verdict.right_id)
                if left is None or right is None:
                    continue
                records.append(_offense_record(
                    formation,
                    play,
                    (left, right),
                    extra_situations=verdict.situational_tags,
                ))
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
    extra_situations: tuple[str, ...] = (),
) -> DesignRecord:
    for primitive in primitives:
        require_primitive(primitive.primitive_id)
    wire = tuple(
        _wire_offense(primitive, audible_to=audible_to if primitive.kind is PrimitiveKind.AUDIBLE else None)
        for primitive in primitives
    )
    situations = tuple(dict.fromkeys(
        (* (look for primitive in primitives for look in primitive.vs), *extra_situations)
    ))
    confidences = [primitive.confidence for primitive in primitives]
    composed = len(primitives) > 1
    purpose = " ".join(_offense_why(primitive.primitive_id) for primitive in primitives).strip()
    macro_id = _digest("ADJ", MacroSide.OFF, formation, play, [f"{item.name}={item.value}" for item in wire])
    candidate = MacroCandidate(
        macro_id=macro_id,
        design_record_id=design_record_id_for(macro_id),
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
        design_record_id=candidate.design_record_id or "",
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
    macro_id = _digest("ADJ", MacroSide.DEF, formation, play, [f"{item.name}={item.value}" for item in wire])
    candidate = MacroCandidate(
        macro_id=macro_id,
        design_record_id=design_record_id_for(macro_id),
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
        design_record_id=candidate.design_record_id or "",
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


def design_record_id_for(macro_id: str) -> str:
    """Stable link from a wire object to its DesignRecord. Not a contract field."""
    if not macro_id:
        raise ValueError("a design record needs a macro id to link")
    return f"DSN-{macro_id}"


def _joint(
    *,
    compatibility: float | None,
    action_count: int,
    reliability: float | None,
    uncertainty: float | None,
) -> MacroJointFeatures:
    """Ranking inputs.

    ``situation`` stays null. The Lead will define it later as a situation
    fit on [0, 1] and version that change. This designer does not invent the scale.
    """
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
    if primitive_id == CHIP_AND_RELEASE_ID:
        return CHIP_AND_RELEASE_WHY
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
    "ArmCapExceeded",
    "DesignRecord",
    "NotLiveSelectable",
    "OffensePairVerdict",
    "active_custom_macros",
    "design_record_id_for",
    "is_baseline_live_legal",
    "offense_pair_verdicts",
    "require_arm_cap",
    "sanctioned_offense_pairs",
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
