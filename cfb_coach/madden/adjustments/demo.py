"""Demo C: original adjustment drafts, a refusal, and a fixture-armed selection.

The commands use a temporary fixture database. They do not open
``~/.cfb-coach/madden27.db`` and they do not create anything on Xbox.

Commands (from the repo root):

    PYTHONPATH=. python3 -m cfb_coach.madden.adjustments.demo generate
    PYTHONPATH=. python3 -m cfb_coach.madden.adjustments.demo refuse
    PYTHONPATH=. python3 -m cfb_coach.madden.adjustments.demo select
    PYTHONPATH=. python3 -m cfb_coach.madden.adjustments.demo all

``all`` runs the three steps, copies the fixture DB before the one meta write,
and checks that snap history and the active macro loadout are unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from cfb_coach.db import CoachDB
from cfb_coach.madden import data as mdata
from cfb_coach.madden.adjustments.candidates import WIRE_VERSION, MacroCandidate, is_live_selectable
from cfb_coach.madden.adjustments.primitives import (
    fields_without_researched_value,
    rejected_non_editor_settings,
    unverified,
)
from cfb_coach.madden.adjustments.synthesis import (
    DESIGN_VERSION,
    DesignRecord,
    MacroState,
    NotLiveSelectable,
    XBOX_DISCLAIMER,
    generate_from_db,
    install_trimmed_book,
    installed_formations,
    offense_pair_verdicts,
    rejected_offense_pairs,
    sanctioned_offense_pairs,
    require_live,
    select_live,
    stage_drafts,
    transition,
)

_EVIDENCE = (
    "FIXTURE: user confirmed every editor row matches this sheet.",
    "FIXTURE: user confirmed the adjustment is saved in Madden.",
    "FIXTURE: user confirmed the adjustment is armed in a loadout slot.",
)


def _fixture() -> tuple[tempfile.TemporaryDirectory[str], CoachDB, str | None]:
    """Build a temp book. Research cache follows ``CFB_COACH_DB``, not ``HOME``.

    ``research_db.load`` reads ``data_dir()``, which is the parent of
    ``CFB_COACH_DB`` or else ``~/.cfb-coach`` (and that fallback creates the
    directory). Pointing the env var at this fixture keeps the demo off
    Aidan's real research cache.
    """
    folder = tempfile.TemporaryDirectory(prefix="madden-demo-c-")
    previous_db = os.environ.get("CFB_COACH_DB")
    os.environ["CFB_COACH_DB"] = str(Path(folder.name) / "coach.db")
    path = Path(folder.name) / "madden27.db"
    db = CoachDB(path, seed=mdata.load_seed())
    install_trimmed_book(db)
    db.log_snap(
        opponent_id="gavin",
        side="defense",
        situation_raw="d 3&6",
        our_call="fixture",
        formation="Nickel Over",
        play="Cover 4 Quarters",
        result="stop",
        macro="TAMPA MABLE",
    )
    from cfb_coach.madden.macros import store_selection

    store_selection(db, "gavin", {"offense": ["MAN", "C2"], "defense": ["TAMPA MABLE", "SKY CONTAIN"]})
    db.conn.commit()
    return folder, db, previous_db


def _snapshot(db: CoachDB) -> dict[str, Any]:
    snaps = [
        tuple(row)
        for row in db.conn.execute(
            "SELECT opponent_id, side, formation, play, macro, result FROM snaps ORDER BY id"
        )
    ]
    return {
        "snaps": snaps,
        "loadout": db.get_meta("active_macros:gavin"),
    }


def _samples(records: list[DesignRecord]) -> tuple[DesignRecord, DesignRecord]:
    offense = next(
        row for row in records
        if row.candidate.side.value == "off" and not row.candidate.is_no_adjustment
    )
    defense = next(
        row for row in records
        if row.candidate.side.value == "def"
        and not row.candidate.is_no_adjustment
        and "combined" in " ".join(note.relation for note in row.catalog_comparison)
    )
    return offense, defense


def _print_record(row: DesignRecord) -> None:
    cand = row.candidate
    print(f"  {cand.macro_id} version {cand.version} state {cand.state.value}")
    print(f"  design_record_id: {row.design_record_id}")
    print(f"  play: {cand.play} ({cand.formation})")
    print(f"  purpose: {cand.tactical_purpose}")
    print(f"  situations: {', '.join(cand.situations) or '(none)'}")
    print(f"  uncertainty: {row.uncertainty_note}")
    print(f"  downside: {row.downside}")
    if row.catalog_comparison:
        note = row.catalog_comparison[0]
        print(f"  catalog comparison: {note.catalog_id} — {note.relation}")
    if cand.editor_fields:
        researched = sum(1 for entry in cand.editor_fields.values() if entry.source.value == "researched")
        print(f"  editor fields: {len(cand.editor_fields)} ({researched} researched, the rest Default)")
    print("  install:")
    for step in row.install_instructions:
        print(f"    {step}")
    print(f"  xbox_created: {row.xbox_created}")


def cmd_generate(records: list[DesignRecord]) -> None:
    offense, defense = _samples(records)
    off_n = sum(1 for row in records if row.candidate.side.value == "off" and not row.candidate.is_no_adjustment)
    def_n = sum(1 for row in records if row.candidate.side.value == "def" and not row.candidate.is_no_adjustment)
    base_n = sum(1 for row in records if row.candidate.is_no_adjustment)
    print(f"Demo C generate  wire {WIRE_VERSION}  design version {DESIGN_VERSION}")
    print(f"drafts: offense {off_n}, defense {def_n}, no-adjustment baselines {base_n}")
    print("Offense draft:")
    _print_record(offense)
    print("Defense draft:")
    _print_record(defense)
    print("Offense pairs Meta Researcher marked legal:")
    for verdict in (*offense_pair_verdicts(), *sanctioned_offense_pairs()):
        if verdict.emit:
            tags = ", ".join(verdict.situational_tags) or "(none)"
            print(f"  {verdict.left_id} + {verdict.right_id}: {verdict.label} {verdict.confidence}; tag {tags}")
    print("Rejected offense combinations:")
    for left, right, reason in rejected_offense_pairs():
        print(f"  {left} + {right}: {reason}")
    print("Primitives not used:")
    for row in unverified():
        print(f"  {row.primitive_id}: {row.block_reason}")
    print("Editor fields with no researched value (left Default):")
    for section, setting in fields_without_researched_value():
        print(f"  {section} / {setting}")
    print("Source rows rejected because they are not editor fields:")
    for row in rejected_non_editor_settings():
        print(f"  {row['macro_id']}: {row['section']} / {row['setting']} = {row['value']}")


def cmd_refuse(records: list[DesignRecord]) -> None:
    draft = next(
        row for row in records
        if not row.candidate.is_no_adjustment and row.candidate.state is MacroState.DRAFT
    )
    print(f"Demo C refuse  {draft.candidate.macro_id} state {draft.candidate.state.value}")
    try:
        require_live(draft.candidate)
    except NotLiveSelectable as exc:
        print(f"refused: {exc}")
        return
    raise SystemExit("unverified draft was selectable")


def cmd_select(records: list[DesignRecord], installed: dict[str, dict[str, list[str]]]) -> MacroCandidate:
    draft = next(
        row for row in records
        if row.candidate.side.value == "def"
        and row.candidate.formation == "Dime 3-2 Odd"
        and row.candidate.play == "DB Fire 2"
        and not row.candidate.is_no_adjustment
    )
    armed = draft
    for state, evidence in (
        (MacroState.EDITOR_VERIFIED, _EVIDENCE[0]),
        (MacroState.INSTALLED, _EVIDENCE[1]),
        (MacroState.ARMED, _EVIDENCE[2]),
    ):
        armed = transition(armed, state, evidence)
    if not is_live_selectable(armed.candidate.state):
        raise SystemExit("armed macro is not live-selectable")
    if armed.xbox_created or any("created the macro on Xbox" in note for note in armed.confirmation_evidence):
        raise SystemExit("selection claimed an Xbox creation")
    updated = [
        armed if row.candidate.macro_id == armed.candidate.macro_id else row
        for row in records
    ]
    chosen = select_live(
        updated, formation="Dime 3-2 Odd", play="DB Fire 2", installed=installed,
    )
    print(
        f"Demo C select  {chosen.macro_id} version {chosen.version} "
        f"state {chosen.state.value}  {chosen.play} ({chosen.formation})"
    )
    print(f"live_selectable: {is_live_selectable(chosen.state)}")
    print(f"evidence steps: {len(armed.confirmation_evidence)}")
    print(XBOX_DISCLAIMER)
    other = select_live(
        records, formation="Pistol Trips", play="HB Dive", installed=installed,
    )
    print(
        f"unarmed play falls back to baseline {other.macro_id} "
        f"is_no_adjustment={other.is_no_adjustment}"
    )
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demo C custom-adjustment designer")
    parser.add_argument("command", choices=("generate", "refuse", "select", "all"))
    args = parser.parse_args(argv)
    folder, db, previous_db = _fixture()
    try:
        before = _snapshot(db)
        records = generate_from_db(db)
        installed = installed_formations(db)
        if args.command in ("generate", "all"):
            cmd_generate(records)
        if args.command in ("refuse", "all"):
            cmd_refuse(records)
        if args.command in ("select", "all"):
            cmd_select(records, installed)
        if args.command == "all":
            staged = stage_drafts(db, backup=True)
            after = _snapshot(db)
            backups = list(Path(db.path).parent.glob(Path(db.path).name + ".bak-*"))
            print(
                f"staged {staged['stored']} drafts on {staged['meta_key']}; "
                f"backup {backups[-1].name}; history_preserved {before == after}"
            )
            if before != after or not backups:
                return 1
        return 0
    finally:
        db.close()
        folder.cleanup()
        if previous_db is None:
            os.environ.pop("CFB_COACH_DB", None)
        else:
            os.environ["CFB_COACH_DB"] = previous_db


if __name__ == "__main__":
    sys.exit(main())
