"""Role 3: original Custom Adjustment drafts, editor completeness, and live gates."""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import fields, replace
from pathlib import Path

from cfb_coach.madden import research_db as rdb
from cfb_coach.madden.adjustments import defense_adjustment, offense_adjustment
from cfb_coach.madden.adjustments.candidates import (
    WIRE_VERSION,
    MacroCandidate,
    MacroSide,
    MacroState,
    candidate_from_json,
    candidate_to_json,
    is_live_selectable,
)
from cfb_coach.madden.adjustments.primitives import (
    CHIP_AND_RELEASE_ID,
    CHIP_AND_RELEASE_SOURCES,
    require_primitive,
    unverified,
)
from cfb_coach.madden.macros import LOADOUT_N, load_selection, store_selection
from cfb_coach.madden.model.schema import defensive_editor_field_names
from cfb_coach.madden.adjustments.synthesis import (
    ArmCapExceeded,
    DesignRecord,
    NotLiveSelectable,
    backup_database,
    is_baseline_live_legal,
    defense_value_conflict,
    generate,
    generate_from_db,
    install_trimmed_book,
    offense_conflict,
    offense_pair_verdicts,
    rejected_offense_pairs,
    require_installed,
    require_live,
    select_live,
    stage_drafts,
    transition,
    trimmed_formations,
)
from tests.test_madden27 import _Isolated


def _book():
    return trimmed_formations()


class TestDesigner(_Isolated):
    @classmethod
    def setUpClass(cls) -> None:
        cls.formations = _book()
        cls.records = generate(cls.formations)

    def test_original_candidates_on_both_sides(self) -> None:
        offense = [
            row for row in self.records
            if row.candidate.side.value == "off" and not row.candidate.is_no_adjustment
        ]
        defense = [
            row for row in self.records
            if row.candidate.side.value == "def" and not row.candidate.is_no_adjustment
        ]
        self.assertTrue(offense)
        self.assertTrue(defense)
        ids = [row.candidate.macro_id for row in self.records]
        self.assertEqual(len(ids), len(set(ids)))
        catalog = {str(macro.get("id") or "").upper() for macro in rdb.defense_macros()}
        for row in offense + defense:
            cand = row.candidate
            side = "defense" if cand.side.value == "def" else "offense"
            self.assertEqual(cand.state, MacroState.DRAFT)
            self.assertEqual(cand.version, "1")
            self.assertTrue(cand.tactical_purpose)
            names = {prim.name for prim in (cand.adjustment_primitives or ())}
            if names == {CHIP_AND_RELEASE_ID}:
                self.assertEqual(cand.situations, ())
                self.assertEqual(row.catalog_comparison, ())
            else:
                self.assertTrue(cand.situations)
                self.assertTrue(row.catalog_comparison)
            self.assertTrue(row.downside)
            self.assertTrue(row.uncertainty_note)
            self.assertTrue(row.install_instructions)
            self.assertFalse(row.xbox_created)
            self.assertNotIn(cand.macro_id.upper(), catalog)
            self.assertIn(cand.play, self.formations[side][cand.formation])
            self.assertIsNone(cand.joint.empirical_results)
            self.assertIsNone(cand.joint.situation)
            self.assertFalse(is_live_selectable(cand.state))
        self.assertTrue(any("combined" in note.relation for row in defense for note in row.catalog_comparison))
        self.assertFalse(any(
            prim.name == "control:offense:motion"
            for row in offense for prim in (row.candidate.adjustment_primitives or ())
        ))

    def test_defense_editor_fields_are_complete(self) -> None:
        fields_doc = rdb.editor_fields()
        self.assertNotIn("offense", fields_doc)
        self.assertEqual(len(defensive_editor_field_names()), 51)
        expected = set(defensive_editor_field_names())
        sources = set(rdb.sources())
        sheets = [
            row.candidate for row in self.records
            if row.candidate.side.value == "def" and not row.candidate.is_no_adjustment
        ]
        self.assertTrue(sheets)
        for cand in sheets:
            self.assertEqual(set(cand.editor_fields), expected)
            for key, entry in cand.editor_fields.items():
                self.assertTrue(entry.value)
                self.assertIn(entry.source.value, ("researched", "default"))
                if entry.source.value == "default":
                    self.assertEqual(entry.value, "Default")
                    self.assertIsNone(entry.provenance)
                else:
                    self.assertNotEqual(entry.value, "Default")
                    self.assertIn(entry.provenance, sources)
                self.assertIsNone(entry.confidence)

    def test_conflicts_and_uninstalled_plays_are_rejected(self) -> None:
        pairs = rejected_offense_pairs()
        self.assertTrue(any("max protect" in reason for _, _, reason in pairs))
        blocked = {
            frozenset((left, right))
            for left, right, reason in pairs
            if "max protect" in reason
        }
        self.assertTrue(blocked)
        for row in self.records:
            ids = {prim.name for prim in (row.candidate.adjustment_primitives or ())}
            self.assertFalse(any(pair <= ids for pair in blocked))
            self.assertEqual(row.candidate.conflicts, ())
        reason = defense_value_conflict([
            {("DL/LB", "DL Technique"): "Pinch"},
            {("DL/LB", "DL Technique"): "Spread"},
        ])
        self.assertIn("DL Technique", reason)
        with self.assertRaises(ValueError):
            require_installed(self.formations["offense"], "Gun Bunch", "Mesh")
        with self.assertRaises(KeyError):
            require_primitive("invented_hot_route")
        self.assertTrue(any(row.primitive_id == "control:offense:motion" for row in unverified()))

    def test_live_gate_and_baseline(self) -> None:
        self.assertFalse(any(item.name in ("live_selectable", "live_eligible", "verified_armed")
                             for item in fields(MacroCandidate)))
        draft = next(
            row for row in self.records
            if row.candidate.side.value == "def"
            and row.candidate.play == "DB Fire 2"
            and not row.candidate.is_no_adjustment
        )
        with self.assertRaises(NotLiveSelectable):
            require_live(draft.candidate)
        with self.assertRaises(ValueError):
            transition(draft, MacroState.ELIGIBLE, "user confirmed the whole chain at once")
        with self.assertRaises(ValueError):
            transition(draft, MacroState.EDITOR_VERIFIED, "  ")
        baseline = select_live(
            self.records,
            formation=draft.candidate.formation,
            play=draft.candidate.play,
            installed=self.formations,
        )
        self.assertTrue(baseline.is_no_adjustment)
        self.assertTrue(is_baseline_live_legal(baseline, self.formations))
        self.assertEqual(baseline.joint.action_count, 0)
        self.assertEqual(baseline.conflicts, ())
        armed = draft
        for state, evidence in (
            (MacroState.EDITOR_VERIFIED, "FIXTURE: editor rows checked"),
            (MacroState.INSTALLED, "FIXTURE: saved in the editor"),
            (MacroState.ARMED, "FIXTURE: armed in a slot"),
        ):
            armed = transition(armed, state, evidence)
        self.assertEqual(armed.candidate.state, MacroState.ARMED)
        self.assertTrue(is_live_selectable(armed.candidate.state))
        self.assertFalse(armed.xbox_created)
        self.assertTrue(all("did not create the macro on Xbox" in note for note in armed.confirmation_evidence))
        updated = [
            armed if row.candidate.macro_id == armed.candidate.macro_id else row
            for row in self.records
        ]
        chosen = select_live(
            updated,
            formation=draft.candidate.formation,
            play=draft.candidate.play,
            installed=self.formations,
        )
        self.assertEqual(chosen.macro_id, armed.candidate.macro_id)
        retired = transition(armed, MacroState.RETIRED, "FIXTURE: retire this sheet")
        self.assertFalse(is_live_selectable(retired.candidate.state))

    def test_baseline_is_live_legal_without_lifecycle(self) -> None:
        row = next(item for item in self.records if item.candidate.is_no_adjustment)
        tree = ast.parse(textwrap.dedent(inspect.getsource(is_baseline_live_legal)))
        function = tree.body[0]
        if (
            function.body
            and isinstance(function.body[0], ast.Expr)
            and isinstance(function.body[0].value, ast.Constant)
        ):
            function.body = function.body[1:]
        source = ast.unparse(function)
        self.assertNotIn("is_live_selectable", source)
        self.assertNotIn("state", source)
        self.assertIn("is_no_adjustment", source)
        for state in (
            MacroState.DRAFT,
            MacroState.EDITOR_VERIFIED,
            MacroState.INSTALLED,
            MacroState.RETIRED,
            MacroState.UNKNOWN,
        ):
            shifted = replace(row.candidate, state=state)
            self.assertNotEqual(state, MacroState.ELIGIBLE)
            self.assertFalse(is_live_selectable(shifted.state))
            self.assertTrue(is_baseline_live_legal(shifted, self.formations))
            require_live(shifted, installed=self.formations)
            chosen = select_live(
                [replace(row, candidate=shifted)],
                formation=shifted.formation,
                play=shifted.play,
                installed=self.formations,
            )
            self.assertEqual(chosen.state, state)
            self.assertTrue(chosen.is_no_adjustment)
            self.assertEqual(chosen.macro_id, row.candidate.macro_id)
        foreign = replace(
            row.candidate,
            formation="Gun Bunch",
            play="Mesh",
            state=MacroState.ELIGIBLE,
        )
        self.assertFalse(is_baseline_live_legal(foreign, self.formations))
        with self.assertRaises(NotLiveSelectable):
            require_live(foreign, installed=self.formations)
        with self.assertRaises(ValueError):
            transition(row, MacroState.EDITOR_VERIFIED, "FIXTURE: baselines do not transition")

    def test_json_adapter_uses_null_for_unknown(self) -> None:
        record = next(row for row in self.records if row.candidate.is_no_adjustment)
        cand = record.candidate
        payload = candidate_to_json(cand)
        self.assertEqual(payload["__schema__"], WIRE_VERSION)
        self.assertEqual(payload["__type__"], "MacroCandidate")
        raw = json.dumps(payload)
        self.assertIn("null", raw)
        self.assertNotIn("None", raw)
        self.assertIsNone(payload["joint"]["empirical_results"])
        self.assertIsNone(payload["joint"]["uncertainty"])
        self.assertIsNone(payload["joint"]["situation"])
        self.assertEqual(payload["design_record_id"], f"DSN-{cand.macro_id}")
        self.assertEqual(cand.design_record_id, f"DSN-{cand.macro_id}")
        self.assertNotIn("downside", payload)
        self.assertNotIn("catalog_comparison", payload)
        self.assertNotIn("install_instructions", payload)
        self.assertNotIn("uncertainty_note", payload)
        restored = candidate_from_json(json.loads(raw))
        self.assertEqual(restored, cand)
        self.assertEqual(restored.design_record_id, f"DSN-{cand.macro_id}")
        legacy = json.loads(raw)
        legacy.pop("design_record_id")
        loaded = candidate_from_json(legacy)
        self.assertIsNone(loaded.design_record_id)
        self.assertEqual(loaded.macro_id, cand.macro_id)
        defense_row = next(
            row for row in self.records
            if row.candidate.side.value == "def" and not row.candidate.is_no_adjustment
        )
        defense_payload = candidate_to_json(defense_row.candidate)
        self.assertEqual(
            defense_payload["design_record_id"],
            f"DSN-{defense_row.candidate.macro_id}",
        )
        self.assertNotIn("downside", defense_payload)
        self.assertNotIn("install_instructions", defense_payload)
        self.assertEqual(
            candidate_from_json(defense_payload),
            defense_row.candidate,
        )

    def test_offense_pair_notes_list_every_evaluated_conflict(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "cfb_coach/data/madden27/notes/offense_primitive_pair_conflicts.md"
        )
        text = path.read_text(encoding="utf-8")
        verdicts = offense_pair_verdicts()
        self.assertEqual(len(verdicts), 6)
        self.assertNotIn("INFERRED — not sourced", text)
        for verdict in verdicts:
            heading = f"## {verdict.left_id} + {verdict.right_id}"
            self.assertIn(heading, text)
            section = text.split(heading, 1)[1].split("\n## ", 1)[0]
            self.assertIn(verdict.label, section)
            self.assertIn(verdict.confidence, section)
            for source in verdict.sources:
                self.assertIn(source, section)
            if verdict.conflict:
                self.assertIn(verdict.conflict, section)
            else:
                self.assertIn("no conflict", section.casefold())
        self.assertIn("ID Mike", text)
        self.assertIn("HB Chip and Release", text)
        self.assertIn("https://www.818madden.com/tools/position-adjustments", text)
        self.assertIn("ClutchPoints", text)
        self.assertIn("left stick", text.casefold())
        self.assertIn("## Not paired", text)

    def test_legal_two_action_drafts_and_chip_alternative(self) -> None:
        def names(row):
            return {prim.name for prim in (row.candidate.adjustment_primitives or ())}

        slant_flat = [
            row for row in self.records
            if names(row) == {"hot_slant_vs_man", "hot_hb_flat_vs_pressure"}
        ]
        slant_max = [
            row for row in self.records
            if names(row) == {"hot_slant_vs_man", "max_protect_vs_pressure"}
        ]
        chip_max = [
            row for row in self.records
            if names(row) == {"max_protect_vs_pressure", CHIP_AND_RELEASE_ID}
        ]
        blocked = [
            row for row in self.records
            if names(row) == {"hot_hb_flat_vs_pressure", "max_protect_vs_pressure"}
        ]
        self.assertTrue(slant_flat)
        self.assertTrue(slant_max)
        self.assertTrue(chip_max)
        self.assertFalse(blocked)
        for row in (*slant_flat, *slant_max, *chip_max):
            self.assertEqual(row.candidate.state, MacroState.DRAFT)
            self.assertFalse(is_live_selectable(row.candidate.state))
            self.assertFalse(row.candidate.is_no_adjustment)
        self.assertTrue(all("man_blitz" in (row.candidate.situations or ()) for row in slant_flat))
        chip = require_primitive(CHIP_AND_RELEASE_ID)
        self.assertEqual(chip.route, "Chip and Release")
        self.assertEqual(chip.target, "HB")
        self.assertEqual(chip.vs, ())
        self.assertEqual(chip.source_ids, CHIP_AND_RELEASE_SOURCES)
        motion = next(row for row in unverified() if row.primitive_id == "control:offense:motion")
        self.assertIn("left stick", motion.block_reason or "")
        self.assertIn("Practice-mode", motion.block_reason or "")

    def test_arm_cap_is_eight_per_side(self) -> None:
        drafts = [
            row for row in self.records
            if row.candidate.state is MacroState.DRAFT and not row.candidate.is_no_adjustment
        ]
        self.assertGreater(len(drafts), LOADOUT_N)

        def probe(index: int, state: MacroState):
            macro_id = f"ADJ-OFF-CAP{index:02d}"
            candidate = MacroCandidate(
                macro_id=macro_id,
                design_record_id=f"DSN-{macro_id}",
                version="1",
                side=MacroSide.OFF,
                state=state,
                formation="Pistol Trips",
                play="HB Dive",
                adjustment_primitives=(),
                editor_fields=None,
                tactical_purpose="cap probe",
                situations=(),
                conflicts=(),
                is_no_adjustment=False,
            )
            return DesignRecord(
                design_record_id=candidate.design_record_id or "",
                candidate=candidate,
                downside=None,
                uncertainty_note=None,
                catalog_comparison=(),
                install_instructions=(),
            )

        armed = [probe(index, MacroState.ARMED) for index in range(LOADOUT_N)]
        ninth = probe(LOADOUT_N, MacroState.INSTALLED)
        with self.assertRaises(ArmCapExceeded):
            transition(ninth, MacroState.ARMED, "FIXTURE: ninth slot", roster=armed)
        with self.assertRaises(ArmCapExceeded):
            select_live(
                [*armed, probe(LOADOUT_N, MacroState.ARMED)],
                formation="Pistol Trips",
                play="HB Dive",
                installed=self.formations,
            )
        defense = next(
            row for row in self.records
            if row.candidate.side is MacroSide.DEF and not row.candidate.is_no_adjustment
        )
        moved = defense
        for state in (MacroState.EDITOR_VERIFIED, MacroState.INSTALLED, MacroState.ARMED):
            moved = transition(moved, state, "FIXTURE: other side still has room", roster=armed)
        self.assertEqual(moved.candidate.state, MacroState.ARMED)

    def test_demo_all_leaves_home_untouched(self) -> None:
        home = Path(tempfile.mkdtemp(prefix="demo-home-"))
        env = os.environ.copy()
        env["HOME"] = str(home)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        env.pop("CFB_COACH_DB", None)
        env.pop("CFB_COACH_MADDEN_DB", None)
        proc = subprocess.run(
            [sys.executable, "-m", "cfb_coach.madden.adjustments.demo", "all"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        created = sorted(path.relative_to(home).as_posix() for path in home.rglob("*"))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:] + proc.stdout[-2000:])
        self.assertEqual(created, [])
        self.assertIn("history_preserved True", proc.stdout)

    def test_existing_fallbacks_still_work(self) -> None:
        kw = dict(
            formation="Gun Doubles Clamp Stack",
            audibles={"Gun Doubles Clamp Stack": ["Texas Y-Stutter Wheel", "Inside Zone"]},
        )
        man = offense_adjustment(
            play="Mesh Post", coverage_class="man", coverage_source="live", repeated=False, **kw,
        )
        self.assertEqual(man["label"], "Hot route WR1 → Slant")
        self.assertIsNone(offense_adjustment(
            play="Mesh Post", coverage_class="man", coverage_source="last", repeated=False, **kw,
        ))
        vert = defense_adjustment("vert")
        self.assertEqual(vert["label"], "Back off + overtop")
        self.assertIsNone(defense_adjustment(None))
        reason = offense_conflict([
            {"type": "hot_route", "target": "HB", "route": "Flat", "vs": ["pressure"], "label": ""},
            {"type": "pass_protection", "target": "protection", "route": "Max protect", "vs": ["pressure"], "label": "Max protect"},
        ])
        self.assertIn("max protect", reason)

    def test_history_and_loadout_are_preserved(self) -> None:
        db = self.madden_db()
        try:
            install_trimmed_book(db)
            db.log_snap(
                opponent_id="gavin", side="defense", situation_raw="d 1&10", our_call="x",
                formation="Nickel Over", play="Cover 4 Quarters", result="stop", macro="TAMPA MABLE",
            )
            store_selection(db, "gavin", {"offense": ["MAN", "C2"], "defense": ["TAMPA MABLE", "SKY CONTAIN"]})
            db.conn.commit()
            before_snaps = db.conn.execute(
                "SELECT opponent_id, formation, play, macro, result FROM snaps ORDER BY id"
            ).fetchall()
            before_loadout = db.get_meta("active_macros:gavin")
            self.assertEqual(load_selection(db, "gavin")["defense"], ["TAMPA MABLE", "SKY CONTAIN"])
            backup = backup_database(db.path)
            staged = stage_drafts(db, backup=False)
            self.assertGreater(staged["stored"], 0)
            self.assertTrue(backup.is_file())
            after_snaps = db.conn.execute(
                "SELECT opponent_id, formation, play, macro, result FROM snaps ORDER BY id"
            ).fetchall()
            self.assertEqual(after_snaps, before_snaps)
            self.assertEqual(db.get_meta("active_macros:gavin"), before_loadout)
            formed = generate_from_db(db)
            self.assertEqual(
                {row.candidate.formation for row in formed if row.candidate.side.value == "off"},
                set(self.formations["offense"]),
            )
        finally:
            db.close()
