"""Role 3: original Custom Adjustment drafts, editor completeness, and live gates."""

from __future__ import annotations

import json
from dataclasses import fields

from cfb_coach.madden import research_db as rdb
from cfb_coach.madden.adjustments import defense_adjustment, offense_adjustment
from cfb_coach.madden.adjustments.candidates import (
    WIRE_VERSION,
    MacroCandidate,
    MacroState,
    candidate_from_json,
    candidate_to_json,
    is_live_selectable,
)
from cfb_coach.madden.adjustments.primitives import require_primitive, unverified
from cfb_coach.madden.model.schema import defensive_editor_field_names
from cfb_coach.madden.adjustments.synthesis import (
    NotLiveSelectable,
    backup_database,
    defense_value_conflict,
    generate,
    generate_from_db,
    install_trimmed_book,
    offense_conflict,
    rejected_offense_pairs,
    require_installed,
    require_live,
    select_live,
    stage_drafts,
    transition,
    trimmed_formations,
)
from cfb_coach.madden.macros import load_selection, store_selection
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
            self.assertTrue(cand.situations)
            self.assertTrue(row.downside)
            self.assertTrue(row.uncertainty_note)
            self.assertTrue(row.catalog_comparison)
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
            self.records, formation=draft.candidate.formation, play=draft.candidate.play,
        )
        self.assertTrue(baseline.is_no_adjustment)
        self.assertTrue(is_live_selectable(baseline.state))
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
        chosen = select_live(updated, formation=draft.candidate.formation, play=draft.candidate.play)
        self.assertEqual(chosen.macro_id, armed.candidate.macro_id)
        retired = transition(armed, MacroState.RETIRED, "FIXTURE: retire this sheet")
        self.assertFalse(is_live_selectable(retired.candidate.state))

    def test_json_adapter_uses_null_for_unknown(self) -> None:
        cand = next(row.candidate for row in self.records if row.candidate.is_no_adjustment)
        payload = candidate_to_json(cand)
        self.assertEqual(payload["__schema__"], WIRE_VERSION)
        self.assertEqual(payload["__type__"], "MacroCandidate")
        raw = json.dumps(payload)
        self.assertIn("null", raw)
        self.assertNotIn("None", raw)
        self.assertIsNone(payload["joint"]["empirical_results"])
        self.assertIsNone(payload["joint"]["uncertainty"])
        self.assertIsNone(payload["joint"]["situation"])
        restored = candidate_from_json(json.loads(raw))
        self.assertEqual(restored, cand)
        defense = next(
            row.candidate for row in self.records
            if row.candidate.side.value == "def" and not row.candidate.is_no_adjustment
        )
        self.assertEqual(candidate_from_json(candidate_to_json(defense)), defense)

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
