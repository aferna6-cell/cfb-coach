"""Sprint 14: opponent isolation and the offline film pipeline."""
from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.madden.model import experimental_live
from cfb_coach.madden.model.film_align import align_segments
from cfb_coach.madden.model.film_evidence import (
    approve_admission, propose_admission, rollback_admission,
)
from cfb_coach.madden.model.film_import import import_recording, locate_candidate_snaps
from cfb_coach.madden.model.film_observe import interpret_frame, pre_snap_features
from cfb_coach.madden.model.film_report import film_report
from cfb_coach.madden.model.film_review import (
    apply_review_action, empty_annotations, render_review_html, save_annotations,
)
from cfb_coach.madden.model.opponent_learning import (
    fit_opponent_model, opponent_learning_report, records_from_memory_events,
    scope_training_rows,
)


def _verified_row(opponent, game, seq, pressure, play="Mesh"):
    return {
        "eligibility": "verified_execution",
        "executed_verification": "verified",
        "executed_status": "identified",
        "executed_play": play,
        "executed_formation": "Gun Bunch",
        "opponent_id": opponent,
        "game_id": game,
        "session_id": game,
        "snap_seq": seq,
        "snap_id": f"{game}-{seq}",
        "down": 3,
        "distance": 8,
        "result": "gain 2" if pressure else "gain 9",
        "yards": 2 if pressure else 9,
        "coverage_seen": "blitz" if pressure else "no pressure",
        "label_available": True,
    }


def _memory_record(seq, game, pressure):
    return {
        "snap_seq": seq,
        "game_id": game,
        "down": 3,
        "distance": 8,
        "yards": 2 if pressure else 9,
        "formation": "Gun Bunch",
        "play": "Mesh",
        "concept_id": "mesh",
        "observation": {
            "pressure": pressure, "blitz": pressure, "legacy_ambiguous": False,
            "coverage_shell": None,
        },
        "verified_execution": True,
        "prelabeled": True,
        "success": not pressure,
        "conversion": not pressure,
        "turnover": False,
        "sack": False,
    }


def _fixture_rgb(*, two_high=True, post=False, yards=6):
    rgb = bytearray(8 * 3)
    rgb[0:3] = bytes((1, 2, 3))
    rgb[3:6] = bytes((2, 3, 8))  # quarter 2, down 3, distance 8
    rgb[6:9] = bytes((90, 0, 0))  # clock 90
    rgb[9:12] = bytes((14, 17, 7))  # score 14-17, box 7
    rgb[12:15] = bytes((2 if two_high else 1, 0, 0))
    rgb[15:18] = bytes((1, 0, 0))  # pressure
    rgb[18:21] = bytes((1, yards, 0))
    rgb[21:24] = bytes((1 if post else 0, 0, 0))
    return bytes(rgb)


def _make_clip(path: Path) -> None:
    colors = ["red", "green", "blue", "red", "green", "blue"]
    command = ["ffmpeg", "-y"]
    for color in colors:
        command += ["-f", "lavfi", "-i", f"color=c={color}:s=64x36:d=1:r=10"]
    command += [
        "-filter_complex", f"concat=n={len(colors)}:v=1:a=0",
        "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(command, check=True, capture_output=True)


class OpponentIsolationTests(unittest.TestCase):
    def test_report_keeps_one_opponent_and_one_game(self):
        rows = [
            _verified_row("cpu", "game-a", 1, False),
            _verified_row("human", "game-a", 2, True),
            _verified_row(None, "game-a", 3, True),
            _verified_row("cpu", "game-b", 4, True),
        ]
        scoped = scope_training_rows(rows, opponent_id="cpu", game_id="game-a")
        self.assertEqual(len(scoped["rows"]), 1)
        self.assertEqual(scoped["excluded_other_opponent"], 1)
        self.assertEqual(scoped["excluded_missing_opponent"], 1)
        self.assertEqual(scoped["excluded_other_game"], 1)
        with mock.patch(
            "cfb_coach.madden.model.dataset.build_rows", return_value=rows,
        ):
            report = opponent_learning_report(object(), opponent_id="cpu", game_id="game-a")
        self.assertEqual(report["usable_verified_snaps"], 1)
        self.assertEqual(report["excluded_other_opponent"], 1)
        self.assertFalse(report["history_modified"])

    def test_change_detection_does_not_cross_games(self):
        early = [_memory_record(i, "game-a", False) for i in range(1, 9)]
        late = [_memory_record(i, "game-b", True) for i in range(1, 9)]
        crossed = fit_opponent_model(records=early + late)
        self.assertFalse(any(
            row["metric"] == "pressure_on_passing_downs" for row in crossed["changes"]
        ))
        inside = [_memory_record(i, "game-a", i > 8) for i in range(1, 17)]
        changed = fit_opponent_model(records=inside)
        self.assertTrue(any(
            row["metric"] == "pressure_on_passing_downs" and row["game_id"] == "game-a"
            for row in changed["changes"]
        ))

    def test_missing_verification_is_not_verified(self):
        unlabeled = dict(_memory_record(1, "game-a", True))
        unlabeled.pop("verified_execution")
        model = fit_opponent_model(records=[unlabeled])
        self.assertEqual(model["usable_verified_snaps"], 0)
        events = records_from_memory_events([
            {"snap_seq": 1, "executed_play": "Mesh", "outcome": "gain 8", "observed_defense": "blitz"},
        ])
        self.assertEqual(events, [])
        ambiguous = []
        for seq in range(1, 13):
            row = _memory_record(seq, "game-a", False)
            row["observation"] = {"legacy_ambiguous": True, "pressure": None, "coverage_shell": None}
            ambiguous.append(row)
        thin = fit_opponent_model(records=ambiguous)
        passing = next(row for row in thin["estimates"] if row["context"] == "passing_downs")
        self.assertEqual(passing["sample_size"], 0)

    def test_later_snap_is_not_in_the_earlier_fit(self):
        rows = [_memory_record(i, "game-a", i > 8) for i in range(1, 17)]
        for index in range(len(rows)):
            model = fit_opponent_model(records=rows[:index])
            self.assertEqual(model["usable_verified_snaps"], index)


class FilmPipelineTests(unittest.TestCase):
    def test_decode_preserves_time_and_finds_more_than_one_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            store = Path(tmp) / "store"
            _make_clip(path)
            first = import_recording(path, game_id="game-a", store=store, dry_run=False)
            self.assertTrue(first["ok"], first)
            self.assertGreater(first["probe"]["duration_s"], 5)
            self.assertEqual(first["probe"]["width"], 64)
            self.assertIsNotNone(first["timestamps"]["first_s"])
            self.assertGreater(len(first["segments"]), 1)
            self.assertFalse(first["identification_claim"])
            self.assertIn("variable_frame_rate", first["probe"])
            self.assertIn("elapsed_ms", first)
            self.assertLess(first["elapsed_ms"], 15000)
            mkv = Path(tmp) / "clip.mkv"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-c", "copy", str(mkv)],
                check=True, capture_output=True,
            )
            mkv_report = import_recording(mkv, game_id="game-a", store=store, dry_run=True)
            self.assertTrue(mkv_report["ok"], mkv_report)
            empty = Path(tmp) / "empty.mp4"
            empty.write_bytes(b"")
            incomplete = import_recording(empty, game_id="game-a", store=store, dry_run=True)
            self.assertEqual(incomplete["probe"]["error"], "incomplete_or_missing_recording")
            odd = Path(tmp) / "notes.txt"
            odd.write_text("x", encoding="utf-8")
            rejected = import_recording(odd, game_id="game-a", store=store, dry_run=True)
            self.assertEqual(rejected["probe"]["error"], "unsupported_container")
            second = import_recording(path, game_id="game-a", store=store, dry_run=True)
            self.assertTrue(second["duplicate"])
            self.assertFalse(second["wrote_manifest"])
            corrupt = Path(tmp) / "bad.mp4"
            corrupt.write_text("not a video", encoding="utf-8")
            broken = import_recording(corrupt, game_id="game-a", store=store, dry_run=True)
            self.assertFalse(broken["ok"])
            self.assertIn(broken["probe"]["error"], ("corrupt_or_unsupported_codec", "incomplete_recording"))

    def test_manual_boundaries_and_alignment(self):
        frames = [
            {"time_s": 0.0, "rgb": bytes([10] * 12)},
            {"time_s": 0.25, "rgb": bytes([10] * 12)},
        ]
        located = locate_candidate_snaps(frames, manual_anchors=[{
            "snap_start": 1.0, "snap_end": 4.0,
            "formation_interval": [1.0, 1.4],
            "presnap_interval": [1.4, 2.0],
            "postsnap_interval": [2.0, 4.0],
        }])
        self.assertEqual(located["segments"][0]["boundary_status"], "corrected")
        self.assertEqual(located["segments"][0]["formation_interval"], [1.0, 1.4])
        logs = [
            {"snap_id": "s1", "snap_seq": 1, "down": 1, "distance": 10, "clock_seconds": 800, "quarter": 1},
            {"snap_id": "s2", "snap_seq": 2, "down": 2, "distance": 7, "clock_seconds": 740, "quarter": 1},
        ]
        segments = [
            {"candidate_id": "menu", "role": "menu_or_replay", "boundary_status": "candidate", "displayed_state": {}},
            {"candidate_id": "play", "boundary_status": "candidate", "displayed_state": {"clock_seconds": 740}},
            {"candidate_id": "late", "boundary_status": "unresolved", "displayed_state": {}},
        ]
        aligned = align_segments(segments, logs, manual_links=[{"candidate_id": "play", "snap_id": "s2"}])
        by_id = {row["candidate_id"]: row for row in aligned["associations"]}
        self.assertEqual(by_id["play"]["status"], "confirmed")
        self.assertEqual(by_id["play"]["snap_id"], "s2")
        self.assertEqual(by_id["menu"]["reason"], "menu_or_replay_not_a_snap")
        self.assertEqual(by_id["late"]["status"], "unresolved")
        self.assertEqual(aligned["verified_executions_created"], 0)
        self.assertEqual(aligned["clock_only_matches_confirmed"], 0)
        clock_only = align_segments(
            [{"candidate_id": "clock", "boundary_status": "candidate", "displayed_state": {"clock_seconds": 800}}],
            logs,
        )
        self.assertEqual(clock_only["associations"][0]["reason"], "clock_similarity_is_not_enough")
        self.assertIsNone(clock_only["associations"][0]["snap_id"])
        repeated = align_segments(
            [
                {"candidate_id": "a", "boundary_status": "candidate", "displayed_state": {"clock_seconds": 740, "down": 2, "distance": 7}},
                {"candidate_id": "b", "boundary_status": "candidate", "displayed_state": {"clock_seconds": 740, "down": 2, "distance": 7}},
            ],
            logs,
        )
        self.assertTrue(all(row["status"] == "unresolved" for row in repeated["associations"]))
        late = next(row for row in aligned["unmatched_log_snaps"] if row["snap_id"] == "s1")
        self.assertEqual(late["reason"], "possible_recording_started_after_kickoff")

    def test_review_html_and_separate_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "history.db"
            db.write_bytes(b"sqlite-history-placeholder")
            digest = hashlib.sha256(db.read_bytes()).hexdigest()
            store = Path(tmp) / "film"
            payload = empty_annotations("game-a", "rec", [
                {"candidate_id": "c1", "start_s": 0, "end_s": 1, "confidence": 0.4, "source": "auto", "boundary_status": "unresolved"},
                {"candidate_id": "c2", "start_s": 1, "end_s": 2, "confidence": 0.5, "source": "auto", "boundary_status": "candidate"},
            ])
            save_annotations(store, payload)
            html = render_review_html(payload)
            self.assertIn("Confirm", html)
            self.assertIn("Leave uncertain", html)
            self.assertIn("Next snap", html)
            corrected = apply_review_action(store, "game-a", "c1", "correct", {
                "snap_start": 0.2, "start_s": 0.2, "end_s": 1.2,
                "snap_id": "s1", "formation": "Gun Bunch", "play": "Mesh",
                "defense_review": "confirmed",
                "defensive_observation": {"safety_depth": "two_high", "pressure": True, "box_count": 7},
                "human_label": {"safety_depth": "two_high"},
            })
            self.assertTrue(corrected["ok"])
            self.assertEqual(corrected["candidate"]["association_status"], "confirmed")
            self.assertFalse(corrected["candidate"]["verified_execution"])
            apply_review_action(store, "game-a", "c2", "uncertain")
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), digest)

    def test_fixture_observation_does_not_invent_coverage(self):
        pre = interpret_frame(
            _fixture_rgb(), width=8, height=1, game_id="game-a", recording_id="rec",
            video_timestamp=3.5, snap_id="s1",
        )
        self.assertTrue(pre["accepted"])
        self.assertEqual(pre["defensive_alignment"]["safety_depth"], "two_high")
        self.assertIsNone(pre["defensive_alignment"]["coverage_shell"])
        self.assertEqual(pre["defensive_alignment"]["structure_label"], "two_high_safety_structure")
        self.assertEqual(pre["observed_game_state"]["down"], 3)
        self.assertTrue(pre["available_before_snap"])
        self.assertIsNone(pre["human_label"])
        self.assertIsNotNone(pre["raw_model_observation"])
        self.assertNotIn("outcome", pre_snap_features(pre)["game_state"])
        post = interpret_frame(
            _fixture_rgb(post=True), width=8, height=1, game_id="game-a", recording_id="rec",
            video_timestamp=6.0, unresolved_snap_association="post-1",
        )
        self.assertFalse(pre_snap_features(post)["available_before_snap"])
        self.assertIn("yards", post["observed_game_state"])
        unknown = interpret_frame(
            bytes(24), width=8, height=1, game_id="game-a", recording_id="rec",
            video_timestamp=1.0, unresolved_snap_association="real-frame",
        )
        self.assertFalse(unknown["fixture_protocol"])
        self.assertIsNone(unknown["defensive_alignment"]["coverage_shell"])
        self.assertFalse(unknown["recognition_accuracy_claim"])

    def test_admission_and_rollback_do_not_verify_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "film"
            db = Path(tmp) / "madden.db"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE snaps (id INTEGER PRIMARY KEY, session_id TEXT, result TEXT)")
            conn.execute("INSERT INTO snaps (session_id, result) VALUES ('game-a', 'gain 4')")
            conn.commit()
            before = hashlib.sha256(db.read_bytes()).hexdigest()
            conn.close()
            payload = empty_annotations("game-a", "rec", [
                {"candidate_id": "c1", "start_s": 0, "end_s": 2, "confidence": 0.8, "boundary_status": "corrected"},
            ])
            save_annotations(store, payload)
            apply_review_action(store, "game-a", "c1", "confirm", {
                "snap_id": "s1",
                "formation": "Gun Bunch",
                "play": "Mesh",
                "observation_time": "pre_snap",
                "game_state": {"down": 3, "distance": 8, "quarter": 1, "clock_seconds": 700},
                "defense_review": "confirmed",
                "execution_review": "confirmed",
                "defensive_observation": {
                    "pressure": True,
                    "safety_depth": "two_high",
                    "coverage_shell": "unknown",
                    "box_count": 7,
                    "front": "unknown",
                    "leverage": "unknown",
                },
                "unknown_fields": ["coverage_shell", "front", "leverage"],
            })
            annotations = json.loads((store / "annotations" / "game-a.json").read_text(encoding="utf-8"))
            logs = [{
                "snap_id": "s1", "snap_seq": 4, "game_id": "game-a", "opponent_id": "cpu",
                "down": 3, "distance": 8, "executed_play": "Stick",
                "executed_verification": "unverified",
            }]
            proposal = propose_admission(annotations, logs, opponent_id="cpu")
            kinds = {row["kind"]: row for row in proposal["proposals"]}
            self.assertTrue(kinds["defensive_observation"]["admit"])
            self.assertFalse(kinds["defensive_observation"]["verified_execution"])
            self.assertEqual(kinds["defensive_observation"]["observation_time"], "pre_snap")
            self.assertTrue(kinds["defensive_observation"]["available_before_snap"])
            self.assertEqual(kinds["execution"]["reason"], "video_cannot_create_verified_execution")
            self.assertEqual(proposal["verified_executions_created"], 0)
            approved = approve_admission(store, proposal)
            self.assertEqual(approved["admitted"], 1)
            learned = fit_opponent_model(records=[])
            from cfb_coach.madden.model.film_evidence import load_admitted_records
            admitted = load_admitted_records(store, game_id="game-a", opponent_id="cpu")
            self.assertEqual(len(admitted), 1)
            model = fit_opponent_model(records=admitted)
            self.assertEqual(model["usable_verified_snaps"], 0)
            self.assertEqual(model["approved_film_observations"], 1)
            self.assertFalse(model["video_observations_included"])
            report = film_report(store, "game-a")
            self.assertEqual(report["admitted_learning_evidence"], ["c1:defense"])
            self.assertFalse(report["admitted_are_verified_executions"])
            self.assertEqual(report["verified_training_eligibility"]["new_verified_executions_from_video"], 0)
            self.assertFalse(report["learned_from_video"])
            rollback_admission(store, "game-a")
            self.assertEqual(load_admitted_records(store, game_id="game-a"), [])
            self.assertEqual(learned["usable_verified_snaps"], 0)
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = list(conn.execute("SELECT result FROM snaps"))
            conn.close()
            self.assertEqual(rows, [("gain 4",)])
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), before)

    def test_live_playcaller_does_not_import_film(self):
        source = inspect.getsource(experimental_live)
        self.assertNotIn("film_import", source)
        self.assertNotIn("film_review", source)
        self.assertNotIn("import_recording", source)


if __name__ == "__main__":
    unittest.main()
