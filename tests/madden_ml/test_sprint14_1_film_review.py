"""Sprint 14.1: usable local film review and read-only coach-log export."""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import sqlite3
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

from cfb_coach.cli import main
from cfb_coach.madden.model import experimental_live
from cfb_coach.madden.model.film_align import align_segments
from cfb_coach.madden.model.film_evidence import (
    approve_admission, load_admitted_records, propose_admission,
)
from cfb_coach.madden.model.film_import import import_recording, locate_candidate_snaps, sample_frames
from cfb_coach.madden.model.film_log import export_coach_log
from cfb_coach.madden.model.film_report import film_report
from cfb_coach.madden.model.film_review import (
    apply_review_action, authorized_recording_path, empty_annotations,
    make_review_server, render_review_html, save_annotations,
)
from cfb_coach.madden.model.opponent_learning import (
    fit_opponent_model, opponent_learning_report,
)


def _explicit(**overrides):
    payload = {
        "snap_id": "s1",
        "formation": "Gun Bunch",
        "play": "Mesh",
        "observation_time": "pre_snap",
        "game_state": {"down": 1, "distance": 10, "quarter": 2, "clock_seconds": 400},
        "defensive_observation": {
            "coverage_shell": "cover_3",
            "safety_depth": "single_high",
            "pressure": False,
            "box_count": 6,
            "front": "over",
            "leverage": "outside",
        },
    }
    payload.update(overrides)
    return payload


def _log(snap_id="s1", game_id="game-a", opponent_id="cpu", **extra):
    row = {
        "snap_id": snap_id,
        "game_id": game_id,
        "session_id": game_id,
        "opponent_id": opponent_id,
        "snap_seq": 1,
        "down": 1,
        "distance": 10,
        "executed_play": None,
        "executed_verification": None,
    }
    row.update(extra)
    return row


def _confirmed_annotations(game_id="game-a", **overrides):
    row = {
        "candidate_id": "c1",
        "start_s": 0,
        "end_s": 2,
        "confidence": 0.8,
        "review_status": "confirmed",
        "association_status": "confirmed",
        "snap_id": "s1",
        "formation": "Gun Bunch",
        "play": "Mesh",
        "observation_time": "pre_snap",
        "defense_review": "confirmed",
        "execution_review": "confirmed",
        "defensive_observation": {
            "pressure": True,
            "safety_depth": "single_high",
            "coverage_shell": "cover_1",
            "box_count": 7,
            "front": "under",
            "leverage": "inside",
        },
        "game_state": {"down": 3, "distance": 8, "quarter": 1, "clock_seconds": 500},
    }
    row.update(overrides)
    return {"game_id": game_id, "recording_id": "rec", "candidates": [row]}


class ReviewInterfaceTests(unittest.TestCase):
    def test_page_has_playback_and_posts_edited_values(self):
        payload = empty_annotations("game-a", "rec", [
            {"candidate_id": "c1", "start_s": 1.0, "end_s": 4.0, "confidence": 0.6, "source": "auto", "boundary_status": "candidate"},
        ])
        html = render_review_html(payload, log_snaps=[_log()])
        for text in (
            "<video", "Seek to snap", "Loop current snap", "Pre-snap frame", "Post-snap frame",
            "Confirm", "Correct", "Reject", "Leave uncertain", "Previous snap", "Next snap",
            "!result.ok", "result.errors",
        ):
            self.assertIn(text, html)
        self.assertIn('value="s1"', html)
        self.assertNotIn('value="s1" selected', html)
        self.assertIn("provenance", html)

    def test_confirm_rejects_missing_fields_and_unknown_snaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "film"
            payload = empty_annotations("game-a", "rec", [
                {"candidate_id": "c1", "start_s": 0, "end_s": 2, "confidence": 0.4, "boundary_status": "candidate"},
            ])
            save_annotations(store, payload)
            before = (store / "annotations" / "game-a.json").read_bytes()
            rejected = apply_review_action(
                store, "game-a", "c1", "confirm", {"play": "Mesh"},
                log_snaps=[_log()], opponent_id="cpu",
            )
            self.assertFalse(rejected["ok"])
            self.assertEqual(rejected["error"], "confirm_rejected")
            self.assertFalse(rejected["saved"])
            self.assertIn("snap_association", rejected["errors"])
            self.assertIn("timing_not_explicit", rejected["errors"])
            self.assertEqual((store / "annotations" / "game-a.json").read_bytes(), before)
            unknown_snap = apply_review_action(
                store, "game-a", "c1", "confirm", _explicit(snap_id="missing"),
                log_snaps=[_log()], opponent_id="cpu",
            )
            self.assertIn("snap_not_in_log", unknown_snap["errors"])
            saved = apply_review_action(
                store, "game-a", "c1", "correct",
                {"snap_id": "not-in-log", "formation": "Gun Trips", "play": "Stick"},
                log_snaps=[_log()], opponent_id="cpu",
            )
            self.assertTrue(saved["ok"])
            self.assertEqual(saved["candidate"]["formation"], "Gun Trips")
            self.assertEqual(saved["candidate"]["association_status"], "unresolved")
            self.assertFalse(saved["candidate"]["verified_execution"])

    def test_explicit_unknown_is_stored_and_a_matching_log_can_confirm(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "film"
            save_annotations(store, empty_annotations("game-a", "rec", [
                {"candidate_id": "c1", "start_s": 0.2, "end_s": 3.0, "confidence": 0.7, "boundary_status": "candidate"},
            ]))
            confirmed = apply_review_action(
                store, "game-a", "c1", "confirm",
                _explicit(
                    defensive_observation={
                        "coverage_shell": "unknown",
                        "safety_depth": "two_high",
                        "pressure": "unknown",
                        "box_count": 6,
                        "front": "unknown",
                        "leverage": "unknown",
                    },
                    unknown_fields=["coverage_shell", "pressure", "front", "leverage"],
                ),
                log_snaps=[_log()], opponent_id="cpu",
            )
            self.assertTrue(confirmed["ok"], confirmed)
            candidate = confirmed["candidate"]
            self.assertEqual(candidate["association_status"], "confirmed")
            self.assertIsNone(candidate["defensive_observation"]["coverage_shell"])
            self.assertIn("coverage_shell", candidate["unknown_fields"])
            self.assertFalse(candidate["verified_execution"])
            self.assertNotEqual(candidate["execution_review"], "confirmed")

    def test_localhost_server_serves_only_the_recording_and_shows_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "game.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x36:d=1:r=10", "-pix_fmt", "yuv420p", str(video)],
                check=True, capture_output=True,
            )
            store = root / "film"
            save_annotations(store, empty_annotations("game-a", "rec", [
                {"candidate_id": "c1", "start_s": 0, "end_s": 1, "confidence": 0.5, "boundary_status": "candidate"},
            ]))
            (store / "recordings").mkdir(parents=True)
            (store / "recordings" / "rec.json").write_text(json.dumps({
                "recording_id": "rec", "path": str(video), "game_id": "game-a",
            }), encoding="utf-8")
            self.assertEqual(authorized_recording_path(store, "game-a"), video)
            with self.assertRaises(ValueError):
                make_review_server(store, "game-a", host="0.0.0.0")
            server = make_review_server(
                store, "game-a", host="127.0.0.1", port=0,
                log_snaps=[_log()], opponent_id="cpu",
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            try:
                media = urllib.request.Request(
                    f"http://127.0.0.1:{port}/media?path=/etc/passwd",
                    headers={"Range": "bytes=0-10"},
                )
                with urllib.request.urlopen(media) as response:
                    self.assertEqual(response.status, 206)
                    self.assertIn("bytes 0-10/", response.headers["Content-Range"])
                    body = response.read()
                self.assertEqual(body, video.read_bytes()[:11])
                self.assertNotEqual(body, Path("/etc/passwd").read_bytes()[:11])
                frame = urllib.request.urlopen(f"http://127.0.0.1:{port}/frame?t=0&path=/etc/passwd")
                self.assertEqual(frame.headers["Content-Type"], "image/jpeg")
                self.assertTrue(frame.read().startswith(b"\xff\xd8"))
                raw = json.dumps({"candidate_id": "c1", "action": "confirm", "updates": {}}).encode()
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/review", data=raw,
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request)
                self.assertEqual(caught.exception.code, 400)
                failure = json.loads(caught.exception.read().decode("utf-8"))
                self.assertFalse(failure["ok"])
                self.assertEqual(failure["error"], "confirm_rejected")
                self.assertFalse(failure["saved"])
                stored = json.loads((store / "annotations" / "game-a.json").read_text(encoding="utf-8"))
                self.assertEqual(stored["candidates"][0]["review_status"], "unreviewed")
                correction = json.dumps({
                    "candidate_id": "c1",
                    "action": "correct",
                    "updates": _explicit(formation="Gun Trips", play="Corner"),
                }).encode()
                posted = urllib.request.Request(
                    f"http://127.0.0.1:{port}/review", data=correction,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(posted) as response:
                    saved = json.loads(response.read().decode("utf-8"))
                self.assertTrue(saved["ok"])
                self.assertEqual(saved["candidate"]["formation"], "Gun Trips")
                self.assertEqual(saved["candidate"]["play"], "Corner")
                self.assertEqual(saved["candidate"]["association_status"], "confirmed")
            finally:
                server.shutdown()
                server.server_close()


class CoachLogTests(unittest.TestCase):
    def _database(self, path: Path) -> None:
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE game_sessions (
                session_id TEXT PRIMARY KEY,
                opponent_id TEXT NOT NULL,
                started_ts TEXT NOT NULL,
                play_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE snaps (
                id INTEGER PRIMARY KEY,
                ts TEXT NOT NULL,
                opponent_id TEXT NOT NULL,
                side TEXT NOT NULL,
                down INTEGER,
                distance INTEGER,
                quarter INTEGER,
                formation TEXT,
                play TEXT,
                result TEXT,
                session_id TEXT,
                ml_snap_id TEXT,
                snap_seq INTEGER,
                executed_status TEXT,
                executed_formation TEXT,
                executed_play TEXT,
                executed_verification TEXT
            );
            CREATE TABLE ml_decisions (
                id INTEGER PRIMARY KEY,
                decision_ts TEXT NOT NULL,
                game_id TEXT,
                snap_id TEXT,
                session_id TEXT,
                snap_seq INTEGER,
                mode TEXT NOT NULL,
                heuristic_play TEXT,
                final_formation TEXT,
                final_play TEXT,
                decision_json TEXT NOT NULL,
                created_ts TEXT NOT NULL
            );
            INSERT INTO game_sessions VALUES ('game-a', 'cpu', 't', 2);
            INSERT INTO snaps VALUES
                (1, 't', 'cpu', 'offense', 1, 10, 1, 'Gun Bunch', 'Inside Zone', 'gain 3', 'game-a', NULL, 1, NULL, NULL, NULL, NULL),
                (2, 't', 'cpu', 'offense', 2, 7, 1, 'Gun Bunch', 'Inside Zone', 'gain 4', 'game-a', 'snap-1', 2, NULL, NULL, NULL, NULL),
                (3, 't', 'cpu', 'offense', 1, 10, 1, 'Gun Bunch', 'Power', 'gain 1', 'game-b', 'snap-other', 1, NULL, NULL, NULL, NULL);
            INSERT INTO ml_decisions VALUES
                (1, 't', 'game-a', 'snap-1', 'game-a', 2, 'shadow', 'Power', 'Gun Bunch', 'Mesh', '{}', 't'),
                (2, 't', 'game-a', 'snap-decision', 'game-a', 3, 'shadow', 'Dive', 'Gun Trips', 'Stick', '{}', 't'),
                (3, 't', 'game-a', NULL, 'game-a', 4, 'shadow', 'Fade', 'Gun Bunch', 'Corner', '{}', 't');
            """
        )
        conn.commit()
        conn.close()

    def test_export_uses_stored_identities_and_leaves_the_database_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "madden27.db"
            self._database(db)
            before = hashlib.sha256(db.read_bytes()).hexdigest()
            report = export_coach_log(db, "game-a", opponent_id="cpu")
            self.assertTrue(report["ok"], report)
            self.assertFalse(report["history_modified"])
            self.assertEqual(report["omitted_missing_snap_id"], 1)
            self.assertEqual(report["invented_snap_identifiers"], 0)
            self.assertEqual(report["invented_executed_plays"], 0)
            by_id = {row["snap_id"]: row for row in report["snaps"]}
            self.assertEqual(set(by_id), {"snap-1", "snap-decision"})
            self.assertIsNone(by_id["snap-1"]["executed_play"])
            self.assertIsNone(by_id["snap-1"]["executed_verification"])
            self.assertEqual(by_id["snap-1"]["logged_play"], "Mesh")
            self.assertEqual(by_id["snap-1"]["logged_play_source"], "decision")
            self.assertNotEqual(by_id["snap-1"]["executed_play"], "Power")
            self.assertEqual(by_id["snap-1"]["down"], 2)
            self.assertEqual(by_id["snap-1"]["distance"], 7)
            self.assertIsNone(by_id["snap-decision"]["executed_play"])
            self.assertEqual(by_id["snap-decision"]["logged_play"], "Stick")
            self.assertIsNone(by_id["snap-decision"]["down"])
            mismatch = export_coach_log(db, "game-a", opponent_id="human")
            self.assertFalse(mismatch["ok"])
            self.assertEqual(mismatch["reason"], "opponent_mismatch")
            self.assertEqual(mismatch["snaps"], [])
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), before)
            out = Path(tmp) / "log-snaps.json"
            old = os.environ.get("CFB_COACH_MADDEN_DB")
            os.environ["CFB_COACH_MADDEN_DB"] = str(db)
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    code = main(["ml", "film-export-log", "--game-id", "game-a", "--out", str(out)])
            finally:
                if old is None:
                    os.environ.pop("CFB_COACH_MADDEN_DB", None)
                else:
                    os.environ["CFB_COACH_MADDEN_DB"] = old
            self.assertEqual(code, 0)
            written = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual({row["snap_id"] for row in written}, {"snap-1", "snap-decision"})
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), before)

    def test_unlinked_snaps_and_a_missing_database_are_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "plain.db"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE snaps (id INTEGER PRIMARY KEY, play TEXT)")
            conn.execute("INSERT INTO snaps VALUES (1, 'Mesh')")
            conn.commit()
            conn.close()
            before = hashlib.sha256(db.read_bytes()).hexdigest()
            report = export_coach_log(db, "game-a")
            self.assertTrue(report["ok"])
            self.assertEqual(report["reason"], "snaps_not_linked_to_game")
            self.assertEqual(report["snaps"], [])
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), before)
            missing = Path(tmp) / "absent" / "madden.db"
            old = os.environ.get("CFB_COACH_MADDEN_DB")
            os.environ["CFB_COACH_MADDEN_DB"] = str(missing)
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    code = main(["ml", "film-export-log", "--game-id", "game-a", "--out", str(Path(tmp) / "nope.json")])
            finally:
                if old is None:
                    os.environ.pop("CFB_COACH_MADDEN_DB", None)
                else:
                    os.environ["CFB_COACH_MADDEN_DB"] = old
            self.assertEqual(code, 2)
            self.assertFalse(missing.parent.exists())
            self.assertFalse((Path(tmp) / "nope.json").exists())


class EvidencePathwayTests(unittest.TestCase):
    def test_approval_checks_game_and_opponent_and_keeps_post_snap_out(self):
        other_game = propose_admission(
            _confirmed_annotations(), [_log(game_id="game-b")], opponent_id="cpu",
        )
        self.assertEqual(other_game["proposals"][0]["reason"], "snap_not_in_requested_game")
        other_opponent = propose_admission(
            _confirmed_annotations(), [_log(opponent_id="human")], opponent_id="cpu",
        )
        self.assertEqual(other_opponent["proposals"][0]["reason"], "opponent_mismatch")
        missing_opponent = propose_admission(
            _confirmed_annotations(), [_log(opponent_id="")], opponent_id="cpu",
        )
        self.assertEqual(missing_opponent["proposals"][0]["reason"], "opponent_missing_on_log_snap")
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "film"
            post = propose_admission(
                _confirmed_annotations(observation_time="post_snap"),
                [_log()], opponent_id="cpu",
            )
            post_row = next(row for row in post["proposals"] if row.get("kind") == "defensive_observation")
            self.assertTrue(post_row["admit"])
            self.assertEqual(post_row["observation_time"], "post_snap")
            self.assertFalse(post_row["available_before_snap"])
            self.assertFalse(post_row["verified_execution"])
            execution = next(row for row in post["proposals"] if row.get("kind") == "execution")
            self.assertFalse(execution["admit"])
            self.assertEqual(execution["reason"], "video_cannot_create_verified_execution")
            approved = approve_admission(store, post)
            self.assertEqual(approved["contributed_to_learning"], [])
            self.assertEqual(approved["withheld_from_learning"][0]["evidence_id"], "c1:defense")
            learned = fit_opponent_model(records=load_admitted_records(store, game_id="game-a", opponent_id="cpu"))
            self.assertEqual(learned["approved_film_observations"], 0)
            self.assertEqual(learned["usable_verified_snaps"], 0)
            pre = propose_admission(_confirmed_annotations(), [_log()], opponent_id="cpu")
            again = approve_admission(store, pre)
            self.assertEqual(again["contributed_to_learning"], ["c1:defense"])
            admitted = load_admitted_records(store, game_id="game-a", opponent_id="cpu")
            self.assertEqual(len(admitted), 1)
            self.assertEqual(admitted[0]["game_id"], "game-a")
            self.assertEqual(admitted[0]["opponent_id"], "cpu")
            self.assertEqual(admitted[0]["snap_id"], "s1")
            self.assertEqual(admitted[0]["observation_time"], "pre_snap")
            self.assertEqual(admitted[0]["confidence"], 0.8)
            self.assertEqual(admitted[0]["provenance"], "human_confirmed_film")
            self.assertFalse(admitted[0]["verified_execution"])
            counted = fit_opponent_model(records=admitted)
            self.assertEqual(counted["approved_film_observations"], 1)
            pressure = next(row for row in counted["estimates"] if row["context"] == "pressure_observations")
            self.assertEqual(pressure["sample_size"], 1)
            self.assertEqual(load_admitted_records(store, game_id="game-b", opponent_id="cpu"), [])
            report = opponent_learning_report(
                None, opponent_id="cpu", game_id="game-a",
                film_store=str(store), include_admitted_film=True,
            )
            self.assertEqual(report["film_observations_contributing"], ["c1:defense"])
            self.assertEqual(report["film_observations_withheld"], [])
            self.assertEqual(report["approved_film_observations"], 1)
            film = film_report(store, "game-a")
            self.assertEqual(film["contributed_to_learning"], ["c1:defense"])
            self.assertIsNone(film["real_recording_evaluation"]["results"])
            self.assertFalse(film["real_recording_evaluation"]["recognition_accuracy_claim"])
            self.assertIn("missed_snaps", film["real_recording_evaluation"]["metrics_to_record"])
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main([
                    "ml", "film-approve", "--game-id", "game-a", "--store", str(store),
                    "--rollback", "--log", str(Path(tmp) / "missing-log.json"),
                ])
            self.assertEqual(code, 0)
            self.assertEqual(load_admitted_records(store, game_id="game-a"), [])
            withdrawn = fit_opponent_model(records=load_admitted_records(store, game_id="game-a"))
            self.assertEqual(withdrawn["approved_film_observations"], 0)


class SynchronizationTests(unittest.TestCase):
    def test_camera_cuts_are_not_snaps_and_irregular_times_are_kept(self):
        frames = [
            {"time_s": 0.0, "rgb": bytes([0] * 12)},
            {"time_s": 0.05, "rgb": bytes([255] * 12)},
        ]
        located = locate_candidate_snaps(frames)
        self.assertEqual(located["segments"][0]["role"], "possible_camera_cut")
        self.assertFalse(located["segments"][0]["snap_claim"])
        aligned = align_segments(located["segments"], [_log()])
        self.assertEqual(aligned["associations"][0]["reason"], "camera_or_menu_cut_not_a_snap")
        self.assertIsNone(aligned["associations"][0]["snap_id"])
        irregular = locate_candidate_snaps([
            {"time_s": 0.0, "rgb": bytes([0] * 12)},
            {"time_s": 0.1, "rgb": bytes([0] * 12)},
            {"time_s": 1.5, "rgb": bytes([255] * 12)},
        ])
        bounds = {(row["start_s"], row["end_s"]) for row in irregular["segments"]}
        self.assertIn((0.0, 1.5), bounds)
        self.assertTrue(all(row["snap_claim"] is False for row in irregular["segments"]))

    def test_variable_frame_rate_samples_keep_decoder_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vfr.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=red:s=64x36:r=10:d=0.2",
                    "-f", "lavfi", "-i", "color=c=blue:s=64x36:r=30:d=0.5",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
                    "-fps_mode", "vfr", str(path),
                ],
                check=True, capture_output=True,
            )
            sampled = sample_frames(path)
            self.assertTrue(sampled["ok"], sampled)
            self.assertEqual(sampled["timestamp_source"], "decoder_pts_time")
            times = [row["time_s"] for row in sampled["frames"]]
            self.assertGreaterEqual(len(times), 2)
            self.assertEqual(times, sorted(times))
            self.assertNotEqual(times, [round(index / 4, 3) for index in range(len(times))])

    def test_duplicate_import_does_not_reset_corrections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            store = Path(tmp) / "film"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=green:s=64x36:d=1:r=10", "-pix_fmt", "yuv420p", str(path)],
                check=True, capture_output=True,
            )
            first = import_recording(path, game_id="game-a", store=store, dry_run=False)
            self.assertTrue(first["ok"], first)
            corrected = apply_review_action(
                store, "game-a", first["segments"][0]["candidate_id"], "correct",
                {"formation": "Gun Bunch", "play": "Mesh"},
            )
            self.assertEqual(corrected["candidate"]["play"], "Mesh")
            second = import_recording(path, game_id="game-a", store=store, dry_run=False)
            self.assertTrue(second["duplicate"])
            stored = json.loads((store / "annotations" / "game-a.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["candidates"][0]["play"], "Mesh")


class PreservationTests(unittest.TestCase):
    def test_live_playcaller_does_not_import_film(self):
        source = inspect.getsource(experimental_live)
        self.assertNotIn("film_import", source)
        self.assertNotIn("film_review", source)
        self.assertNotIn("import_recording", source)


if __name__ == "__main__":
    unittest.main()
