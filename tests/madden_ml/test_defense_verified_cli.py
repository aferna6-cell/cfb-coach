"""Verified defense train/eval and unarmed-macro legality.

Synthetic rows only. The repo root comes from this file, not a fixed path.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.madden import data as mdata
from cfb_coach.madden.model import defense_coordinator as defense
from cfb_coach.madden.model.defense_verified import (
    MIN_CONCEPT_SEEN_CONFIDENCE,
    VERIFIED_DEFINITION,
    evaluate_adjustments,
    evaluate_defense,
    record_defensive_execution,
    training_view,
    verification_source,
)
from cfb_coach.madden.model.dual_control import install_primary_lions_books
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.situation import parse_madden_situation

_REPO = Path(__file__).resolve().parents[2]


def _env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["PYTHONPATH"] = str(_REPO)
    env.pop("CFB_COACH_DB", None)
    env.pop("CFB_COACH_MADDEN_DB", None)
    return env


def _run(argv: list[str], cwd: Path, home: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cfb_coach", *argv],
        cwd=cwd,
        env=_env(home),
        capture_output=True,
        text=True,
        check=False,
    )


def _books() -> dict:
    from cfb_coach.madden.playbook import make_record

    return {
        "offense": make_record("offense", "stock", "Buccaneers")["formations"],
        "defense": make_record("defense", "stock", "49ers")["formations"],
    }


def _live(match: str, play: str, success: bool, **extra: object) -> dict:
    row = {
        "side": "defense",
        "game_id": match,
        "match_id": match,
        "down": 3,
        "distance": 10,
        "executed_play": play,
        "success": success,
        "executed_verification": "verified",
        "executed_status": "identified",
        "accepted": "accepted",
        "opponent_id": "synth",
    }
    row.update(extra)
    return row


def _labeled_rows() -> list[dict]:
    rows: list[dict] = []
    for match in ("match-a", "match-b"):
        for _ in range(4):
            rows.append(_live(match, "Cover 4 Quarters", True))
            rows.append(_live(match, "Cover 2", False))
    rows.append({
        "side": "defense",
        "game_id": "vod-match",
        "match_id": "vod-match",
        "vod_id": "vod-match",
        "trusted_vod": True,
        "provenance": "vod:synthetic",
        "review_status": "reviewed",
        "permitted_use": "training",
        "trainable": True,
        "concept_seen": "Four Verticals",
        "concept_seen_confidence": 0.8,
        "executed_play": "Cover 4 Quarters",
        "success": True,
        "down": 3,
        "distance": 10,
    })
    rows.append({
        "side": "defense",
        "vod_id": "vod-skip",
        "trusted_vod": True,
        "review_status": "unreviewed",
        "concept_seen": "Mesh",
        "provenance": "vod:synthetic",
    })
    rows.append({
        "side": "defense",
        "recommendation_only": True,
        "executed_verification": "verified",
        "executed_status": "identified",
        "executed_play": "Cover 2",
        "success": True,
        "game_id": "match-a",
    })
    rows.append({
        "side": "offense",
        "executed_verification": "verified",
        "executed_status": "identified",
        "accepted": "accepted",
        "executed_play": "Inside Zone",
        "game_id": "match-a",
    })
    rows.append({
        "side": "defense",
        "trusted_vod": True,
        "vod_id": "F8Hm324SMtg",
        "source": "F8Hm324SMtg",
        "review_status": "reviewed",
        "permitted_use": "training",
        "trainable": True,
        "concept_seen": "Four Verticals",
        "game_id": "blocked",
    })
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_verified_definition_counts_live_and_reviewed_vod_only() -> None:
    rows = _labeled_rows()
    assert "concept_seen" in VERIFIED_DEFINITION
    assert "executed/accepted" in VERIFIED_DEFINITION
    assert verification_source(rows[0]) == "live"
    vod = next(row for row in rows if row.get("vod_id") == "vod-match")
    assert verification_source(vod) == "vod"
    assert verification_source(next(row for row in rows if row.get("vod_id") == "vod-skip")) is None
    assert verification_source(next(row for row in rows if row.get("recommendation_only"))) is None
    assert verification_source(next(row for row in rows if row.get("side") == "offense")) is None
    assert verification_source(next(row for row in rows if row.get("source") == "F8Hm324SMtg")) is None
    permitted = {
        "side": "defense",
        "trusted_vod": True,
        "vod_id": "vod-ok",
        "review_status": "unreviewed",
        "permitted_use": "training",
        "concept_seen": "Inside Zone",
        "concept_seen_confidence": 0.9,
    }
    assert verification_source(permitted) is None


def _vod(match: str, **extra: object) -> dict:
    row = {
        "side": "defense",
        "match_id": match,
        "video_id": f"clip-{match}",
        "trusted_vod": True,
        "review_status": "reviewed",
        "concept_seen": "Four Verticals",
        "concept_seen_confidence": MIN_CONCEPT_SEEN_CONFIDENCE,
        "observed_defense": "Cover 4 Quarters",
        "executed_play": "Cover 4 Quarters",
        "success": True,
        "down": 3,
        "distance": 10,
    }
    row.update(extra)
    return row


def test_vod_verified_requires_review_or_trainable_and_confidence() -> None:
    assert MIN_CONCEPT_SEEN_CONFIDENCE == 0.5
    assert verification_source(_vod("m", trainable=True)) == "vod"
    assert verification_source(_vod("m", review_status="confirmed", trainable=False)) == "vod"
    assert verification_source(_vod(
        "m", review_status="auto", trainable=True, concept_seen_confidence=0.8,
    )) == "vod"
    auto = _vod(
        "m", review_status="auto", trainable=False, concept_seen_confidence=0.3,
        concept_seen="Mesh",
    )
    assert verification_source(auto) is None
    assert verification_source(_vod("m", concept_seen_confidence=0.3)) is None
    assert verification_source(_vod("m", concept_seen_confidence=0.49)) is None
    missing = _vod("m")
    missing.pop("concept_seen_confidence")
    assert verification_source(missing) is None
    assert verification_source(_vod("m", excluded_from_training=True)) is None
    assert verification_source(_vod("m", excluded_from_training="true")) is None
    rights_only = _vod(
        "m", review_status="unreviewed", permitted_use="training", trainable=False,
    )
    rights_only.pop("concept_seen_confidence")
    assert verification_source(rights_only) is None
    nested = _vod("m", review_status="corrected", trainable=False)
    nested.pop("concept_seen_confidence")
    nested["field_confidence"] = {"concept_seen": 0.7}
    assert verification_source(nested) == "vod"


def test_train_defense_accepts_vod_rows_without_game_id() -> None:
    rows = []
    for match in ("vod-a", "vod-b"):
        for _ in range(4):
            rows.append(_vod(match, executed_play="Cover 4 Quarters", success=True))
            rows.append(_vod(match, executed_play="Cover 2", success=False, concept_seen="Inside Zone"))
    rows.append(_vod(
        "auto-match", review_status="auto", trainable=False,
        concept_seen_confidence=0.3, concept_seen="Mesh",
    ))
    assert "game_id" not in rows[0]
    assert rows[0]["video_id"] == "clip-vod-a"
    assert verification_source(rows[-1]) is None
    verified = [training_view(row) for row in rows if verification_source(row)]
    art = defense.train_defense(verified, preverified=True)
    assert art["verified_human_games"] == 2
    video_only = {
        "side": "defense", "video_id": "only-video",
        "executed_play": "Cover 2", "success": False,
    }
    solo = defense.train_defense([video_only], preverified=True)
    assert solo["verified_human_games"] == 1


def test_eval_reports_unlabelled_drops_and_baseline_fallback() -> None:
    labelled = []
    for match in ("m1", "m2"):
        labelled.append(_live(match, "Base Align", True))
        labelled.append(_live(match, "Base Align", False))
    missing = _live("m1", "Cover 2", True)
    missing.pop("success")
    garbled = _live("m2", "Cover 2", True)
    garbled["success"] = "maybe"
    report = evaluate_defense(labelled + [missing, garbled], seed=1, n_boot=30)
    assert report["n_dropped_unlabelled"] == 2
    assert report["unlabelled_drop_reasons"]["missing_success"] == 1
    assert report["unlabelled_drop_reasons"]["unparseable_success"] == 1
    assert report["situations"]
    for bucket in report["situations"].values():
        assert bucket["n_model_used"] == 0
        assert bucket["n_baseline_fallback"] == bucket["n"]
        assert bucket["diff"] == 0.0


def test_train_defense_refuses_zero_and_does_not_enable(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    out = tmp_path / "defense.json"
    empty = tmp_path / "empty.jsonl"
    _write_jsonl(empty, [{
        "side": "defense", "concept_seen": "Mesh", "trusted_vod": True,
        "review_status": "unreviewed", "game_id": "g",
    }])
    refused = _run(
        ["ml", "train-defense", "--out", str(out), "--path", str(empty), "--seed", "7"],
        tmp_path, home,
    )
    assert refused.returncode != 0
    assert "verified_live: 0" in refused.stdout
    assert "verified_vod: 0" in refused.stdout
    assert "zero verified" in refused.stderr
    assert not out.exists()

    rows = tmp_path / "snaps.jsonl"
    _write_jsonl(rows, _labeled_rows())
    trained = _run(
        ["ml", "train-defense", "--out", str(out), "--path", str(rows), "--seed", "7"],
        tmp_path, home,
    )
    assert trained.returncode == 0, trained.stderr
    assert "verified_live: 16" in trained.stdout
    assert "verified_vod: 1" in trained.stdout
    assert "defense_mode: off" in trained.stdout
    assert "auto_promote: false" in trained.stdout
    assert VERIFIED_DEFINITION in trained.stdout
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["verified_snaps"] == 17
    assert art["auto_promote"] is False
    assert art["eval_report"]["gate"] == "pass"
    assert art["eval_report"]["ci_excludes_0"] is True
    assert art["eval_report"]["beats_baseline"] is True
    assert art["eval_report"]["p95_ms"] < 150
    assert art["eval_report"]["n_dropped_unlabelled"] == 0
    assert art["eval_report"]["situations"]
    for bucket in art["eval_report"]["situations"].values():
        assert "n_model_used" in bucket
        assert bucket["n_model_used"] + bucket["n_baseline_fallback"] == bucket["n"]
        assert bucket["n_model_used"] > 0

    evaluated = _run(
        ["ml", "eval-defense", "--model", str(out), "--path", str(rows), "--seed", "7"],
        tmp_path, home,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert "gate: pass" in evaluated.stdout
    assert "auto_promote: false" in evaluated.stdout
    assert "dropped_unlabelled: 0" in evaluated.stdout


def test_enable_refuses_without_a_passing_report(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    db_path = tmp_path / "madden.db"
    db = CoachDB(db_path, seed=mdata.load_seed())
    try:
        install_primary_lions_books(db)
        bare = tmp_path / "bare.json"
        defense.save_model(defense.train_defense([], preverified=True), bare)
        db.set_meta(defense.MODEL_META, str(bare))
        assert defense.mode(db) == "off"
    finally:
        db.close()
    refused = _run(
        ["ml", "defense-experimental", "--enable", "--db", str(db_path)],
        tmp_path, home,
    )
    assert refused.returncode != 0
    assert "no eval report" in refused.stderr
    db = CoachDB(db_path, seed=mdata.load_seed())
    try:
        assert defense.mode(db) == "off"
        art = json.loads(bare.read_text(encoding="utf-8"))
        art["eval_report"] = {
            "beats_baseline": True,
            "ci_excludes_0": True,
            "ci95": [0.01, 0.2],
            "p95_ms": 1.0,
            "p95_under_150": True,
            "gate": "pass",
        }
        bare.write_text(json.dumps(art), encoding="utf-8")
    finally:
        db.close()
    enabled = _run(
        ["ml", "defense-experimental", "--enable", "--db", str(db_path)],
        tmp_path, home,
    )
    assert enabled.returncode == 0, enabled.stderr
    db = CoachDB(db_path, seed=mdata.load_seed())
    try:
        assert defense.mode(db) == "experimental"
    finally:
        db.close()


def test_established_caller_drops_unarmed_qtrs_and_safe(tmp_path: Path) -> None:
    db = CoachDB(tmp_path / "madden.db", seed=mdata.load_seed())
    try:
        armed = {"offense": ["MAN"], "defense": ["TAMPA MABLE"]}
        books = _books()
        for _ in range(2):
            db.log_snap(
                opponent_id="gavin", side="defense", situation_raw="d 2&6",
                our_call="x", formation="Nickel", play="Cover 4 Quarters",
                result="+20", concept_seen="Four Verticals",
            )
        verts = make_call(
            parse_madden_situation("d 2&6 showing 4 verts"), "gavin", db,
            rng=random.Random(1), playbook=books, active_macros=armed,
        )
        shown = verts.format()
        assert "QTRS OVERTOP" not in shown
        assert "SAFE DEEP" not in shown
        assert verts.macro != "QTRS OVERTOP"
        assert "dropped_unarmed_macro:QTRS OVERTOP" in (verts.adjustment_dropped or "")

        lead = make_call(
            parse_madden_situation("d 1&10 score 24-10 q4"), "james", db,
            rng=random.Random(1), playbook=books, active_macros=armed,
        )
        assert "SAFE DEEP" not in lead.format()
        assert "QTRS OVERTOP" not in lead.format()
        assert lead.macro != "SAFE DEEP"
        assert "dropped_unarmed_macro:SAFE DEEP" in (lead.adjustment_dropped or "")

        armed_deep = {"offense": ["MAN"], "defense": ["TAMPA MABLE", "SAFE DEEP"]}
        kept = make_call(
            parse_madden_situation("d 1&10 score 24-10 q4"), "james", db,
            rng=random.Random(1), playbook=books, active_macros=armed_deep,
        )
        assert kept.macro == "SAFE DEEP"
    finally:
        db.close()


def test_model_and_offense_drop_unarmed_macros(tmp_path: Path, monkeypatch) -> None:
    db = CoachDB(tmp_path / "madden.db", seed=mdata.load_seed())
    try:
        defense.set_mode(db, "experimental")
        books = _books()
        armed = {"offense": ["MAN"], "defense": ["TAMPA MABLE"]}

        def _plan(*_args, **_kwargs):
            return {
                "kind": "macro", "id": "QTRS OVERTOP",
                "macro": {"name": "QTRS OVERTOP", "settings": []},
                "why": "synthetic", "influence": 1.0, "executable": True,
            }

        monkeypatch.setattr(
            "cfb_coach.madden.model.defense_action_policy.select_defensive_action",
            _plan,
        )
        call = make_call(
            parse_madden_situation("d 1&10"), "gavin", db,
            rng=random.Random(0), playbook=books, active_macros=armed,
        )
        assert "QTRS OVERTOP" not in call.format()
        assert call.macro != "QTRS OVERTOP"
        assert "dropped_unarmed_macro:QTRS OVERTOP" in (call.adjustment_dropped or "")

        def _offense(**_kwargs):
            return {"id": "O-RUN", "name": "O-RUN", "why": "synthetic", "buttons": "LB → O-RUN"}

        monkeypatch.setattr(
            "cfb_coach.madden.offense_macros.suggest_for_snap", _offense,
        )
        offense = make_call(
            parse_madden_situation("1&10"), "gavin", None,
            rng=random.Random(0), playbook=books, active_macros={"offense": ["MAN"], "defense": []},
        )
        assert offense.macro != "O-RUN"
        assert "O-RUN" not in offense.format()
        assert "dropped_unarmed_macro:O-RUN" in (offense.adjustment_dropped or "")
    finally:
        db.close()


def test_defense_accept_execute_is_optional_and_defaults_unknown(tmp_path: Path) -> None:
    from cfb_coach.live_server import LivePlayController

    bare = record_defensive_execution({
        "side": "defense", "game_id": "g", "executed_play": "Cover 2",
    })
    assert bare["accepted"] == "unknown"
    assert bare["executed_status"] == "unknown"
    assert "executed_adjustment_id" not in bare
    assert verification_source(bare) is None

    stamped = record_defensive_execution(
        {
            "side": "defense", "game_id": "g", "play": "Cover 2",
            "macro": "TAMPA MABLE",
        },
        executed_status="used_recommended",
        executed_play="Cover 2",
        recommended_play="Cover 2",
        touched=True,
        adjustment_confirmed=True,
        executed_adjustment_id="TAMPA MABLE",
    )
    assert stamped["accepted"] == "true"
    assert stamped["executed_status"] == "identified"
    assert stamped["executed_verification"] == "verified"
    assert stamped["executed_adjustment_id"] == "TAMPA MABLE"
    assert verification_source(stamped) == "live"

    class _DefenseCall:
        side = "defense"
        formation = "Nickel"
        play = "Cover 2"
        macro = "TAMPA MABLE"
        macro_info = {"name": "TAMPA MABLE", "buttons": "LB → TAMPA MABLE", "why": "synth"}
        adjustment = None

        def format(self) -> str:
            return "Nickel — Cover 2"

    db = CoachDB(tmp_path / "madden.db", seed=mdata.load_seed())
    try:
        ctrl = LivePlayController(
            db=db,
            opponent_id="synth",
            make_call=lambda _sit, **_kwargs: _DefenseCall(),
            parse_situation=parse_madden_situation,
            learn_summary=lambda: "",
            brand="Madden 27 Franchise",
            cpu_only=False,
            enable_execution_verify=True,
            session_id="synth-defense",
        )
        opened = ctrl.call_only("d 1&10", side="defense")
        assert opened["ok"]
        assert ctrl._pending_defense_action()["id"] == "TAMPA MABLE"
        assert ctrl._pending_offense_action() is None
        default = ctrl.result_and_call(
            outcome="+3", sit_raw="d 2&7", side="defense",
        )
        assert default["ok"]
        snap = db.conn.execute("SELECT * FROM snaps ORDER BY id DESC LIMIT 1").fetchone()
        assert snap["accepted"] == "unknown"
        assert snap["executed_status"] == "unknown"
        assert snap["executed_adjustment_id"] is None

        ctrl.call_only("d 2&7", side="defense")
        opted = ctrl.result_and_call(
            outcome="+1", sit_raw="d 3&6", side="defense",
            executed_status="used_recommended",
            applied_recommended_action=True,
            verify_touched=True,
        )
        assert opted["ok"]
        confirmed = db.conn.execute("SELECT * FROM snaps ORDER BY id DESC LIMIT 1").fetchone()
        assert confirmed["accepted"] == "true"
        assert confirmed["executed_status"] == "identified"
        assert confirmed["executed_adjustment_id"] == "TAMPA MABLE"
        assert verification_source(dict(confirmed)) == "live"
    finally:
        db.close()


def test_eval_adjustments_unknown_and_small_n() -> None:
    rows = [
        {"side": "defense", "game_id": "m1", "recommended_adjustment_id": "SAFE DEEP"},
        {"side": "defense", "game_id": "m1", "macro": "TAMPA MABLE", "success": True},
        {
            "side": "offense", "game_id": "m1", "recommended_adjustment_id": "O-RUN",
            "accepted": "accepted", "executed_adjustment_id": "O-RUN", "success": True,
        },
        {
            "side": "offense", "game_id": "m2", "recommended_adjustment_id": "MAN",
            "accepted": "rejected", "success": False,
        },
    ]
    report = evaluate_adjustments(rows, seed=1, n_boot=50)
    defense_side = report["sides"]["defense"]
    offense_side = report["sides"]["offense"]
    assert defense_side["applied"] == "unknown"
    assert defense_side["outcome"] == "unknown"
    assert defense_side["recommended"] == 2
    assert offense_side["applied"] == 1
    assert offense_side["outcome"] != "unknown"
    assert any(line.startswith("small-n warning:") for line in report["warnings"])
    text = json.dumps(report)
    assert "unknown" in text
    home_note = report["definition"]
    assert "unknown" in home_note
