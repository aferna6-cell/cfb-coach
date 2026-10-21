"""Shadow offense train/eval, opponent profiles, and snap-log gaps.

Synthetic rows only. The repo root comes from this file. Nothing here writes
a database, a trained artifact, or a real game into the repository.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import offense_shadow
from cfb_coach.madden.model.call_outcome_log import NO_RESULT, NO_SNAP_RECORDED
from cfb_coach.madden.model.dataset import EXCLUDED_VOD_SOURCE
from cfb_coach.madden.model.inference import resolve_mode
from cfb_coach.madden.model.schema import CoachingMode
from cfb_coach.session import start_session

ROOT = Path(__file__).resolve().parents[2]


def _env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["CFB_COACH_DB"] = str(home / "coach.db")
    env["CFB_COACH_MADDEN_DB"] = str(home / "madden.db")
    env["CFB_COACH_AI_RESEARCH"] = "off"
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not prior else str(ROOT) + os.pathsep + prior
    return env


def _cli(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cfb_coach", *args],
        cwd=home,
        env=_env(home),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


def _live(play: str, yards: int, *, kind: str, match: str, player: str) -> dict:
    return {
        "side": "offense",
        "down": 1,
        "distance": 10,
        "yardline": 25,
        "yards": yards,
        "result": f"+{yards}",
        "match_id": match,
        "player_id": player,
        "session_id": match,
        "opponent_id": "cpu-synth" if kind == "cpu" else "human-synth",
        "opponent_type": "cpu" if kind == "cpu" else "human",
        "executed_status": "identified",
        "executed_verification": "verified",
        "executed_formation": "Gun",
        "executed_play": play,
    }


def _vod(play: str, yards: int, version: str, *, match: str, player: str, vod_id: str) -> dict:
    return {
        "side": "offense",
        "down": 1,
        "distance": 10,
        "yardline": 55,
        "yards": yards,
        "result": f"+{yards}",
        "trusted_vod": True,
        "label_version": version,
        "vod_id": vod_id,
        "match_id": match,
        "player_id": player,
        "recommended_formation": "Gun",
        "recommended_play": play,
        "executed_formation": "Gun",
        "executed_play": play,
        "executed_status": "identified",
        "executed_verification": "verified",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _opponent(db: CoachDB, oid: str) -> None:
    db.conn.execute(
        """
        INSERT OR REPLACE INTO opponents
            (id, display_name, team_now, skill, confidence, profile_json)
        VALUES (?, ?, NULL, NULL, NULL, ?)
        """,
        (oid, oid, json.dumps({"id": oid, "display_name": oid})),
    )
    db.conn.commit()


def _tendencies(db: CoachDB) -> list[tuple]:
    return [
        tuple(row)
        for row in db.conn.execute(
            "SELECT opponent_id, bucket, key, count, success FROM tendencies ORDER BY 1, 2, 3"
        )
    ]


def test_repo_root_comes_from_this_file() -> None:
    assert offense_shadow.repo_root() == ROOT
    assert ROOT == Path(__file__).resolve().parents[2]


def test_success_rate_thresholds_and_turnovers() -> None:
    base = {"down": 1, "distance": 10, "yards": 4, "result": "+4"}
    assert offense_shadow.snap_success(base) is True
    assert offense_shadow.snap_success({**base, "yards": 3, "result": "+3"}) is False
    assert offense_shadow.snap_success({"down": 2, "distance": 10, "yards": 6, "result": "+6"}) is True
    assert offense_shadow.snap_success({"down": 2, "distance": 10, "yards": 5, "result": "+5"}) is False
    assert offense_shadow.snap_success({"down": 3, "distance": 10, "yards": 10, "result": "+10"}) is True
    assert offense_shadow.snap_success({"down": 3, "distance": 10, "yards": 9, "result": "+9"}) is False
    assert offense_shadow.snap_success({"down": 4, "distance": 2, "yards": 2, "result": "+2"}) is True
    assert offense_shadow.snap_success(
        {"down": 1, "distance": 10, "yards": 20, "result": "interception"}
    ) is False


def test_sources_keep_verified_live_and_vod_v034(tmp_path: Path) -> None:
    rows = [
        _live("Mesh", 8, kind="human", match="h1", player="p1"),
        _live("Mesh", 8, kind="cpu", match="c1", player="p2"),
        _vod("Mesh", 8, "v0.34", match="v1", player="p3", vod_id="clip-a"),
        _vod("Mesh", 8, "v0.33", match="v0", player="p4", vod_id="clip-b"),
        _vod("Mesh", 8, "v0.40", match="vx", player="p5", vod_id=EXCLUDED_VOD_SOURCE),
        {
            **_live("Mesh", 8, kind="human", match="h2", player="p6"),
            "data_source": "session_close",
        },
        {
            "side": "offense",
            "down": 1,
            "distance": 10,
            "yardline": 25,
            "yards": 8,
            "result": "+8",
            "recommended_formation": "Gun",
            "recommended_play": "Mesh",
            "executed_status": "unknown",
            "executed_verification": "unknown",
        },
    ]
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, rows)
    from cfb_coach.madden.model.dataset import import_path

    chosen, counts = offense_shadow.select_training_rows(import_path(str(path)))
    assert counts == {"live_human": 1, "live_cpu": 1, "vod": 1}
    assert {row["training_source"] for row in chosen} == {"live_human", "live_cpu", "vod"}
    assert all(row.get("label_version") != "v0.33" for row in chosen)


def test_situation_call_shrinks_toward_the_situation_model() -> None:
    rows = []
    for index in range(24):
        rows.append(_live("Mesh", 8, kind="human", match="same", player="p"))
        rows.append(_live("Stretch", 1, kind="human", match="same", player="p"))
        rows[-2]["snap_id"] = f"mesh-{index}"
        rows[-1]["snap_id"] = f"stretch-{index}"
    artifact = offense_shadow.fit_artifact(rows)
    probe = {"down": 1, "distance": 10, "yardline": 25, "call_key": "mesh"}
    situation = offense_shadow.predict_situation(artifact, probe)
    mesh = offense_shadow.predict_success(artifact, {**probe, "call_key": "mesh"})
    stretch = offense_shadow.predict_success(artifact, {**probe, "call_key": "stretch"})
    assert situation == pytest.approx(
        offense_shadow.predict_situation(artifact, {**probe, "call_key": "stretch"})
    )
    assert mesh > stretch
    assert 0.05 < mesh < 0.95
    assert 0.05 < stretch < 0.95
    assert artifact["shadow_only"] is True
    assert artifact["auto_promote"] is False
    assert artifact["heuristic_remains_default"] is True
    interactions = (artifact["full_model"]["coef"].get("down_x_call") or {})
    assert "1|mesh" in interactions
    assert "1|stretch" in interactions


def test_holdout_keeps_a_player_together_and_opponents_of_matches_apart() -> None:
    rows = [
        {"match_id": "m1", "player_id": "p1"},
        {"match_id": "m2", "player_id": "p1"},
        {"match_id": "m3", "player_id": "p2"},
        {"match_id": "m4"},
    ]
    groups = offense_shadow.holdout_groups(rows)
    assert len(groups) == 3
    joined = {tuple(sorted(group)) for group in groups}
    assert (0, 1) in joined
    assert (2,) in joined
    assert (3,) in joined


def test_save_artifact_refuses_the_repo(tmp_path: Path) -> None:
    artifact = offense_shadow.fit_artifact([
        _live("Mesh", 8, kind="human", match="m", player="p"),
    ])
    inside = ROOT / "tests" / "madden_ml" / "offense_shadow_refused.json"
    try:
        with pytest.raises(ValueError, match="inside the repo"):
            offense_shadow.save_artifact(artifact, inside)
        assert not inside.exists()
    finally:
        inside.unlink(missing_ok=True)
    written = offense_shadow.save_artifact(artifact, tmp_path / "offense.json")
    assert not offense_shadow.inside_repo(written)
    assert written.is_file()


def test_train_offense_prints_counts_and_refuses_zero_rows(tmp_path: Path) -> None:
    empty = _cli(tmp_path, "ml", "train-offense", "--seed", "7")
    assert empty.returncode == 1
    assert "training_rows live_human: 0" in empty.stdout
    assert "training_rows live_cpu: 0" in empty.stdout
    assert "training_rows vod: 0" in empty.stdout
    assert "zero verified offense rows" in empty.stderr
    assert not (tmp_path / "madden.db").exists()

    vod = [
        _vod("Mesh", 8, "v0.34", match="vod-a", player="viewer", vod_id="clip-new"),
        _vod("Stretch", 1, "v0.34", match="vod-a", player="viewer", vod_id="clip-new"),
        _vod("Mesh", 8, "v0.33", match="vod-old", player="viewer", vod_id="clip-old"),
        _vod("Mesh", 8, "v0.34", match="vod-x", player="viewer", vod_id=EXCLUDED_VOD_SOURCE),
    ]
    vod_path = tmp_path / "vod.jsonl"
    _write_jsonl(vod_path, vod)
    db_path = tmp_path / "live.db"
    db = CoachDB(db_path)
    try:
        _opponent(db, "human-synth")
        _opponent(db, "cpu-synth")
        for play, yards, oid in (
            ("Mesh", 8, "human-synth"),
            ("Stretch", 1, "human-synth"),
            ("Mesh", 8, "cpu-synth"),
        ):
            db.log_snap(
                opponent_id=oid,
                side="offense",
                situation_raw="1&10 my 25",
                our_call=play,
                down=1,
                distance=10,
                yardline=25,
                result=f"+{yards}",
                yards=yards,
                session_id=f"sess-{oid}",
                executed_status="identified",
                executed_verification="verified",
                executed_formation="Gun",
                executed_play=play,
            )
        assert resolve_mode(db) is CoachingMode.HEURISTIC
    finally:
        db.close()

    out = tmp_path / "out" / "offense.json"
    trained = _cli(
        tmp_path,
        "ml",
        "train-offense",
        "--seed",
        "7",
        "--db",
        str(db_path),
        "--path",
        str(vod_path),
        "--out",
        str(out),
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    assert "training_rows live_human: 2" in trained.stdout
    assert "training_rows live_cpu: 1" in trained.stdout
    assert "training_rows vod: 2" in trained.stdout
    assert "shadow_only: true" in trained.stdout
    assert "auto_promoted: false" in trained.stdout
    assert "heuristic_remains_default: true" in trained.stdout
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["kind"] == offense_shadow.MODEL_KIND
    assert artifact["auto_promote"] is False
    assert not offense_shadow.inside_repo(out)

    refused = ROOT / "tests" / "madden_ml" / "offense_shadow_refused.json"
    try:
        blocked = _cli(
            tmp_path,
            "ml",
            "train-offense",
            "--seed",
            "7",
            "--path",
            str(vod_path),
            "--out",
            str(refused),
        )
        assert blocked.returncode == 2
        assert "inside the repo" in blocked.stderr
        assert not refused.exists()
    finally:
        refused.unlink(missing_ok=True)

    checked = CoachDB(db_path)
    try:
        assert resolve_mode(checked) is CoachingMode.HEURISTIC
    finally:
        checked.close()


def test_eval_offense_writes_a_gate_report_without_promoting(tmp_path: Path) -> None:
    rows = []
    for match, player in (("match-a", "player-a"), ("match-b", "player-b")):
        for _ in range(8):
            rows.append(_live("Mesh", 9, kind="human", match=match, player=player))
            rows.append(_live("Stretch", 1, kind="human", match=match, player=player))
    path = tmp_path / "eval.jsonl"
    _write_jsonl(path, rows)
    out = tmp_path / "report"
    result = _cli(
        tmp_path,
        "ml",
        "eval-offense",
        "--seed",
        "7",
        "--path",
        str(path),
        "--out",
        str(out),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "auto_promoted: False" in result.stdout
    assert "heuristic_remains_default: true" in result.stdout
    report = json.loads((out / "offense_eval.json").read_text(encoding="utf-8"))
    markdown = (out / "offense_eval.md").read_text(encoding="utf-8")
    assert report["auto_promoted"] is False
    assert report["heuristic_remains_default"] is True
    assert report["holdout"] == "match_and_player"
    assert report["n_groups"] == 2
    gate = report["promotion_gate"]
    assert gate["auto_promoted"] is False
    assert set(gate) >= {
        "passed",
        "beats_situation_only",
        "beats_prior_driven",
        "ci_excludes_zero",
        "p95_latency_ms",
        "p95_under_150ms",
        "reason",
    }
    assert gate["p95_under_150ms"] is True
    assert gate["p95_latency_ms"] < 150
    for name in ("artifact", "situation_only", "prior_driven"):
        block = report["metrics"][name]
        assert set(block) >= {"log_loss", "top1", "top3", "success_rate", "n"}
        assert block["n"] == 32
    assert report["deltas"]["vs_situation_only"]["ci95"][0] is not None
    assert report["per_situation"]
    assert "Shadow only" in markdown
    assert "does not promote" in markdown
    assert "Per situation" in markdown



# Cross-opponent fold and synthetic session-close tests remain in PR #77
# pending reconciliation with the newer PR #80/81 session logging semantics.
