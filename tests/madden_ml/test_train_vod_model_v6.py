"""Synthetic-fixture tests for the v6 VOD trainer. No real label rows."""

from __future__ import annotations

import json

import pytest

from cfb_coach.vod_success.train_vod_model_v6 import (
    CONF_MIN,
    CXS_GROUPS,
    DEFAULT_HP,
    DEFAULT_LAMBDA_CXS,
    LAMBDA_CXS_GRID,
    MIN_CXS_ROWS,
    SIT_GROUPS,
    GroupLR,
    apply_nice,
    build_parser,
    call_class,
    ci_below_zero,
    constant_pick_check,
    dist_bucket,
    exclusion_reason,
    group_penalty,
    leave_one_group_predictions,
    logloss,
    main,
    map_items,
    norm_key,
    prep,
    prepare_row,
    row_features,
    select_lambda_cxs,
    split_excluded,
    time_bucket,
    train,
    v6_beats_v5,
    zone_bucket,
)

TINY_HP = ((2.0, 16.0, 64.0),)
TINY_LAMBDA = (64.0, 256.0)


def _conf(**overrides: float) -> dict[str, float]:
    base = {
        "play": 0.95,
        "def_call": 0.95,
        "coverage_seen": 0.95,
        "def_overlay": 0.95,
        "success": 0.9,
    }
    base.update(overrides)
    return base


def _row(**overrides):
    row = {
        "snap_id": "s",
        "video_id": "vod-1",
        "match_id": "m1",
        "player_id": "p1",
        "game": "madden27",
        "opponent_type": "cpu",
        "side": "offense",
        "success": True,
        "down": 1,
        "distance": 10,
        "yardline": 25,
        "quarter": 1,
        "clock": "12:00",
        "score_left": 7,
        "score_right": 7,
        "offense_call": {"play": "Mesh"},
        "defense_look": {
            "def_call": "Cover 3",
            "coverage_seen": "Cover 2",
            "def_overlay": "Cover 4",
        },
        "conf": _conf(),
        "clean_pair": True,
        "review_status": "unreviewed",
        "permitted_use": "training",
    }
    for key, value in overrides.items():
        if key in ("offense_call", "defense_look", "conf", "adjustments") and isinstance(value, dict):
            merged = dict(row.get(key) or {})
            merged.update(value)
            row[key] = merged
        else:
            row[key] = value
    return row


def _manual_keys(rows, min_rows=MIN_CXS_ROWS):
    """Count call|bucket keys without the trainer's counter."""
    counts = {group: {} for group in CXS_GROUPS}
    buckets = {
        "call_x_down": "down",
        "call_x_dist": "dist",
        "call_x_zone": "zone",
    }
    for raw in rows:
        prepared = prepare_row(raw)
        call = prepared["call"]
        if not call or prepared["y"] is None:
            continue
        for group, field in buckets.items():
            key = f"{call}|{prepared[field]}"
            counts[group][key] = counts[group].get(key, 0) + 1
    return {
        group: sorted(key for key, count in counts[group].items() if count >= min_rows)
        for group in CXS_GROUPS
    }


def _integration_rows():
    rows = []
    snap = 0

    def add(play, match, player, down, distance, yardline, success):
        nonlocal snap
        snap += 1
        rows.append(_row(
            snap_id=f"s{snap}",
            video_id=f"vod-{match}",
            match_id=match,
            player_id=player,
            offense_call={"play": play},
            down=down,
            distance=distance,
            yardline=yardline,
            success=success,
        ))

    # HB Dive: 2 per match and 2 per player, so every fold still has >= 5.
    players = ("p1", "p2", "p3", "p4")
    i = 0
    for match in ("m1", "m2", "m3", "m4"):
        for _k in range(2):
            add("HB Dive", match, players[i % 4], 2, 10, 25, i % 2 == 0)
            i += 1
    # Mesh: 5 rows, two of them in m1. No match-fold training set reaches 5.
    add("Mesh", "m1", "p1", 1, 1, 85, True)
    add("Mesh", "m1", "p2", 1, 1, 85, False)
    add("Mesh", "m2", "p3", 1, 1, 85, True)
    add("Mesh", "m3", "p4", 1, 1, 85, False)
    add("Mesh", "m4", "p1", 1, 1, 85, True)
    # Slant: 5 rows, all player p1. Holding p1 out drops the key; holding p2 out keeps it.
    for match, success in (("m1", True), ("m2", False), ("m3", True), ("m4", False), ("m1", True)):
        add("Slant", match, "p1", 3, 15, 10, success)
    for match_i, match in enumerate(("m1", "m2", "m3", "m4")):
        add("Curl", match, players[match_i], 1, 8, 55, True)
        add("Curl", match, players[(match_i + 1) % 4], 1, 8, 55, False)
    rows.append(_row(
        snap_id="blocked-vod",
        video_id="F8Hm324SMtg",
        match_id="m-blocked",
        player_id="p9",
        offense_call={"play": "Blocked Play"},
    ))
    rows.append(_row(
        snap_id="blocked-url",
        video_id="clean-vod",
        url="https://example.test/watch?v=F8Hm324SMtg",
        match_id="m-url",
        offense_call={"play": "Blocked Url Play"},
    ))
    rows.append(_row(
        snap_id="audit-row",
        video_id="vod-audit",
        match_id="audit-77",
        game_id="audit-77",
        player_id="p9",
        offense_call={"play": "Audit Play"},
    ))
    rows.append(_row(
        snap_id="cfb-row",
        video_id="vod-cfb",
        match_id="cfb-m",
        player_id="p-cfb",
        game="cfb27",
        offense_call={"play": "Inside Zone"},
    ))
    rows.append(_row(
        snap_id="human-row",
        video_id="vod-human",
        match_id="human-m",
        player_id="p-human",
        opponent_type="human",
        offense_call={"play": "Human Only"},
    ))
    rows.append(_row(
        snap_id="defense-row",
        video_id="vod-def",
        match_id="def-m",
        player_id="p-def",
        side="defense",
        offense_call={"play": "Defense Only"},
    ))
    rows.append(_row(
        snap_id="h2h-row",
        video_id="vod-h2h",
        match_id="h2h-m",
        player_id="p-h2h",
        opponent_type="h2h",
        offense_call={"play": "H2H Only"},
    ))
    return rows


@pytest.fixture(scope="module")
def v6_report():
    return train(
        _integration_rows(),
        audit_game_ids=["audit-77"],
        workers=1,
        hp_grid=TINY_HP,
        lambda_cxs_grid=TINY_LAMBDA,
        n_boot=30,
        min_groups=2,
    )


def test_min_count_is_training_fold_only():
    assert MIN_CXS_ROWS == 5
    rows = [
        _row(snap_id=f"t{i}", match_id="train", player_id="p", down=1, distance=1, yardline=85)
        for i in range(4)
    ]
    held = [_row(snap_id="held", match_id="held", player_id="p", down=1, distance=1, yardline=85)]
    train_only = [prepare_row(row) for row in rows]
    both = train_only + [prepare_row(row) for row in held]
    short = GroupLR(DEFAULT_HP, enable_cxs=True, lambda_cxs=64).fit(train_only)
    full = GroupLR(DEFAULT_HP, enable_cxs=True, lambda_cxs=64).fit(both)
    assert "mesh|1" not in short.allowed_cxs["call_x_down"]
    assert "mesh|short" not in short.allowed_cxs["call_x_dist"]
    assert "mesh|red_zone" not in short.allowed_cxs["call_x_zone"]
    assert short.allowed_cxs["call_x_down"] == set()
    assert "mesh|1" in full.allowed_cxs["call_x_down"]
    assert "mesh|short" in full.allowed_cxs["call_x_dist"]
    assert "mesh|red_zone" in full.allowed_cxs["call_x_zone"]
    exact = [
        _row(snap_id=f"e{i}", match_id="train", down=1, distance=10, yardline=25)
        for i in range(5)
    ]
    fitted = GroupLR(DEFAULT_HP, enable_cxs=True, lambda_cxs=64).fit([prepare_row(row) for row in exact])
    assert fitted.allowed_cxs["call_x_down"] == {"mesh|1"}
    assert fitted.allowed_cxs["call_x_dist"] == {"mesh|long"}
    assert fitted.allowed_cxs["call_x_zone"] == {"mesh|own"}


def test_report_fold_counts_ignore_held_out_rows(v6_report):
    stratum = v6_report["strata"]["madden27/cpu/offense"]
    assert "mesh|1" in stratum["keys_survived"]["call_x_down"]
    assert "mesh|short" in stratum["keys_survived"]["call_x_dist"]
    assert "mesh|red_zone" in stratum["keys_survived"]["call_x_zone"]
    assert "hb dive|2" in stratum["keys_survived"]["call_x_down"]
    assert stratum["n_keys_survived"]["call_x_down"] == len(stratum["keys_survived"]["call_x_down"])
    kept = [
        row for row in _integration_rows()
        if exclusion_reason(row, ["audit-77"]) is None and row["game"] == "madden27"
        and row["opponent_type"] == "cpu" and row["side"] == "offense"
    ]
    manual_all = _manual_keys(kept)
    assert manual_all["call_x_down"] == stratum["keys_survived"]["call_x_down"]
    match_folds = {fold["held_out"]: fold for fold in stratum["holdouts"]["match"]["folds"]}
    for held, fold in match_folds.items():
        manual = _manual_keys([row for row in kept if row["match_id"] != held])
        assert fold["keys_survived"]["call_x_down"] == manual["call_x_down"]
        assert fold["keys_survived"]["call_x_dist"] == manual["call_x_dist"]
        assert fold["keys_survived"]["call_x_zone"] == manual["call_x_zone"]
        assert "mesh|1" not in fold["keys_survived"]["call_x_down"]
        assert "hb dive|2" in fold["keys_survived"]["call_x_down"]
        assert fold["n_keys_survived"]["call_x_dist"] == len(fold["keys_survived"]["call_x_dist"])
        assert set(fold["n_keys_survived"]) == set(CXS_GROUPS)
    listed = stratum["holdouts"]["match"]["n_keys_survived_by_fold"]
    assert [item["held_out"] for item in listed] == [fold["held_out"] for fold in stratum["holdouts"]["match"]["folds"]]
    player_folds = {fold["held_out"]: fold for fold in stratum["holdouts"]["player"]["folds"]}
    assert "slant|3" in stratum["keys_survived"]["call_x_down"]
    assert "slant|3" not in player_folds["p1"]["keys_survived"]["call_x_down"]
    assert "slant|3" in player_folds["p2"]["keys_survived"]["call_x_down"]
    manual_p2 = _manual_keys([row for row in kept if row["player_id"] != "p2"])
    assert player_folds["p2"]["keys_survived"] == manual_p2


def test_disable_flag_drops_keyed_groups_on_v024_features():
    raw = _row(
        down=1,
        distance=10,
        yardline=25,
        quarter=1,
        clock="12:00",
        score_left=7,
        score_right=7,
        offense_call={"play": "Mesh"},
        defense_look={"def_call": "Cover 3 Blitz", "coverage_seen": "Cover 2", "def_overlay": "Nickel"},
        adjustments={
            "audible": {"value": "hot"},
            "def_adjustments": {"value": "shade"},
        },
    )
    prepared = prep(raw)
    # def_call never feeds the look; score/off_adj/def_adj are dropped (recipe O2, O4).
    expected = [
        ("down", "1"),
        ("dist", "long"),
        ("zone", "own"),
        ("down_x_dist", "1|long"),
        ("side", "offense"),
        ("time", "off"),
        ("look", "cover 2"),
        ("look_family", "cover_2"),
        ("call_class", "pass"),
        ("call", "mesh"),
        ("call_x_family", "mesh|cover_2"),
        ("class_x_family", "pass|cover_2"),
    ]
    assert row_features(prepared, enable_cxs=False) == expected
    assert SIT_GROUPS == ("side", "down", "dist", "zone", "time", "down_x_dist")
    situation = [
        ("down", "1"),
        ("dist", "long"),
        ("zone", "own"),
        ("down_x_dist", "1|long"),
    ]
    allowed = {
        "call_x_down": {"mesh|1"},
        "call_x_dist": {"mesh|long"},
        "call_x_zone": {"mesh|own"},
    }
    assert row_features(prepared, situation_only=True, enable_cxs=True, allowed_cxs=allowed) == situation
    with_cxs = expected + [
        ("class_x_down_x_dist", "pass|1|long"),
        ("class_x_zone", "pass|own"),
        ("call_x_down", "mesh|1"),
        ("call_x_dist", "mesh|long"),
        ("call_x_zone", "mesh|own"),
    ]
    assert row_features(prepared, enable_cxs=True, allowed_cxs=allowed) == with_cxs
    groups = {group for group, _level in row_features(prepared, enable_cxs=True, allowed_cxs=allowed)}
    assert groups.isdisjoint({"score", "off_adj", "def_adj"})
    copies = [prepare_row(_row(snap_id=f"c{i}", success=i % 2 == 0)) for i in range(5)]
    disabled = GroupLR(DEFAULT_HP, enable_cxs=False, lambda_cxs=64).fit(copies)
    enabled = GroupLR(DEFAULT_HP, enable_cxs=True, lambda_cxs=256).fit(copies)
    assert set(CXS_GROUPS).isdisjoint(disabled.groups())
    assert {"call_x_down", "call_x_dist", "call_x_zone"} <= set(enabled.groups())
    assert set(enabled.vocab) - set(disabled.vocab) == {
        ("call_x_down", "mesh|1"),
        ("call_x_dist", "mesh|long"),
        ("call_x_zone", "mesh|own"),
        ("class_x_down_x_dist", "pass|1|long"),
        ("class_x_zone", "pass|own"),
    }
    assert disabled.penalty_by_group["call_x_family"] == DEFAULT_HP[2]
    assert "call_x_down" not in disabled.penalty_by_group
    assert enabled.penalty_by_group["call_x_down"] == 256
    assert enabled.penalty_by_group["call_x_family"] == DEFAULT_HP[2]
    assert enabled.penalty_by_group["class_x_down_x_dist"] == 2 * DEFAULT_HP[0]
    hp = (1.0, 4.0, 16.0)
    assert group_penalty("down", hp, 64) == 1
    assert group_penalty("down_x_dist", hp, 64) == 2
    assert group_penalty("call", hp, 64) == 4
    assert group_penalty("call_x_family", hp, 64) == 16
    assert group_penalty("call_x_down", hp, 64) == 64
    assert group_penalty("call_x_dist", hp, 1024) == 1024
    assert group_penalty("call_x_zone", hp, 4096) == 4096
    assert group_penalty("look_x_zone", hp, 4096) == 4096
    assert group_penalty("class_x_zone", hp, 4096) == 2
    assert group_penalty("family_x_down_x_dist", hp, 4096) == 2


def test_defense_rows_use_our_candidate_and_drop_offense_call():
    raw = _row(
        side="defense",
        offense_call={"play": "HB Dive"},
        defense_look={"def_call": "DB Fire 2", "coverage_seen": "Cover 1", "def_overlay": ""},
        down=3,
        distance=2,
        yardline=None,
        yards_to_goal=None,
    )
    prepared = prep(raw)
    feats = dict(row_features(prepared, enable_cxs=True, allowed_cxs={
        "look_x_down": {"db fire 2|3"}, "look_x_dist": {"db fire 2|short"}, "look_x_zone": set(),
    }))
    assert prepared["look"] == "db fire 2"
    assert prepared["look_family"] == "cover_2"
    assert prepared["family_src"] == "candidate"
    assert feats["pressure"] == "yes"
    assert feats["zone"] == "unk"
    assert prepared["location_missing"] is True
    for dropped in ("call", "call_x_family", "call_class", "class_x_family", "score", "off_adj", "def_adj"):
        assert dropped not in feats
    assert feats["look_x_down"] == "db fire 2|3"
    assert feats["look_x_dist"] == "db fire 2|short"
    assert "look_x_zone" not in feats and "family_x_zone" not in feats
    assert feats["family_x_down_x_dist"] == "cover_2|3|short"
    assert feats["pressure_x_down_x_dist"] == "yes|3|short"


def test_cxs_key_format_and_shared_buckets():
    assert dist_bucket(None) == "unk"
    assert dist_bucket(2) == "short"
    assert dist_bucket(3) == "medium"
    assert dist_bucket(6) == "medium"
    assert dist_bucket(7) == "long"
    assert dist_bucket(10) == "long"
    assert dist_bucket(11) == "xlong"
    assert zone_bucket(None) == "unk"
    assert zone_bucket(0) == "own_deep"
    assert zone_bucket(19.9) == "own_deep"
    assert zone_bucket(20) == "own"
    assert zone_bucket(49) == "own"
    assert zone_bucket(50) == "opp"
    assert zone_bucket(79) == "opp"
    assert zone_bucket(80) == "red_zone"
    assert zone_bucket(None, 15) == "red_zone"
    assert zone_bucket(25, yardline_touched=False) == "unk"
    assert time_bucket(2, "1:30") == "on"
    assert time_bucket(2, "3:00") == "off"
    assert time_bucket(5, "1:00") == "off"
    assert time_bucket("", "1:00") == "unknown"
    assert time_bucket(4, "") == "unknown"
    prepared = prepare_row(_row(
        offense_call={"play": "Mesh"},
        down=1,
        distance=1,
        yardline=85,
    ))
    features = row_features(
        prepared,
        enable_cxs=True,
        allowed_cxs={
            "call_x_down": {f"{prepared['call']}|{prepared['down']}"},
            "call_x_dist": {f"{prepared['call']}|{prepared['dist']}"},
            "call_x_zone": {f"{prepared['call']}|{prepared['zone']}"},
        },
    )
    keys = {group: level for group, level in features if group in CXS_GROUPS}
    assert keys == {
        "call_x_down": "mesh|1",
        "call_x_dist": "mesh|short",
        "call_x_zone": "mesh|red_zone",
    }
    for level in keys.values():
        call, bucket = level.split("|")
        assert call == norm_key("Mesh")
        assert bucket
        assert level.count("|") == 1


class _DownSwitch:
    def predict(self, rows):
        out = []
        for row in rows:
            prefer = "mesh" if row["down"] == "1" else "hb dive"
            out.append(0.9 if row["call"] == prefer else 0.1)
        return out


def test_constant_pick_uses_shared_look_order():
    assert CONF_MIN == 0.5
    primary = _row(
        snap_id="primary",
        offense_call={"play": "Mesh"},
        defense_look={
            "def_call": "Cover 3",
            "coverage_seen": "Cover 2 Man",
            "def_overlay": "Nickel",
        },
        personnel="nickel",
        package="dime",
        base="base",
        empty="empty",
    )
    low_coverage = _row(
        snap_id="low-coverage",
        offense_call={"play": "HB Dive"},
        defense_look={"def_call": "Cover 0", "coverage_seen": "Cover 4", "def_overlay": "Cover 6"},
        conf=_conf(def_call=0.9, coverage_seen=0.1, def_overlay=0.9),
    )
    call_only = _row(
        snap_id="call-only",
        offense_call={"play": "Mesh"},
        defense_look={"def_call": "Cover 1", "coverage_seen": "", "def_overlay": ""},
        conf=_conf(def_call=0.9, coverage_seen=0.0, def_overlay=0.0),
    )
    personnel_rows = [
        _row(
            snap_id=f"personnel-{token}",
            offense_call={"play": "Mesh"},
            defense_look={"def_call": "Cover 3", "coverage_seen": token, "def_overlay": "Cover 4"},
            conf=_conf(),
        )
        for token in ("Nickel", "Base", "Dime", "Empty")
    ]
    prepared = prep(primary)
    assert prepared["look"] == "cover 2 man"
    assert prepared["look_src"] == "coverage_seen"
    assert prepared["look_family"] == "cover_2_man"
    assert prepared["look_family_src"] == "coverage_seen"
    assert prep(call_only)["look"] is None
    assert prep(call_only)["look_family"] == "none"
    report = constant_pick_check(
        _DownSwitch(),
        [primary, low_coverage, call_only, *personnel_rows],
    )
    by_look = {row["look"]: row for row in report["looks"]}
    assert set(by_look) == {"cover 2 man", "cover 6"}
    assert by_look["cover 6"]["look_src"] == "def_overlay"
    assert by_look["cover 6"]["look_family"] == "cover_6"
    assert report["skipped_personnel_looks"] == ["base", "dime", "empty", "nickel"]
    for token in ("nickel", "base", "dime", "empty"):
        assert token not in by_look
    assert by_look["cover 2 man"]["changed_down"] is True
    assert by_look["cover 2 man"]["picks_down"] == ["mesh", "hb dive", "hb dive", "hb dive"]
    assert by_look["cover 2 man"]["changed_distance"] is False
    assert by_look["cover 2 man"]["changed_zone"] is False
    assert by_look["cover 6"]["changed_down"] is True
    assert report["n_looks"] == 2
    assert report["rate_changed_down"] == 1
    assert report["rate_changed_distance"] == 0
    assert report["rate_changed_zone"] == 0
    half = prep(_row(
        defense_look={"def_call": "Cover 3", "coverage_seen": "Cover 3", "def_overlay": ""},
        conf=_conf(def_call=0.0, coverage_seen=0.5, def_overlay=0.0),
    ))
    below = prep(_row(
        defense_look={"def_call": "Cover 3", "coverage_seen": "Cover 3", "def_overlay": ""},
        conf=_conf(def_call=0.9, coverage_seen=0.49, def_overlay=0.0),
    ))
    assert half["look_src"] == "coverage_seen"
    assert half["look"] == "cover 3"
    assert below["look"] is None


def test_kneel_rows_are_never_training_targets():
    rows = [_row(snap_id=f"k{i}", match_id=f"m{i % 2}", offense_call={"play": "QB Kneel"}) for i in range(3)]
    rows += [_row(snap_id=f"r{i}", match_id=f"m{i % 2}", success=i % 2 == 0) for i in range(6)]
    report = train(rows, workers=1, hp_grid=TINY_HP, lambda_cxs_grid=TINY_LAMBDA, n_boot=0, min_groups=2)
    assert report["n_kneel_excluded"] == 3
    assert report["strata"]["madden27/cpu/offense"]["n_rows"] == 6
    assert report["feature_module"]["recipe_sha256"].startswith("f7c17d16")
    assert report["feature_module"]["gate_sha256"].startswith("4d4234aa")


def test_excludes_blocked_vod_and_audit_games():
    rows = [
        _row(snap_id="keep", video_id="ok", match_id="m1", game_id="g-keep"),
        _row(snap_id="vod", video_id="F8Hm324SMtg", match_id="m2"),
        _row(snap_id="url", video_id="other", url="https://youtu.be/F8Hm324SMtg", match_id="m3"),
        _row(snap_id="audit", video_id="aud", game_id="audit-9", match_id="m4"),
        _row(snap_id="title", video_id="title-vod", match_id="m5", game="madden27"),
    ]
    kept, summary = split_excluded(rows, audit_game_ids=["audit-9", "madden27"])
    assert [row["snap_id"] for row in kept] == ["keep", "title"]
    assert summary["by_reason"]["excluded_vod_F8Hm324SMtg"] == 2
    assert summary["by_reason"]["audit_game_id"] == 1
    assert exclusion_reason(rows[4], ["madden27"]) is None
    assert exclusion_reason(_row(match_id="m1"), ["m1"]) == "audit_game_id"
    caller = list(rows)
    split_excluded(caller, ["audit-9"])
    assert caller == rows


def test_strata_keep_games_opponents_and_sides_apart(v6_report):
    assert set(v6_report["strata"]) >= {
        "madden27/cpu/offense",
        "madden27/cpu/defense",
        "madden27/human/offense",
        "cfb27/cpu/offense",
    }
    offense = v6_report["strata"]["madden27/cpu/offense"]
    defense = v6_report["strata"]["madden27/cpu/defense"]
    human = v6_report["strata"]["madden27/human/offense"]
    cfb = v6_report["strata"]["cfb27/cpu/offense"]
    assert offense["n_rows"] > defense["n_rows"] == human["n_rows"] == cfb["n_rows"] == 1
    offense_snaps = {
        row["snap_id"]
        for row in offense["holdouts"]["match"]["predictions"]
    }
    assert "cfb-row" not in offense_snaps
    assert "human-row" not in offense_snaps
    assert "defense-row" not in offense_snaps
    assert "h2h-row" not in offense_snaps
    assert "blocked-vod" not in offense_snaps
    assert "audit-row" not in offense_snaps
    joined = " ".join(offense["keys_survived"]["call_x_down"])
    assert "inside zone" not in joined
    assert "human only" not in joined
    assert "defense only" not in joined
    assert "h2h only" not in joined
    assert v6_report["not_trained"]["madden27/h2h/offense"]["n_rows"] == 1
    assert "human" in v6_report["not_trained"]["madden27/h2h/offense"]["reason"]
    assert human["keys_survived"]["call_x_down"] == []
    assert cfb["keys_survived"]["call_x_down"] == []
    assert defense["keys_survived"]["call_x_down"] == []


def test_holdout_report_metrics_pass_rule_and_scored_on(v6_report):
    assert v6_report["scored_on"] == "unreviewed"
    assert v6_report["training_report"]["tags"]["scored_on"] == "unreviewed"
    assert v6_report["script_version"] == "train_vod_model.v6"
    stratum = v6_report["strata"]["madden27/cpu/offense"]
    assert stratum["scored_on"] == "unreviewed"
    for name in ("match", "player"):
        holdout = stratum["holdouts"][name]
        assert holdout["status"] == "evaluated"
        for model_name in ("v6", "v5_refit", "situation_only"):
            metrics = holdout["metrics"][model_name]
            for field in ("log_loss", "brier", "ece", "log_loss_ci95", "brier_ci95", "ece_ci95"):
                assert field in metrics
            assert metrics["log_loss"] is not None
            lo, hi = metrics["log_loss_ci95"]
            assert lo <= hi
        for axis in ("down", "distance", "zone"):
            assert holdout["breakdowns"][axis]
            for level in holdout["breakdowns"][axis].values():
                assert level["n"] > 0
                assert "log_loss" in level["v6"]
                assert "brier" in level["v5_refit"]
                assert "ece" in level["situation_only"]
        diff = holdout["logloss_diff_v6_minus_v5"]
        assert diff["estimate"] is not None
        assert len(diff["ci95"]) == 2
        assert holdout["passes_against_v5"] is ci_below_zero(diff["ci95"])
        for row in holdout["predictions"]:
            assert row["scored_on"] == "unreviewed"
        for fold in holdout["folds"]:
            assert set(CXS_GROUPS).isdisjoint(fold["v5_groups"])
            if fold["n_keys_survived"]["call_x_down"]:
                assert "call_x_down" in fold["v6_groups"]
            assert fold["lambda_cxs"] in TINY_LAMBDA
            best = min(fold["lambda_cxs_inner"], key=lambda item: (item["inner_logloss"], item["lambda_cxs"]))
            assert fold["lambda_cxs"] == best["lambda_cxs"]
    match_ci = stratum["holdouts"]["match"]["logloss_diff_v6_minus_v5"]["ci95"]
    player_ci = stratum["holdouts"]["player"]["logloss_diff_v6_minus_v5"]["ci95"]
    assert stratum["v6_better"] is v6_beats_v5(match_ci, player_ci)
    look = stratum["final_constant_pick"]["looks"][0]
    assert look["look"] == "cover 2"
    assert look["look_src"] == "coverage_seen"
    assert look["look_family"] == "cover_2"
    assert look["look_family_src"] == "coverage_seen"
    json.dumps(v6_report)


def test_pass_rule_requires_both_cis_below_zero():
    assert v6_beats_v5([-0.2, -0.01], [-0.3, -0.02]) is True
    assert v6_beats_v5([-0.2, 0.01], [-0.3, -0.02]) is False
    assert v6_beats_v5([-0.2, -0.01], None) is False
    assert v6_beats_v5([-0.2, 0.0], [-0.2, -0.01]) is False
    assert ci_below_zero([-0.1, -0.2]) is False


def test_lambda_cxs_grid_and_inner_cv():
    assert LAMBDA_CXS_GRID == (64.0, 256.0, 1024.0, 4096.0)
    assert DEFAULT_LAMBDA_CXS == 256.0
    short = [prepare_row(_row(snap_id=f"a{i}", match_id=f"m{i % 2}", success=i % 2 == 0)) for i in range(4)]
    chosen, detail = select_lambda_cxs(short, "match_id", DEFAULT_HP, LAMBDA_CXS_GRID)
    assert chosen == DEFAULT_LAMBDA_CXS
    assert detail["scores"] == []
    rows = []
    for group in ("g1", "g2", "g3"):
        for i in range(4):
            rows.append(prepare_row(_row(
                snap_id=f"{group}-{i}",
                match_id=group,
                player_id=group,
                down=1 if i < 2 else 3,
                success=(group == "g1" and i < 2) or (group != "g1" and i >= 2),
            )))
    chosen, detail = select_lambda_cxs(rows, "match_id", DEFAULT_HP, (64.0, 4096.0))
    assert {item["lambda_cxs"] for item in detail["scores"]} == {64.0, 4096.0}
    manual = leave_one_group_predictions(
        rows,
        "match_id",
        lambda train_rows: GroupLR(DEFAULT_HP, enable_cxs=True, lambda_cxs=64.0).fit(train_rows),
    )
    stored = {item["lambda_cxs"]: item["inner_logloss"] for item in detail["scores"]}
    assert stored[64.0] == round(logloss([row["y"] for row in rows], manual), 6)
    assert chosen == min(detail["scores"], key=lambda item: (item["inner_logloss"], item["lambda_cxs"]))["lambda_cxs"]
    assert call_class("Mesh") == "pass"
    assert call_class("HB Dive") == "run"


def test_prep_fallback_success_matches_v5_rule():
    raw = _row(down=1, distance=10, yardline=25, yards=5, result_kind="gain")
    del raw["success"]
    made = prep(raw)
    assert made["success_src"] == "fallback_rule"
    assert made["y"] == 1
    raw["yards"] = 3
    missed = prep(raw)
    assert missed["y"] == 0
    assert prepare_row(raw)["scored_on"] == "unreviewed"


def test_workers_and_nice(monkeypatch):
    assert map_items(lambda value: value * value, [1, 2, 3, 4], workers=2) == [1, 4, 9, 16]
    assert map_items(lambda value: value + 1, [1, 2], workers=1) == [2, 3]
    with pytest.raises(ValueError):
        map_items(lambda value: value, [1], workers=0)
    calls = {}

    def _setpriority(_which, _who, level):
        calls["level"] = level

    def _getpriority(_which, _who):
        return calls.get("level", 0)

    monkeypatch.setattr("cfb_coach.vod_success.train_vod_model_v6.os.setpriority", _setpriority)
    monkeypatch.setattr("cfb_coach.vod_success.train_vod_model_v6.os.getpriority", _getpriority)
    assert apply_nice(7) == 7
    assert calls["level"] == 7
    assert apply_nice(None) is None
    with pytest.raises(ValueError):
        apply_nice(-1)
    report = train(
        [_row(snap_id="only")],
        workers=1,
        nice=4,
        hp_grid=TINY_HP,
        lambda_cxs_grid=TINY_LAMBDA,
        n_boot=1,
        min_groups=2,
    )
    assert report["runtime"] == {"workers": 1, "nice": 4}
    parser = build_parser()
    args = parser.parse_args([
        "--rows", "rows.json",
        "--workers", "3",
        "--nice", "5",
        "--audit-game-id", "g1",
        "--disable-cxs",
    ])
    assert args.workers == 3
    assert args.nice == 5
    assert args.audit_game_id == ["g1"]
    assert args.disable_cxs is True


def test_cli_refuses_vod_data_path():
    with pytest.raises(SystemExit, match="vod_data"):
        main(["--rows", "/workspace/vod_data/labels.json", "--workers", "1"])


def test_train_records_the_defense_book_and_restores_it():
    from cfb_coach.vod_success import shared_features as sf

    rows = [_row(snap_id=f"r{i}", match_id=f"m{i % 2}", success=i % 2 == 0) for i in range(6)]
    book = sf.DefenseBook(["Cover 3 Sky", "Free Fire"], source="synthetic")
    before = sf.active_defense_book()
    report = train(rows, workers=1, hp_grid=TINY_HP, lambda_cxs_grid=TINY_LAMBDA, n_boot=0, min_groups=2,
                   defense_book=book)
    assert report["feature_module"]["defense_book"] == book.describe()
    assert set(report["feature_module"]["defense_family_rows"]) == {"on_book", "off_book_other", "blank"}
    assert sf.active_defense_book() is before
    assert "--defense-book" in build_parser().format_help()
