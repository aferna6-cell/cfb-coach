"""v0.24 shared feature module: parity, isolation, latency and rule regressions.

Synthetic fixtures only. No label rows, DB rows or held-out rows are used.
Recipe f7c17d16..., gate v2.2 4d4234aa... (box-only spec files, not committed).
"""

from __future__ import annotations

import ast
import inspect
import itertools
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cfb_coach.madden.model import inference, policy
from cfb_coach.vod_success import shared_features as sf

_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _ROOT / "cfb_coach" / "vod_success" / "shared_features.py"

# Play names the recipe lists by name (section 6, D3). Names only, no data.
_RECIPE_23 = {
    "Clear Deep": "pass", "HB Off Tackle": "run", "HB Zone WK": "run",
    "Mtn Choice Gos Y-Shake": "pass", "Mtn Drag Sluggo": "pass", "Mtn HB Swing": "pass",
    "Mtn Salem Drag": "pass", "Mtn Slot Fades": "pass", "Mtn Triple Slants": "pass",
    "Mtn Zone Split": "run", "PA Boot": "pass", "PA Boot Lt": "pass", "PA Boot Y Leak": "pass",
    "PA HB Flat": "pass", "PA Z Over": "pass", "Portland Ywheel": "pass",
    "RPO Alert Bubble": "rpo", "Read Option": "run", "Return PA Z Over": "pass",
    "Return RPO Zone Alert": "rpo", "Same Side Zone": "run", "Slants": "pass", "Y Ohio Return": "pass",
}
_BLITZ_7 = {
    "DBL Safety Blitz": "pressure_other",
    "FS Zone Blitz": "pressure_other",
    "Free Fire": "pressure_other",
    "Overload Blitz": "pressure_other",
    "SS Zone Blitz": "pressure_other",
    "FS Will Blitz 0": "cover_0",
    "Mid Blitz 0": "cover_0",
}
_SYNTH_OFFENSE_BOOK = sorted(set(_RECIPE_23) | {
    "Mesh", "Flood", "HB Dive", "Inside Zone", "Counter", "QB Kneel", "Four Verticals",
    "Smash", "Stick", "Y Cross", "Bench", "Curl Flat", "Drive", "Dagger", "Power O",
})
_SYNTH_DEFENSE_BOOK = sorted(set(_BLITZ_7) | {
    "DB Fire 2", "OLB Fire Man", "Cover 3 Sky", "Cover 2 Man", "Tampa 2", "Cover 4 Quarters",
    "Cover 6", "Cover 1 Robber", "Cover 0 Blitz", "Cover 3 Buzz", "Cover 2 Invert", "Mid Blitz",
})
_LOOKS = ("Cover 3", "Cover 2 Man", "Cover 1 Hole", "Tampa 2", "Quarters", "Fire Zone 3", "", None)


def _label_row(side, play, *, look=None, down=1, distance=10, yardline=25, ytg=None,
               quarter=1, clock="12:00", touched=None, look_conf=0.9):
    """A synthetic label row in the v0.24 label shape."""
    row = {
        "side": side,
        "down": down,
        "distance": distance,
        "yardline": yardline,
        "yards_to_goal": ytg,
        "quarter": quarter,
        "clock": clock,
        "conf": {"play": 0.9, "def_call": 0.9, "coverage_seen": look_conf, "def_overlay": 0.0},
    }
    if touched is not None:
        row["yardline_touched"] = touched
    if side == "defense":
        row["offense_call"] = {"play": "HB Dive"}  # opponent's call: never a defense input
        row["defense_look"] = {"def_call": play, "coverage_seen": look}
    else:
        # def_call is set to a decoy shell: it must never reach the family.
        row["offense_call"] = {"play": play}
        row["defense_look"] = {"def_call": "Cover 0 Blitz", "coverage_seen": look}
    return row


def _live_kwargs(row):
    """What the live path supplies for that snap: typed look, situation, no def_call/coverage_seen."""
    look = None
    if row["side"] != "defense":
        conf = row["conf"].get("coverage_seen", 0)
        look = row["defense_look"].get("coverage_seen") if conf >= 0.5 else None
    yardline = row["yardline"]
    if yardline is None and row.get("yards_to_goal") is not None:
        yardline = 100 - row["yards_to_goal"]
    touched = row.get("yardline_touched")
    if yardline is None:
        yardline, touched = 25, False  # live default own 25, nobody edited it
    return {
        "opp_look": look,
        "down": row["down"],
        "distance": row["distance"],
        "yardline": yardline,
        "yardline_touched": touched,
        "quarter": row["quarter"],
        "clock": row["clock"],
    }


def _cases():
    situations = [
        dict(),  # default own 25, Q1 clock
        dict(yardline=None),  # no location -> untouched live default
        dict(yardline=25, touched=False),  # untouched yardline case
        dict(yardline=None, ytg=12),  # location from yards_to_goal
        dict(quarter=4, clock="1:30", down=3, distance=2, yardline=82),
        dict(quarter=2, clock=""),  # blank clock -> unknown
        dict(quarter=None, clock=None, down=None, distance=None),
        dict(quarter=4, clock=95, down=4, distance=15, yardline=5),
        dict(look_conf=0.2),  # low-confidence look -> blank
    ]
    for side, book in (("offense", _SYNTH_OFFENSE_BOOK), ("defense", _SYNTH_DEFENSE_BOOK)):
        for play, look, sit in itertools.product(book, _LOOKS, situations):
            yield _label_row(side, play, look=look, **sit)


def test_parity_zero_mismatches_on_live_shaped_inputs(capsys):
    """G6: training features == the features inference.vod_candidate_features builds."""
    n = 0
    mismatches = []
    for row in _cases():
        trained = sf.feature_levels(sf.training_inputs(row))
        live = inference.vod_candidate_features(
            row["side"], row["defense_look"]["def_call"] if row["side"] == "defense" else row["offense_call"]["play"],
            **_live_kwargs(row),
        )
        n += 1
        if trained != live:
            mismatches.append((row, trained, live))
    with capsys.disabled():
        print(f"\n[parity] live-shaped synthetic rows={n} mismatches={len(mismatches)}")
    assert n > 1000
    assert mismatches == []


def test_parity_every_book_play_both_sides():
    for side, book in (("offense", _SYNTH_OFFENSE_BOOK), ("defense", _SYNTH_DEFENSE_BOOK)):
        for play in book:
            row = _label_row(side, play, look="Cover 3")
            trained = sf.feature_levels(sf.training_inputs(row))
            live = inference.vod_candidate_features(side, play, **_live_kwargs(row))
            assert trained == live, play


def test_untouched_yardline_is_the_missing_bucket():
    live = inference.vod_live_inputs("offense", "Mesh", opp_look="Cover 3", down=1, distance=10,
                                     yardline=25, yardline_touched=False)
    assert live["zone"] == "unk" and live["location_missing"] is True
    feats = inference.vod_candidate_features("offense", "Mesh", opp_look="Cover 3", down=1,
                                             distance=10, yardline=25, yardline_touched=False)
    assert feats["zone"] == "unk"
    assert "call_x_zone" not in feats and "class_x_zone" not in feats
    touched = inference.vod_candidate_features("offense", "Mesh", opp_look="Cover 3", down=1,
                                               distance=10, yardline=25, yardline_touched=True)
    assert touched["zone"] == "own"
    flag_absent = inference.vod_candidate_features("offense", "Mesh", opp_look="Cover 3", down=1,
                                                   distance=10, yardline=25)
    assert flag_absent["zone"] == "own"  # until the UI flag ships, own 25 is indistinguishable
    trained = sf.training_inputs(_label_row("offense", "Mesh", look="Cover 3", yardline=None))
    assert trained["zone"] == "unk" and trained["location_missing"] is True


def test_two_minute_flag_unknown_is_never_off():
    assert sf.two_minute_flag(4, "1:59") == "on"
    assert sf.two_minute_flag("Q2", "2:00") == "on"
    assert sf.two_minute_flag(2, "2:01") == "off"
    assert sf.two_minute_flag(1, "0:30") == "off"
    assert sf.two_minute_flag(4, 90) == "on"
    for quarter, clock in ((4, ""), (4, None), (None, "1:00"), ("", ""), (2, "garbage")):
        assert sf.two_minute_flag(quarter, clock) == "unknown"
    feats = inference.vod_candidate_features("offense", "Mesh", opp_look="Cover 3", quarter=4, clock="")
    assert feats["time"] == "unknown"
    # Situation term only: no group crosses time with a candidate or class.
    assert not any("time" in group for group in sf.ALL_GROUPS if group != "time")


def test_score_time_family_reads_only_typed_look_or_our_candidate():
    params = set(inspect.signature(inference.vod_candidate_features).parameters)
    assert "def_call" not in params and "coverage_seen" not in params
    params = set(inspect.signature(sf.live_inputs).parameters)
    assert "def_call" not in params and "coverage_seen" not in params

    from scripts.madden_ml import eval_harness

    class Strict(dict):
        def get(self, key, default=None):
            if key in ("def_call", "coverage_seen", "def_overlay"):
                raise AssertionError(f"score path read {key}")
            return super().get(key, default)

        def __getitem__(self, key):
            if key in ("def_call", "coverage_seen", "def_overlay"):
                raise AssertionError(f"score path read {key}")
            return super().__getitem__(key)

    row = Strict(down=1, distance=10, yardline=30, look="Cover 4", def_call="Cover 0", coverage_seen="Cover 0")
    off = eval_harness._candidate_feature_map(row, formation="Gun", play="Mesh", side="offense")
    assert off["look_family"] == "cover_4"
    dfn = eval_harness._defense_feature_map(row, formation="Nickel", play="DB Fire 2", side="defense")
    assert dfn["look_family"] == "cover_2" and dfn["look"] == "db fire 2"
    # Training side: def_call never feeds the offense family.
    only_call = sf.training_inputs(_label_row("offense", "Mesh", look=None))
    assert only_call["look_family"] == "none" and only_call["look"] is None


def test_db_fire_2_family_regression():
    for path in ("train", "live"):
        if path == "train":
            inputs = sf.training_inputs(_label_row("defense", "DB Fire 2"))
        else:
            inputs = inference.vod_live_inputs("defense", "DB Fire 2", down=1, distance=10, yardline=25)
        assert inputs["look_family"] == "cover_2", path
        assert inputs["pressure"] is True, path
    assert sf.family_and_pressure("Free Fire") == ("pressure_other", False)
    assert sf.family_and_pressure("OLB Fire Man") == ("man_other", True)
    assert sf.family_and_pressure("Cover 3 Sky") == ("cover_3", False)


def test_is_pressure_seven_blitz_plays(capsys):
    lines = []
    for play, family in _BLITZ_7.items():
        got_family, flag = sf.family_and_pressure(play)
        assert got_family == family, play
        assert sf.is_pressure(got_family, flag) is True, play
        lines.append(f"{play}: {got_family} (pressure flag {'on' if flag else 'off'})")
    assert sf.is_pressure("cover_3", False) is False
    assert sf.is_pressure("cover_2", True) is True
    assert sf.is_pressure("none", None) is False
    assert policy.blitz_share(list(_BLITZ_7)) == 1.0
    assert policy.blitz_share(["Cover 3 Sky", "Free Fire"]) == 0.5
    with capsys.disabled():
        print("\n[is_pressure] " + "; ".join(lines))


def test_rpo_is_its_own_class_and_folds_only_for_binary():
    for play, cls in _RECIPE_23.items():
        assert sf.call_class(play) == cls, play
    assert sf.call_class("RPO Bubble") == "rpo"
    assert sf.binary_run_pass("rpo") == "pass"
    assert sf.binary_run_pass("run") == "run"
    feats = inference.vod_candidate_features("offense", "RPO Alert Bubble", opp_look="Cover 3")
    assert feats["call_class"] == "rpo"
    assert feats["class_x_family"] == "rpo|cover_3"


def test_kneel_is_never_a_candidate_or_target():
    assert sf.call_class("QB Kneel") == "none"
    assert sf.is_kneel("QB Kneel") and sf.is_kneel("Victory") and not sf.is_kneel("Mesh")
    assert sf.drop_kneel(["Mesh", "QB Kneel", "HB Dive"]) == ["Mesh", "HB Dive"]
    feats = inference.vod_candidate_features("offense", "QB Kneel", opp_look="Cover 3")
    assert "call" not in feats and "call_x_family" not in feats
    trained = sf.training_inputs(_label_row("offense", "QB Kneel", look="Cover 3"))
    assert trained["kneel"] is True and trained["call"] is None
    # Clock rule helper: applied before ranking, never fires on unknown inputs.
    assert sf.kneel_clock_rule(quarter=4, clock="0:35", down=1, score_us=21, score_them=14, opp_timeouts=0) is True
    assert sf.kneel_clock_rule(quarter=4, clock="1:50", down=1, score_us=21, score_them=14, opp_timeouts=3) is False
    assert sf.kneel_clock_rule(quarter=4, clock="0:20", down=1, score_us=14, score_them=21) is False
    assert sf.kneel_clock_rule(quarter=4, clock="", down=1, score_us=21, score_them=14) is None

    from cfb_coach.madden.model import adapters
    from cfb_coach.madden.model.schema import OpponentKind, Possession, candidate_in_book
    from cfb_coach.situation import Situation

    sit = Situation(raw="1&10 my 35", side="offense", down=1, distance=10, yardline=35)
    state, obs = adapters.situation_to_state(sit, opponent_type=OpponentKind.CPU)
    book = {"Victory": ["QB Kneel"], "Gun Bunch": ["Mesh"]}
    cands = [
        candidate_in_book(book, side=Possession.OFFENSE, formation="Victory", play="QB Kneel"),
        candidate_in_book(book, side=Possession.OFFENSE, formation="Gun Bunch", play="Mesh"),
    ]
    ranking = policy.rank_plays(state, obs, cands, None, None, ())
    assert [item.candidate.play for item in ranking.plays] == ["Mesh"]


def test_no_score_or_adjustment_features_and_defense_drops():
    off = inference.vod_candidate_features("offense", "Mesh", opp_look="Cover 3", down=2, distance=4)
    dfn = inference.vod_candidate_features("defense", "Cover 3 Sky", down=2, distance=4)
    for feats in (off, dfn):
        assert not {"score", "off_adj", "def_adj"} & set(feats)
    assert not {"call", "call_x_family", "call_class", "class_x_family"} & set(dfn)
    assert set(sf.groups_for("offense", situation_only=True)) == {"down", "dist", "zone", "down_x_dist"}


def test_ties_and_blank_looks_go_to_heuristic():
    assert inference.vod_fallback_reason(None, 0.2, side="offense") == "blank_look"
    assert inference.vod_fallback_reason("   ", 0.2, side="offense") == "blank_look"
    assert inference.vod_fallback_reason("Cover 3", 0.0099, side="offense") == "tie"
    assert inference.vod_fallback_reason("Cover 3", None, side="offense") == "tie"
    assert inference.vod_fallback_reason("Cover 3", 0.01, side="offense") is None
    assert sf.top_two_margin([0.61, 0.6, 0.2]) == pytest.approx(0.01)
    assert sf.top_two_margin([0.5]) is None


def test_defense_fallback_checks_its_own_input_not_opp_look():
    # Defense ranks our own calls: an empty opp_look is not a blank look.
    assert inference.vod_fallback_reason(None, 0.2, side="defense") is None
    assert inference.vod_fallback_reason("", 0.2, side="defense", candidate="Cover 3 Sky") is None
    assert inference.vod_fallback_reason("Cover 2", 0.2, side="defense", candidate="") == "blank_look"
    assert inference.vod_fallback_reason(None, 0.005, side="defense", candidate="Cover 3 Sky") == "tie"
    assert inference.vod_fallback_reason(None, None, side="defense", candidate="Cover 3 Sky") == "tie"
    with pytest.raises(ValueError):
        inference.vod_fallback_reason("Cover 3", 0.2, side="special_teams")
    with pytest.raises(TypeError):
        inference.vod_fallback_reason("Cover 3", 0.2)  # side is required


def test_top_two_margin_dedupes_the_same_play_across_formations():
    # Same play in two formations scores the same: it must not be a tie with itself.
    pairs = [("Mesh", 0.62), ("mesh ", 0.62), ("Flood", 0.50), ("HB Dive", 0.40)]
    assert sf.top_two_margin(pairs) == pytest.approx(0.12)
    assert sf.top_two_margin({"Mesh": 0.62, "Flood": 0.5}) == pytest.approx(0.12)
    assert inference.vod_fallback_reason("Cover 3", sf.top_two_margin(pairs), side="offense") is None
    # Keeps the best score per play name.
    assert sf.top_two_margin([("Mesh", 0.3), ("Mesh", 0.7), ("Flood", 0.6)]) == pytest.approx(0.1)
    # Only one distinct play: no margin, so a tie.
    assert sf.top_two_margin([("Mesh", 0.6), ("Mesh", 0.6)]) is None
    assert inference.vod_fallback_reason("Cover 3", None, side="offense") == "tie"
    # A real near-tie between two different plays stays a tie.
    assert sf.top_two_margin([("Mesh", 0.605), ("Flood", 0.6)]) == pytest.approx(0.005)
    # Bare numbers are still distinct candidates.
    assert sf.top_two_margin([0.6, 0.6]) == pytest.approx(0.0)


def test_eval_fallback_uses_deduped_margin_and_side():
    from scripts.madden_ml import eval_harness, eval_scoring

    ctx = {"row": {"look": None}}
    dup = [("Cover 3 Sky", 0.7), ("Cover 3 Sky", 0.7), ("Tampa 2", 0.6)]
    got = eval_harness._vod_fallback("defense", ctx, dup, "Cover 3 Sky")
    assert got["vod_fallback"] is None and got["top_two_margin"] == pytest.approx(0.1)
    got = eval_harness._vod_fallback("offense", ctx, [("Mesh", 0.7), ("Mesh", 0.7), ("Flood", 0.6)], "Mesh")
    assert got["vod_fallback"] == "blank_look"
    got = eval_harness._vod_fallback("offense", {"row": {"look": "Cover 3"}},
                                     [("Mesh", 0.7), ("Mesh", 0.7), ("Flood", 0.6)], "Mesh")
    assert got["vod_fallback"] is None
    got = eval_scoring.model_fallback("defense", None, {"Cover 3 Sky": 0.7, "Tampa 2": 0.695}, candidate="Cover 3 Sky")
    assert got["vod_fallback"] == "tie"


@pytest.mark.parametrize("play,cls", [
    ("Jet Sweep", "run"), ("Jet Toss", "run"), ("Bunch HB Draw", "run"),
    ("Double Wing Power", "run"), ("Switch HB Dive", "run"), ("Flea Flicker", "pass"),
    # Formation/motion tags still mean pass next to a pass concept, or alone.
    ("PA Jet Sweep", "pass"), ("Spacing Switch", "pass"), ("Double Post", "pass"),
    ("Bunch Mesh", "pass"), ("Jet HB Screen", "pass"), ("Switch", "pass"),
    ("FK Jet HB Shovel", "pass"), ("RPO Jet Sweep", "rpo"),
    # Run-concept words override jet/motion/formation tags.
    ("FK Jet Inside Zone", "run"), ("Mtn Jet HB Zone", "run"), ("End Around", "run"),
    ("Slash FK End Arnd", "run"), ("QB Blast", "run"), ("Mtn QB Blast", "run"), ("Blast Rt", "run"),
    ("QB Wrap", "run"), ("Tush Push", "run"),
    ("Crossers", "pass"), ("Clearout", "pass"), ("Clearout Crosser", "pass"), ("Tight Crosses", "pass"),
    ("Empty Vertical", "pass"), ("TE Hook", "pass"), ("TE In", "pass"),
    ("PA End Around Sift", "pass"), ("Dash Flood", "pass"),
    # Glued names stay unknown (not special-cased).
    ("Mtn Paboot Lt", "unknown"), ("Mtn Paslide", "unknown"), ("Ystick", "unknown"), ("Wrdig", "unknown"),
])
def test_run_pass_formation_tags_do_not_override_run_concepts(play, cls):
    assert sf.call_class(play) == cls


_GLUED = ("Cover Hole1 Rpo Stretch Slant", "Cover 1 Hole Mid Blitz 0", "Cover 3 Sky Tampa 2")


_NONSENSE = "Zzyzx Qwerty Plonk"


def test_defense_family_book_then_pattern_then_other():
    book = sf.DefenseBook(_SYNTH_DEFENSE_BOOK, source="synthetic")
    assert sf.defense_family_path("Cover 3 Sky", book) == ("cover_3", False, "book")
    assert sf.defense_family_path("  cover 3   SKY ", book) == ("cover_3", False, "book")  # case/space only
    assert sf.defense_family_path("DB Fire 2", book) == ("cover_2", True, "book")
    # Off-book names fall back to the text pattern (addendum 3).
    assert sf.defense_family_path("Cover 3 Sky Wk", book) == ("cover_3", False, "pattern")
    assert sf.defense_family_path("Cover 1 Hole", book) == ("cover_1", False, "pattern")
    assert sf.defense_family_path("Nickel Fire Zone", book) == ("pressure_other", False, "pattern")
    assert sf.defense_on_book("Cover 1 Hole", book) is False
    # Only a name neither step resolves is other.
    assert sf.defense_family_path(_NONSENSE, book) == ("other", False, "other")
    assert sf.is_pressure(*sf.defense_family(_NONSENSE, book)) is False
    assert sf.defense_family_path("", book) == ("none", False, None)
    assert sf.defense_family(None, book) == ("none", False)
    assert sf.defense_on_book(None, book) is None


def test_off_book_and_unblanked_glued_names_map_by_pattern_on_every_path():
    """No glued special case: the trainer's frozen filter blanks glued calls upstream.
    A glued name that reaches the module unblanked maps by pattern like any other."""
    book = sf.DefenseBook(_SYNTH_DEFENSE_BOOK, source="synthetic")
    for name in _GLUED + ("Cover 3 Sky Wk", _NONSENSE):
        expected = sf.family_and_pressure(name)
        path = "other" if expected[0] == "other" else "pattern"
        train = sf.training_inputs(_label_row("defense", name), defense_book=book)
        live = inference.vod_live_inputs("defense", name, down=1, distance=10, yardline=25, defense_book=book)
        for inputs in (train, live):
            assert (inputs["look_family"], inputs["pressure"]) == expected, name
            assert inputs["family_path"] == path, name
        assert sf.feature_levels(train) == inference.vod_candidate_features(
            "defense", name, down=1, distance=10, yardline=25, quarter=1, clock="12:00", defense_book=book,
        )
        swapped = sf.swap_candidate(sf.training_inputs(_label_row("defense", "Cover 3 Sky"), defense_book=book),
                                    name, defense_book=book)
        assert (swapped["look_family"], swapped["family_path"]) == (expected[0], path)
        assert sf.defense_family(name) == expected  # catalog default: same answer
    assert sf.defense_family_path("Cover Hole1 Rpo Stretch Slant", book)[::2] == ("cover_1", "pattern")


def test_book_lookup_matches_pattern_on_every_book_name_and_keeps_blitz_pressure():
    assert len(sf.FAMILY_OVERRIDES) == 1
    for book in (sf.DefenseBook(_SYNTH_DEFENSE_BOOK), sf.catalog_defense_book()):
        mismatches = [name for name in book.table if sf.defense_family(name, book) != sf.pattern_family(name)[:2]]
        assert mismatches == []
    book = sf.DefenseBook(_BLITZ_7)
    for play, family in _BLITZ_7.items():
        got_family, flag = sf.defense_family(play, book)
        assert got_family == family, play
        assert sf.is_pressure(got_family, flag) is True, play
    for play in _BLITZ_7:
        assert play in sf.catalog_defense_book()


def test_defense_book_reads_playbook_shapes(tmp_path):
    side = {"formations": {"Nickel 2-4": ["Cover 3 Sky", "Free Fire"], "Dime": ["Cover 3 Sky"]}}
    blob = {"applied": {"offense": {"formations": {"Gun": ["Mesh"]}}, "defense": side}}
    for source in (side, blob, ["Cover 3 Sky", "Free Fire"]):
        book = sf.DefenseBook.from_playbook(source)
        assert sorted(book.table) == ["cover 3 sky", "free fire"]
    path = tmp_path / "book.json"
    path.write_text(__import__("json").dumps(blob))
    book = sf.DefenseBook.from_file(path)
    assert len(book) == 2 and "mesh" not in book
    assert book.describe()["plays"] == 2
    previous = sf._ACTIVE_BOOK
    try:
        sf.set_defense_book(book)
        assert sf.defense_family_path("Cover 2 Invert") == ("cover_2", False, "pattern")
        assert sf.defense_family_path("Free Fire") == ("pressure_other", False, "book")
    finally:
        sf.set_defense_book(previous)
    assert sf.defense_family_path("Cover 2 Invert") == ("cover_2", False, "book")


def test_eval_labels_executed_def_call_through_the_book_lookup():
    from scripts.madden_ml import eval_scoring

    book = sf.DefenseBook(_SYNTH_DEFENSE_BOOK)
    got = eval_scoring.executed_call_label("defense", "Cover 1 Hole", defense_book=book)
    assert got["family"] == "cover_1" and got["family_path"] == "pattern" and got["is_pressure"] is False
    got = eval_scoring.executed_call_label("defense", _NONSENSE, defense_book=book)
    assert got["family"] == "other" and got["family_path"] == "other"
    got = eval_scoring.executed_call_label("defense", "Free Fire", defense_book=book)
    assert got["family"] == "pressure_other" and got["is_pressure"] is True and got["family_path"] == "book"
    assert eval_scoring.executed_call_label("defense", "", defense_book=book)["scored"] is False
    assert eval_scoring.executed_call_label("offense", "Jet Sweep")["call_class"] == "run"


def test_trainer_prep_and_row_features_match_inference(capsys):
    """G6 through the trainer's own path: prep() + row_features() vs vod_candidate_features."""
    pytest.importorskip("numpy")
    from cfb_coach.vod_success import train_vod_model_v6 as v6

    book = sf.DefenseBook(_SYNTH_DEFENSE_BOOK, source="synthetic")
    previous = sf._ACTIVE_BOOK
    n = 0
    mismatches = []
    try:
        sf.set_defense_book(book)
        for row in list(_cases()) + [_label_row("defense", name) for name in _GLUED]:
            row = dict(row, video_id="synthetic")
            prepared = v6.prep(row)
            allowed = v6.allowed_cxs_keys([prepared], min_rows=1)
            cand = row["defense_look"]["def_call"] if row["side"] == "defense" else row["offense_call"]["play"]
            for keyed in (True, False):
                trained = dict(v6.row_features(prepared, enable_cxs=keyed, allowed_cxs=allowed))
                live = inference.vod_candidate_features(
                    row["side"], cand, enable_keyed=keyed, defense_book=book, **_live_kwargs(row),
                )
                n += 1
                if trained != live:
                    mismatches.append((row["side"], cand, keyed))
    finally:
        sf.set_defense_book(previous)
    with capsys.disabled():
        print(f"\n[parity-trainer] prep+row_features rows={n} mismatches={len(mismatches)}")
    assert n > 1000
    assert mismatches == []


def test_shared_module_is_stdlib_only():
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    assert imported <= stdlib, imported - stdlib


def test_inference_and_policy_import_without_training_libs():
    code = (
        "import sys\n"
        "import cfb_coach.vod_success.shared_features\n"
        "import cfb_coach.madden.model.policy\n"
        "import cfb_coach.madden.model.inference\n"
        "bad = sorted(m for m in ('numpy', 'pandas', 'sklearn', 'scipy', 'torch') if m in sys.modules)\n"
        "trainers = sorted(m for m in sys.modules if m.startswith(('cfb_coach.vod_success.train', 'cfb_coach.vod_success.glm')))\n"
        "print(bad, trainers)\n"
        "sys.exit(1 if bad or trainers else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_full_book_feature_build_and_score_under_150ms(capsys):
    offense = [f"{play} {i}" for i in range(4) for play in _SYNTH_OFFENSE_BOOK][:93]
    defense = [f"{play} {i}" for i in range(4) for play in _SYNTH_DEFENSE_BOOK][:78]
    coef = {"look_family": {"cover_3": 0.2}, "call_class": {"pass": 0.1}, "down_x_dist": {"1|long": 0.1}}

    def snap() -> None:
        for side, book in (("offense", offense), ("defense", defense)):
            scores = []
            for play in book:
                feats = inference.vod_candidate_features(
                    side, play, opp_look="Cover 3", down=1, distance=10, yardline=25,
                    yardline_touched=True, quarter=4, clock="",
                )
                scores.append(sum(coef.get(g, {}).get(k, 0.0) for g, k in feats.items()))
            assert len(scores) == len(book)

    # Budget is asserted on CPU time (time.process_time), so other jobs on a
    # loaded box can't fail it. Wall-clock is printed as a diagnostic only.
    wall0, cpu0 = time.perf_counter(), time.process_time()
    snap()
    cold_cpu = (time.process_time() - cpu0) * 1000.0
    cold_wall = (time.perf_counter() - wall0) * 1000.0
    cpu_runs, wall_runs = [], []
    for _ in range(30):
        wall0, cpu0 = time.perf_counter(), time.process_time()
        snap()
        cpu_runs.append((time.process_time() - cpu0) * 1000.0)
        wall_runs.append((time.perf_counter() - wall0) * 1000.0)
    cpu_runs.sort()
    wall_runs.sort()

    def pct(runs):
        return runs[len(runs) // 2], runs[int(len(runs) * 0.95) - 1]

    p50, p95 = pct(cpu_runs)
    w50, w95 = pct(wall_runs)
    with capsys.disabled():
        print(f"\n[latency] 93 offense + 78 defense candidates (CPU, asserted): cold={cold_cpu:.2f}ms "
              f"p50={p50:.2f}ms p95={p95:.2f}ms max={cpu_runs[-1]:.2f}ms (budget 150ms); "
              f"wall diagnostic: cold={cold_wall:.2f}ms p50={w50:.2f}ms p95={w95:.2f}ms max={wall_runs[-1]:.2f}ms")
    assert p95 < 150.0
    assert cold_cpu < 150.0


# Names whose family the Strategist re-checked and may differ from the 1b85600
# snapshot. Every other catalog defensive play (a superset of the applied book)
# must keep its 1b85600 family and pressure flag.
_STRATEGIST_CHECKED: frozenset[str] = frozenset({"roll 23 cloud"})


def test_book_families_keep_their_1b85600_values():
    """Negative test: no book play's family or pressure flag drifts from the 1b85600 snapshot."""
    import json

    snap = json.loads((_ROOT / "tests" / "madden_ml" / "fixtures" / "defense_family_snapshot_1b85600.json")
                      .read_text(encoding="utf-8"))["plays"]
    book = sf.catalog_defense_book()
    assert len(snap) == len(book) == 332
    drift = []
    for name, (family, pressure) in snap.items():
        if name in _STRATEGIST_CHECKED:
            continue
        got = sf.defense_family_path(name, book)
        if got != (family, pressure, "book"):
            drift.append((name, (family, pressure), got))
    assert drift == []


def test_roll_23_cloud_is_cover_3_through_the_named_override():
    assert sf.FAMILY_OVERRIDES == {"roll 23 cloud": ("cover_3", False)}
    # The cloud pattern alone does not resolve it ("23" is not a single-digit shell).
    assert sf.family_and_pressure("Roll 23 Cloud") == ("other", False)
    assert sf.pattern_family("Roll 23 Cloud") == ("cover_3", False, "override")
    for book in (sf.DefenseBook(["Roll 23 Cloud", "Cover 3 Sky"]), sf.catalog_defense_book(), sf.DefenseBook([])):
        assert sf.defense_family_path("Roll 23 Cloud", book) == ("cover_3", False, "override")
        assert sf.defense_family_path("  roll 23   CLOUD ", book) == ("cover_3", False, "override")
    assert sf.defense_on_book("Roll 23 Cloud", sf.DefenseBook([])) is False
    assert sf.defense_on_book("Roll 23 Cloud", sf.catalog_defense_book()) is True
    train = sf.training_inputs(_label_row("defense", "Roll 23 Cloud"))
    live = inference.vod_live_inputs("defense", "Roll 23 Cloud", down=1, distance=10, yardline=25)
    for inputs in (train, live):
        assert (inputs["look_family"], inputs["family_path"]) == ("cover_3", "override")
