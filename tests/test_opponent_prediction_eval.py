import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "madden_ml"))
spec = importlib.util.spec_from_file_location("ope", ROOT / "scripts" / "madden_ml" / "opponent_prediction_eval.py")
ope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ope)


def _game(concepts, results=None):
    rows = []
    for i, c in enumerate(concepts):
        rows.append({"id": i, "ts": f"2026-01-01T00:00:{i:02d}", "side": "defense", "opponent_id": "opp_a",
                     "session_id": "s1", "down": 1 + i % 3, "distance": 10, "yardline": 60, "quarter": 1,
                     "situation_raw": "1&10 my 40", "formation": "Nickel", "play": "Cover 3",
                     "concept_seen": c, "result": (results or {}).get(i, "gain 3"),
                     "score_for": 0, "score_against": 0})
    return rows


def test_a_snaps_own_observation_never_enters_its_features():
    base = _game(["HB Inside Zone", "Mesh", "Four Verticals", "HB Inside Zone"])
    flipped = _game(["HB Inside Zone", "Mesh", "Four Verticals", "Mesh"], results={3: "td"})
    a = ope.build_examples([base])
    b = ope.build_examples([flipped])
    assert a[-1]["x"] == b[-1]["x"]
    assert a[-1]["y"] != b[-1]["y"]


def test_earlier_observations_do_enter_later_features():
    a = ope.build_examples([_game(["HB Inside Zone", "HB Inside Zone", "Mesh"])])
    b = ope.build_examples([_game(["Mesh", "Mesh", "Mesh"])])
    assert a[-1]["x"]["game_opp_pass_share"] < b[-1]["x"]["game_opp_pass_share"]


def test_our_own_call_is_not_an_opponent_label():
    rows = _game([None, None])
    for r in rows:
        r["play"] = "HB Dive"
    assert ope.build_examples([rows]) == []
