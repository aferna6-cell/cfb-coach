"""The dataset's recommendation is the call that was displayed, not the heuristic's pick."""

from __future__ import annotations

import sqlite3
import unittest
from types import SimpleNamespace

from cfb_coach.madden.model import dataset as dataset_mod


def _db(decision_final: tuple[str, str] | None) -> SimpleNamespace:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE snaps (id INTEGER PRIMARY KEY, opponent_id TEXT, side TEXT, down INT, distance INT, "
        "yardline INT, quarter INT, situation_raw TEXT, our_call TEXT, formation TEXT, play TEXT, result TEXT, "
        "session_id TEXT, ml_snap_id TEXT, executed_status TEXT, executed_verification TEXT)"
    )
    conn.execute(
        "CREATE TABLE ml_decisions (id INTEGER PRIMARY KEY, snap_id TEXT, game_id TEXT, "
        "heuristic_formation TEXT, heuristic_play TEXT, final_formation TEXT, final_play TEXT)"
    )
    conn.execute(
        "INSERT INTO snaps (opponent_id, side, down, distance, yardline, quarter, situation_raw, our_call, "
        "formation, play, result, session_id, ml_snap_id, executed_status, executed_verification) "
        "VALUES ('opp_a', 'defense', 1, 10, 30, 1, '1&10', 'Nickel — Cover 3', 'Nickel', 'Cover 3', '+4', "
        "'g1', 'g1-0001', 'unknown', 'unknown')"
    )
    if decision_final:
        conn.execute(
            "INSERT INTO ml_decisions (snap_id, game_id, heuristic_formation, heuristic_play, final_formation, "
            "final_play) VALUES ('g1-0001', 'g1', 'Dime', 'Cover 6', ?, ?)", decision_final,
        )
    return SimpleNamespace(conn=conn, path=":memory:")


class DisplayedRecommendationTests(unittest.TestCase):
    def test_recommendation_is_the_final_displayed_pick(self):
        rows = dataset_mod.build_rows(db=_db(("Nickel", "Cover 3")))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["recommended_formation"], "Nickel")
        self.assertEqual(rows[0]["recommended_play"], "Cover 3")

    def test_unverified_recommendation_is_never_an_action(self):
        row = dataset_mod.build_rows(db=_db(("Nickel", "Cover 3")))[0]
        self.assertIsNone(row["action_play"])
        self.assertIsNone(row["executed_play"])
        self.assertEqual(row["eligibility"], "outcome_known_execution_uncertain")

    def test_without_a_decision_the_snap_call_stands(self):
        row = dataset_mod.build_rows(db=_db(None))[0]
        self.assertEqual(row["recommended_play"], "Cover 3")


if __name__ == "__main__":
    unittest.main()
