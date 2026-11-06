"""Execution-evidence tiers for human-game snaps."""

from __future__ import annotations

import json
import sqlite3
import unittest

from cfb_coach.madden.model import attestation_audit as aa


def _snap(**kw) -> dict:
    base = {
        "id": 1, "session_id": "g1", "opponent_id": "opp_a", "side": "defense", "down": 1, "distance": 10,
        "yardline": 30, "quarter": 1, "situation_raw": "1&10", "formation": "Nickel", "play": "Cover 3", "result": "+2",
        "ml_snap_id": "g1-0001", "executed_status": None, "executed_verification": None,
        "executed_formation": None, "executed_play": None, "play_mismatch": None, "accepted": None,
        "recommended_adjustment_id": None, "executed_adjustment_id": None,
    }
    base.update(kw)
    return base


def _decision(form="Nickel", play="Cover 3") -> dict:
    return {"final_formation": form, "final_play": play, "heuristic_formation": "Dime", "heuristic_play": "Cover 6"}


def _outcome(status="unknown", verification="unknown", form=None, play=None, source=None, accepted=None) -> dict:
    return {"executed_status": status, "executed_verification": verification, "executed_formation": form,
            "executed_play": play, "accepted": accepted,
            "outcome_json": json.dumps({"execution_attestation_source": source}) if source else None}


class TierTests(unittest.TestCase):
    def test_retrospective_attestation_with_a_matching_decision_is_high_confidence(self):
        r = aa.classify(_snap(), _decision(), _outcome())
        self.assertEqual((r.tier, r.attestation_source, r.confidence), (aa.TIER_ATTESTED, "retrospective_statement", "high"))
        self.assertEqual((r.executed_formation, r.executed_play), ("Nickel", "Cover 3"))

    def test_snap_call_without_a_decision_is_medium_confidence(self):
        r = aa.classify(_snap(ml_snap_id=None), None, None)
        self.assertEqual((r.tier, r.confidence, r.recommendation_source), (aa.TIER_ATTESTED, "medium", "legacy_snap_call"))

    def test_heuristic_pick_is_not_the_recommendation(self):
        r = aa.classify(_snap(), _decision(), _outcome())
        self.assertNotEqual(r.executed_play, "Cover 6")

    def test_decision_that_disagrees_with_the_displayed_call_is_excluded(self):
        r = aa.classify(_snap(), _decision(play="Cover 1"), _outcome())
        self.assertEqual(r.tier, aa.TIER_EXCLUDED)
        self.assertIn("decision_disagrees_with_displayed_call", r.reasons)

    def test_missing_call_or_outcome_never_fabricates_an_execution(self):
        r = aa.classify(_snap(play=None), None, None)
        self.assertEqual(r.tier, aa.TIER_EXCLUDED)
        self.assertIsNone(r.executed_play)
        r = aa.classify(_snap(result=None), _decision(), _outcome())
        self.assertIn("missing_outcome_label", r.reasons)

    def test_per_snap_unknown_and_contrary_status_are_excluded(self):
        r = aa.classify(_snap(), _decision(), _outcome(source="per_snap_user_confirmation"))
        self.assertIn("user_marked_execution_unknown_for_this_snap", r.reasons)
        r = aa.classify(_snap(), _decision(), _outcome(status="other"))
        self.assertIn("executed_status_other", r.reasons)
        r = aa.classify(_snap(play_mismatch=1), _decision(), _outcome())
        self.assertIn("play_mismatch_flag", r.reasons)
        r = aa.classify(_snap(), _decision(), _outcome(accepted="false"))
        self.assertIn("recommendation_not_accepted", r.reasons)

    def test_per_snap_verified_execution_is_independent_even_if_different(self):
        r = aa.classify(_snap(), _decision(), _outcome("identified", "verified", "Dime", "Cover 6",
                                                       source="per_snap_user_confirmation"))
        self.assertEqual(r.tier, aa.TIER_VERIFIED)
        self.assertEqual(r.executed_play, "Cover 6")

    def test_standing_toggle_is_attested_not_verified(self):
        r = aa.classify(_snap(side="offense"), _decision(),
                        _outcome("identified", "verified", "Nickel", "Cover 3", source="standing_user_attestation"))
        self.assertEqual((r.tier, r.attestation_source), (aa.TIER_ATTESTED, "standing_prospective"))

    def test_unconfirmed_adjustment_is_flagged_but_play_stays_attested(self):
        r = aa.classify(_snap(recommended_adjustment_id="RZ"), _decision(), _outcome())
        self.assertEqual(r.tier, aa.TIER_ATTESTED)
        self.assertTrue(r.adjustment_unconfirmed)


class AuditTests(unittest.TestCase):
    def test_cpu_games_are_skipped_and_counts_are_grouped(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        cols = list(_snap().keys())
        conn.execute(f"CREATE TABLE snaps ({', '.join(cols)})")
        for i, opp in enumerate(("opp_a", "cpu", "opp_b"), start=1):
            row = _snap(id=i, opponent_id=opp, ml_snap_id=None)
            conn.execute(f"INSERT INTO snaps VALUES ({', '.join('?' * len(cols))})", [row[c] for c in cols])
        rows = aa.audit_rows(conn)
        self.assertEqual(len(rows), 2)
        summary = aa.summarize(rows, alias=aa.opponent_aliases(rows))
        self.assertEqual(summary["by_tier"], {"user_attested:retrospective_statement:medium": 2})
        self.assertEqual(set(summary["by_opponent"]), {"human_1|defense", "human_2|defense"})


if __name__ == "__main__":
    unittest.main()
