"""Persona aliases must not substring-match NFL team names (t387u)."""
from __future__ import annotations

import unittest

from cfb_coach.madden.human_opponents import resolve_madden_opponent
from cfb_coach.opponents import resolve_opponent


# These names used to resolve to "cpu" because the alias "nd" sits inside them.
_ND_TEAMS = (
    "Indianapolis Colts",
    "indianapolis colts",
    "Indianapolis",
    "Colts",
    "IND",
    "Cleveland Browns",
    "Browns",
    "CLE",
    "New England Patriots",
    "Patriots",
    "NE",
    "Washington Commanders",
    "Commanders",
    "WAS",
)
# "houston" is a college alias for quen; "sc" sits inside "San Francisco".
_OTHER_TEAMS = (
    ("Houston Texans", None),
    ("Texans", None),
    ("HOU", None),
    ("San Francisco 49ers", None),
    ("49ers", None),
    ("SF", None),
    ("niners", None),
)


class NflTeamsDoNotSoftMatchPersonas(unittest.TestCase):
    def test_listed_nfl_teams_do_not_resolve_to_a_persona(self) -> None:
        for raw in _ND_TEAMS:
            with self.subTest(raw=raw):
                self.assertIsNone(resolve_opponent(raw), raw)
                self.assertIsNone(resolve_madden_opponent(raw), raw)
        for raw, _ in _OTHER_TEAMS:
            with self.subTest(raw=raw):
                self.assertIsNone(resolve_opponent(raw), raw)
                self.assertIsNone(resolve_madden_opponent(raw), raw)

    def test_exact_short_aliases_still_work(self) -> None:
        self.assertEqual(resolve_opponent("nd"), "cpu")
        self.assertEqual(resolve_opponent("ND"), "cpu")
        self.assertEqual(resolve_opponent("notre dame"), "cpu")
        self.assertEqual(resolve_opponent("houston"), "quen")
        self.assertEqual(resolve_opponent("sc"), "michael")
        self.assertEqual(resolve_opponent("gavin"), "gavin")
        self.assertEqual(resolve_opponent("auburn"), "gavin")

    def test_whole_token_soft_match_still_works(self) -> None:
        # "south carolina" is a two-token alias; a longer phrase may still match.
        self.assertEqual(resolve_opponent("south carolina gamecocks"), "michael")

    def test_short_alias_is_not_a_substring_hit(self) -> None:
        self.assertIsNone(resolve_opponent("indianapolis"))
        self.assertIsNone(resolve_opponent("commanders"))
        self.assertIsNone(resolve_opponent("england"))
        self.assertIsNone(resolve_opponent("francisco"))
