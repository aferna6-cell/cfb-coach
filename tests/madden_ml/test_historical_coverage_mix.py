import sqlite3

import pytest

from cfb_coach.madden.model import football_priors as fp
from cfb_coach.madden.model.historical_coverage_mix import extract_opponent_history_mix, map_coverage_seen


@pytest.mark.parametrize("raw,key", [
    ("Cover 2 Man", "cover_2_man"), ("cover-2 man", "cover_2_man"), ("Cover 3 Match", "cover_3_match"),
    ("Cover 3 Sky", "cover_3_sky"), ("Cover 4 Quarters", "cover_4_quarters"), ("Cover 4 Palms", "cover_4_palms"),
    ("Cover 1 Robber", "cover_1_robber"), ("Cover 1 Hole", "cover_1_hole"), ("Trap 2", "trap_2"),
    ("Tampa 2", "tampa_2"), ("Cover 2 Invert", "cover_2_invert"), ("Cover 6", "cover_6"), ("Cover 0", "cover_0"),
    ("Cover 1 Blitz", "cover_1_pressure"),
    ("Cover 1", "cover_1_unspecified"), ("man", "cover_1_unspecified"), ("single_high", "cover_1_unspecified"),
    ("Cover 2", "cover_2_unspecified"), ("Cover 3", "cover_3_unspecified"), ("Cover 4", "cover_4_unspecified"),
])
def test_mapping_to_descriptor_keys(raw, key):
    assert map_coverage_seen(raw) == key
    assert fp.descriptor(key) is not None


@pytest.mark.parametrize("raw", [None, "", "  ", "???", "Cover 9", "pressure", "zone"])
def test_unknown_is_none(raw):
    assert map_coverage_seen(raw) is None


def _db():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE snaps (id INTEGER PRIMARY KEY, opponent_id TEXT, side TEXT, coverage_seen TEXT)")
    rows = [("cpu", "offense", "Cover 2 Man"), ("cpu", "offense", "Cover 2 Man"), ("cpu", "offense", "Cover 3"),
            ("cpu", "offense", None), ("cpu_dynasty", "offense", "mystery"), ("cpu", "defense", "Cover 4"),
            ("opp_a", "offense", "Cover 1"), ("opp_a", "offense", "Cover 1 Robber")]
    db.executemany("INSERT INTO snaps (opponent_id, side, coverage_seen) VALUES (?,?,?)", rows)
    return db


def test_cpu_offense_filter_and_normalization():
    out = extract_opponent_history_mix(_db())
    assert out["mix_source"] == "opponent_history"
    assert out["total_snaps"] == 5 and out["mapped_snaps"] == 3
    assert out["mix"] == pytest.approx({"cover_2_man": 2 / 3, "cover_3_unspecified": 1 / 3})
    assert sum(out["mix"].values()) == pytest.approx(1.0)
    assert out["unknown_mass"] == pytest.approx(2 / 5)
    assert out["raw_inventory"]["mystery"] == {"key": "UNKNOWN", "count": 1}
    assert "cover_4_unspecified" not in out["mix"]


def test_specific_opponent_id_and_empty():
    out = extract_opponent_history_mix(_db(), opponent_id="opp_a")
    assert out["mix"] == pytest.approx({"cover_1_unspecified": 0.5, "cover_1_robber": 0.5})
    none = extract_opponent_history_mix(_db(), opponent_id="nobody")
    assert none["mix"] == {} and none["unknown_mass"] == 0.0 and none["mix_source"] == "opponent_history"


@pytest.mark.parametrize("raw,key", [
    ("quarterback scramble", None),
    ("whole field zone", None),
    ("palm tree", None),
    ("Cover 9 trap", None),
    ("Cover 4 robber", "cover_4_unspecified"),
])
def test_unrelated_words_and_conflicting_variants(raw, key):
    assert map_coverage_seen(raw) == key


@pytest.mark.parametrize("raw,key", [
    ("quarter", "cover_4_quarters"),
    ("quarters", "cover_4_quarters"),
    ("palms", "cover_4_palms"),
    ("Cover 4 palm", "cover_4_palms"),
    ("Cover 4 hole", "cover_4_unspecified"),
    ("Cover 4 trap", "cover_4_unspecified"),
    ("Cover 2 quarters", "cover_2_unspecified"),
    ("Cover 3 palms", "cover_3_unspecified"),
    ("Cover 2 robber", "cover_2_unspecified"),
    ("Cover 3 hole", "cover_3_unspecified"),
    ("Cover 9 robber", None),
    ("Cover 9 hole", None),
    ("Cover 9 Tampa 2", None),
    ("Cover 9 Cover 4 Quarters", None),
    ("Cover 1 Cover 4 robber", None),
    ("Cover 1 pressure", "cover_1_pressure"),
    ("Cover 1 blitzing", "cover_1_unspecified"),
    ("Cover 2 many", "cover_2_unspecified"),
    ("trapdoor", None),
    ("robbery", None),
])
def test_complete_tokens_and_variant_families(raw, key):
    assert map_coverage_seen(raw) == key


def test_unrecognized_word_does_not_create_historical_mix():
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE snaps (opponent_id TEXT, side TEXT, coverage_seen TEXT)")
        db.execute("INSERT INTO snaps VALUES (?, ?, ?)", ("cpu", "offense", "quarterback scramble"))
        out = extract_opponent_history_mix(db)
    assert out["mix"] == {}
    assert out["unknown_mass"] == 1.0
    assert out["mix_source"] == "opponent_history"
