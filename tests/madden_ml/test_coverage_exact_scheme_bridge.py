"""Run inside the full CFB Coach checkout (PR #110+)."""
from __future__ import annotations
import pytest
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import historical_defense_baseline as hb


def test_exact_palms_vs_quarters_keeps_separate_probabilities():
    d, source = pf.scheme_distribution({"cover_4_palms":0.6,"cover_4_quarters":0.4})
    assert source == "opponent_coverage_mix_uniform_within_bucket"
    assert d["cover_4_palms"] == pytest.approx(0.6)
    assert d["cover_4_quarters"] == pytest.approx(0.4)
    assert set(d)=={"cover_4_palms","cover_4_quarters"}


def test_generic_cover_four_remains_ambiguous():
    d, _=pf.scheme_distribution({"cover4":1.0})
    assert set(d) == {"cover_4_quarters","cover_4_palms","cover_4_drop"}
    assert sum(d.values())==pytest.approx(1.0)


def test_coverage_mix_preserves_named_tampa_and_split_field():
    probs={"probabilities":{"tampa_2":0.5,"cover_6":0.2,"cover_9":0.2,"cover_2_unspecified":0.1}}
    mixed=hb.offense_model_mix(probs)
    d,_=pf.scheme_distribution(mixed)
    assert d["tampa_2"]==pytest.approx(0.5)
    assert d["cover_6"]==pytest.approx(0.2)
    assert d["cover_9"]==pytest.approx(0.2)
    assert d["cover_2_man"]>0 and d["cover_2_spot"]>0  # unspecified Cover2 stays ambiguous


def test_historical_model_does_not_make_a_certain_cover2_claim():
    rows=[{"coverage":"cover_2_unspecified","opponent_id":"cpu","down":1,"distance":10}]
    p=hb.predict_coverage(rows,{"opponent_id":"cpu","down":1,"distance":10})
    mix=hb.offense_model_mix(p)
    assert "cover_2_spot" not in mix and "cover2" in mix
