"""Label resolution and structural features: different defensive structures give different inputs; nothing is guessed."""
from __future__ import annotations

import itertools
import json

import pytest

from cfb_coach.madden.model import defense_catalog as dc
from cfb_coach.madden.model import defense_semantics as ds
from cfb_coach.madden.model import historical_defense_baseline as hb


def axes(label, formation=None):
    r = ds.resolve(label, formation)
    assert r.key, label
    return ds.key_axes(r.key)


def diff(a, b):
    fa, fb = axes(a), axes(b)
    return {n for n in fa if fa[n] != fb[n]}


# ------------------------------------------------------------------------------------------------ resolution

def test_all_123_installed_entries_resolve_where_the_old_recognizer_named_38():
    snap = json.loads(dc.SNAPSHOT_PATH.read_text())["sides"]["defense"]["formations"]
    rows = [(f, p) for f, ps in snap.items() for p in ps]
    assert len(rows) == 123
    assert sum(hb.normalize_label(p) in hb.KNOWN_COVERAGES for _f, p in rows) == 38           # the baseline from the gap audit
    assert all(ds.resolve(p, f).key is not None for f, p in rows)
    assert all(ds.resolve(p, f).resolution == "exact_by_formation" for f, p in rows)


def test_all_external_variants_resolve():
    ext = [e["raw_name"] for e in ds.catalog()["entries"] if e["status"] == "external_identified"]
    assert len(ext) == 27 and all(ds.resolve(n).resolution == "exact_external" for n in ext)


def test_raw_label_is_preserved_exactly():
    for raw in ("Cover 3 Sky", "  cover 3   SKY ", "Cov 3 Buzz Match", "weird label"):
        assert ds.resolve(raw).raw == raw


def test_spelling_variants_share_a_key_and_formation_picks_the_exact_entry():
    assert ds.resolve("Cov 3 Buzz Match").key == ds.resolve("Cover 3 Buzz Match").key == "dk:cover_3_buzz_match"
    assert ds.resolve("cover 3  sky").key == ds.resolve("Cover 3 Sky").key
    r = ds.resolve("Cover 3 Sky", "Nickel Over")
    assert r.resolution == "exact_by_formation" and r.entry_ids == ("inst:Nickel Over|Cover 3 Sky",)
    assert ds.resolve("Cover 3 Sky").n_formations == 3        # same name, three installed formations: assignments may differ


def test_every_name_the_old_recognizer_accepted_resolves_to_the_same_parent_class():
    for alias in hb.ALIASES:
        r = ds.resolve(alias)
        assert r.key is not None, alias
        lab = hb.normalize_label(alias)
        assert r.legacy_bucket == hb.GENERIC_TO_BUCKET.get(lab, lab), (alias, r.legacy_bucket)


def test_unrecognized_ambiguous_and_empty_labels_stay_unknown_never_guessed():
    for raw in ("Cover 3 Sky (strong)", "Cover 3 Weirdo", "something", "", None, "Cover 7", "Man blitz maybe"):
        r = ds.resolve(raw)
        assert r.key is None and r.resolution == "unknown" and r.legacy_bucket == "unknown", raw


def test_generic_pressure_is_kept_as_pressure_without_inventing_cover_0_or_a_blitz_type():
    r = ds.resolve("pressure")
    assert r.key == ds.GENERIC_PRESSURE_KEY and r.legacy_bucket == "unknown"
    a = ds.key_axes(r.key)
    assert a["dk_press_unspecified"] == 1.0 and a["dk_press_man_blitz"] == a["dk_press_zone_blitz"] == 0.0
    assert a["dk_shell_0"] == 0.0 and a["dk_shell_unk"] == 1.0


def test_generic_cover_labels_get_family_level_features_only():
    r = ds.resolve("Cover 2")
    assert r.resolution == "parsed_generic" and r.legacy_bucket == "cover2"
    a = ds.key_axes(r.key)
    assert a["dk_shell_2"] == 1.0 and a["dk_var_invert"] == a["dk_type_man"] == 0.0 and a["dk_ev_exact"] == 0.0


# ------------------------------------------------------------------------------------------------ different structures -> different inputs

COVER3 = ["Cover 3", "Cover 3 Sky", "Cover 3 Cloud", "Cover 3 Buzz Match", "Cover 3 Match", "Cover 3 Hard Flat", "Cover 3 Mable"]
COVER2 = ["Cover 2 Man", "Cover 2 Invert", "Nickel 2 Trap", "Tampa 2", "Cover 2 Hard Flat", "Cover 2 Roll"]
SHELLS = ["Cover 4 Quarters", "Cover 4 Palms", "Cover 4 Drop", "Cover 6", "Cover 9 Show 2"]
PRESSURE = ["Fire Zone 3", "Zero Blitz", "Nickel Sim 2", "Blitz Loop 3", "Edge Blitz 3"]


@pytest.mark.parametrize("group", [COVER3, COVER2, SHELLS, PRESSURE, ["Cover 1 Hole", "Cover 1 Robber Press", "Cover 1 Contain"]])
def test_distinct_calls_within_a_family_have_distinct_feature_vectors(group):
    for a, b in itertools.combinations(group, 2):
        assert diff(a, b), f"{a} and {b} produced identical inputs"


def test_cloud_sky_buzz_match_differ_on_the_axes_that_name_them():
    assert diff("Cover 3 Sky", "Cover 3 Cloud") == {"dk_var_sky", "dk_var_cloud"}
    assert {"dk_var_buzz", "dk_type_match", "dk_type_spot_zone"} <= diff("Cover 3 Sky", "Cover 3 Buzz Match")
    assert axes("Cover 3 Match")["dk_type_match"] == 1.0 and axes("Cover 3 Sky")["dk_type_spot_zone"] == 1.0


def test_cover2_man_spot_invert_trap_tampa_are_not_one_class():
    assert axes("Cover 2 Man")["dk_type_man"] == 1.0 and axes("Tampa 2")["dk_type_spot_zone"] == 1.0
    assert axes("Cover 2 Invert")["dk_var_invert"] == 1.0 and axes("Nickel 2 Trap")["dk_var_trap"] == 1.0
    assert len({tuple(sorted(axes(n).items())) for n in COVER2}) == len(COVER2)


def test_quarters_palms_and_split_field_are_distinguished():
    assert axes("Cover 4 Quarters")["dk_var_quarters"] == 1.0 and axes("Cover 4 Palms")["dk_var_palms"] == 1.0
    assert axes("Cover 6")["dk_shell_split"] == axes("Cover 9 Show 2")["dk_shell_split"] == 1.0
    assert axes("Cover 6")["dk_type_match"] == 1.0 and axes("Cover 9 Show 2")["dk_type_unk"] == 1.0     # EA documents 6, not 9


def test_zone_blitz_man_blitz_sim_and_stunt_are_different_pressure_axes():
    assert axes("Fire Zone 3")["dk_press_zone_blitz"] == 1.0
    assert axes("Zero Blitz")["dk_press_man_blitz"] == 1.0 and axes("Zero Blitz")["dk_shell_0"] == 1.0
    assert axes("Nickel Sim 2")["dk_press_sim"] == 1.0 and axes("Blitz Loop 3")["dk_press_stunt"] == 1.0
    assert axes("Edge Blitz 3")["dk_press_unspecified"] == 1.0
    assert axes("Cover 3 Sky")["dk_press_any"] == 0.0 and axes("Fire Zone 3")["dk_press_any"] == 1.0


def test_match_pressure_hybrids_retain_both_axes():
    a = axes("FZ 3 Sky")
    assert a["dk_shell_3"] == 1.0 and a["dk_var_sky"] == 1.0 and a["dk_press_zone_blitz"] == 1.0
    b = axes("Cover 3 Sim Pressure")
    assert b["dk_shell_3"] == 1.0 and b["dk_press_sim"] == 1.0


def test_pressure_name_digit_is_a_separate_low_trust_hint_not_a_coverage_shell():
    a = axes("Fire Zone 3")
    assert a["dk_hint_3"] == 1.0 and a["dk_shell_3"] == 0.0 and a["dk_shell_unk"] == 1.0
    assert axes("Mid Blitz 0")["dk_hint_0"] == 1.0 and axes("Mid Blitz 0")["dk_shell_0"] == 0.0


def test_unestablished_axes_are_unknown_not_filled_in():
    assert axes("Cover 3 Mable")["dk_type_unk"] == 1.0                         # match-vs-zone conflict is not resolved by fiat
    assert axes("Cover 9 Show 2")["dk_type_unk"] == 1.0
    t = axes("Thunder Smoke")
    assert t["dk_ev_unknown_axes"] == 1.0 and t["dk_shell_unk"] == t["dk_type_unk"] == 1.0


def test_features_never_use_claims_that_are_not_feature_eligible():
    for e in ds.catalog()["entries"]:
        a = ds.key_axes("dk:" + e["name_key"])
        if not e["coverage"]["deep_shell"]["feature_eligible"] or e["coverage"]["deep_shell"]["value"] == "unknown":
            assert a["dk_shell_unk"] == 1.0, e["id"]
        if not e["coverage"]["type"]["feature_eligible"] or e["coverage"]["type"]["value"] == "unknown":
            assert a["dk_type_unk"] == 1.0, e["id"]
        assert a["dk_ev_ledger_known"] == 0.0


def test_axis_vectors_are_valid_one_hot_groups():
    for e in ds.catalog()["entries"]:
        a = ds.key_axes("dk:" + e["name_key"])
        assert sum(a[f"dk_shell_{s}"] for s in ds.SHELLS) == 1.0
        assert sum(a[f"dk_type_{t}"] for t in ds.TYPES) == 1.0
        assert a["dk_press_any"] == (0.0 if a["dk_press_standard"] else 1.0)


# ------------------------------------------------------------------------------------------------ mixes

def test_expected_axes_is_mass_weighted_and_not_renormalized():
    e = ds.expected_axes({"dk:cover_3_sky": 0.5, "dk:fire_zone_3": 0.25})
    assert e["dk_ev_info"] == pytest.approx(0.75) and e["dk_shell_3"] == pytest.approx(0.5)
    assert e["dk_press_zone_blitz"] == pytest.approx(0.25) and e["dk_hint_3"] == pytest.approx(0.25)
    assert ds.expected_axes({})["dk_ev_info"] == 0.0 and ds.expected_axes(None)["dk_shell_3"] == 0.0


def test_exact_family_counts_resolve_typed_text_and_drop_unrecognized_ones():
    c = ds.family_counts(["Cover 3 Sky", "cover 3 sky", "pressure", "nonsense", "Fire Zone 3", None], stored=False)
    assert c == {"dk:cover_3_sky": 2, ds.GENERIC_PRESSURE_KEY: 1, "dk:fire_zone_3": 1}


# ------------------------------------------------------------------------------------------------ the post-play form is a lossy channel

def test_the_true_call_is_always_among_the_readings_of_what_the_form_stored():
    """Property: for every catalog name, whatever the CURRENT parser stores for it, the name's own key is a possible reading."""
    names = sorted({e["raw_name"] for e in ds.catalog()["entries"]})
    for n in names:
        stored = ds.form_store(n)
        if stored is None:
            continue
        w = ds.stored_label_weights(stored)
        true_key = ds.resolve(n).key
        if stored.lower() == "pressure":
            assert w == {ds.GENERIC_PRESSURE_KEY: 1.0}
        else:
            assert true_key in w, (n, stored)


def test_stored_label_weights_form_a_distribution_and_unusable_text_adds_none():
    for lab in ("Cover 3 Sky", "Cover 1", "Cover 2 Man", "Cover 0", "pressure", "Cover 3 Cloud", "Fire Zone 3", "Cover 4 Quarters"):
        w = ds.stored_label_weights(lab)
        assert w and sum(w.values()) == pytest.approx(1.0) and all(v > 0 for v in w.values()), lab
    assert ds.stored_label_weights("gibberish") == {} and ds.stored_label_weights(None) == {} and ds.stored_label_weights("") == {}


def test_a_label_the_form_cannot_produce_is_read_exactly_and_one_it_can_is_a_mixture():
    assert ds.stored_label_weights("Cover 3 Cloud") == {"dk:cover_3_cloud": 1.0}          # lossless capture: exact
    sky = ds.stored_label_weights("Cover 3 Sky")                                            # a collapsed label: many possible calls
    assert len(sky) >= 5 and max(sky.values()) <= 0.2 and "dk:cover_3_sky" in sky


def test_collapsed_labels_keep_family_structure_and_blur_only_the_variant():
    e = ds.expected_axes(ds.stored_label_weights("Cover 3 Sky"))
    assert e["dk_shell_3"] >= 0.85 and e["dk_var_sky"] <= 0.2                                # still a three-deep family; Sky NOT claimed
    assert e["dk_press_any"] <= 0.15                                                         # nearly all candidates are coverage calls
    cov1 = ds.expected_axes(ds.stored_label_weights("Cover 1"))
    assert cov1["dk_shell_1"] >= 0.8 and cov1["dk_var_hole"] < 0.5 and cov1["dk_var_robber"] < 0.5


def test_stored_pressure_stays_untyped_instead_of_becoming_a_guessed_blitz_type():
    e = ds.expected_axes(ds.stored_label_weights("pressure"))
    assert e["dk_press_unspecified"] == 1.0 and e["dk_press_zone_blitz"] == e["dk_press_man_blitz"] == 0.0


def test_channel_audit_quantifies_the_loss_today():
    """Characterizes TODAY's parser. If capture becomes lossless this should change, deliberately."""
    a = ds.channel_audit()
    assert a["unique_names"] == 109
    assert a["stored_exactly_as_typed"] < 15 and a["rewritten_to_coarser_label"] > 60 and a["nothing_stored"] > 15
    assert a["stored_exactly_as_typed"] + a["rewritten_to_coarser_label"] + a["nothing_stored"] == 109
    assert a["largest_collapse"][1] >= 15
    assert "Cover 3 Cloud" in a["stored_label_to_possible_typed_names"]["Cover 3 Sky"]
    assert "Cover 1 Hole" in a["stored_label_to_possible_typed_names"]["Cover 1"] and "Cover 1 Robber Press" in a["stored_label_to_possible_typed_names"]["Cover 1"]


def test_project_legacy_sums_parent_classes_and_passes_legacy_keys_through():
    out = ds.project_legacy({"dk:cover_3_sky": 0.3, "dk:cover_3_cloud": 0.2, "dk:fire_zone_3": 0.1, "cover_6": 0.4})
    assert out == pytest.approx({"cover3": 0.5, "unknown": 0.1, "cover_6": 0.4})


# ------------------------------------------------------------------------------------------------ our own candidates (offline)

def test_candidate_features_separate_our_123_calls_and_carry_personnel():
    snap = json.loads(dc.SNAPSHOT_PATH.read_text())["sides"]["defense"]["formations"]
    vecs = {}
    for f, ps in snap.items():
        for p in ps:
            c = ds.candidate_features(f, p)
            assert c["dk_cand_in_catalog"] == 1.0
            vecs[(f, p)] = tuple(ds.candidate_vector(f, p))
    assert len(set(vecs.values())) > 90                         # distinct structures stay distinct (name-identical calls differ by formation)
    nick = ds.candidate_features("Nickel Over", "Cover 3 Sky")
    dime = ds.candidate_features("Dime 3-2 Odd", "Cover 3 Cloud")
    assert nick["dk_pers_nickel_db5"] == 1.0 and dime["dk_pers_dime_db6"] == 1.0 and dime["dk_pers_odd_front"] == 1.0
    assert ds.candidate_features("46 Normal", "Cover 3")["dk_pers_46"] == 1.0
    assert ds.candidate_vector("Nickel Over", "not a play") == [0.0] * len(ds.CANDIDATE_NAMES)


# ------------------------------------------------------------------------------------------------ explanation never recommends

def test_explain_separates_known_hypothesis_and_unknown_and_never_recommends():
    x = ds.explain("Cover 3 Sky", "Nickel Over")
    assert x["known"] == [] and x["hypotheses"] and any("11 defender" in u for u in x["unknown"])
    assert not {"recommended_play", "beater", "counter", "best_play"} & set(x)
    y = ds.explain("Cover 6", "Dime 3-2 Odd")
    assert any("ea_documented" in k for k in y["known"])
    z = ds.explain("totally unknown")
    assert z["resolution"] == "unknown" and z["known"] == [] and z["unknown"]
