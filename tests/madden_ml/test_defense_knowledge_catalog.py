"""The defensive knowledge catalog: complete, provenance-backed, reproducible, and honest about what is NOT known."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from cfb_coach.madden.model import defense_catalog as dc
from cfb_coach.madden.model import defense_lexicon as L

# The issue's required coverage, restated independently of the snapshot so a silent change to either is caught.
INSTALLED_COUNTS = {"4-3 Over": 18, "46 Normal": 9, "Dime 3-2 Odd": 27, "Nickel Double Mug": 15, "Nickel Over": 42, "Nickel Single Mug": 12}
EXTERNAL = ["Cover 1 Spy", "Cover 1 Thief", "Cover 1 Sting", "Cover 2 Match", "Cover 2 Drop", "Cover 2 Spy", "Cover 2 Man Spy", "Cover 3 Mable",
            "3 DBL Mable Cloud", "3 Double Mable", "Cover 3 Buzz Seam", "Cover 3 Buzz Press", "Cover 3 Cloud Show 2", "Cover 3 Sim Pressure",
            "Cover 3 Sammie", "Cover 4 Show 2", "Cover 4 Zone", "Prevent", "Man 3 Deep", "3 Double Buzz", "Cover 3 Drop", "Cover 3 Buzz",
            "3 Deep Blitz", "Fire Zone 2", "FZ 3 Sky", "Buck Zone Blitz", "SS 2 Trap"]


@pytest.fixture(scope="module")
def cat():
    return dc.load_catalog()


@pytest.fixture(scope="module")
def by_id(cat):
    return {e["id"]: e for e in cat["entries"]}


def entry(by_id, formation, name):
    return by_id[f"inst:{formation}|{name}"]


# ------------------------------------------------------------------------------------------------ completeness

def test_committed_catalog_is_valid_and_reproducible(cat):
    assert dc.validate_catalog(cat) == []
    assert dc.build_catalog()["digest"] == cat["digest"], "committed catalog differs from a fresh build: run scripts/madden_ml/build_defense_catalog.py"


def test_every_installed_entry_exists_with_its_raw_name_untouched(cat):
    snap = json.loads(dc.SNAPSHOT_PATH.read_text())["sides"]["defense"]
    assert snap["inventory_fingerprint"] == "DEF-F1CF-1862" == cat["built_for_fingerprint"]
    installed = [e for e in cat["entries"] if e["status"] == "installed"]
    assert len(installed) == 123
    counts: dict[str, int] = {}
    for e in installed:
        counts[e["raw_formation"]] = counts.get(e["raw_formation"], 0) + 1
    assert counts == INSTALLED_COUNTS
    assert {(e["raw_formation"], e["raw_name"]) for e in installed} == {(f, p) for f, ps in snap["formations"].items() for p in ps}


def test_every_external_variant_is_present(cat):
    ext = [e for e in cat["entries"] if e["status"] == "external_identified"]
    assert sorted(e["raw_name"] for e in ext) == sorted(EXTERNAL) and len(ext) == 27
    assert all(e["raw_formation"] is None and e["source_book"] is None for e in ext)


def test_same_name_in_different_formations_is_not_collapsed(by_id):
    sky = [e for e in by_id.values() if e["status"] == "installed" and e["raw_name"] == "Cover 3 Sky"]
    assert sorted(e["raw_formation"] for e in sky) == ["4-3 Over", "Nickel Double Mug", "Nickel Over"]
    assert len({e["id"] for e in sky}) == 3
    assert set(sky[0]["same_name_other_formations"]) == {e["raw_formation"] for e in sky} - {sky[0]["raw_formation"]}


def test_near_duplicate_spellings_are_kept_as_separate_entries_but_cross_linked(by_id):
    a = entry(by_id, "Nickel Over", "Cov 3 Buzz Match")
    b = entry(by_id, "Nickel Over", "Cover 3 Buzz Match")
    assert a["id"] != b["id"] and a["name_key"] == b["name_key"]
    assert a["possible_duplicate_of"] == b["id"] and b["possible_duplicate_of"] == a["id"]


# ------------------------------------------------------------------------------------------------ provenance

def test_every_record_has_provenance_and_known_sources(cat):
    srcs = cat["sources"]
    for e in cat["entries"]:
        assert e["evidence"], e["id"]
        for ev in e["evidence"]:
            assert ev["source"] in srcs and ev["claim"]
        for ax in (e["coverage"]["deep_shell"], e["coverage"]["type"], e["pressure"]["concept"]):
            assert ax["status"] in L.STATUS_ORDER
            assert all(s in srcs for s in ax["sources"])
        assert e["review"]["status"] == "unreviewed" and e["review"]["reviewer"] is None
        assert e["licensed_for_training"]["third_party_text"] is False


def test_the_eleven_slot_ledger_is_unknown_everywhere_and_nothing_claims_xbox_verification(cat):
    for e in cat["entries"]:
        assert len(e["ledger"]["slots"]) == 11
        assert all(s["status"] == "unknown" and s["position"] is None and s["assignment"] is None for s in e["ledger"]["slots"])
        assert e["ledger"]["known_count"] == 0 and e["ledger"]["verified_count"] == 0
        assert e["verification"]["needs_xbox_screenshot"] is True and e["verification"]["screenshot_sha256"] is None
        assert e["verification"]["level"] != "role_verified"
        assert any("11 defender" in u for u in e["unknowns"]), e["id"]
        assert any("unobservable" in u for u in e["unknowns"]), "opponent adjustments must be recorded as UNKNOWN, never as absent"
    assert cat["summary"]["ledger_known_slots"] == 0 and cat["summary"]["ledger_verified_slots"] == 0


def test_catalog_coverage_and_responsibility_verification_are_reported_separately(cat):
    s = cat["summary"]
    assert s["entries"] == 150 and sum(s["verification_levels"].values()) == 150
    assert s["verification_levels"].get("role_verified", 0) == 0
    assert s["verification_levels"]["role_partially_documented"] == 3          # the three third-party-annotated defensive plays only


def test_third_party_per_position_jobs_are_research_display_only(by_id):
    for name in ("Cover 3 Sky", "Tampa 2", "Cover 4 Quarters"):
        ref = entry(by_id, "Nickel Over", name)["ledger"]["third_party_reference"]
        assert ref and ref["model_input"] is False and ref["jobs"]
    assert entry(by_id, "4-3 Over", "Cover 3 Sky")["ledger"]["third_party_reference"] is None


# ------------------------------------------------------------------------------------------------ honesty of individual claims

def test_football_level_definitions_are_labelled_hypotheses_not_madden_facts(by_id):
    sky = entry(by_id, "Nickel Over", "Cover 3 Sky")
    assert sky["coverage"]["deep_shell"]["value"] == 3 and sky["coverage"]["deep_shell"]["status"] == "football_hypothesis"
    tok = next(v for v in sky["coverage"]["variants"] if v["token"] == "sky")
    assert tok["meaning_status"] == "football_hypothesis" and tok["meaning_feature_eligible"] is False   # unfetched, conflicting source
    assert tok["present"]["status"] == "name_derived"                                                     # the TOKEN is a fact about the label


def test_ea_documented_mechanics_are_distinguished_from_hypotheses(by_id):
    assert entry(by_id, "Nickel Over", "Cover 3 Match")["coverage"]["type"]["status"] == "ea_documented"
    assert entry(by_id, "Dime 3-2 Odd", "Cover 6")["coverage"]["type"]["status"] == "ea_documented"
    assert entry(by_id, "Nickel Over", "Cover 4 Palms")["coverage"]["type"]["status"] == "ea_documented"
    assert entry(by_id, "Nickel Over", "Cover 2 Man")["coverage"]["type"]["status"] == "name_derived"
    assert entry(by_id, "Nickel Over", "Cover 9 Show 2")["coverage"]["type"]["value"] == "unknown"        # EA does not define Cover 9


def test_pre_release_third_party_education_is_not_treated_as_madden_27_verification(by_id, cat):
    hole = next(v for v in entry(by_id, "4-3 Over", "Cover 1 Hole")["coverage"]["variants"] if v["token"] == "hole")
    assert hole["meaning_status"] == "third_party_unverified" and hole["meaning_feature_eligible"] is False
    assert cat["sources"]["mutgg_cover1"]["published"] == "2026-01-09"
    assert "BEFORE" in cat["sources"]["mutgg_cover1"]["note"]


def test_a_trailing_digit_is_a_hint_not_a_coverage_claim(by_id):
    fz = entry(by_id, "46 Normal", "Fire Zone 3")
    assert fz["coverage"]["deep_shell"]["value"] == "unknown" and fz["coverage"]["deep_shell"]["status"] == "unknown"
    assert fz["pressure"]["digit_hint"]["value"] == 3 and "UNVERIFIED" in fz["pressure"]["digit_hint"]["meaning"]
    assert fz["pressure"]["rusher_count"]["value"] == "unknown" and fz["pressure"]["dropping_defenders"]["value"] == "unknown"


def test_coverage_and_pressure_stay_separate_axes_for_hybrids(by_id):
    fz = by_id["ext:FZ 3 Sky"]
    assert fz["coverage"]["base"] == "c3" and any(v["token"] == "sky" for v in fz["coverage"]["variants"])
    assert fz["pressure"]["concept"]["value"] == "zone_blitz"
    sim = by_id["ext:Cover 3 Sim Pressure"]
    assert sim["coverage"]["base"] == "c3" and sim["pressure"]["concept"]["value"] == "sim"
    assert entry(by_id, "Dime 3-2 Odd", "Zero Blitz")["coverage"]["base"] == "c0"


def test_pressure_types_are_distinguished_not_collapsed(by_id):
    kinds = {n: entry(by_id, f, n)["pressure"]["concept"]["value"] for f, n in
             [("46 Normal", "Fire Zone 3"), ("Dime 3-2 Odd", "FS Zone Blitz"), ("Dime 3-2 Odd", "Zero Blitz"), ("Nickel Over", "Nickel Sim 2"),
              ("Nickel Double Mug", "Blitz Loop 3"), ("Nickel Over", "Edge Blitz 3"), ("Nickel Over", "Cover 2 Man")]}
    assert kinds == {"Fire Zone 3": "zone_blitz", "FS Zone Blitz": "zone_blitz", "Zero Blitz": "man_blitz", "Nickel Sim 2": "sim",
                     "Blitz Loop 3": "stunt", "Edge Blitz 3": "unspecified", "Cover 2 Man": "standard"}


def test_unexplainable_names_are_recorded_as_such_not_decomposed_by_guesswork(by_id):
    e = entry(by_id, "4-3 Over", "Thunder Smoke")
    assert e["verification"]["level"] == "catalog_only"
    assert any("thunder smoke" in u for u in e["unknowns"])
    assert e["coverage"]["deep_shell"]["value"] == "unknown"


def test_siblings_state_documented_and_undocumented_differences_separately(by_id):
    sky = entry(by_id, "4-3 Over", "Cover 3 Sky")
    names = {s["raw_name"] for s in sky["nearest_siblings"]}
    assert {"Cover 3", "Cover 3 Cloud Str"} & names
    docs = [d for s in sky["nearest_siblings"] for d in s["documented_difference"]]
    assert any("Sky vs" in d or "Cloud vs Sky" in d for d in docs) and all("unverified" in d.lower() or "convention" in d.lower() for d in docs
                                                                           if "Sky" in d)
    assert all("unknown" in s["unverified"] for s in sky["nearest_siblings"])


def test_stress_statements_are_hypotheses_and_never_model_inputs(cat):
    n = 0
    for e in cat["entries"]:
        for h in e["offensive_stress_hypotheses"]:
            n += 1
            assert h["status"] == "football_hypothesis" and h["model_input"] is False and "Hypothesis only" in h["limitation"]
            assert "guaranteed" not in h["stress"].lower() and "always" not in h["stress"].lower()
    assert n >= 150                                   # every entry with a recognizable trait carries at least one
    assert all(e["offensive_stress_hypotheses"] for e in cat["entries"] if e["coverage"]["base"] in ("c0", "c1", "c2", "t2", "c3", "c4", "c6"))
    # no stress statement is invented where none is sourced
    assert not any(e["offensive_stress_hypotheses"] for e in cat["entries"]
                   if e["coverage"]["base"] in ("c9", "m3d", "prev") and e["pressure"]["concept"]["value"] == "standard"
                   and not e["coverage"]["variants"])


# ------------------------------------------------------------------------------------------------ the validator really gates

def _bad(cat, mutate):
    c = copy.deepcopy(cat)
    mutate(c)
    c["digest"] = dc.digest_of(c)          # re-seal so ONLY the content rule under test can fail
    return dc.validate_catalog(c)


def test_validator_rejects_a_missing_installed_entry(cat):
    errs = _bad(cat, lambda c: c["entries"].pop(0))
    assert any("installed" in e for e in errs)


def test_validator_rejects_an_altered_raw_name(cat):
    def m(c):
        c["entries"][0]["raw_name"] = c["entries"][0]["raw_name"] + "X"
    assert any("installed entries differ" in e for e in _bad(cat, m))


def test_validator_rejects_a_missing_or_extra_external_variant(cat):
    errs = _bad(cat, lambda c: c["entries"].pop(-1))
    assert any("external variants" in e for e in errs)


def test_validator_rejects_a_verified_claim_without_a_screenshot_hash(cat):
    def m(c):
        c["entries"][5]["coverage"]["deep_shell"]["status"] = "xbox_verified"
    assert any("xbox_verified claim without a screenshot hash" in e for e in _bad(cat, m))


def test_validator_rejects_verified_slots_without_a_hash_and_inconsistent_counts(cat):
    def m(c):
        s = c["entries"][7]["ledger"]["slots"][0]
        s.update(status="xbox_verified", position="FS", assignment="deep middle")
    errs = _bad(cat, m)
    assert any("ledger counts inconsistent" in e for e in errs)


def test_validator_rejects_feature_eligibility_resting_on_third_party_text(cat):
    def m(c):
        ax = c["entries"][3]["coverage"]["deep_shell"]
        ax.update(feature_eligible=True, sources=["mutgg_cover1"], status="football_hypothesis")
    assert any("non-trainable source" in e for e in _bad(cat, m))


def test_validator_rejects_third_party_jobs_flagged_as_model_input(cat):
    idx = next(i for i, e in enumerate(cat["entries"]) if e["ledger"]["third_party_reference"])
    def m(c):
        c["entries"][idx]["ledger"]["third_party_reference"]["model_input"] = True
    assert any("third-party jobs flagged" in e for e in _bad(cat, m))


def test_validator_rejects_a_stress_hypothesis_marked_as_model_input(cat):
    def m(c):
        c["entries"][0]["offensive_stress_hypotheses"][0]["model_input"] = True
    assert any("stress hypothesis" in e for e in _bad(cat, m))


def test_validator_rejects_a_hand_edited_catalog_with_a_stale_digest(cat):
    c = copy.deepcopy(cat)
    c["entries"][0]["unknowns"] = []
    assert any("digest" in e for e in dc.validate_catalog(c))


def test_building_the_catalog_never_modifies_its_inputs():
    paths = [dc.SNAPSHOT_PATH, dc.ANNOTATIONS_PATH]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    dc.build_catalog()
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


# ------------------------------------------------------------------------------------------------ reuse: consistent with the PR #107 classifier

def test_catalog_never_contradicts_the_pr107_scheme_classifier_where_both_commit(cat):
    from cfb_coach.madden.model import coverage_intelligence as ci

    def shell_of(s):
        if s.startswith("cover_1"):
            return "1"
        if s in ("cover_2_man", "cover_2_spot", "cover_2_invert", "tampa_2", "trap_2"):
            return "2"
        if s.startswith("cover_3"):
            return "3"
        if s.startswith("cover_4"):
            return "4"
        return "split" if s in ("cover_6", "cover_9") else None

    type_of = {"cover_2_man": "man", "cover_1_hole": "man", "cover_1_robber": "man", "cover_1_contain": "man", "cover_1_man_other": "man",
               "cover_3_match": "match", "cover_6": "match", "cover_4_quarters": "match", "cover_4_palms": "match",
               "cover_3_spot": "spot_zone", "cover_2_spot": "spot_zone", "tampa_2": "spot_zone", "cover_4_drop": "spot_zone"}
    compared = 0
    for e in (x for x in cat["entries"] if x["status"] == "installed"):
        c = ci.classify_defensive_play(e["raw_name"])
        sh, mine = shell_of(c.scheme_id), str(e["coverage"]["deep_shell"]["value"])
        if sh and mine != "unknown":
            compared += 1
            assert sh == mine, (e["id"], c.scheme_id, mine)
        t, mt = type_of.get(c.scheme_id), e["coverage"]["type"]["value"]
        if t and mt != "unknown":
            assert t == mt, (e["id"], c.scheme_id, mt)
    assert compared >= 60                      # the comparison is not vacuous
