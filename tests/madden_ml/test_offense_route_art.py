"""PR #110: deterministic Madden 27 route-art features, no copyrighted images shipped."""
from __future__ import annotations

import copy
import hashlib

import pytest

from cfb_coach.madden.model import offense_art_features as art
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import offense_structural_model as sm


def test_93_art_features_and_two_unverified_name_aliases():
    x = art.load()
    assert x['coverage']['total'] == 93
    assert x['coverage']['model_eligible'] == 91
    assert x['coverage']['quarantined_name_conflicts'] == 2
    assert len({(r['formation'],r['play']) for r in x['plays']}) == 93


def test_madden_file_provenance_and_no_image_binaries():
    d = art.load()
    assert d['inventory_fingerprint'] == 'OFF-3C6D-E543'
    assert all(len(p['image_sha256']) == 64 for p in d['plays'])
    assert all(set(p['features']) == set(art.FEATURE_NAMES) for p in d['plays'])
    assert art.data_digest() == hashlib.sha256(art.DATA.read_bytes()).hexdigest()


def test_name_mismatches_are_quarantined():
    for p in ('L Spot','Zl Spot Shake'):
        assert art.status('Gun 5WR Tight', p) == 'label_discrepancy_quarantined'
        assert not any(art.art_features('Gun 5WR Tight',p).values())
    assert art.status('Gun 5WR Tight', 'Mesh') == 'visual_layout_available'


def test_unknown_play_cannot_inherit_some_other_play_art():
    assert art.status('Unknown','Mesh') == 'not_in_installed_book'
    assert art.art_features('Unknown','Mesh') == {k: 0.0 for k in art.FEATURE_NAMES}


def test_two_similar_named_plays_have_distinct_route_geometry():
    assert art.art_features('Gun 5WR Tight','Mesh') != art.art_features('Gun 5WR Tight','Mesh Post')
    assert art.art_features('Gun 5WR Tight','Mesh Post') != art.art_features('Gun 5WR Tight','QB Draw')


def test_art_features_enter_real_structural_model_vector():
    s = pf.structure_lookup('Gun 5WR Tight','Mesh')
    assert s is not None
    names = pf.feature_names()
    vec = pf.full_vector(s, {'unknown':1.0})
    assert len(names) == len(vec)
    fv = dict(zip(names,vec))
    for k,v in art.art_features(s.formation,s.play).items():
        assert fv[k] == pytest.approx(v)
    assert sm.LAMBDA['art'] > 0
    assert pf.group_of('art_deep_share') == 'art'


def test_visual_art_is_stock_context_not_fictional_modified_route_art():
    s = pf.structure_lookup('Pistol Trips','Slants')
    modified = pf.apply_hot_route(s, 'WR1', 'Streak')
    stock = dict(zip(pf.feature_names(),pf.full_vector(s, {'unknown':1.0})))
    adj = dict(zip(pf.feature_names(),pf.full_vector(modified, {'unknown':1.0})))
    for k in art.FEATURE_NAMES:
        assert stock[k] == adj[k]
    assert modified.adjustments


def test_artifact_retraining_required_when_visual_bank_changes():
    a = sm.StructuralArtifact()
    a.art_digest = 'stale'
    assert any('route-art' in e for e in sm.validate_artifact(a))


def test_corrupt_visual_values_rejected():
    bad = copy.deepcopy(art.load())
    bad['plays'][0]['features']['art_spread'] = -1
    with pytest.raises(art.RouteArtError):
        art.validate(bad)


def test_documented_source_does_not_claim_exact_player_route():
    for p in art.load()['plays']:
        assert not any(k.startswith('WR') or k.startswith('OL') for k in p['features'])
        assert 'source' in p and 'site_play_name' in p


def test_unlicensed_third_party_route_rows_not_used_for_model_training():
    # Source-documented routes are kept for review, but the ML vector uses only
    # screenshot-derived visual geometry or first-party verified named routes.
    s = pf.structure_lookup('Gun Trips X Nasty', 'Mtn Stick Wheel')
    assert s.n_known > 0
    values = dict(zip(pf.feature_names(),pf.full_vector(s, {'unknown':1.0})))
    assert values['known_route_frac'] == 0.0
    assert values['art_present'] == 1.0


def test_visual_coverage_interactions_are_specific_and_nonheuristic():
    s = pf.structure_lookup('Gun Trips X Nasty','Mtn Stick Wheel')
    names = pf.feature_names()
    palms = dict(zip(names,pf.full_vector(s, {'cover_4_palms':1.0})))
    quarters = dict(zip(names,pf.full_vector(s, {'cover_4_quarters':1.0})))
    assert palms['am_palms_shallow_outside'] > 0
    assert palms['am_quarters_deep'] == 0
    assert quarters['am_palms_shallow_outside'] == 0
    assert quarters['am_quarters_deep'] > 0
    # Interactions are descriptive quantities; the model learns their weights.
    assert sm.LAMBDA['art'] > 0 and pf.group_of('am_palms_shallow_outside') == 'art'


def test_cover6_cover9_receive_separate_route_art_interactions():
    s = pf.structure_lookup('Gun 5WR Tight', 'Mesh')
    names = pf.feature_names()
    a = dict(zip(names,pf.full_vector(s, {'cover_6':1.0})))
    b = dict(zip(names,pf.full_vector(s, {'cover_9':1.0})))
    assert a['am_cover6_deep_left'] > 0 and a['am_cover9_deep_right'] == 0
    assert b['am_cover9_deep_right'] > 0 and b['am_cover6_deep_left'] == 0


def test_unknown_opponent_coverage_gives_zero_art_coverage_features():
    s = pf.structure_lookup('Gun 5WR Tight','Mesh')
    f=dict(zip(pf.feature_names(),pf.full_vector(s, {'unknown':1.0})))
    assert all(f[n]==0.0 for n in pf.feature_names() if n.startswith('am_'))
    assert f['art_present']==1.0