"""Synthetic geometry only: no downloaded art or football-name inference."""
import io

import pytest
from PIL import Image, ImageDraw
from cfb_coach.madden.model import play_art_extract as pe


def panel(draw, side):
    image = Image.new('RGB', pe.PANEL, 'black')
    draw(ImageDraw.Draw(image))
    data = io.BytesIO()
    image.save(data, 'PNG')
    return pe.analyze(data.getvalue(), side)


def marker(d, x, y):
    d.rectangle((x-4, y-4, x+4, y+4), fill='white')


def test_two_crossing_same_colour_routes_keep_both_starts():
    def draw(d):
        d.line([(100, 180), (300, 40)], fill='yellow', width=4)
        d.line([(300, 180), (100, 40)], fill='yellow', width=4)
        marker(d, 100, 185)
        marker(d, 300, 185)
    ex = panel(draw, 'offense')
    assert len(ex['routes']) == 2
    assert {r['owner_marker_index'] for r in ex['routes']} == {0, 1}
    for r in ex['routes']:
        assert r['end'][1] < 50
        assert r['suggested_shape'] == 'diagonal_upfield'


def test_zone_nearer_marker_and_pale_fill():
    def draw(d):
        d.ellipse((70, 70, 170, 120), fill=(70, 80, 45))
        marker(d, 120, 150)
        marker(d, 330, 150)
    ex = panel(draw, 'defense')
    assert len(ex['zones']) == 1
    assert ex['zones'][0]['colour'] == 'yellow'
    assert ex['zones'][0]['owner_marker_index'] == 0


def test_rush_starts_at_nearby_square():
    def draw(d):
        d.line([(120, 185), (120, 216)], fill='red', width=4)
        marker(d, 120, 180)
    ex = panel(draw, 'defense')
    assert len(ex['rush_arrows']) == 1
    assert ex['rush_arrows'][0]['owner_marker_index'] == 0
    assert ex['rush_arrows'][0]['starts_at_square_marker']


def test_equal_distance_zone_stays_unmatched():
    def draw(d):
        d.ellipse((150, 60, 250, 110), fill=(35, 45, 110))
        marker(d, 160, 140)
        marker(d, 240, 140)
    ex = panel(draw, 'defense')
    assert ex['zones'][0]['owner_marker_index'] is None
    assert 'owner_ambiguous' in ex['zones'][0]['association_flags']


def test_zone_fragments_merge_across_thin_field_line():
    def draw(d):
        d.ellipse((100, 60, 220, 120), fill=(35, 45, 110))
        d.line([(100, 90), (220, 90)], fill='black', width=2)
        marker(d, 160, 150)
    ex = panel(draw, 'defense')
    assert len(ex['zones']) == 1
    assert ex['zones'][0]['area_px'] > 4000


def test_grass_and_chrome_do_not_become_zones():
    def draw(d):
        d.rectangle((0, 0, 555, 257), fill=(10, 22, 8))
        d.rectangle((0, 230, 555, 257), fill=(50, 70, 130))
        d.line([(0, 190), (555, 190)], fill=(100, 100, 100), width=2)
    assert panel(draw, 'defense')['zones'] == []


@pytest.mark.parametrize('path,label', [
    ([(100,180),(102,140),(98,100),(100,30)], 'vertical'),
    ([(100,180),(100,120),(110,80),(130,75),(160,76)], 'break_in'),
    ([(100,180),(100,80),(110,70),(125,95)], 'curl_or_comeback'),
    ([(100,180),(120,195)], 'backward_or_swing'),
])
def test_geometric_shape_suggestions(path, label):
    assert pe._family(path, path[0][0])['suggested_shape'] == label


def test_no_verification_even_for_clear_geometry():
    ex = panel(lambda d: marker(d, 120, 180), 'offense')
    assert ex['extractor'] == 'play_art_extract.v2'
    assert ex['tier'] == 'diagram_extracted_unreviewed'
    assert ex['needs_review'] is True
    assert ex['model_feature_eligible'] is False
    assert ex['verified'] == 0


def test_short_collinear_gap_is_joined_not_separate_routes():
    def draw(d):
        d.line([(120, 170), (120, 120)], fill='yellow', width=4)
        d.line([(120, 105), (120, 30)], fill='yellow', width=4)
        marker(d, 120, 180)
    ex = panel(draw, 'offense')
    assert len(ex['routes']) == 1
    assert ex['routes'][0]['owner_marker_index'] == 0
    assert ex['routes'][0]['end'][1] < 40


def test_nearby_parallel_routes_are_not_chained():
    def draw(d):
        d.line([(120, 172), (120, 30)], fill='yellow', width=4)
        d.line([(135, 172), (135, 30)], fill='yellow', width=4)
        marker(d, 120, 180)
        marker(d, 135, 180)
    ex = panel(draw, 'offense')
    assert len(ex['routes']) == 2
    assert {r['owner_marker_index'] for r in ex['routes']} == {0, 1}


def test_route_start_beyond_old_22_pixel_gate_and_far_zone_flags():
    def draw(d):
        d.line([(100, 150), (100, 30)], fill='yellow', width=4)
        marker(d, 100, 180)
    ex = panel(draw, 'offense')
    assert ex['routes'][0]['owner_marker_index'] == 0
    assert ex['routes'][0]['owner_distance_px'] > 22
    assert pe._owner((10, 10), ex['players'], 120)['association_flags'] == ['owner_too_far']


def test_white_link_does_not_erase_defender_or_become_chrome_zone():
    def draw(d):
        d.ellipse((100, 70, 220, 120), fill=(110, 110, 110))
        d.line([(160, 95), (230, 145)], fill='white', width=3)
        d.ellipse((226, 141, 234, 149), fill='white')
        marker(d, 160, 135)
    ex = panel(draw, 'defense')
    assert len(ex['players']) == 2
    assert len(ex['zones']) == 1
    zone = ex['zones'][0]
    assert zone['association_basis'] == 'visible_white_connector_to_marker'
    assert zone['owner_marker_index'] == 1


def test_dilation_does_not_wrap_opposite_edges():
    import numpy as np
    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 0] = True
    assert not pe._dilate(mask, 2)[:, -1].any()
