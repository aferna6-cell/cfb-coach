"""Compile *user-supplied* Madden 27 AceMadden play-art crops to compact visual route features.

Research-only. Does not scrape, download, rehost images, extract conditional blocking, or
pretend to identify player-to-route assignments. The output is numerical route-layout data
for a regularized ML residual. Image processing is OFFLINE; the runtime uses only JSON.

Example:
  python build_offense_art_features.py --captures ~/Madden27_93_offense_play_art_captures \
      --snapshot installed_playbook_snapshot.json --out offense_play_art_features.json

Only exact formation+play source-label matches become model-eligible. Ambiguous cross-site
aliases are captured but quarantined unless a future in-game verification resolves them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit('Pillow is required for offline compilation: pip install Pillow') from exc

SCHEMA = 'madden27.offense_route_art_features.v1'
# 16 fairly low-dimensional, conservative summaries; no per-player labels are inferred.
FEATURE_NAMES = (
    'art_present',
    'art_ink_density',
    'art_yellow_share', 'art_red_share', 'art_cyan_share',
    'art_left_share', 'art_middle_share', 'art_right_share',
    'art_shallow_share', 'art_intermediate_share', 'art_deep_share',
    'art_deep_left', 'art_deep_middle', 'art_deep_right',
    'art_horizontal_share', 'art_spread',
)


def _color(r: int, g: int, b: int) -> str | None:
    # Anti-aliased route strokes are separated from gray/white formation sprites and grid.
    if r > 82 and g > 82 and abs(r - g) < 32 and b * 1.25 < min(r, g):
        return 'yellow'
    if r > 85 and r > 1.46 * max(1, g) and r > 1.32 * max(1, b) and g < 110:
        return 'red'
    if b > 100 and g > 75 and (g > 1.15 * max(1, r) or b > 1.25 * max(1, r)) and b > 1.07 * max(1, r):
        return 'cyan'
    return None


def extract(image_path: Path) -> dict:
    """Extract graphical *occupancy*, not unverified player assignments or route names."""
    im = Image.open(image_path).convert('RGB')
    w, h = im.size
    if not (300 <= w <= 500 and 180 <= h <= 260):
        raise ValueError(f'unexpected AceMadden tile size {w}x{h} for {image_path}')
    # Standardized 0..149 artwork region, excluding bottom labels/call statistics.
    artwork_bottom = 150
    px = im.load()
    pixels: set[tuple[int, int]] = set()
    by_color: Counter[str] = Counter()
    by_lateral: Counter[int] = Counter()
    by_depth: Counter[int] = Counter()
    by_deep_lateral: Counter[int] = Counter()
    x_values: list[float] = []
    for y in range(2, artwork_bottom):
        for x in range(4, w-4):
            c = _color(*px[x, y])
            if c is None:
                continue
            # Remove a small green BREAKDOWN banner if a bright anticorrelated pixel
            # could otherwise be misinterpreted in the artwork's upper left.
            if x < 80 and y < 25 and px[x,y][1] > 1.3 * px[x,y][0]:
                continue
            pixels.add((x, y))
            by_color[c] += 1
            lateral = min(2, int(3*x/w))
            # y=135 corresponds approximately to the LOS, y=0 deep downfield.
            depth = 2 if y < 49 else 1 if y < 92 else 0
            by_lateral[lateral] += 1
            by_depth[depth] += 1
            if depth == 2:
                by_deep_lateral[lateral] += 1
            x_values.append(x / w)
    n = len(pixels)
    if n < 10:
        raise ValueError(f'insufficient nonwhite route ink ({n} pixels): {image_path}')
    # Adjacent pixels encode how horizontal/vertical the route lines actually are.
    horiz = sum((x+1,y) in pixels for x,y in pixels)
    vert = sum((x,y+1) in pixels for x,y in pixels)
    spatial = lambda c: round(c / n, 5)
    mean_x = sum(x_values) / n
    lateral_spread = math.sqrt(max(0.0, sum((x - mean_x) ** 2 for x in x_values) / n))
    values = {
        'art_present': 1.0,
        'art_ink_density': round(min(1.0, n / 2000.0),5),
        'art_yellow_share': spatial(by_color['yellow']),
        'art_red_share': spatial(by_color['red']),
        'art_cyan_share': spatial(by_color['cyan']),
        'art_left_share': spatial(by_lateral[0]),
        'art_middle_share': spatial(by_lateral[1]),
        'art_right_share': spatial(by_lateral[2]),
        'art_shallow_share': spatial(by_depth[0]),
        'art_intermediate_share': spatial(by_depth[1]),
        'art_deep_share': spatial(by_depth[2]),
        'art_deep_left': spatial(by_deep_lateral[0]),
        'art_deep_middle': spatial(by_deep_lateral[1]),
        'art_deep_right': spatial(by_deep_lateral[2]),
        'art_horizontal_share': round(horiz / max(1, horiz+vert),5),
        'art_spread': round(min(1.0, lateral_spread * 3), 5),
    }
    return {'features': values, 'image_size': [w, h], 'route_ink_pixels': n,
            'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
            'graphic_layers': dict(by_color)}


def build(captures: Path, snapshot: Path) -> dict:
    manifest = json.loads((captures / 'manifest.json').read_text())
    original = json.loads(snapshot.read_text())
    offense = original['sides']['offense']
    if offense['inventory_fingerprint'] != 'OFF-3C6D-E543':
        raise ValueError('offensive installed-playbook fingerprint changed: re-audit before training')
    formations = offense['formations']
    expected = {(f, name) for f, names in formations.items() for name in names}
    if len(expected) != 93 or len(manifest['plays']) != 93:
        raise ValueError('expected precisely 93 installed / uploaded play-art entries')
    bykey = {}
    for e in manifest['plays']:
        key = (e['formation'], e['installed_play_name'])
        if key not in expected or key in bykey:
            raise ValueError(f'Unexpected or duplicate image identity: {key}')
        path = captures / e['capture_path']
        if not path.is_file():
            raise FileNotFoundError(path)
        stats = extract(path)
        # Minor apostrophe/spacing variations are not substantive; true label changes are quarantined.
        norm = lambda z: re.sub(r"[^a-z0-9]", "", z.lower())
        match = norm(e['site_play_name']) == norm(e['installed_play_name'])
        bykey[key] = {
            'formation': key[0], 'play': key[1],
            'site_play_name': e['site_play_name'], 'exact_name_match': match,
            'source': 'user_supplied_screenshot_of_third_party_website',
            'source_url': e.get('website_url',''),
            'manifest_crop': e['capture_path'],
            'model_eligible': bool(match),
            'reason': '' if match else 'MISMATCHED SITE LABEL: confirm play identity on Xbox before modeling',
            **stats,
        }
    if set(bykey) != expected:
        raise ValueError(f'missing installed play-art entries: {sorted(expected-set(bykey))}')
    rows = [bykey[(f, play)] for f, plays in formations.items() for play in plays]
    ready = sum(x['model_eligible'] for x in rows)
    return {'schema':SCHEMA,'feature_names':list(FEATURE_NAMES),
            'inventory_fingerprint':'OFF-3C6D-E543',
            'extraction':'HSV-like pixel classification of top play-art region; fixed normalized spatial statistics, no OCR',
            'limitations':'Route-colored occupancy, NOT per-player route names, OL blocks, route order, or adjusted play-art effects.',
            'training_rights':'Images were user-supplied for private analysis; original website images not reproduced in repository.',
            'coverage':{'total':len(rows),'model_eligible':ready,'quarantined_name_conflicts':len(rows)-ready},
            'plays':rows}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise SystemExit(f'refuse overwrite: {args.out}')
    data=build(args.captures,args.snapshot)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf8')
    print(json.dumps({'out':str(args.out),**data['coverage']},indent=2))

if __name__=='__main__':
    main()