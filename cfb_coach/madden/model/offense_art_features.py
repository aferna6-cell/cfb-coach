"""Fast, offline-compiled Madden 27 visual route-art feature bank.

No image, OCR or external lookup is accessed at inference time. Features encode
*shape and spacing*, not named player routes. Two cross-site name aliases are
model-eligible solely on the user's explicit assumption; they are not Xbox-verified.
"""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

DATA = Path(__file__).resolve().parents[2] / 'data' / 'madden27' / 'offense_play_art_features.json'
SCHEMA = 'madden27.offense_route_art_features.v1'
EXPECTED_INVENTORY = 'OFF-3C6D-E543'
# Exact user-approved cross-source assumptions; never generalize to unknown aliases.
USER_APPROVED_ALIASES = {
    ('Gun 5WR Tight', 'L Spot'): 'Z Spot',
    ('Gun 5WR Tight', 'Zl Spot Shake'): 'Z Spot Shake',
}
FEATURE_NAMES = (
    'art_present', 'art_ink_density', 'art_yellow_share', 'art_red_share', 'art_cyan_share',
    'art_left_share', 'art_middle_share', 'art_right_share',
    'art_shallow_share', 'art_intermediate_share', 'art_deep_share',
    'art_deep_left', 'art_deep_middle', 'art_deep_right',
    'art_horizontal_share', 'art_spread',
)


class RouteArtError(ValueError):
    """Structural route feature bank is corrupt or mismatched."""


def validate(data: Mapping[str, Any]) -> None:
    if data.get('schema') != SCHEMA or data.get('inventory_fingerprint') != EXPECTED_INVENTORY:
        raise RouteArtError('wrong route-art feature schema or installed playbook fingerprint')
    if data.get('feature_names') != list(FEATURE_NAMES):
        raise RouteArtError('route-art features or their order changed; rebuild and retrain')
    rows = data.get('plays')
    if not isinstance(rows, list) or len(rows) != 93:
        raise RouteArtError('expected 93 plays from the installed offensive playbook')
    keys: set[tuple[str, str]] = set()
    for p in rows:
        key = (p.get('formation'), p.get('play'))
        if not all(isinstance(a,str) and a for a in key) or key in keys:
            raise RouteArtError(f'missing or duplicate play key: {key}')
        keys.add(key)
        vals = p.get('features')
        if not isinstance(vals,dict) or set(vals) != set(FEATURE_NAMES):
            raise RouteArtError(f'{key}: incomplete/incompatible visual features')
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and
                   math.isfinite(v) and 0 <= v <= 1.001 for v in vals.values()):
            raise RouteArtError(f'{key}: invalid visual feature value')
        if not isinstance(p.get('model_eligible'),bool):
            raise RouteArtError(f'{key}: missing source-name approval')
        if not p['model_eligible'] and p.get('exact_name_match'):
            raise RouteArtError(f'{key}: incongruent approval flags')
        # Mismatched labels must have explicit, scoped user approval; otherwise
        # even a manually flipped model_eligible flag cannot smuggle in art.
        approved_alias = bool(p.get('user_approved_alias', False))
        expected_alias = USER_APPROVED_ALIASES.get(key)
        if not p.get('exact_name_match') and p['model_eligible']:
            if not approved_alias or not expected_alias or p.get('site_play_name') != expected_alias:
                raise RouteArtError(f'{key}: unapproved or mismatched source alias')
        if approved_alias and (p.get('exact_name_match') or
                               not p['model_eligible'] or not expected_alias or
                               p.get('site_play_name') != expected_alias):
            raise RouteArtError(f'{key}: inconsistent user-approved source alias')
    usable = sum(p['model_eligible'] for p in rows)
    counts = data.get('coverage', {})
    if counts.get('total') != 93 or counts.get('model_eligible') != usable or \
            counts.get('quarantined_name_conflicts') != 93-usable:
        raise RouteArtError('visual feature-bank approval counts do not match records')


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    raw = json.loads(DATA.read_text(encoding='utf8'))
    validate(raw)
    return raw


@lru_cache(maxsize=1)
def _index() -> dict[tuple[str,str],dict[str,Any]]:
    return {(p['formation'],p['play']): p for p in load()['plays']}


def art_features(formation: str, play: str) -> dict[str,float]:
    """Visual geometry for stock plays (including two approved assumptions); unknown => zeros."""
    rec = _index().get((formation,play))
    if rec is None or not rec['model_eligible']:
        return {name:0.0 for name in FEATURE_NAMES}
    return dict(rec['features'])


def status(formation: str, play: str) -> str:
    rec = _index().get((formation,play))
    if rec is None:
        return 'not_in_installed_book'
    return 'visual_layout_available' if rec['model_eligible'] else 'label_discrepancy_quarantined'


def data_digest() -> str:
    """Training binds to exact feature bank; refuses silent feature changes."""
    return hashlib.sha256(DATA.read_bytes()).hexdigest()