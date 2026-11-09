"""Strict first-party eleven-player Madden defensive role verification."""
from __future__ import annotations

import json
import pytest
from cfb_coach.madden.model import defense_v2_roles as roles


def fixture():
    return {"schema": roles.SCHEMA, "plays": [{
        "formation": "Nickel Over", "play": "Cover 3 Sky",
        "verified_in_madden": True, "source": "owner_xbox_play_art",
        "verified_on": "2026-11-09",
        "slots": [{"slot_id": f"player_{i}", "role": "observed",
                   "responsibility": f"verified responsibility {i}", "verified": True}
                  for i in range(11)],
    }]}


def test_roles_unknown_until_exact_eleven_player_verification(tmp_path):
    assert roles.verified_roles(None, "Nickel Over", "Cover 3 Sky")["missing_roles"] == 11
    p = tmp_path / "verified-roles.json"
    p.write_text(json.dumps(fixture()))
    found = roles.verified_roles(p, "Nickel Over", "Cover 3 Sky")
    assert found["verified_defenders"] == 11
    assert found["verification"] == "OWNER_XBOX_VERIFIED"
    assert roles.verified_roles(p, "Dime", "Cover 3 Sky")["verification"] == "UNKNOWN"


def test_roles_invalid_verification_fails_closed():
    data = fixture()
    data["plays"][0]["slots"][0]["verified"] = False
    with pytest.raises(ValueError):
        roles.verify_document(data)
    data = fixture()
    data["plays"][0]["slots"][1]["slot_id"] = "player_0"
    with pytest.raises(ValueError):
        roles.verify_document(data)
    data = fixture()
    data["plays"][0]["slots"].pop()
    with pytest.raises(ValueError):
        roles.verify_document(data)
