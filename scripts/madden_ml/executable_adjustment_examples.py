"""Print Madden 27 executable-adjustment examples, the catalog status and the old-control audit.

Every loadout, confirmation and threat distribution below is an EXAMPLE, not the user's
real setup. Nothing here reads or writes the Madden database or live settings.

    python scripts/madden_ml/executable_adjustment_examples.py
"""

from __future__ import annotations

import json

from cfb_coach.madden.adjustments import executable as ex

EXAMPLE_LOADOUT = ex.Loadout.from_dict({
    "game_label": "EXAMPLE - not your loadout",
    "offense": [{"name": "Slants", "slot": "LB menu, 2nd item", "installed": True, "active": True,
                 "confirmed_by_user": True, "preset_id": "off.preset.slants"}],
    "defense": [{"name": "Play Short Routes", "slot": "LB menu, 1st item", "installed": True, "active": True,
                 "confirmed_by_user": True, "preset_id": "def.preset.play_short_routes"},
                {"name": "No Deep Passes", "slot": "LB menu, 2nd item", "installed": True, "active": True,
                 "confirmed_by_user": True, "preset_id": "def.preset.no_deep_passes"}],
})
EXAMPLE_CONFIRMATIONS = {
    "off.hot_route": {"confirmed": True}, "off.hot_route|Slant": {"confirmed": True},
    "def.double_team": {"confirmed": True},
}

OFFENSE = ex.PlayContext(
    side="offense", formation="Gun Trips TE", play="Mesh", play_type="pass",
    eligible_receivers=("WR1", "WR2", "WR3", "TE1", "HB"),
    formation_audibles={"Mesh": "pass", "Inside Zone": "run"}, play_clock_remaining=22)
DEFENSE = ex.PlayContext(
    side="defense", formation="Nickel 2-4-5", play="Cover 3 Sky", play_type="coverage",
    eligible_receivers=("WR1", "WR2", "WR3", "TE1", "HB"), defenders=("CB1", "CB2", "SS", "FS", "MLB"),
    play_clock_remaining=22)

SCENARIOS = [
    ("offense vs expected man coverage", OFFENSE, {"man": 0.7, "pressure": 0.2, "two_high": 0.1}),
    ("offense vs balanced/uncertain shell", OFFENSE, {"man": 0.25, "two_high": 0.25, "single_high": 0.25, "cover2": 0.25}),
    ("defense vs crossers/flood heavy", DEFENSE, {"cross": 0.5, "flood": 0.3, "run": 0.1, "vert": 0.1}),
    ("defense vs vertical heavy", DEFENSE, {"vert": 0.6, "flood": 0.15, "cross": 0.1, "run": 0.15}),
]


def main() -> None:
    print("=== EXAMPLE coaching cards (example loadout and confirmations, not yours) ===")
    for label, ctx, dist in SCENARIOS:
        for mode, lo, conf in (("nothing confirmed - today's real state", None, None),
                               ("EXAMPLE direct controls confirmed, no loadout", None, EXAMPLE_CONFIRMATIONS),
                               ("EXAMPLE loadout + direct controls confirmed", EXAMPLE_LOADOUT, EXAMPLE_CONFIRMATIONS)):
            rec = ex.recommend(ctx, dist, loadout=lo, confirmations=conf)
            print(f"\n--- {label} [{mode}] ---")
            print(ex.coaching_card(rec, ctx, loadout=lo))
    print("\n=== Catalog status with NO user confirmations (the real state today) ===")
    print(json.dumps(ex.catalog_report(), indent=2))
    print("\n=== Old research_db control table vs EA Madden 27 map ===")
    for (side, cat), (verdict, why) in sorted(ex.RESEARCH_CONTROL_AUDIT.items()):
        print(f"{side:8} {cat:22} {verdict:10} {why}")
    print("\n=== EXAMPLE pre-game defensive loadout plan (to install, not installed) ===")
    for row in ex.plan_loadout("defense", [(0.5, {"cross": 0.5, "flood": 0.3, "vert": 0.2}),
                                           (0.3, {"vert": 0.6, "run": 0.4}), (0.2, {"scram": 0.6, "run": 0.4})], limit=4):
        print(row)


if __name__ == "__main__":
    main()
