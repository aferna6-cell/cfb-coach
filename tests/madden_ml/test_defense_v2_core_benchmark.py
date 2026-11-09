"""Read-only V2 core benchmark measures actual scorer + adjustments + USER reads."""
from __future__ import annotations

from types import SimpleNamespace

from cfb_coach.madden.model import defense_v2_pilot as v2
from scripts.madden_ml import bench_defense_v2_scoring as bench


def test_core_benchmark_uses_installed_stock_calls_only(monkeypatch):
    monkeypatch.setattr(bench, "active_books", lambda db, sides: {
        "defense": {"formations": {"Nickel Over": ["Cover 3 Sky"]},
                    "locked_ts": "2026-11-09"}
    })
    monkeypatch.setattr(bench, "eligible", lambda raw: {
        "defense": raw["defense"]["formations"]
    })
    monkeypatch.setattr(bench.live, "_confirmed_book", lambda db, book: [
        {"formation": "Nickel Over", "play": "Cover 3 Sky",
         "installed": True, "legal": True}
    ])
    report = bench.benchmark(SimpleNamespace(), v2.cold_start("cpu-test"), n=12)
    assert report["runs"] == 12
    assert report["installed_forms"] == 1
    assert report["installed_calls"] == 1
    assert len(report["samples"]) == 6
    assert all(x["user_read"] for x in report["samples"])
    assert report["complete_live_p95_verified"] is False
    assert report["release_status"] == "CORE_MEASUREMENT_ONLY_NOT_GAME_READY"
