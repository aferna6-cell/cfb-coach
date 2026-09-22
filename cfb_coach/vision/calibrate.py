"""Interactive / file calibration — crop & window title → ~/.cfb-coach/vision_calib.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def calib_path() -> Path:
    d = Path.home() / ".cfb-coach"
    d.mkdir(parents=True, exist_ok=True)
    return d / "vision_calib.json"


def load_calib(path: Path | None = None) -> dict[str, Any]:
    p = path or calib_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_calib(data: dict[str, Any], path: Path | None = None) -> Path:
    p = path or calib_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def run_calibrate(*, window: str | None = None, region: str | None = None) -> int:
    """CLI entry: collect window title substring + optional crop region."""
    data = load_calib()
    print("CFB Coach vision calibrate")
    print("Saves crop/window to", calib_path())
    print("SIDE-CAR ONLY — never controls Xbox.\n")

    win = window
    if not win:
        default = data.get("window_substring") or "Xbox"
        try:
            raw = input(f"Window title substring [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        win = raw or default
    data["window_substring"] = win

    reg = region
    if not reg:
        prev = data.get("crop") or data.get("screen_region")
        hint = ""
        if isinstance(prev, dict):
            hint = f" [{prev.get('left')},{prev.get('top')},{prev.get('width')},{prev.get('height')}]"
        try:
            raw = input(
                f"Crop region left,top,width,height (blank=full window){hint}: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        reg = raw or None

    if reg:
        parts = [p.strip() for p in reg.replace("x", ",").split(",")]
        if len(parts) != 4:
            print("Expected left,top,width,height", file=sys.stderr)
            return 2
        try:
            left, top, width, height = (int(x) for x in parts)
        except ValueError:
            print("Region values must be integers", file=sys.stderr)
            return 2
        data["crop"] = {"left": left, "top": top, "width": width, "height": height}
        data["screen_region"] = dict(data["crop"])
    else:
        # keep existing crop unless explicitly cleared — if blank and no prev, none
        if "crop" not in data:
            data["crop"] = None

    # Default ROI stubs (relative 0..1 of crop)
    data.setdefault(
        "rois",
        {
            "down_distance": {"x": 0.40, "y": 0.88, "w": 0.20, "h": 0.08},
            "quarter_clock": {"x": 0.42, "y": 0.02, "w": 0.16, "h": 0.06},
            "score": {"x": 0.35, "y": 0.0, "w": 0.30, "h": 0.05},
            "field": {"x": 0.05, "y": 0.15, "w": 0.90, "h": 0.70},
        },
    )
    data["version"] = 1
    path = save_calib(data)
    print(f"Saved → {path}")
    print(json.dumps(data, indent=2))
    return 0


def parse_screen_region(spec: str) -> dict[str, int]:
    """Parse 'left,top,width,height'."""
    parts = [p.strip() for p in spec.replace("x", ",").split(",")]
    if len(parts) != 4:
        raise ValueError("screen-region must be left,top,width,height")
    left, top, width, height = (int(x) for x in parts)
    return {"left": left, "top": top, "width": width, "height": height}
