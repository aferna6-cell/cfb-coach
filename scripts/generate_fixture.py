#!/usr/bin/env python3
"""Generate tiny synthetic PNG + player-coord JSON for vision tests."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path


def _png_rgb(w: int, h: int, rgb: tuple[int, int, int] = (34, 139, 34)) -> bytes:
    """Minimal RGB PNG without PIL/cv2."""
    r, g, b = rgb
    raw = b""
    for y in range(h):
        raw += b"\x00"
        row = bytearray()
        for x in range(w):
            # simple gradient + "player" dots for visual variety
            if (x + y) % 17 == 0:
                row.extend([20, 20, 200])
            else:
                row.extend([r, g, b])
        raw += bytes(row)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    root.mkdir(parents=True, exist_ok=True)
    png = root / "synthetic_field.png"
    data = _png_rgb(320, 240, (34, 139, 34))
    png.write_bytes(data)
    coords = {
        "bunch_r": [
            (0.48, 0.34), (0.50, 0.35), (0.52, 0.34),
            (0.68, 0.26), (0.70, 0.27), (0.72, 0.25),
            (0.18, 0.22),
        ],
        "trips_l": [
            (0.48, 0.34), (0.50, 0.35), (0.52, 0.34),
            (0.12, 0.22), (0.22, 0.26), (0.32, 0.24),
            (0.78, 0.22),
        ],
        "empty": [
            (0.50, 0.36),
            (0.05, 0.18), (0.15, 0.20), (0.28, 0.17),
            (0.72, 0.17), (0.85, 0.20), (0.95, 0.18),
        ],
        "two_by_two": [
            (0.50, 0.34),
            (0.18, 0.24), (0.28, 0.22),
            (0.72, 0.24), (0.82, 0.22),
        ],
        "defense_two_high": [
            (0.30, 0.55), (0.40, 0.52), (0.50, 0.53), (0.60, 0.52), (0.70, 0.55),
            (0.35, 0.80), (0.65, 0.80),
        ],
    }
    (root / "player_coords.json").write_text(json.dumps(coords, indent=2), encoding="utf-8")
    print(f"wrote {png} ({png.stat().st_size} bytes)")
    print(f"wrote {root / 'player_coords.json'}")


if __name__ == "__main__":
    main()
