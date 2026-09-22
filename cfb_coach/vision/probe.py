"""`cfb-coach watch --probe` — one-shot capture diagnosis on Aidan's machine.

Answers, per backend (wgc / dxcam / mss), with evidence instead of guesses:
  - did frames arrive at all? (and the exact exception if not)
  - are they BLACK?   → protected surface / wrong or cloaked window
  - are they FROZEN?  → Remote Play paused (focus left the stream) / static menu
  - is there GRASS?   → the crop actually shows the field (vision can't work without it)
  - how many player blobs does the classical CV find?

Saves one PNG per backend to ~/.cfb-coach/probe/ so the crop can be eyeballed.
Sidecar-only: reads pixels, never sends input.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

# HSV grass range — kept identical to players.extract_player_centroids.
_GRASS_LO = (35, 40, 40)
_GRASS_HI = (90, 255, 255)


def probe_dir() -> Path:
    d = Path.home() / ".cfb-coach" / "probe"
    d.mkdir(parents=True, exist_ok=True)
    return d


def frame_stats(bgr: Any, prev_bgr: Any = None) -> dict[str, float]:
    """Pure numpy/cv2 stats for one frame (unit-tested with synthetic arrays)."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(
        hsv, np.array(_GRASS_LO, np.uint8), np.array(_GRASS_HI, np.uint8)
    )
    stats = {
        "mean": float(gray.mean()),
        "black_frac": float((gray < 16).mean()),
        "grass_frac": float((grass > 0).mean()),
        "change": -1.0,
    }
    if prev_bgr is not None and getattr(prev_bgr, "shape", None) == bgr.shape:
        a = cv2.resize(gray, (160, 90))
        b = cv2.resize(cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY), (160, 90))
        stats["change"] = float(cv2.absdiff(a, b).mean())
    return stats


def verdict(n_frames: int, stats: dict[str, float] | None, error: str | None) -> str:
    """Human verdict from probe evidence. Order matters: worst failure first."""
    if n_frames == 0:
        return f"NO FRAMES — {error}" if error else "NO FRAMES — backend returned nothing"
    if stats is None:
        return "NO STATS"
    if stats["black_frac"] > 0.95:
        return (
            "BLACK — surface is capture-protected, or the matched window is "
            "hidden/minimized/wrong"
        )
    if 0.0 <= stats["change"] < 0.5:
        return (
            "FROZEN — identical frames for the whole probe: Remote Play paused "
            "(focus left the stream?) or a static menu"
        )
    if stats["grass_frac"] < 0.10:
        return (
            "NO FIELD — frames arrive but <10% grass: crop is wrong, or a "
            "menu/play-call screen/pause overlay is up. Vision cannot read looks here."
        )
    return "OK — live field frames; the capture path works"


def _dep(name: str) -> str:
    return "yes" if importlib.util.find_spec(name) is not None else "no"


def _probe_backend(
    label: str, factory: Any, seconds: float, out: Path
) -> tuple[int, dict[str, float] | None, str | None, int]:
    """Grab for `seconds`; returns (n_new_frames, stats(last vs first), error, blobs)."""
    from cfb_coach.vision.players import extract_player_centroids

    n = 0
    first = last = None
    err: str | None = None
    cap = None
    try:
        cap = factory()
        t_end = time.time() + seconds
        while time.time() < t_end:
            try:
                f = cap.grab()
            except Exception as e:  # capture the real reason
                err = f"{type(e).__name__}: {e}"
                break
            bgr = getattr(f, "bgr", None) if f is not None else None
            if bgr is not None:
                n += 1
                if first is None:
                    first = bgr
                last = bgr
            time.sleep(1 / 30)
        err = err or getattr(cap, "last_error", None)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        if cap is not None:
            try:
                cap.close()
            except Exception:
                pass
    if last is None:
        return 0, None, err, 0
    stats = frame_stats(last, first if n > 1 else None)
    blobs = len(extract_player_centroids(last))
    try:
        import cv2  # type: ignore

        cv2.imwrite(str(out / f"probe_{label}.png"), last)
    except Exception:
        pass
    return n, stats, err, blobs


def run_probe(
    window: str | None = None,
    *,
    seconds: float = 3.0,
    countdown: float = 5.0,
) -> int:
    from cfb_coach.vision.winenum import (
        enumerate_windows,
        ensure_dpi_aware,
        foreground_title,
        primary_monitor_size,
        rank_matches,
        window_search_needles,
    )

    print("CFB Coach capture probe (sidecar — reads pixels only, never sends input)")
    print(f"  python {sys.version.split()[0]} on {sys.platform}")
    if sys.platform != "win32":
        print("  Not Windows — run this from NATIVE Windows Python (not WSL).")
        return 2
    print(f"  DPI awareness: {ensure_dpi_aware()}   primary monitor: {primary_monitor_size()}")
    print(
        "  deps: numpy={} cv2={} mss={} dxcam={} windows_capture={}".format(
            _dep("numpy"), _dep("cv2"), _dep("mss"), _dep("dxcam"), _dep("windows_capture")
        )
    )
    if _dep("numpy") == "no" or _dep("cv2") == "no":
        print('  Missing numpy/opencv — pip install -e ".[vision]"')
        return 2

    wins = enumerate_windows(include_hidden=True) or []
    needle_hits: list[Any] = []
    print(f"\nWindows matching {window or 'Xbox'!r} (+aliases), incl. hidden ones:")
    for needle in window_search_needles(window or "Xbox"):
        for w in wins:
            if needle.lower() in w.title.lower() and w not in needle_hits:
                needle_hits.append(w)
                print(f"  {w.describe()}")
    if not needle_hits:
        print("  (none) — is Remote Play open? run: cfb-coach watch --list-windows")
        return 1
    target = None
    for needle in window_search_needles(window or "Xbox"):
        hits = rank_matches(wins, needle)
        if hits:
            target = hits[0]
            break
    if target is None:
        print("  All matches are CLOAKED/MINIMIZED/terminal — restore the stream window.")
        return 1
    print(f"\nTarget: {target.describe()}")

    print(
        f"\n>>> CLICK THE XBOX STREAM NOW (keep it focused). Probing in {countdown:.0f}s…"
    )
    time.sleep(max(0.0, countdown))
    print(f"  foreground window during probe: {foreground_title()!r}")

    from cfb_coach.vision.capture import (
        DxcamWindowCapture,
        MssRegionCapture,
        ltrb_to_mss_region,
    )

    backends: list[tuple[str, Any]] = []
    if _dep("windows_capture") == "yes":
        from cfb_coach.vision.wgc_capture import WgcWindowCapture

        backends.append(("wgc", lambda: WgcWindowCapture(hwnd=target.hwnd, title=target.title)))
    if _dep("dxcam") == "yes":
        backends.append(("dxcam", lambda: DxcamWindowCapture(region=target.client, window_substring=None)))
    if _dep("mss") == "yes":
        backends.append(("mss", lambda: MssRegionCapture(region=ltrb_to_mss_region(target.client))))

    out = probe_dir()
    any_ok = False
    print("")
    for label, factory in backends:
        n, stats, err, blobs = _probe_backend(label, factory, seconds, out)
        v = verdict(n, stats, err)
        any_ok = any_ok or v.startswith("OK")
        fps = n / seconds if seconds > 0 else 0.0
        s = (
            f"mean={stats['mean']:.0f} black={stats['black_frac']:.0%} "
            f"grass={stats['grass_frac']:.0%} change={stats['change']:.1f} blobs={blobs}"
            if stats
            else ""
        )
        print(f"  {label:6s} fps~{fps:4.1f}  {s}")
        print(f"         → {v}")
    if _dep("windows_capture") == "no":
        print("  wgc    (skipped — pip install windows-capture to test Windows Graphics Capture)")
    print(f"\nSnapshots: {out}  (open them — is it the FIELD, full-frame, no browser chrome?)")
    print("Read docs/vision-capture-rca.md for what each verdict means.")
    return 0 if any_ok else 1
