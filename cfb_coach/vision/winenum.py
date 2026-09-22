"""Win32 window enumeration for capture targeting — ctypes only (no pywin32).

Why this exists (see docs/vision-capture-rca.md):
- The old path read GetWindowRect while the process was still DPI-unaware, so
  on a 125–150% scaled laptop every rect was in *logical* px while dxcam / mss
  capture *physical* px → wrong crop or out-of-range region.
- GetWindowRect includes the invisible ~7px resize border. A window snapped to
  the left half starts at x=-7; snapped right ends 7px past the monitor edge.
  dxcam rejects both ("Invalid Region") and ThreadedCapture swallowed that
  exception → endless "waiting for frames".
- IsWindowVisible() is True for DWM-*cloaked* windows (suspended UWP apps,
  Xbox Game Bar, other virtual desktops). Matching one of those captures
  whatever happens to be on screen at its stale rect.

Everything here is a no-op returning None/False off Windows so Linux/WSL tests
and the typing-only `play` flow never touch it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

LTRB = tuple[int, int, int, int]

# Window classes we never want to capture as "the game".
_TERMINAL_CLASSES = frozenset(
    {
        "ConsoleWindowClass",  # conhost PowerShell / cmd
        "CASCADIA_HOSTING_WINDOW_CLASS",  # Windows Terminal
        "PseudoConsoleWindow",
    }
)

_DWMWA_EXTENDED_FRAME_BOUNDS = 9
_DWMWA_CLOAKED = 14
_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4

_dpi_state: str | None = None


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    rect: LTRB  # visible frame (DWM extended bounds), physical screen px
    client: LTRB  # client (drawable) area, physical screen px
    cloaked: bool = False
    minimized: bool = False
    is_terminal: bool = False

    @property
    def capturable(self) -> bool:
        return not (self.cloaked or self.minimized or self.is_terminal)

    @property
    def area(self) -> int:
        l, t, r, b = self.client
        return max(0, r - l) * max(0, b - t)

    def describe(self) -> str:
        flags = []
        if self.cloaked:
            flags.append("CLOAKED")
        if self.minimized:
            flags.append("MINIMIZED")
        if self.is_terminal:
            flags.append("terminal")
        l, t, r, b = self.client
        f = f" [{' '.join(flags)}]" if flags else ""
        return (
            f"{self.title!r}  class={self.class_name}  "
            f"client=({l},{t})-({r},{b}) {r - l}x{b - t}{f}"
        )


def ensure_dpi_aware() -> str:
    """Make the process per-monitor DPI aware BEFORE any coordinate is read.

    Returns a short state string for diagnostics ("per-monitor-v2", ...,
    "unavailable"). Idempotent.
    """
    global _dpi_state
    if _dpi_state is not None:
        return _dpi_state
    if sys.platform != "win32":
        _dpi_state = "n/a"
        return _dpi_state
    import ctypes

    state = "unaware"
    try:
        user32 = ctypes.windll.user32
        fn = getattr(user32, "SetProcessDpiAwarenessContext", None)
        if fn is not None:
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_int
            if fn(ctypes.c_void_p(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)):
                state = "per-monitor-v2"
        if state == "unaware":
            try:
                # 0 == S_OK; E_ACCESSDENIED means already set (also fine)
                hr = ctypes.windll.shcore.SetProcessDpiAwareness(2)
                state = "per-monitor" if hr in (0, -2147024891) else state
            except Exception:
                pass
        if state == "unaware" and user32.SetProcessDPIAware():
            state = "system"
    except Exception:
        state = "unavailable"
    _dpi_state = state
    return state


def enumerate_windows(*, include_hidden: bool = False) -> list[WindowInfo] | None:
    """Top-level titled windows in Z-order (front first). None off Windows."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    ensure_dpi_aware()
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        dwmapi = ctypes.WinDLL("dwmapi")
    except Exception:
        return None

    HWND = wintypes.HWND
    user32.IsWindowVisible.argtypes = [HWND]
    user32.IsIconic.argtypes = [HWND]
    user32.GetWindowTextLengthW.argtypes = [HWND]
    user32.GetWindowTextW.argtypes = [HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.argtypes = [HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowRect.argtypes = [HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.argtypes = [HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [HWND, ctypes.POINTER(wintypes.POINT)]
    dwmapi.DwmGetWindowAttribute.argtypes = [
        HWND,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long

    out: list[WindowInfo] = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, HWND, wintypes.LPARAM)

    def _cb(hwnd, _lparam):  # noqa: ANN001 — ctypes callback
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            n = user32.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            title = (buf.value or "").strip()
            if not title:
                return True
            cls_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls_buf, 256)
            cls = cls_buf.value or ""

            cloaked = wintypes.DWORD(0)
            dwmapi.DwmGetWindowAttribute(
                hwnd, _DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
            )
            is_cloaked = bool(cloaked.value)
            if is_cloaked and not include_hidden:
                return True

            r = wintypes.RECT()
            hr = dwmapi.DwmGetWindowAttribute(
                hwnd, _DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(r), ctypes.sizeof(r)
            )
            if hr != 0:
                user32.GetWindowRect(hwnd, ctypes.byref(r))
            c = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(c))
            pt = wintypes.POINT(0, 0)
            user32.ClientToScreen(hwnd, ctypes.byref(pt))
            out.append(
                WindowInfo(
                    hwnd=int(hwnd or 0),
                    title=title,
                    class_name=cls,
                    rect=(int(r.left), int(r.top), int(r.right), int(r.bottom)),
                    client=(
                        int(pt.x),
                        int(pt.y),
                        int(pt.x + c.right),
                        int(pt.y + c.bottom),
                    ),
                    cloaked=is_cloaked,
                    minimized=bool(user32.IsIconic(hwnd)),
                    is_terminal=cls in _TERMINAL_CLASSES,
                )
            )
        except Exception:
            pass
        return True

    try:
        user32.EnumWindows(WNDENUMPROC(_cb), 0)
    except Exception:
        return None
    return out


def rank_matches(windows: list[WindowInfo], needle: str) -> list[WindowInfo]:
    """Capturable windows whose title contains needle (case-insensitive).

    Order: exact title > title starts with needle > contains; ties → larger
    client area, then original Z-order. Pure function (unit-tested).
    """
    n = (needle or "").strip().lower()
    if not n:
        return []
    scored: list[tuple[int, int, int, WindowInfo]] = []
    for z, w in enumerate(windows):
        if not w.capturable:
            continue
        t = w.title.lower()
        if n not in t:
            continue
        tier = 0 if t == n else (1 if t.startswith(n) else 2)
        scored.append((tier, -w.area, z, w))
    scored.sort(key=lambda s: (s[0], s[1], s[2]))
    return [s[3] for s in scored]


def clamp_ltrb(rect: LTRB, bounds: LTRB) -> LTRB | None:
    """Intersect rect with bounds (both l,t,r,b). None if empty."""
    l = max(int(rect[0]), int(bounds[0]))
    t = max(int(rect[1]), int(bounds[1]))
    r = min(int(rect[2]), int(bounds[2]))
    b = min(int(rect[3]), int(bounds[3]))
    if r <= l or b <= t:
        return None
    return (l, t, r, b)


def to_output_local(rect: LTRB, output_origin: tuple[int, int], size: tuple[int, int]) -> LTRB | None:
    """Virtual-desktop rect → output-local rect clamped to (0,0,w,h).

    dxcam requires 0 <= left < right <= width (same for y) in *output* px.
    """
    ox, oy = int(output_origin[0]), int(output_origin[1])
    w, h = int(size[0]), int(size[1])
    local = (rect[0] - ox, rect[1] - oy, rect[2] - ox, rect[3] - oy)
    return clamp_ltrb(local, (0, 0, w, h))


def primary_monitor_size() -> tuple[int, int] | None:
    """Physical px of the primary monitor (after DPI awareness). None off Windows."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        ensure_dpi_aware()
        u = ctypes.windll.user32
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        return None


# Common Xbox / Remote Play / Game Bar title fragments (case-insensitive match).
WINDOW_TITLE_ALIASES: tuple[str, ...] = (
    "Xbox",
    "Remote Play",
    "Remote play",
    "Xbox Remote Play",
    "Game Bar",
    "Xbox Game Bar",
    "Widget",
)


def window_search_needles(substring: str | None) -> list[str]:
    """Primary substring then aliases when looking for Xbox/Remote Play.

    Case-insensitive substring match is applied by the finder; this only
    expands which needles to try when the primary fails.
    """
    primary = (substring or "Xbox").strip() or "Xbox"
    needles = [primary]
    low = primary.lower()
    try_aliases = (
        low in {"xbox", "remote", "remote play", "game bar"}
        or "xbox" in low
        or "remote" in low
        or "game bar" in low
    )
    if try_aliases:
        for alias in WINDOW_TITLE_ALIASES:
            if alias.lower() == low:
                continue
            if alias not in needles:
                needles.append(alias)
    return needles


def list_visible_windows() -> list[str] | None:
    """Windows-only: capturable top-level window titles. None if unavailable.

    Uses the ctypes enumerator (DPI-aware, skips DWM-cloaked / minimized /
    terminal windows). Falls back to pywin32 if ctypes enumeration fails.
    """
    wins = enumerate_windows()
    if wins is not None:
        titles = [w.title for w in wins if w.capturable]
    else:
        try:
            import win32gui  # type: ignore
        except ImportError:
            return None
        titles = []

        def _enum(hwnd: int, _: Any) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = (win32gui.GetWindowText(hwnd) or "").strip()
            if title:
                titles.append(title)

        try:
            win32gui.EnumWindows(_enum, None)
        except Exception:
            return None
    # Stable unique order
    seen: set[str] = set()
    out: list[str] = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def find_window(substring: str | None = "Xbox") -> Any:
    """Best capturable WindowInfo for substring (+ Xbox aliases), else None."""
    wins = enumerate_windows()
    if not wins:
        return None
    for needle in window_search_needles(substring):
        hits = rank_matches(wins, needle)
        if hits:
            return hits[0]
    return None


def _enum_windows_matching(needle: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Case-insensitive substring match → [(title, client l,t,r,b), ...].

    Client rect in physical px: no title bar, no invisible resize border.
    Cloaked (suspended UWP / Game Bar / other desktop), minimized and
    terminal windows are never returned.
    """
    wins = enumerate_windows()
    if not wins:
        return []
    return [(w.title, w.client) for w in rank_matches(wins, needle)]


def foreground_title() -> str | None:
    """Title of the window that currently has keyboard focus. None off Windows."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        u = ctypes.windll.user32
        u.GetForegroundWindow.restype = wintypes.HWND
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        hwnd = u.GetForegroundWindow()
        buf = ctypes.create_unicode_buffer(512)
        u.GetWindowTextW(hwnd, buf, 512)
        return buf.value
    except Exception:
        return None
