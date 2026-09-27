"""Open URLs (and shared WSL helpers) without xdg-open spam on WSL.

Prep opens file:// paths via prep_browser.open_prep_html; live HTML play opens
http:// URLs via open_url. Both detect WSL and prefer Windows-side launchers.
"""

from __future__ import annotations

import os
import subprocess
import webbrowser
from pathlib import Path


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        ver = Path("/proc/version").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "microsoft" in ver.lower()


def try_cmd(argv: list[str]) -> bool:
    try:
        r = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
        )
        return r.returncode == 0
    except (FileNotFoundError, OSError):
        return False


def open_url(url: str) -> None:
    """Open an http(s) URL in the user's browser.

    On WSL, skip webbrowser/xdg-open first — they often "succeed" while flooding
    the terminal with xdg-open errors and never show a window. Prefer wslview,
    then cmd.exe start / powershell Start-Process, then print a clear manual URL.
    """
    if is_wsl():
        if try_cmd(["wslview", url]):
            return
        # empty "" is the START window title so the URL is not treated as title
        if try_cmd(["cmd.exe", "/c", "start", "", url]):
            return
        if try_cmd(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                f"Start-Process '{url}'",
            ]
        ):
            return
        print(f"Open in Windows browser: {url}")
        return

    opened = False
    try:
        opened = bool(webbrowser.open(url))
    except Exception:
        opened = False
    if opened:
        return
    if try_cmd(["open", url]):  # macOS
        return
    if try_cmd(["xdg-open", url]):
        return
    print(f"Could not auto-open browser. Open manually:\n  {url}")


__all__ = ["is_wsl", "open_url", "try_cmd"]
