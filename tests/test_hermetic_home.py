"""The suite must resolve home and both coach DBs inside the session tmp dir."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from cfb_coach.copilot import default_overlay_path
from cfb_coach.db import default_db_path
from cfb_coach.prep_browser import default_prep_dir


def _session_conftest() -> ModuleType:
    """The conftest module pytest already loaded, not a second import."""
    for name, module in sys.modules.items():
        if not (name == "conftest" or name.endswith(".conftest")):
            continue
        if getattr(module, "_redirected", False) and getattr(module, "_root", None) is not None:
            return module
    raise RuntimeError("hermetic conftest is not the active session module")


def _roots() -> tuple[Path, Path]:
    root = Path(os.environ["CFB_COACH_HERMETIC_ROOT"]).resolve()
    real = Path(os.environ["CFB_COACH_REAL_HOME"]).resolve()
    return root, real


def _assert_off_real_home(path: Path, root: Path, real: Path) -> None:
    resolved = path.resolve()
    assert resolved == root or root in resolved.parents, (resolved, root)
    assert resolved != real
    assert real not in resolved.parents, (resolved, real)


class TestHermeticHome(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.home = Path.home()
        cls.db = Path(os.environ["CFB_COACH_DB"])
        cls.madden_db = Path(os.environ["CFB_COACH_MADDEN_DB"])

    def test_setupclass_paths_are_inside_the_tmp_dir(self) -> None:
        root, real = _roots()
        _assert_off_real_home(self.home, root, real)
        _assert_off_real_home(self.db, root, real)
        _assert_off_real_home(self.madden_db, root, real)
        self.assertNotEqual(self.madden_db.resolve(), self.db.resolve())
        self.assertEqual(self.madden_db.resolve(), root / ".cfb-coach" / "madden27.db")
        self.assertEqual(os.environ.get("CFB_COACH_AI_RESEARCH"), "off")


def test_path_home_and_cfb_coach_db_are_inside_the_tmp_dir() -> None:
    root, real = _roots()
    home = Path.home()
    db = Path(os.environ["CFB_COACH_DB"])
    madden_db = Path(os.environ["CFB_COACH_MADDEN_DB"])
    _assert_off_real_home(home, root, real)
    _assert_off_real_home(db, root, real)
    _assert_off_real_home(madden_db, root, real)
    assert home.resolve() == root
    assert db.resolve().parent == root
    assert madden_db.resolve() != db.resolve()
    assert madden_db.resolve() == root / ".cfb-coach" / "madden27.db"
    for path in (default_db_path(), default_prep_dir(), default_overlay_path()):
        _assert_off_real_home(path, root, real)
    assert os.environ.get("CFB_COACH_AI_RESEARCH") == "off"


def test_playwright_browsers_path_survives_redirect_and_preset_is_kept() -> None:
    """A discovered cache stays pinned across the HOME redirect.

    An already-set ``PLAYWRIGHT_BROWSERS_PATH`` is left unchanged, and a
    missing cache is not created.
    """
    session = _session_conftest()
    had_browsers = "PLAYWRIGHT_BROWSERS_PATH" in os.environ
    previous_browsers = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    try:
        with tempfile.TemporaryDirectory(prefix="pw-browsers-") as name:
            real = Path(name)
            cache = real / ".cache" / "ms-playwright"
            cache.mkdir(parents=True)
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
            session._pin_playwright_browsers(real)
            pinned = os.environ["PLAYWRIGHT_BROWSERS_PATH"]
            assert Path(pinned) == cache.resolve()
            assert Path(pinned).is_absolute()

            os.environ["HOME"] = str(real / "redirected-home")
            Path(os.environ["HOME"]).mkdir()
            session._redirect_home()
            assert Path(os.environ["HOME"]).resolve() != real.resolve()
            assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == pinned

            preset = str(real / "already-set")
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = preset
            session._pin_playwright_browsers(real)
            session._redirect_home()
            assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == preset

            bare = real / "no-cache"
            bare.mkdir()
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
            session._pin_playwright_browsers(bare)
            assert "PLAYWRIGHT_BROWSERS_PATH" not in os.environ
            assert not (bare / ".cache").exists()
            assert list(bare.iterdir()) == []
    finally:
        if had_browsers:
            assert previous_browsers is not None
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = previous_browsers
        else:
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
        session._redirect_home()
