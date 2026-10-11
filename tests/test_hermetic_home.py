"""The suite must resolve home and the coach DB inside the session tmp dir."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from cfb_coach.copilot import default_overlay_path
from cfb_coach.db import default_db_path
from cfb_coach.prep_browser import default_prep_dir


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

    def test_setupclass_paths_are_inside_the_tmp_dir(self) -> None:
        root, real = _roots()
        _assert_off_real_home(self.home, root, real)
        _assert_off_real_home(self.db, root, real)
        self.assertEqual(os.environ.get("CFB_COACH_AI_RESEARCH"), "off")


def test_path_home_and_cfb_coach_db_are_inside_the_tmp_dir() -> None:
    root, real = _roots()
    home = Path.home()
    db = Path(os.environ["CFB_COACH_DB"])
    _assert_off_real_home(home, root, real)
    _assert_off_real_home(db, root, real)
    assert home.resolve() == root
    assert db.resolve().parent == root
    for path in (default_db_path(), default_prep_dir(), default_overlay_path()):
        _assert_off_real_home(path, root, real)
    assert os.environ.get("CFB_COACH_AI_RESEARCH") == "off"
