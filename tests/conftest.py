"""Session hooks for the cfb-coach test suite.

``CFB_COACH_AI_RESEARCH=off`` stops prep from downloading research JSON.
GitHub-research tests call ``cfb_coach.ai_research`` with a fake fetch.

Hermetic home
-------------
Session-scoped fixtures are set up after collection. Test modules are imported
during collection, so module-level code runs before any session fixture.
``unittest.TestCase.setUpClass`` is itself a class-scoped autouse fixture on
pytest 9, which means a session fixture *does* run before ``setUpClass``, but
it is still too late for module-level code.

``HOME`` and ``CFB_COACH_DB`` are therefore installed in ``pytest_configure``,
which runs before collection. The session fixture re-applies the same paths
and restores both variables afterwards. The real-home guard is separate: it
records ``<real HOME>/.cfb-coach`` before that redirect and fails the session
if the tree was created or modified, including when the redirect is turned off.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# Prep must not download research/*.json from GitHub during tests.
os.environ.setdefault("CFB_COACH_AI_RESEARCH", "off")

_Snapshot = tuple[bool, tuple[tuple[str, int, int, int], ...]]

_real_home: Path | None = None
_home_was_set = False
_orig_home: str | None = None
_db_was_set = False
_orig_db: str | None = None
_root: Path | None = None
_tmp: tempfile.TemporaryDirectory[str] | None = None
_before: _Snapshot | None = None
_redirected = False


def _snapshot(real_home: Path) -> _Snapshot:
    """Listing and mtime snapshot of ``<real HOME>/.cfb-coach``, plus existence."""
    target = real_home / ".cfb-coach"
    try:
        target.lstat()
    except OSError:
        return (False, ())

    rows: list[tuple[str, int, int, int]] = []
    pending = [target]
    while pending:
        current = pending.pop()
        try:
            info = current.lstat()
        except OSError:
            continue
        try:
            rel = current.relative_to(real_home).as_posix()
        except ValueError:
            rel = current.as_posix()
        rows.append((rel, info.st_mtime_ns, info.st_size, stat.S_IFMT(info.st_mode)))
        if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
            try:
                children = list(current.iterdir())
            except OSError:
                continue
            pending.extend(sorted(children, key=lambda path: path.name, reverse=True))
    rows.sort(key=lambda row: row[0])
    return (True, tuple(rows))


def _entry_map(snapshot: _Snapshot) -> dict[str, tuple[int, int, int]]:
    return {rel: (mtime, size, mode) for rel, mtime, size, mode in snapshot[1]}


def _guard_diff(before: _Snapshot, after: _Snapshot, real_home: Path) -> str | None:
    if before[0] == after[0] and _entry_map(before) == _entry_map(after):
        return None
    lines = [
        "hermetic home guard: "
        f"{real_home / '.cfb-coach'} was created or modified during the session"
    ]
    if before[0] != after[0]:
        lines.append(f"exists: {before[0]} -> {after[0]}")
    before_map = _entry_map(before)
    after_map = _entry_map(after)
    for rel in sorted(set(before_map) | set(after_map)):
        if rel not in before_map:
            lines.append(f"created {rel}")
        elif rel not in after_map:
            lines.append(f"removed {rel}")
        elif before_map[rel] != after_map[rel]:
            lines.append(f"modified {rel}")
    return "\n".join(lines)


def _remember_real_home() -> None:
    """Record the process home and ``.cfb-coach`` before any redirect."""
    global _real_home, _home_was_set, _orig_home, _db_was_set, _orig_db, _before
    if _before is not None:
        return
    _home_was_set = "HOME" in os.environ
    _orig_home = os.environ.get("HOME")
    _db_was_set = "CFB_COACH_DB" in os.environ
    _orig_db = os.environ.get("CFB_COACH_DB")
    if _home_was_set:
        real = Path(_orig_home or "")
    else:
        import pwd

        real = Path(pwd.getpwuid(os.getuid()).pw_dir)
    _real_home = real
    _before = _snapshot(real)


def _hermetic_tmp(real: Path) -> tempfile.TemporaryDirectory[str]:
    tmp = tempfile.TemporaryDirectory(prefix="cfb-coach-hermetic-", ignore_cleanup_errors=True)
    root = Path(tmp.name).resolve()
    real_resolved = real.resolve()
    if root == real_resolved or real_resolved in root.parents:
        tmp.cleanup()
        tmp = tempfile.TemporaryDirectory(
            prefix="cfb-coach-hermetic-",
            dir="/tmp",
            ignore_cleanup_errors=True,
        )
    return tmp


def _redirect_home() -> None:
    """Point ``HOME`` and ``CFB_COACH_DB`` at one session tmp dir."""
    global _root, _tmp, _redirected
    _remember_real_home()
    assert _real_home is not None
    if _redirected:
        assert _root is not None
        os.environ["HOME"] = str(_root)
        os.environ["CFB_COACH_DB"] = str(_root / "coach.db")
        os.environ["CFB_COACH_HERMETIC_ROOT"] = str(_root)
        os.environ["CFB_COACH_REAL_HOME"] = str(_real_home)
        return
    _tmp = _hermetic_tmp(_real_home)
    _root = Path(_tmp.name)
    os.environ["HOME"] = str(_root)
    os.environ["CFB_COACH_DB"] = str(_root / "coach.db")
    os.environ["CFB_COACH_HERMETIC_ROOT"] = str(_root)
    os.environ["CFB_COACH_REAL_HOME"] = str(_real_home)
    _redirected = True


def _restore_home() -> None:
    """Put ``HOME`` and ``CFB_COACH_DB`` back to the values captured before the redirect."""
    global _redirected, _tmp
    if not _redirected:
        return
    if _home_was_set:
        assert _orig_home is not None
        os.environ["HOME"] = _orig_home
    else:
        os.environ.pop("HOME", None)
    if _db_was_set:
        assert _orig_db is not None
        os.environ["CFB_COACH_DB"] = _orig_db
    else:
        os.environ.pop("CFB_COACH_DB", None)
    os.environ.pop("CFB_COACH_HERMETIC_ROOT", None)
    os.environ.pop("CFB_COACH_REAL_HOME", None)
    _redirected = False
    if _tmp is not None:
        _tmp.cleanup()
        _tmp = None


def pytest_configure(config: pytest.Config) -> None:
    # Before collection, so module-level code and setUpClass cannot see the real home.
    _remember_real_home()
    _redirect_home()


@pytest.fixture(scope="session", autouse=True)
def hermetic_coach_home() -> Iterator[None]:
    """Keep ``HOME`` and ``CFB_COACH_DB`` inside a session tmp dir, then restore both.

    Installed early from ``pytest_configure`` because this fixture runs after
    test modules are imported. Re-applied here so unittest cases, which cannot
    request fixtures, still inherit the redirected environment, and torn down
    here so the process home is restored when the session ends.
    """
    _redirect_home()
    yield
    _restore_home()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    del exitstatus
    if _real_home is None or _before is None:
        return
    failure = _guard_diff(_before, _snapshot(_real_home), _real_home)
    if failure is None:
        return
    session.shouldfail = "hermetic home guard: real HOME/.cfb-coach was created or modified"
    if session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    sys.stderr.write(failure + "\n")
