"""Protected-asset guard: checksum the owner's primary data before and after, refuse to write there.

Protected: the primary-franchise DB (``~/.cfb-coach/*.db`` unless it is an explicitly isolated path),
installed-book/state files, champion model artifacts and production config passed in ``extra``.
Nothing in the unified package writes to any of these; this module proves it.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Mapping

DEFAULT_HOME = Path(os.environ.get("CFB_COACH_HOME", Path.home() / ".cfb-coach"))
PROTECTED_REPO_GLOBS = ("cfb_coach/data/madden27/*.json",)


class ProtectedPathError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_paths(extra: Iterable[str | Path] = (), repo_root: Path | None = None) -> list[Path]:
    out: list[Path] = [Path(p) for p in extra]
    if DEFAULT_HOME.exists():
        out += [p for p in DEFAULT_HOME.rglob("*") if p.is_file() and p.suffix in (".db", ".json", ".sqlite")]
    if repo_root is not None:
        for g in PROTECTED_REPO_GLOBS:
            out += sorted(repo_root.glob(g))
    return sorted({p.resolve() for p in out if Path(p).exists() and Path(p).is_file()})


def snapshot(paths: Iterable[Path]) -> dict[str, str]:
    return {str(p): sha256_file(p) for p in paths}


def diff(before: Mapping[str, str], after: Mapping[str, str]) -> dict[str, list[str]]:
    return {"changed": sorted(k for k in before if k in after and before[k] != after[k]),
            "removed": sorted(k for k in before if k not in after), "added": sorted(k for k in after if k not in before)}


def assert_isolated(path: str | Path, *, protected: Iterable[str | Path] = ()) -> Path:
    """Refuse any store path inside the owner's default data dir or equal to a protected file."""
    p = Path(path).expanduser().resolve()
    home = DEFAULT_HOME.resolve()
    if p == home or home in p.parents:
        raise ProtectedPathError(f"{p} is inside the primary data directory {home}; use an isolated path")
    for q in protected:
        if p == Path(q).expanduser().resolve():
            raise ProtectedPathError(f"{p} is a protected asset")
    return p
