"""Where the sample CRYSTAL files the tests read come from.

Small ones live in ``tests/data`` and ship with the repository, so the suite is
the same suite on every machine. The larger runs — wavefunctions, density grids,
a 250 MB dispersion set — do not, and the tests that need those look for a
directory given by ``CRYSTALLINE_TEST_DATA`` instead, skipping cleanly when it
is not set:

    CRYSTALLINE_TEST_DATA=~/my-crystal-runs pytest

``tools/collect_test_data.py`` is what put the shipped files here, and refreshes
them from a local copy of their sources.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest

_SHIPPED = Path(__file__).resolve().parent / "data"


def path(name: str) -> Optional[str]:
    """The file's path, or ``None`` when this machine has not got it.

    ``name`` is relative: ``"silicon/BAND.DAT"`` for a shipped file, or any path
    under ``CRYSTALLINE_TEST_DATA`` for one that is too big to ship.
    """
    shipped = _SHIPPED / name
    if shipped.is_file():
        return str(shipped)
    root = os.environ.get("CRYSTALLINE_TEST_DATA")
    if root:
        candidate = Path(root).expanduser() / name
        if candidate.is_file():
            return str(candidate)
    return None


def folder(name: str) -> Optional[str]:
    """The directory's path, for a test that reads a whole run rather than a file."""
    shipped = _SHIPPED / name
    if shipped.is_dir():
        return str(shipped)
    root = os.environ.get("CRYSTALLINE_TEST_DATA")
    if root:
        candidate = Path(root).expanduser() / name
        if candidate.is_dir():
            return str(candidate)
    return None


def matches(pattern: str) -> list:
    """Every file matching a relative glob, shipped or configured.

    Used where a test reads whatever runs a folder happens to hold, rather than
    one file by name.
    """
    hits = sorted(str(hit) for hit in _SHIPPED.glob(pattern) if hit.is_file())
    if hits:
        return hits
    root = os.environ.get("CRYSTALLINE_TEST_DATA")
    if root:
        return sorted(str(hit) for hit in Path(root).expanduser().glob(pattern)
                      if hit.is_file())
    return []


def has(*names: str) -> bool:
    """Whether every one of ``names`` is available here."""
    return all(path(name) is not None for name in names)


def need(*names: str) -> str:
    """The path to ``names[0]``, skipping the test when any of them is absent."""
    if not has(*names):
        pytest.skip(f"sample data not on this machine: {', '.join(names)} "
                    f"(set CRYSTALLINE_TEST_DATA to a directory holding it)")
    return str(path(names[0]))


def needs(*names: str):
    """A ``skipif`` marker for tests that read ``names``."""
    return pytest.mark.skipif(
        not has(*names),
        reason=f"sample data not on this machine: {', '.join(names)} "
               f"(set CRYSTALLINE_TEST_DATA to a directory holding it)",
    )


__all__ = ["path", "folder", "matches", "has", "need", "needs"]
