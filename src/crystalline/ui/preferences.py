"""Choices the app remembers from one session to the next.

Stored with ``QSettings`` under the same organisation and application name as
the appearance (``ui.theme``), so they live in one place per platform. Every
read falls back to the default and every write is best-effort: a machine whose
settings cannot be read or written still runs, it only forgets.
"""

from __future__ import annotations

from crystalline.core.cell_setting import CELL_CHOICES, DEFAULT_CHOICE

_CELL_KEY = "crystallography/cell"
_SECTION_KEY = "sections/{}"


def _settings():
    from PySide6.QtCore import QSettings

    from crystalline.ui.theme import _SETTINGS_APPLICATION, _SETTINGS_ORGANISATION

    return QSettings(_SETTINGS_ORGANISATION, _SETTINGS_APPLICATION)


def cell_choice() -> str:
    """Which cell the Info panel and the input builder start on.

    ``"computed"`` — the cell as calculated, in its own setting — or
    ``"standard"``, pymatgen's conventional standard cell.
    """
    try:
        value = str(_settings().value(_CELL_KEY, DEFAULT_CHOICE))
    except Exception:  # noqa: BLE001 - unreadable settings: the default still works
        return DEFAULT_CHOICE
    return value if value in dict(CELL_CHOICES) else DEFAULT_CHOICE


def set_cell_choice(choice: str) -> None:
    """Remember ``choice`` for the next session (silently, if it can't be)."""
    if choice not in dict(CELL_CHOICES):
        raise ValueError(f"unknown cell choice {choice!r}; expected one of {dict(CELL_CHOICES)}")
    try:
        _settings().setValue(_CELL_KEY, choice)
    except Exception:  # noqa: BLE001 - can't persist; it still applies this session
        pass


def section_open(name: str, default: bool = True) -> bool:
    """Whether the panel section ``name`` (``"geometry/measure"``, say) was left open."""
    try:
        value = _settings().value(_SECTION_KEY.format(name), default)
    except Exception:  # noqa: BLE001 - unreadable settings: the default still works
        return default
    # QSettings hands booleans back as the strings "true"/"false" on some platforms.
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "")
    return bool(value)


def set_section_open(name: str, open_: bool) -> None:
    """Remember whether the section ``name`` is open (silently, if it can't be)."""
    try:
        _settings().setValue(_SECTION_KEY.format(name), bool(open_))
    except Exception:  # noqa: BLE001 - can't persist; it still applies this session
        pass


__all__ = ["cell_choice", "section_open", "set_cell_choice", "set_section_open"]
