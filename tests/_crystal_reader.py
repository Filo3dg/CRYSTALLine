"""A stand-in for how CRYSTAL reads the geometry block of a 3D ``.d12``.

Used by the tests to check a deck the only way that matters: rebuild the
crystal from it the way the program will, and compare. It follows the manual's
geometry input (record ``IFLAG IFHR IFSO``):

* ``IFLAG = 0``: the space-group number, in the standard setting of the
  International Tables; ``IFLAG = 1``: the Hermann–Mauguin symbol of the
  setting, blanks between directions.
* ``IFSO = 0``: the second origin setting where a group has two; ``1``: the first.
* ``IFHR = 0``: a rhombohedral group on hexagonal axes; ``1``: on rhombohedral ones.

Operations come from spglib's table of the 530 Hall settings, so the only thing
taken from the code under test is its spelling of a symbol — and that the
spelling picks out exactly one setting is itself checked here.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from crystalline.core import cell_setting


def geometry_record(deck: str) -> List[str]:
    """The lines of a 3D geometry block, from the flags record to the last atom."""
    lines = [line.strip() for line in deck.splitlines()]
    start = lines.index("CRYSTAL") + 1
    count = int(lines[start + 3])
    return lines[start:start + 4 + count]


def setting_read(flags: str, group: str) -> cell_setting.Setting:
    """The one Hall setting CRYSTAL takes ``flags`` and ``group`` to mean."""
    iflag, ifhr, ifso = (int(v) for v in flags.split())
    if iflag == 0:
        halls = cell_setting.halls_of(int(group))
        chosen = [h for h in halls if cell_setting.setting_from_hall(h).is_standard_axes]
    else:
        wanted = " ".join(group.upper().split())
        chosen = []
        for hall in range(1, 531):
            symbol = cell_setting.setting_from_hall(hall)._crystal_symbol()
            if symbol is not None and " ".join(symbol.upper().split()) == wanted:
                chosen.append(hall)
    settings = [cell_setting.setting_from_hall(h) for h in chosen]
    if any(s.origin_choice for s in settings):
        settings = [s for s in settings if s.origin_choice == (1 if ifso == 1 else 2)]
    if any(s.axes in ("H", "R") for s in settings):
        settings = [s for s in settings if s.axes == ("R" if ifhr else "H")]
    if len(settings) != 1:
        raise AssertionError(f"{flags!r} {group!r} names {len(settings)} settings, not one")
    return settings[0]


def lattice_read(setting: cell_setting.Setting, values: List[float]):
    """The cell CRYSTAL builds from the parameters the record carries."""
    from pymatgen.core import Lattice

    n = setting.number
    if n <= 2:
        return Lattice.from_parameters(*values)
    if n <= 15:
        a, b, c, angle = values
        angles = {"a": (angle, 90, 90), "b": (90, angle, 90), "c": (90, 90, angle)}
        return Lattice.from_parameters(a, b, c, *angles[setting.unique_axis])
    if n <= 74:
        return Lattice.from_parameters(*values, 90, 90, 90)
    if n <= 142:
        a, c = values
        return Lattice.from_parameters(a, a, c, 90, 90, 90)
    if n <= 167 and setting.rhombohedral_axes:
        a, alpha = values
        return Lattice.from_parameters(a, a, a, alpha, alpha, alpha)
    if n <= 194:
        a, c = values
        return Lattice.from_parameters(a, a, c, 90, 90, 120)
    (a,) = values
    return Lattice.from_parameters(a, a, a, 90, 90, 90)


def crystal_reads(deck: str) -> Tuple[cell_setting.Setting, object, np.ndarray, np.ndarray]:
    """``(setting, lattice, atomic numbers, fractional coordinates)`` CRYSTAL builds."""
    record = geometry_record(deck)
    setting = setting_read(record[0], record[1])
    lattice = lattice_read(setting, [float(v) for v in record[2].split()])
    rotations, translations = cell_setting.operations(setting.hall_number)
    numbers: List[int] = []
    points: List[np.ndarray] = []
    for row in record[4:]:
        fields = row.split()
        z, point = int(fields[0]), np.array([float(v) for v in fields[1:4]])
        for image in point @ rotations.transpose(0, 2, 1) + translations:
            image = image - np.floor(image + 1e-6)
            if any(zz == z and _close(image, p, lattice.matrix) for zz, p in zip(numbers, points)):
                continue
            numbers.append(z)
            points.append(image)
    return setting, lattice, np.asarray(numbers), np.asarray(points)


def same_crystal(deck: str, lattice_matrix, numbers, fractional, tol: float = 1e-3) -> bool:
    """Whether the deck rebuilds these atoms in this cell (up to a common origin shift).

    The cell is compared by its parameters, the atoms by their fractional
    coordinates — so the check is exact where it needs to be, and blind only to
    the orientation of the axes in space and the choice of origin, which CRYSTAL
    fixes for itself.
    """
    from pymatgen.core import Lattice

    _setting, lattice, got_numbers, got = crystal_reads(deck)
    wanted = Lattice(np.asarray(lattice_matrix, dtype=float))
    if not np.allclose(lattice.parameters, wanted.parameters, atol=1e-4):
        return False
    numbers = np.asarray(numbers)
    fractional = np.asarray(fractional, dtype=float)
    if len(got) != len(fractional) or sorted(got_numbers) != sorted(numbers):
        return False
    # the written origin may differ from the atoms' by a constant: try each
    # shift that maps the first written atom onto an atom of its element
    for candidate in fractional[numbers == got_numbers[0]]:
        shift = candidate - got[0]
        moved = got + shift
        if all(any(z == zz and _close(p, q, wanted.matrix, tol) for zz, q in zip(numbers, fractional))
               for z, p in zip(got_numbers, moved)):
            return True
    return False


def _close(p, q, matrix, tol: float = 1e-3) -> bool:
    delta = np.asarray(p) - np.asarray(q)
    delta -= np.round(delta)
    return float(np.linalg.norm(delta @ matrix)) < tol * 10
