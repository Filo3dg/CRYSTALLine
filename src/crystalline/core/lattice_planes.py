"""Lattice planes named by their Miller indices: where they lie, and what lies on them.

The plane ``(hkl)`` of a cell is the one whose normal is the reciprocal lattice
vector ``G = h a* + k b* + l c*`` (without the 2π), and the planes of the family
are ``d = 1/|G|`` apart. With that convention a point at fractional coordinates
``(x, y, z)`` sits on the plane of the family at

    offset = h x + k y + l z

counted in units of ``d`` from the plane through the origin. That one line is
most of this module: it places a plane through a chosen atom, lists the planes
of a family that cross a region, and says which atoms lie on one.

It is the one place the app does this: the density slice, the Geometry panel's
lattice planes and the a*/b*/c* views all take their reciprocal lattice, their
planes and their "on the plane" test from here.

Indices are always quoted in a *reference* cell — the conventional cell, in the
app, as the density map's lattice planes are — which need not be the cell on
screen: a supercell or a primitive view must not change what (001) means. So
the geometry here takes that cell explicitly and works in Cartesian space, and
the region a plane is drawn across is another, independent parallelepiped.

Hexagonal and trigonal crystals are indexed by four Miller–Bravais indices
``(h k i l)``, where ``i = -(h + k)`` is redundant but is how such planes are
written; :func:`four_index` gives it when the reference cell is hexagonal.

Qt- and renderer-free, like the rest of ``core``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# An atom this close (Å) to a plane is taken to lie on it — counted in the
# Geometry panel, marked on a density slice.
ON_PLANE_TOLERANCE = 0.15

# A family is drawn across the region, and a fine one — (7 3 11) through a
# 3×3×3 supercell — would otherwise be hundreds of sheets.
MAX_FAMILY_PLANES = 60

# How opaque a plane's sheet is drawn unless told otherwise: enough to see, not
# enough to hide the atoms behind it. Its outline is always drawn solid.
DEFAULT_OPACITY = 0.35


@dataclass(frozen=True)
class LatticePlane:
    """One plane ``(hkl)`` to show, or its whole family.

    ``offset`` places it along the normal in units of ``d(hkl)`` from the plane
    through the reference cell's origin, so ``0`` and ``1`` are neighbouring
    planes of the family and ``0.5`` lies halfway between. ``family`` draws
    every plane of the family that crosses the region instead of just this one.
    ``opacity`` is the sheet's, from 0 (only the outline is seen) to 1.
    """

    miller: Tuple[int, int, int]
    offset: float = 0.0
    family: bool = False
    color: Optional[str] = None
    opacity: float = DEFAULT_OPACITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "miller", miller_indices(self.miller))
        object.__setattr__(self, "offset", float(self.offset))
        object.__setattr__(self, "opacity", min(1.0, max(0.0, float(self.opacity))))

    def drawn_the_same(self, other: "LatticePlane") -> bool:
        """Whether ``other`` is this plane in the same place — differing, at
        most, in how it looks. Such a change needs no new geometry."""
        return (self.miller, self.offset, self.family) == (other.miller, other.offset, other.family)


# ── indices ───────────────────────────────────────────────────────────────
def miller_indices(miller: Sequence[int]) -> Tuple[int, int, int]:
    """``miller`` as three ints, refusing (0 0 0), which names no plane."""
    indices = tuple(int(v) for v in miller)
    if len(indices) != 3:
        raise ValueError(f"a plane has three Miller indices, got {tuple(miller)!r}")
    if indices == (0, 0, 0):
        raise ValueError("(000) is not a plane: give at least one non-zero index.")
    return indices


def is_hexagonal(cell) -> bool:
    """Whether ``cell`` has hexagonal axes: a = b and γ = 120°, α = β = 90°.

    Such a cell is indexed with four Miller–Bravais indices.
    """
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        return False
    a, b, c = (float(np.linalg.norm(v)) for v in cell)
    if min(a, b, c) < 1e-8 or abs(a - b) > 1e-3 * a:
        return False
    cosine = lambda u, v: float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))  # noqa: E731
    return (abs(cosine(cell[0], cell[1]) + 0.5) < 1e-3
            and abs(cosine(cell[0], cell[2])) < 1e-3
            and abs(cosine(cell[1], cell[2])) < 1e-3)


def four_index(miller: Sequence[int]) -> Tuple[int, int, int, int]:
    """``(h, k, l)`` -> ``(h, k, i, l)`` with ``i = -(h + k)``."""
    h, k, l = (int(v) for v in miller)
    return h, k, -(h + k), l


def three_index(miller: Sequence[int]) -> Tuple[int, int, int]:
    """``(h, k, i, l)`` -> ``(h, k, l)``, checking that ``h + k + i = 0``."""
    h, k, i, l = (int(v) for v in miller)
    if h + k + i != 0:
        raise ValueError(f"({h} {k} {i} {l}) is not a Miller–Bravais index: h + k + i must be 0")
    return h, k, l


def label(miller: Sequence[int], cell=None) -> str:
    """``"(1 1 0)"``, or ``"(1 0 -1 0)"`` for a hexagonal cell."""
    indices = four_index(miller) if cell is not None and is_hexagonal(cell) else tuple(miller)
    return "(" + " ".join(str(int(v)) for v in indices) + ")"


# ── geometry ──────────────────────────────────────────────────────────────
def reciprocal_lattice(cell) -> np.ndarray:
    """The reciprocal lattice vectors a*, b*, c* as rows, in Å⁻¹ and without the 2π.

    ``a* = (b × c) / V`` and cyclically — the inverse transpose of the cell —
    so that ``a*·a = 1`` and a vector's fractional coordinates multiply these
    rows directly. Raises ``ValueError`` for a cell with no volume.
    """
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("the cell has no volume, so no reciprocal lattice")
    return np.linalg.inv(cell).T


def reciprocal_vector(cell, miller: Sequence[int]) -> np.ndarray:
    """``G = h a* + k b* + l c*``: the normal of ``(hkl)``, ``1/d`` long."""
    return np.asarray(miller_indices(miller), dtype=float) @ reciprocal_lattice(cell)


def plane_frame(cell, miller: Sequence[int], offset: float = 0.0):
    """``(point, unit normal, spacing)`` of the plane ``(hkl)`` at ``offset``.

    ``point`` is the foot of the normal from the origin, ``offset`` spacings
    along it: ``0`` is the plane through the origin, ``0.5`` the one halfway
    to the next. ``cell`` should be the conventional cell — what Miller indices
    are quoted in: MgO's (001) in its primitive fcc cell is a {111} plane
    holding one kind of atom.
    """
    g = reciprocal_vector(cell, miller)
    length = float(np.linalg.norm(g))
    normal = g / length
    return float(offset) / length * normal, normal, 1.0 / length


def spacing(cell, miller: Sequence[int]) -> float:
    """``d(hkl)`` in Å: the distance between neighbouring planes of the family."""
    return plane_frame(cell, miller)[2]


def offset_of(cell, miller: Sequence[int], point) -> float:
    """Where the plane of the family through ``point`` (Cartesian) sits, in units of d.

    ``r · G`` — equal to ``h x + k y + l z`` in the reference cell's fractional
    coordinates. Use it to put a plane through a chosen atom.
    """
    return float(np.asarray(point, dtype=float) @ reciprocal_vector(cell, miller))


def offsets_across(cell, miller: Sequence[int], offset: float, points,
                   limit: int = MAX_FAMILY_PLANES) -> List[float]:
    """The planes of the family through ``offset`` that cross the hull of ``points``.

    ``points`` are Cartesian — the corners of the drawn region, and the atoms.
    Returned nearest the plane asked for first, so capping the count keeps
    the planes around it rather than those at one end.
    """
    values = np.asarray(points, dtype=float) @ reciprocal_vector(cell, miller)
    if values.size == 0:
        return [float(offset)]
    low, high = float(values.min()), float(values.max())
    first = int(np.ceil(low - offset - 1e-9))
    last = int(np.floor(high - offset + 1e-9))
    found = [float(offset + n) for n in range(first, last + 1)]
    found.sort(key=lambda value: abs(value - offset))
    return found[:limit] if found else [float(offset)]


def distances_to(cell, plane: LatticePlane, positions) -> np.ndarray:
    """Signed distance (Å) of each position from ``plane``, along its normal.

    For a family, from the nearest plane of the family. ``(r·G - offset) d``,
    so a tolerance on it does not change meaning with the spacing.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    g = reciprocal_vector(cell, plane.miller)
    distance = positions @ g - plane.offset
    if plane.family:
        distance = distance - np.round(distance)
    return distance / float(np.linalg.norm(g))


def atoms_on(cell, plane: LatticePlane, positions,
             tolerance: float = ON_PLANE_TOLERANCE) -> np.ndarray:
    """Indices of the atoms lying on ``plane`` (on any plane of it, for a family)."""
    return np.flatnonzero(np.abs(distances_to(cell, plane, positions)) < tolerance)


def region_corners(origin, vectors) -> np.ndarray:
    """The eight corners of the parallelepiped at ``origin`` spanned by ``vectors``."""
    origin = np.asarray(origin, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    return np.asarray([origin + sum(c * v for c, v in zip(combo, vectors))
                       for combo in itertools.product((0, 1), repeat=3)])


__all__ = [
    "DEFAULT_OPACITY",
    "LatticePlane",
    "MAX_FAMILY_PLANES",
    "ON_PLANE_TOLERANCE",
    "atoms_on",
    "distances_to",
    "four_index",
    "is_hexagonal",
    "label",
    "miller_indices",
    "offset_of",
    "offsets_across",
    "plane_frame",
    "reciprocal_lattice",
    "reciprocal_vector",
    "region_corners",
    "spacing",
    "three_index",
]
