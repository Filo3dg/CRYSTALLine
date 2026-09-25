"""Lattice planes named by their Miller indices: where they lie, and what lies on them.

The plane ``(hkl)`` of a cell is the one whose normal is the reciprocal lattice
vector ``G = h a* + k b* + l c*`` (without the 2π), and the planes of the family
are ``d = 1/|G|`` apart. With that convention a point at fractional coordinates
``(x, y, z)`` sits on the plane of the family at

    offset = h x + k y + l z

counted in units of ``d`` from the plane through the origin. That one line is
most of this module: it places a plane through a chosen atom, lists the planes
of a family that cross a region, and says which atoms lie on one.

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

# Two atoms this close (Å) to a plane are taken to lie on it.
ON_PLANE_TOLERANCE = 0.1

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
        indices = tuple(int(v) for v in self.miller)
        if len(indices) != 3:
            raise ValueError(f"a plane has three Miller indices, got {self.miller!r}")
        if indices == (0, 0, 0):
            raise ValueError("(0 0 0) is not a plane: give at least one non-zero index")
        object.__setattr__(self, "miller", indices)
        object.__setattr__(self, "offset", float(self.offset))
        object.__setattr__(self, "opacity", min(1.0, max(0.0, float(self.opacity))))

    def drawn_the_same(self, other: "LatticePlane") -> bool:
        """Whether ``other`` is this plane in the same place — differing, at
        most, in how it looks. Such a change needs no new geometry."""
        return (self.miller, self.offset, self.family) == (other.miller, other.offset, other.family)


# ── indices ───────────────────────────────────────────────────────────────
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
def reciprocal_vector(cell, miller: Sequence[int]) -> np.ndarray:
    """``G = h a* + k b* + l c*`` in Å⁻¹, without the 2π."""
    cell = np.asarray(cell, dtype=float)
    reciprocal = np.linalg.inv(cell).T          # rows a*, b*, c*
    return np.asarray(miller, dtype=float) @ reciprocal


def spacing(cell, miller: Sequence[int]) -> float:
    """``d(hkl)`` in Å: the distance between neighbouring planes of the family."""
    return float(1.0 / np.linalg.norm(reciprocal_vector(cell, miller)))


def normal(cell, miller: Sequence[int]) -> np.ndarray:
    """The unit normal of ``(hkl)``, pointing along ``G``."""
    g = reciprocal_vector(cell, miller)
    return g / np.linalg.norm(g)


def offset_of(cell, miller: Sequence[int], point) -> float:
    """Where the plane of the family through ``point`` (Cartesian) sits, in units of d.

    ``r · G`` — equal to ``h x + k y + l z`` in the reference cell's fractional
    coordinates. Use it to put a plane through a chosen atom.
    """
    return float(np.asarray(point, dtype=float) @ reciprocal_vector(cell, miller))


def plane_point(cell, miller: Sequence[int], offset: float) -> np.ndarray:
    """A point on the plane at ``offset`` (the foot of the normal from the origin)."""
    g = reciprocal_vector(cell, miller)
    return float(offset) * g / float(g @ g)


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


def atoms_on(cell, plane: LatticePlane, positions,
             tolerance: float = ON_PLANE_TOLERANCE) -> np.ndarray:
    """Indices of the atoms lying on ``plane`` (on any plane of it, for a family).

    Measured as a distance in Å — ``|r·G - offset| d`` — so the tolerance does
    not change meaning with the spacing of the family.
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) == 0:
        return np.empty(0, dtype=int)
    g = reciprocal_vector(cell, plane.miller)
    distance = positions @ g - plane.offset
    if plane.family:
        distance = distance - np.round(distance)
    return np.flatnonzero(np.abs(distance) / np.linalg.norm(g) < tolerance)


# ── the region a plane is drawn across ────────────────────────────────────
def region_corners(origin, vectors) -> np.ndarray:
    """The eight corners of the parallelepiped at ``origin`` spanned by ``vectors``."""
    origin = np.asarray(origin, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    return np.asarray([origin + sum(c * v for c, v in zip(combo, vectors))
                       for combo in itertools.product((0, 1), repeat=3)])


def polygon_in_region(point, unit_normal, origin, vectors) -> Optional[np.ndarray]:
    """The polygon a plane cuts out of a parallelepiped, corners in order.

    ``None`` when the plane misses it. The parallelepiped is the region drawn —
    the cell on screen, or the supercell — so a plane in a monoclinic or
    hexagonal cell stops at the cell's own faces instead of filling the
    axis-aligned box around it, which is what a clip to that box would do.

    A large square in the plane is cut against the six faces in turn
    (Sutherland–Hodgman), which keeps the corners in winding order.
    """
    point = np.asarray(point, dtype=float)
    n = np.asarray(unit_normal, dtype=float)
    n = n / np.linalg.norm(n)
    origin = np.asarray(origin, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if abs(np.linalg.det(vectors)) < 1e-12:
        return None

    # An in-plane frame, and a square big enough to cover the region.
    helper = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    corners = region_corners(origin, vectors)
    centre = corners.mean(axis=0)
    foot = centre - n * float((centre - point) @ n)   # the region's centre, dropped onto the plane
    size = 2.0 * float(np.linalg.norm(corners - centre, axis=1).max()) + 1.0
    polygon = [foot + size * (su * u + sw * w) for su, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1))]

    # Each face as an inward half-space  m · x >= c.
    volume_sign = np.sign(np.linalg.det(vectors))
    for axis in range(3):
        other = [vectors[j] for j in range(3) if j != axis]
        m = np.cross(other[0], other[1]) * volume_sign
        if axis == 1:
            m = -m      # keep the cyclic order so every m points into the region
        m /= np.linalg.norm(m)
        for base, sign in ((origin, 1.0), (origin + vectors[axis], -1.0)):
            polygon = _clip(polygon, sign * m, sign * float(m @ base))
            if len(polygon) < 3:
                return None
    # Clipping through a corner repeats it; a plane that only grazes an edge or
    # a corner leaves no area at all, and is not drawn.
    kept = [polygon[0]]
    for vertex in polygon[1:]:
        if np.linalg.norm(vertex - kept[-1]) > 1e-7:
            kept.append(vertex)
    if len(kept) > 1 and np.linalg.norm(kept[0] - kept[-1]) <= 1e-7:
        kept.pop()
    if len(kept) < 3:
        return None
    kept = np.asarray(kept, dtype=float)
    centre = kept.mean(axis=0)
    area = sum(np.linalg.norm(np.cross(kept[i] - centre, kept[(i + 1) % len(kept)] - centre))
               for i in range(len(kept))) / 2.0
    return kept if area > 1e-6 else None


def _clip(polygon, m: np.ndarray, c: float, eps: float = 1e-9):
    """Keep the part of a convex polygon where ``m · x >= c``."""
    out = []
    count = len(polygon)
    for index in range(count):
        current, following = polygon[index], polygon[(index + 1) % count]
        a, b = float(m @ current) - c, float(m @ following) - c
        if a >= -eps:
            out.append(current)
        if (a >= -eps) != (b >= -eps):
            t = a / (a - b)
            out.append(current + t * (following - current))
    return out


__all__ = [
    "DEFAULT_OPACITY",
    "LatticePlane",
    "MAX_FAMILY_PLANES",
    "ON_PLANE_TOLERANCE",
    "atoms_on",
    "four_index",
    "is_hexagonal",
    "label",
    "normal",
    "offset_of",
    "offsets_across",
    "plane_point",
    "polygon_in_region",
    "reciprocal_vector",
    "region_corners",
    "spacing",
    "three_index",
]
