"""Lattice planes from Miller indices: spacing, placement, atoms on them, outline."""

import numpy as np
import pytest

from crystalline.core import lattice_planes as lp
from crystalline.core.lattice_planes import LatticePlane


def _cell(a, b, c, alpha, beta, gamma):
    from ase.geometry import cellpar_to_cell

    return np.asarray(cellpar_to_cell([a, b, c, alpha, beta, gamma]), dtype=float)


def _area(polygon):
    centre = polygon.mean(axis=0)
    return sum(np.linalg.norm(np.cross(polygon[i] - centre, polygon[(i + 1) % len(polygon)] - centre))
               for i in range(len(polygon))) / 2.0


# ── indices ───────────────────────────────────────────────────────────────
def test_a_plane_needs_three_indices_not_all_zero():
    with pytest.raises(ValueError):
        LatticePlane((0, 0, 0))
    with pytest.raises(ValueError):
        LatticePlane((1, 0))
    plane = LatticePlane((1.0, -2, 3), offset=1)
    assert plane.miller == (1, -2, 3) and isinstance(plane.miller[0], int)
    assert plane.offset == 1.0 and not plane.family


def test_hexagonal_axes_are_recognised_and_indexed_with_four_indices():
    hexagonal = _cell(3.2, 3.2, 5.2, 90, 90, 120)
    assert lp.is_hexagonal(hexagonal)
    assert not lp.is_hexagonal(_cell(3.2, 3.2, 5.2, 90, 90, 90))
    assert not lp.is_hexagonal(_cell(3.2, 3.3, 5.2, 90, 90, 120))
    assert lp.four_index((1, 0, 0)) == (1, 0, -1, 0)
    assert lp.three_index((1, 1, -2, 0)) == (1, 1, 0)
    with pytest.raises(ValueError):
        lp.three_index((1, 1, 1, 0))
    assert lp.label((1, 0, 0), hexagonal) == "(1 0 -1 0)"
    assert lp.label((1, 1, 0), 4 * np.eye(3)) == "(1 1 0)"
    assert lp.label((1, -1, 2)) == "(1 -1 2)"


# ── spacing ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("miller", [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 3), (0, 0, 2)])
def test_cubic_spacing_is_a_over_the_root_of_h2_k2_l2(miller):
    a = 5.43
    assert lp.spacing(a * np.eye(3), miller) == pytest.approx(a / np.sqrt(np.dot(miller, miller)))


@pytest.mark.parametrize("miller", [(1, 0, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (2, -1, 3)])
def test_hexagonal_spacing_matches_the_textbook_formula(miller):
    a, c = 3.21, 5.21
    h, k, l = miller
    expected = 1.0 / np.sqrt(4.0 / 3.0 * (h * h + h * k + k * k) / a**2 + l * l / c**2)
    assert lp.spacing(_cell(a, a, c, 90, 90, 120), miller) == pytest.approx(expected)


@pytest.mark.parametrize("miller", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, -1, 2)])
def test_monoclinic_spacing_matches_the_textbook_formula(miller):
    a, b, c, beta = 8.24, 13.45, 8.97, 105.48
    h, k, l = miller
    s = np.sin(np.radians(beta))
    inverse = (h * h / a**2 + k * k * s * s / b**2 + l * l / c**2
               - 2 * h * l * np.cos(np.radians(beta)) / (a * c)) / (s * s)
    assert lp.spacing(_cell(a, b, c, 90, beta, 90), miller) == pytest.approx(1 / np.sqrt(inverse))


def test_the_normal_is_perpendicular_to_every_lattice_vector_in_the_plane():
    cell = _cell(5.1, 6.3, 7.2, 81, 97, 103)
    n = lp.normal(cell, (1, 2, 3))
    assert np.linalg.norm(n) == pytest.approx(1.0)
    # (1 2 3) contains the lattice directions [2 -1 0] and [3 0 -1]
    for direction in ((2, -1, 0), (3, 0, -1), (0, 3, -2)):
        assert np.dot(n, np.asarray(direction) @ cell) == pytest.approx(0.0, abs=1e-12)


# ── placement ─────────────────────────────────────────────────────────────
def test_the_offset_of_a_point_is_h_x_plus_k_y_plus_l_z():
    cell = _cell(5.1, 6.3, 7.2, 81, 97, 103)
    point = np.array([0.25, 0.5, 0.75]) @ cell
    assert lp.offset_of(cell, (1, 2, 3), point) == pytest.approx(0.25 + 1.0 + 2.25)


def test_a_plane_placed_through_a_point_contains_it():
    cell = _cell(5.1, 6.3, 7.2, 81, 97, 103)
    point = np.array([0.31, 0.12, 0.77]) @ cell
    miller = (2, -1, 1)
    offset = lp.offset_of(cell, miller, point)
    foot = lp.plane_point(cell, miller, offset)
    assert np.dot(point - foot, lp.normal(cell, miller)) == pytest.approx(0.0, abs=1e-12)


def test_neighbouring_offsets_are_one_spacing_apart():
    cell = _cell(5.1, 6.3, 7.2, 81, 97, 103)
    miller = (1, 1, 2)
    step = lp.plane_point(cell, miller, 1.0) - lp.plane_point(cell, miller, 0.0)
    assert np.linalg.norm(step) == pytest.approx(lp.spacing(cell, miller))


def test_a_family_across_a_region_counts_every_plane_that_crosses_it():
    cell = 4.0 * np.eye(3)
    corners = lp.region_corners(np.zeros(3), cell)
    assert lp.offsets_across(cell, (1, 1, 0), 0.0, corners) == [0.0, 1.0, 2.0]
    assert sorted(lp.offsets_across(cell, (1, 1, 1), 0.5, corners)) == [0.5, 1.5, 2.5]
    # a 2×2×2 supercell holds twice as many along each index
    big = lp.region_corners(np.zeros(3), 2 * cell)
    assert len(lp.offsets_across(cell, (1, 0, 0), 0.0, big)) == 3
    # nearest the plane asked for first, and capped
    assert lp.offsets_across(cell, (1, 1, 1), 1.5, corners)[0] == 1.5
    assert len(lp.offsets_across(cell, (7, 5, 3), 0.0, big, limit=10)) == 10


# ── atoms on a plane ──────────────────────────────────────────────────────
def test_atoms_on_a_plane_and_on_its_family():
    from ase.build import bulk

    nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    cell, positions = nacl.cell[:], nacl.positions
    # (1 1 1) through the origin holds one atom of the cell; the family holds
    # the Na layers (integer offsets) — the Cl layers lie halfway between.
    assert list(lp.atoms_on(cell, LatticePlane((1, 1, 1)), positions)) == [0]
    family = lp.atoms_on(cell, LatticePlane((1, 1, 1), family=True), positions)
    assert set(nacl.get_chemical_symbols()[i] for i in family) == {"Na"} and len(family) == 4
    halfway = lp.atoms_on(cell, LatticePlane((1, 1, 1), 0.5, family=True), positions)
    assert set(nacl.get_chemical_symbols()[i] for i in halfway) == {"Cl"} and len(halfway) == 4
    # (1 0 0) planes are a apart and hold the x = 0 layer, Na and Cl mixed;
    # (2 0 0), half as far apart, holds every atom
    x0 = lp.atoms_on(cell, LatticePlane((1, 0, 0), family=True), positions)
    assert len(x0) == 4 and set(nacl.get_chemical_symbols()[i] for i in x0) == {"Na", "Cl"}
    assert len(lp.atoms_on(cell, LatticePlane((2, 0, 0), family=True), positions)) == 8
    assert len(lp.atoms_on(cell, LatticePlane((1, 0, 0), 0.25), positions)) == 0


def test_the_tolerance_is_a_distance_not_a_fraction_of_d():
    cell = 10.0 * np.eye(3)
    near = np.array([[0.05, 3.0, 3.0]])      # 0.05 Å off (1 0 0)
    far = np.array([[0.15, 3.0, 3.0]])       # 0.15 Å off it
    for miller in ((1, 0, 0), (4, 0, 0)):    # d = 10 Å and d = 2.5 Å alike
        assert len(lp.atoms_on(cell, LatticePlane(miller), near)) == 1
        assert len(lp.atoms_on(cell, LatticePlane(miller), far)) == 0
    assert len(lp.atoms_on(cell, LatticePlane((1, 0, 0)), np.empty((0, 3)))) == 0


# ── the outline drawn ─────────────────────────────────────────────────────
def test_a_cube_face_plane_is_a_square_and_a_111_plane_a_hexagon_or_triangle():
    cell = 4.0 * np.eye(3)
    origin = np.zeros(3)

    def outline(miller, offset):
        return lp.polygon_in_region(lp.plane_point(cell, miller, offset),
                                    lp.normal(cell, miller), origin, cell)

    square = outline((0, 0, 1), 0.25)
    assert len(square) == 4 and _area(square) == pytest.approx(16.0)
    assert np.allclose(square[:, 2], 1.0)
    hexagon = outline((1, 1, 1), 1.5)
    assert len(hexagon) == 6 and _area(hexagon) == pytest.approx(3 * np.sqrt(3) / 2 * 8, rel=1e-6)  # side 2√2
    triangle = outline((1, 1, 1), 0.75)
    assert len(triangle) == 3 and _area(triangle) == pytest.approx(np.sqrt(3) / 4 * 18, rel=1e-6)  # side 3√2
    # through a corner only, or outside altogether: nothing to draw
    assert outline((1, 1, 1), 0.0) is None
    assert outline((1, 1, 1), 3.2) is None


def test_a_plane_stops_at_the_faces_of_a_slanted_cell():
    cell = _cell(8.24, 13.45, 8.97, 90, 105.48, 90)
    miller = (0, 0, 1)
    polygon = lp.polygon_in_region(lp.plane_point(cell, miller, 0.5), lp.normal(cell, miller),
                                   np.zeros(3), cell)
    # the (0 0 1) section of a monoclinic cell is the a×b parallelogram, not the
    # axis-aligned box around the cell
    assert len(polygon) == 4
    assert _area(polygon) == pytest.approx(np.linalg.norm(np.cross(cell[0], cell[1])))
    fractional = polygon @ np.linalg.inv(cell)
    assert np.all(fractional > -1e-9) and np.all(fractional < 1 + 1e-9)
    assert np.allclose(fractional[:, 2], 0.5)


def test_the_region_can_be_moved_off_the_origin():
    cell = 4.0 * np.eye(3)
    origin = np.array([0.0, 0.0, -2.0])
    polygon = lp.polygon_in_region(lp.plane_point(cell, (0, 0, 1), -0.25),
                                   lp.normal(cell, (0, 0, 1)), origin, cell)
    assert polygon is not None and np.allclose(polygon[:, 2], -1.0)
    assert lp.polygon_in_region(lp.plane_point(cell, (0, 0, 1), 0.75),
                                lp.normal(cell, (0, 0, 1)), origin, cell) is None


def test_a_plane_carries_its_opacity_kept_between_0_and_1():
    assert LatticePlane((1, 0, 0)).opacity == lp.DEFAULT_OPACITY
    assert LatticePlane((1, 0, 0), opacity=1.7).opacity == 1.0
    assert LatticePlane((1, 0, 0), opacity=-0.2).opacity == 0.0
    plane = LatticePlane((1, 1, 0), 0.5, family=True)
    # restyled, it is still drawn in the same place; moved, it is not
    assert plane.drawn_the_same(LatticePlane((1, 1, 0), 0.5, family=True, color="#000000", opacity=0.9))
    assert not plane.drawn_the_same(LatticePlane((1, 1, 0), 0.25, family=True))
    assert not plane.drawn_the_same(LatticePlane((1, 1, 0), 0.5))
