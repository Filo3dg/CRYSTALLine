"""Lattice planes from Miller indices: spacing, placement, atoms on them, outline."""

import numpy as np
import pytest

from crystalline.core import lattice_planes as lp
from crystalline.core.lattice_planes import LatticePlane


def _cell(a, b, c, alpha, beta, gamma):
    from ase.geometry import cellpar_to_cell

    return np.asarray(cellpar_to_cell([a, b, c, alpha, beta, gamma]), dtype=float)


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


@pytest.mark.parametrize("miller", [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 1), (2, -1, 3)])
def test_orthorhombic_and_tetragonal_spacing_match_the_textbook_formula(miller):
    h, k, l = miller
    a, b, c = 4.0, 5.0, 6.0                                      # orthorhombic
    expected = 1.0 / np.sqrt(h * h / a**2 + k * k / b**2 + l * l / c**2)
    assert lp.spacing(_cell(a, b, c, 90, 90, 90), miller) == pytest.approx(expected)
    a, c = 4.0, 6.0                                              # tetragonal
    expected = 1.0 / np.sqrt((h * h + k * k) / a**2 + l * l / c**2)
    assert lp.spacing(_cell(a, a, c, 90, 90, 90), miller) == pytest.approx(expected)


@pytest.mark.parametrize("miller", [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, -1, 0), (2, 1, -1)])
def test_rhombohedral_spacing_matches_the_textbook_formula(miller):
    a, alpha = 4.75, np.radians(57.2)
    h, k, l = miller
    cos, sin = np.cos(alpha), np.sin(alpha)
    inverse = (((h * h + k * k + l * l) * sin * sin + 2 * (h * k + k * l + h * l) * (cos * cos - cos))
               / (a * a * (1 - 3 * cos * cos + 2 * cos**3)))
    cell = _cell(a, a, a, 57.2, 57.2, 57.2)
    assert lp.spacing(cell, miller) == pytest.approx(1 / np.sqrt(inverse))


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


def test_the_reciprocal_lattice_is_dual_to_the_cell():
    cell = _cell(5.1, 6.3, 7.2, 81, 97, 103)
    assert np.allclose(lp.reciprocal_lattice(cell) @ cell.T, np.eye(3))   # a*·a = 1, a*·b = 0
    with pytest.raises(ValueError):
        lp.reciprocal_lattice([[1, 0, 0], [2, 0, 0], [0, 0, 1]])


def test_the_normal_is_perpendicular_to_every_lattice_vector_in_the_plane():
    cell = _cell(5.1, 6.3, 7.2, 81, 97, 103)
    n = lp.plane_frame(cell, (1, 2, 3))[1]
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
    foot, normal, _d = lp.plane_frame(cell, miller, offset)
    assert np.dot(point - foot, normal) == pytest.approx(0.0, abs=1e-12)


def test_neighbouring_offsets_are_one_spacing_apart():
    cell = _cell(5.1, 6.3, 7.2, 81, 97, 103)
    miller = (1, 1, 2)
    step = lp.plane_frame(cell, miller, 1.0)[0] - lp.plane_frame(cell, miller, 0.0)[0]
    assert np.linalg.norm(step) == pytest.approx(lp.spacing(cell, miller))


def test_the_plane_frame_of_a_cube_and_a_hexagonal_cell():
    cubic = 4.21 * np.eye(3)
    hexagonal = np.array([[2.29, 0.0, 0.0], [-1.145, 1.9832, 0.0], [0.0, 0.0, 3.59]])
    point, normal, d = lp.plane_frame(cubic, (0, 0, 1), offset=0.5)
    assert np.allclose(normal, [0, 0, 1]) and d == pytest.approx(4.21)
    assert np.allclose(point, [0, 0, 2.105])
    assert lp.spacing(cubic, (1, 1, 1)) == pytest.approx(4.21 / np.sqrt(3))
    assert lp.spacing(cubic, (2, 0, 0)) == pytest.approx(4.21 / 2)
    assert lp.spacing(hexagonal, (0, 0, 1)) == pytest.approx(3.59)


def test_000_is_refused_everywhere_with_the_same_words():
    for ask in (lambda: lp.plane_frame(np.eye(3), (0, 0, 0)),
                lambda: lp.spacing(np.eye(3), (0, 0, 0)),
                lambda: LatticePlane((0, 0, 0))):
        with pytest.raises(ValueError, match=r"\(000\) is not a plane"):
            ask()


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
    far = np.array([[0.2, 3.0, 3.0]])        # 0.2 Å off it
    for miller in ((1, 0, 0), (4, 0, 0)):    # d = 10 Å and d = 2.5 Å alike
        assert len(lp.atoms_on(cell, LatticePlane(miller), near)) == 1
        assert len(lp.atoms_on(cell, LatticePlane(miller), far)) == 0
    assert len(lp.atoms_on(cell, LatticePlane((1, 0, 0)), np.empty((0, 3)))) == 0


def test_distances_are_signed_along_the_normal_and_wrap_for_a_family():
    cell = 4.0 * np.eye(3)
    points = np.array([[0.0, 0.0, 1.1], [0.0, 0.0, 4.9], [0.0, 0.0, 5.0]])
    single = lp.distances_to(cell, LatticePlane((0, 0, 1), 0.25), points)   # the plane z = 1
    assert np.allclose(single, [0.1, 3.9, 4.0])
    family = lp.distances_to(cell, LatticePlane((0, 0, 1), 0.25, family=True), points)
    assert np.allclose(family, [0.1, -0.1, 0.0])


def test_a_plane_carries_its_opacity_kept_between_0_and_1():
    assert LatticePlane((1, 0, 0)).opacity == lp.DEFAULT_OPACITY
    assert LatticePlane((1, 0, 0), opacity=1.7).opacity == 1.0
    assert LatticePlane((1, 0, 0), opacity=-0.2).opacity == 0.0
    plane = LatticePlane((1, 1, 0), 0.5, family=True)
    # restyled, it is still drawn in the same place; moved, it is not
    assert plane.drawn_the_same(LatticePlane((1, 1, 0), 0.5, family=True, color="#000000", opacity=0.9))
    assert not plane.drawn_the_same(LatticePlane((1, 1, 0), 0.25, family=True))
    assert not plane.drawn_the_same(LatticePlane((1, 1, 0), 0.5))


# ── planes fitted to atoms ────────────────────────────────────────────────
def test_a_fit_is_exact_for_coplanar_atoms():
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    plane = lp.fit_plane(square)
    assert plane.point == pytest.approx((0.5, 0.5, 0.0))
    assert plane.normal == pytest.approx((0.0, 0.0, 1.0))    # sign fixed: largest component positive
    assert plane.rms == pytest.approx(0.0, abs=1e-12)
    assert len(plane.points) == 4 and not plane.family


def test_a_fit_reports_how_far_from_planar_the_atoms_are():
    puckered = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0.4]], dtype=float)
    assert lp.fit_plane(puckered).rms > 0.1


def test_a_fit_needs_three_atoms():
    with pytest.raises(ValueError):
        lp.fit_plane(np.eye(2, 3))


def test_atoms_on_a_fitted_plane_need_no_cell():
    plane = lp.fit_plane([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
    positions = np.array([[5, 5, 1.0], [0, 0, 1.1], [0, 0, 2.0]])
    assert lp.distances_to(None, plane, positions) == pytest.approx([0.0, 0.1, 1.0])
    assert list(lp.atoms_on(None, plane, positions)) == [0, 1]


@pytest.mark.parametrize("cell", [
    _cell(5.43, 5.43, 5.43, 90, 90, 90),     # cubic
    _cell(4.0, 4.0, 6.0, 90, 90, 90),        # tetragonal
    _cell(4.0, 5.0, 6.0, 90, 90, 90),        # orthorhombic
    _cell(5.0, 6.0, 7.0, 90, 105, 90),       # monoclinic
    _cell(5.1, 6.3, 7.2, 81, 97, 103),       # triclinic
    _cell(4.55, 4.55, 11.86, 90, 90, 120),   # trigonal / hexagonal axes
    _cell(4.75, 4.75, 4.75, 57.2, 57.2, 57.2),  # trigonal, rhombohedral axes
    _cell(3.99, 3.99, 3.99, 60, 60, 60),     # primitive cell of an fcc crystal
], ids=["cubic", "tetragonal", "orthorhombic", "monoclinic", "triclinic",
        "hexagonal-axes", "rhombohedral-axes", "fcc-primitive"])
def test_the_nearest_hkl_of_a_fit_is_its_own_when_it_is_a_lattice_plane(cell):
    """A plane's normal is along [hkl] only when the axes are orthogonal *and*
    equally long: (1 1 0) of an orthorhombic or tetragonal-c cell, or any plane
    of an oblique one, is where finding (hkl) from a normal can go wrong."""
    for miller in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 1, 1),
                   (2, -1, 3), (1, 0, -2)]:
        point, normal, _d = lp.plane_frame(cell, miller, 0.5)
        found, angle = lp.nearest_miller(cell, normal)
        assert found in (miller, tuple(-v for v in miller))     # (hkl) and (h̄k̄l̄) are one plane
        assert angle == pytest.approx(0.0, abs=1e-6)
        assert lp.nearest_miller(cell, -normal)[0] == found


def test_the_nearest_hkl_of_a_tilted_fit_says_how_far_off_it_is():
    cubic = np.eye(3) * 4.0
    tilt = np.radians(3.0)
    miller, angle = lp.nearest_miller(cubic, [np.sin(tilt), 0.0, np.cos(tilt)])
    assert miller == (0, 0, 1) and angle == pytest.approx(3.0, abs=1e-6)
    # among equally good planes, the one with the smaller indices
    assert lp.nearest_miller(cubic, [1, 1, 0])[0] == (1, 1, 0)


def test_a_fit_restyled_is_drawn_the_same_and_never_as_an_indexed_plane():
    import dataclasses

    plane = lp.fit_plane([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert plane.drawn_the_same(dataclasses.replace(plane, color="#000000", opacity=0.9))
    assert not plane.drawn_the_same(lp.fit_plane([[0, 0, 1], [1, 0, 1], [0, 1, 1]]))
    assert not plane.drawn_the_same(LatticePlane((0, 0, 1)))
    assert not LatticePlane((0, 0, 1)).drawn_the_same(plane)
