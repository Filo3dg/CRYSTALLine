"""The a/b/c and a*/b*/c* view-alignment buttons.

Looking down a lattice axis needs an up direction, and which one is chosen is
what makes the three views agree or disagree with each other. Taking the other
lattice vectors in plain index order gave the b view a→up — the *previous* axis
where a and c both got the next one — which mirrored that one view against the
other two and threw the a/b/c gizmo to the opposite side of the screen.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

from crystalline.ui.viewport import Viewport  # noqa: E402

_CUBIC = np.eye(3) * 5.0
# ZnO's hexagonal cell: a and b are 120° apart, so the up vector has to be
# projected perpendicular to the view direction rather than used as-is.
_HEXAGONAL = np.array(
    [[2.85292, -1.647134, 0.0], [0.0, 3.294269, 0.0], [0.0, 0.0, 5.270251]]
)


def _screen_frame(cell, axis):
    """Which lattice vector points up and which points right, for this view."""
    direction = cell[axis] / np.linalg.norm(cell[axis])
    up = Viewport._up_for(direction, cell, axis)
    right = np.cross(direction, up)  # VTK: right = view direction × up
    def nearest(vector):
        dots = [float(np.dot(cell[i] / np.linalg.norm(cell[i]), vector)) for i in range(3)]
        return "abc"[int(np.argmax(dots))]
    return nearest(up), nearest(right)


@pytest.mark.parametrize("cell", [_CUBIC, _HEXAGONAL], ids=["cubic", "hexagonal"])
def test_the_three_axis_views_are_cyclic(cell):
    """Down a puts b up and c right; down b puts c up and a right; down c puts a
    up and b right. Any other pattern makes one view disagree with the others."""
    assert _screen_frame(cell, 0) == ("b", "c")
    assert _screen_frame(cell, 1) == ("c", "a")
    assert _screen_frame(cell, 2) == ("a", "b")


@pytest.mark.parametrize("cell", [_CUBIC, _HEXAGONAL], ids=["cubic", "hexagonal"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_up_vector_is_a_unit_vector_across_the_view(cell, axis):
    direction = cell[axis] / np.linalg.norm(cell[axis])
    up = Viewport._up_for(direction, cell, axis)

    assert np.linalg.norm(up) == pytest.approx(1.0)
    assert float(np.dot(up, direction)) == pytest.approx(0.0, abs=1e-12)


def test_a_degenerate_lattice_vector_falls_through_to_the_next_one():
    """A slab's aperiodic direction can be a zero (or vacuum) vector; the up
    choice has to skip it rather than normalising a zero."""
    cell = np.array([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])

    up = Viewport._up_for(np.array([1.0, 0.0, 0.0]), cell, 0)

    assert np.linalg.norm(up) == pytest.approx(1.0)
    assert np.allclose(np.abs(up), [0.0, 0.0, 1.0])  # fell through b to c


def test_without_a_cell_the_up_vector_comes_from_a_world_axis():
    """A molecule has no a/b/c; the view still has to be well defined."""
    for axis in range(3):
        direction = np.eye(3)[axis]
        up = Viewport._up_for(direction, None, axis)

        assert np.linalg.norm(up) == pytest.approx(1.0)
        assert float(np.dot(up, direction)) == pytest.approx(0.0, abs=1e-12)



# ── a*, b*, c* ────────────────────────────────────────────────────────────
_MONOCLINIC = np.array([[8.24, 0.0, 0.0], [0.0, 13.45, 0.0],
                        [8.967 * np.cos(np.radians(105.48)), 0.0,
                         8.967 * np.sin(np.radians(105.48))]])
_TRICLINIC = np.array([[5.1, 0.0, 0.0], [1.3, 6.1, 0.0], [0.9, 1.7, 6.6]])


@pytest.mark.parametrize("cell", [_MONOCLINIC, _TRICLINIC, _HEXAGONAL],
                         ids=["monoclinic", "triclinic", "hexagonal"])
def test_a_star_is_normal_to_b_and_c_and_on_a_s_side(cell):
    for axis in range(3):
        star = Viewport._reciprocal_direction(cell, axis)
        others = [cell[(axis + 1) % 3], cell[(axis + 2) % 3]]
        assert all(abs(float(np.dot(star, v))) < 1e-10 for v in others)
        assert float(np.dot(star, cell[axis])) == pytest.approx(1.0)   # a*·a = 1
    assert np.allclose(np.stack([Viewport._reciprocal_direction(cell, i) for i in range(3)]),
                       np.linalg.inv(cell).T)


def test_in_an_orthogonal_cell_a_star_is_just_a():
    for axis in range(3):
        star = Viewport._reciprocal_direction(_CUBIC, axis)
        assert np.allclose(star / np.linalg.norm(star), np.eye(3)[axis])


def test_a_left_handed_cell_keeps_the_sense_of_its_axes():
    left = _MONOCLINIC[[1, 0, 2]]           # a and b swapped: det < 0
    for axis in range(3):
        assert float(np.dot(Viewport._reciprocal_direction(left, axis), left[axis])) > 0


@pytest.mark.parametrize("cell", [_MONOCLINIC, _TRICLINIC], ids=["monoclinic", "triclinic"])
def test_down_a_star_the_bc_face_is_seen_square_on(cell):
    """What the button is for: b straight up, and c lying in the screen — both
    at their true length, where looking down a foreshortens the face."""
    star = Viewport._reciprocal_direction(cell, 0)
    view = star / np.linalg.norm(star)
    up = Viewport._up_for(view, cell, 0)
    assert np.allclose(up, cell[1] / np.linalg.norm(cell[1]))       # b is up, exactly
    assert abs(float(np.dot(cell[2], view))) < 1e-10               # c is in the screen
    right = np.cross(view, up)
    assert float(np.dot(cell[2], right)) > 0                         # and to the right
    # while down a, c is tipped out of the screen in a non-orthogonal cell
    along_a = cell[0] / np.linalg.norm(cell[0])
    assert abs(float(np.dot(cell[2], along_a))) > 0.1


def test_c_star_of_a_slab_with_no_c_is_the_view_onto_the_layer():
    slab = np.array([[4.0, 0.0, 0.0], [2.0, 3.5, 0.0], [0.0, 0.0, 0.0]])
    star = Viewport._reciprocal_direction(slab, 2)
    assert np.allclose(star / np.linalg.norm(star), [0.0, 0.0, 1.0])
    a_star = Viewport._reciprocal_direction(slab, 0)
    assert abs(float(np.dot(a_star, slab[1]))) < 1e-10 and abs(a_star[2]) < 1e-12


def test_a_lattice_with_no_volume_gives_no_reciprocal_axis():
    flat = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    assert Viewport._reciprocal_direction(flat, 0) is None
