"""Tests for StructurePanel's multi-selection model and editing gate.

Pure-Qt widget (no VTK), so it runs headless; skips without PySide6.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui.panels.structure_panel import StructurePanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _panel(qapp, n=4):
    s = Structure.empty()
    for i in range(n):
        s.add_atom("C", [i, 0, 0])
    return StructurePanel(s), s


def test_multiselect_toggle_replace_and_range_filter(qapp):
    panel, _ = _panel(qapp)
    emitted = []
    panel.selection_changed.connect(lambda idx: emitted.append(list(idx)))

    panel.set_selection([1, 3])
    assert panel.selected_indices() == [1, 3]

    panel.select_atom(2, additive=True)  # add
    assert panel.selected_indices() == [1, 2, 3]

    panel.select_atom(1, additive=True)  # toggle off
    assert panel.selected_indices() == [2, 3]

    panel.select_atom(0, additive=False)  # replace
    assert panel.selected_indices() == [0]

    panel.set_selection([0, 9, 5])  # out-of-range filtered away
    assert panel.selected_indices() == [0]

    assert emitted  # selection_changed fired for the UI/highlights


def test_editing_gate_controls(qapp):
    panel, _ = _panel(qapp)
    # editing off by default: add + coordinate editor disabled
    assert not panel.add_btn.isEnabled()

    panel.set_editing_enabled(True)
    assert panel.add_btn.isEnabled()

    panel.set_selection([2])  # single selection while editing -> editor on
    assert panel.editor.isEnabled()

    panel.set_selection([1, 2])  # multi-selection -> single-atom editor off
    assert not panel.editor.isEnabled()


def test_an_atom_added_to_a_slab_lands_on_the_slab(qapp):
    """The Add atom button, on a structure that is not a crystal.

    CRYSTAL writes a formal 500 Å across the direction a slab does not repeat
    in, and the button used to drop the atom at the centre of that cell — 250 Å
    above the surface. It was added and selected, off screen and out of reach of
    the picker, which is indistinguishable from nothing having happened.
    """
    import numpy as np
    from ase.build import fcc111

    slab = fcc111("Pt", size=(2, 2, 3), vacuum=0.0)
    cell = np.asarray(slab.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, 500.0]
    slab.set_cell(cell)
    slab.pbc = [True, True, False]
    structure = Structure.from_ase(slab)
    surface = np.asarray(structure.positions)[:, 2].max()

    panel = StructurePanel(structure)
    panel.set_editing_enabled(True)
    panel.element_box.setCurrentText("O")
    panel.add_btn.click()

    assert len(structure) == len(slab) + 1
    added = np.asarray(structure.positions)[-1]
    assert panel.selected_indices() == [len(structure) - 1]  # selected, ready to drag
    assert added[2] < surface + 20.0, "the atom is out in the vacuum, not on the slab"


def test_both_ways_of_adding_an_atom_agree_on_where_it_goes(qapp):
    """The panel's button and the element picker's signal are separate paths,
    and each had its own copy of "the centre of the cell" to get wrong."""
    import numpy as np
    from ase.build import fcc111

    from crystalline.ui.main_window import MainWindow

    slab = fcc111("Pt", size=(2, 2, 3), vacuum=0.0)
    cell = np.asarray(slab.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, 500.0]
    slab.set_cell(cell)
    slab.pbc = [True, True, False]
    structure = Structure.from_ase(slab)

    class _StubWindow:  # the method reads nothing else
        pass

    _StubWindow.structure = structure
    _StubWindow._new_atom_position = MainWindow._new_atom_position

    panel = StructurePanel(structure)
    assert _StubWindow()._new_atom_position() == pytest.approx(panel._default_position())
