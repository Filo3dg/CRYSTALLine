"""Geometry panel: measure the selection, and edit atoms with the mouse.

Docked beside Info and Display. Three parts, each a section that folds away
under its header, as the Display panel's do; which are open is remembered:

* **Measure** — turn the current 3D selection into a position (1 atom), a
  distance (2), an angle (3) or a dihedral (4), keep a list of them, and draw
  the ones that are ticked over the structure as points and lines, each line
  in its own colour and thickness.
* **Lattice planes** — draw a crystallographic plane, or its whole family,
  from its Miller indices: placed at a typed position along the normal or
  through the selected atom, with d(hkl) and the atoms lying on it. Or fit a
  plane to three or more selected atoms, kept exactly as fitted and listed with
  its rms deviation and the (hkl) nearest to it.
* **Atoms** — add, delete, duplicate, re-element, translate the selection and
  place a single atom at typed cartesian coordinates, without going to the Edit
  menu. These mirror the Edit-menu actions and are
  gated the same way: they need **Editing mode**, which is exposed here as a
  checkbox so it is obvious (and one click away) why a button is greyed out.

The panel owns no editing logic — it emits intent and :class:`MainWindow` runs
the same slots the menu uses, so undo/redo and the selection model behave
identically whichever route the user takes.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from crystalline.core import lattice_planes as lp
from crystalline.core import measure as measure_mod
from crystalline.core.structure import Structure
from crystalline.ui import preferences
from crystalline.ui.panels.controls import Section
from crystalline.ui.safety import guard
from crystalline.ui.widgets.miller import MillerIndices

# Same starter palette the (hidden) structure panel used; free text is allowed.
_ELEMENTS = ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Na", "Mg", "Al", "Ca", "Ti", "Fe"]
# Range of the position boxes. Fractional coordinates sit in [0, 1) for a plain
# cell and a little outside it for the images boundary completion adds, but the
# aperiodic axis of a slab is shown in Å against CRYSTAL's formal 500 Å vector,
# so the range has to cover that too.
_COORD_RANGE = 1000.0
# Spinner step: a hundredth of a cell edge, or a tenth of an Ångström.
_FRACTIONAL_STEP = 0.01
_ANGSTROM_STEP = 0.1
# Below this (Å) a typed coordinate counts as unchanged, so tabbing out of a box
# without editing it does not record an undo step for a move of nothing.
_COORD_EPS = 1e-9
# How far a coordinate box may be squeezed (px). Their *preferred* width is
# ignored entirely — see the size policy in _build_atoms_group.
_COORD_BOX_MIN_WIDTH = 56

# Lattice planes. Where a typed plane sits, in units of d(hkl) from the plane through the origin;
# wide enough for a large supercell.
_PLANE_OFFSET_RANGE = 100.0
_PLANE_OFFSET_STEP = 0.25
# Each new plane takes the next colour, so two planes on screen can be told
# apart and matched to their line in the list. Chosen away from the measurement
# and symmetry-element colours.
_PLANE_COLORS = ("#e6550d", "#3182bd", "#31a354", "#756bb1", "#d6616b", "#8c6d31")
# The height (px) the two lists ask for: a few rows. They grow into any spare
# height; asking for Qt's default 192 px each put the panel into scrolling in a
# dock of ordinary height.
_LIST_HINT_HEIGHT = 96
# The opacity box, as wide as the value boxes of the Display panel.
_OPACITY_BOX_WIDTH = 66
# The range of a measurement line's thickness (Å): from a hairline to about four
# times a bond at the Display panel's default radius (0.06 Å, so 0.12 Å thick).
_THICKNESS_MIN = 0.02
_THICKNESS_MAX = 0.50
# The panel's sections: the key each is remembered under, and its header.
_SECTIONS = (("measure", "Measure"), ("planes", "Lattice planes"), ("atoms", "Atoms"))


class GeometryPanel(QWidget):
    """Measurements and atom-editing tools for the current selection."""

    # editing intent — MainWindow runs the matching Edit-menu slot
    editing_toggled = Signal(bool)
    add_atom_requested = Signal(str)      # element symbol
    delete_requested = Signal()
    duplicate_requested = Signal()
    translate_requested = Signal()
    set_element_requested = Signal(str)   # element symbol
    set_position_requested = Signal(object)  # cartesian (x, y, z) in Å for the one selected atom
    # the measurements that should be drawn in 3D (possibly empty)
    annotations_changed = Signal(list)
    # the lattice planes to draw (possibly empty), and the cell their indices are in
    lattice_planes_changed = Signal(list, object)
    # atoms (indices into the shown structure) lying on the chosen planes, to select
    select_atoms_requested = Signal(list)

    def __init__(self, structure: Structure, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._structure = structure
        self._selection: List[int] = []
        self._measurements: List[measure_mod.Measurement] = []
        self._editing = False
        self._planes: List[lp.LatticePlane] = []
        self._miller_cell: Optional[np.ndarray] = None
        self._next_plane_colour = 0

        # Three sections open do not fit a short dock, and a dock's minimum
        # height is the window's: scroll, as the Display panel does, rather than
        # make the window refuse to get shorter. Downwards only — the rows compress.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)

        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 4, 14, 18)  # the Display panel's
        layout.setSpacing(0)  # sections space themselves
        # Each a header that folds the section away, as in the Display panel:
        # someone who only ever measures need not scroll past the atom tools.
        builders = {"measure": self._build_measure_group,
                    "planes": self._build_planes_group,
                    "atoms": self._build_atoms_group}
        self._sections = {}
        self._restoring_sections = False
        widest = 0
        for key, title in _SECTIONS:
            section = Section(layout, title, collapsed=not self._remembered_open(key))
            content = builders[key]()
            widest = max(widest, content.minimumSizeHint().width(),
                         section.header.minimumSizeHint().width())
            section.add_wide(content)
            section.header.toggled.connect(
                lambda open_, key=key: self._remember_section(key, open_)
            )
            self._sections[key] = section
        layout.addStretch(1)
        # Never narrower than the rows plus a scroll bar, which would cut their
        # right edge off once the scroll bar appears. Taken from each section's
        # rows rather than from the body, which leaves a folded section out —
        # and, before the panel is first shown, an open one too — so unfolding
        # a section never asks the dock to widen.
        margins = layout.contentsMargins()
        scroll.setMinimumWidth(
            widest + margins.left() + margins.right()
            + scroll.verticalScrollBar().sizeHint().width()
        )
        self._sync_plane_form()
        self._sync_buttons()

    # ── sections ────────────────────────────────────────────────────────
    @staticmethod
    def _remembered_open(key: str) -> bool:
        return preferences.section_open(f"geometry/{key}")

    def _remember_section(self, key: str, open_: bool) -> None:
        if not self._restoring_sections:
            preferences.set_section_open(f"geometry/{key}", open_)

    def section_open(self, key: str) -> bool:
        """Whether the section ``key`` (``"measure"``, ``"planes"``, ``"atoms"``) is open."""
        return self._sections[key].header.isChecked()

    def _restore_sections(self) -> None:
        """Open and close the sections as they were last left — in any tab."""
        self._restoring_sections = True
        try:
            for key, section in self._sections.items():
                section.header.setChecked(self._remembered_open(key))
        finally:
            self._restoring_sections = False

    @guard()
    def showEvent(self, event) -> None:  # noqa: N802 - Qt's name
        """Each tab has a panel of its own: one shown again takes up the
        sections as they were last left in whichever tab that was."""
        self._restore_sections()
        super().showEvent(event)

    # ── construction ────────────────────────────────────────────────────
    def _build_measure_group(self) -> QWidget:
        group = QWidget()
        box = QVBoxLayout(group)
        box.setContentsMargins(0, 0, 0, 0)

        self._hint = QLabel(measure_mod.selection_hint(0))
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: palette(mid);")
        box.addWidget(self._hint)

        self._measure_btn = QPushButton("Measure selection")
        self._measure_btn.setToolTip(
            "1 atom → position, 2 → distance, 3 → angle, 4 → dihedral. "
            "A plane through the atoms is fitted under Lattice planes."
        )
        self._measure_btn.clicked.connect(self._measure_selection)
        box.addWidget(self._measure_btn)

        # Ticked measurements are drawn in the 3D view.
        self._list = _CompactList()
        self._list.setToolTip("Tick a measurement to show it in the 3D view")
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        self._list.itemChanged.connect(lambda _item: self._emit_annotations())
        box.addWidget(self._list, 1)

        row = QHBoxLayout()
        self._color_btn = QPushButton("Colour…")
        self._color_btn.setToolTip("Set the colour of the selected measurement(s)")
        self._color_btn.clicked.connect(self._set_measurement_colour)
        row.addWidget(self._color_btn)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_selected_measurements)
        row.addWidget(self._remove_btn)
        self._clear_btn = QPushButton("Clear all")
        self._clear_btn.clicked.connect(self.clear_measurements)
        row.addWidget(self._clear_btn)
        row.addStretch(1)
        box.addLayout(row)
        # The colour button follows the list selection, not just "any measurement".
        self._list.itemSelectionChanged.connect(self._sync_buttons)
        self._list.itemSelectionChanged.connect(self._show_selected_thickness)

        # How thick the lines are: the default reads well on a small cell and
        # vanishes on a big supercell seen whole. Works like the planes' opacity
        # below — on the measurements selected, else on all, and on the next.
        row = QHBoxLayout()
        row.addWidget(QLabel("Thickness (Å)"))
        tip = ("How thick the lines of distances, angles and dihedrals are drawn: those "
               "selected in the list — all of them when none is selected — and the ones "
               "measured next")
        self._thickness_slider = QSlider(Qt.Horizontal)
        self._thickness_slider.setRange(int(round(_THICKNESS_MIN * 100)),
                                        int(round(_THICKNESS_MAX * 100)))
        self._thickness_slider.setValue(int(round(measure_mod.DEFAULT_THICKNESS * 100)))
        self._thickness_slider.setToolTip(tip)
        self._thickness_slider.valueChanged.connect(
            lambda value: self._set_measurement_thickness(value / 100.0)
        )
        row.addWidget(self._thickness_slider, 1)
        self._thickness_box = QDoubleSpinBox()
        self._thickness_box.setRange(_THICKNESS_MIN, _THICKNESS_MAX)
        self._thickness_box.setSingleStep(0.01)
        self._thickness_box.setDecimals(2)
        self._thickness_box.setValue(measure_mod.DEFAULT_THICKNESS)
        self._thickness_box.setToolTip(tip)
        self._thickness_box.setFixedWidth(_OPACITY_BOX_WIDTH)
        self._thickness_box.valueChanged.connect(self._set_measurement_thickness)
        row.addWidget(self._thickness_box)
        box.addLayout(row)
        return group

    def _build_planes_group(self) -> QWidget:
        self._planes_group = group = QWidget()
        box = QVBoxLayout(group)
        box.setContentsMargins(0, 0, 0, 0)

        # Which cell the indices are in: the one thing that decides what (001)
        # means, so it is said rather than left to be guessed.
        self._plane_hint = QLabel()
        self._plane_hint.setWordWrap(True)
        self._plane_hint.setToolTip(
            "Miller indices are quoted in the conventional cell — as for the density "
            "slice — whatever cell, primitive view or supercell is on screen"
        )
        self._plane_hint.setStyleSheet("color: palette(mid);")
        box.addWidget(self._plane_hint)

        row = QHBoxLayout()
        # The same boxes the density slice asks for a plane with.
        self._miller = MillerIndices((1, 0, 0))
        self._miller.changed.connect(self._sync_plane_form)
        self._miller_boxes = self._miller.boxes
        self._i_label = self._miller.i_label
        row.addWidget(self._miller, 1)
        self._spacing_label = QLabel()
        self._spacing_label.setToolTip("Distance between neighbouring planes of the family")
        row.addWidget(self._spacing_label)
        box.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Position"))
        self._plane_offset = QDoubleSpinBox()
        self._plane_offset.setRange(-_PLANE_OFFSET_RANGE, _PLANE_OFFSET_RANGE)
        self._plane_offset.setDecimals(3)
        self._plane_offset.setSingleStep(_PLANE_OFFSET_STEP)
        self._plane_offset.setSuffix(" d")
        self._plane_offset.setToolTip(
            "Along the normal, in units of d(hkl) from the plane through the origin: "
            "0 and 1 are neighbouring planes of the family, 0.5 lies halfway between"
        )
        self._plane_offset.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._plane_offset.setMinimumWidth(_COORD_BOX_MIN_WIDTH)
        row.addWidget(self._plane_offset, 1)
        self._family_check = QCheckBox("Whole family")
        self._family_check.setToolTip(
            "Draw every plane of the family across the cell on screen, d(hkl) apart"
        )
        row.addWidget(self._family_check)
        box.addLayout(row)

        row = QHBoxLayout()
        self._add_plane_btn = QPushButton("Add plane")
        self._add_plane_btn.setToolTip("Draw (hkl) at the position typed")
        self._add_plane_btn.clicked.connect(self._add_typed_plane)
        row.addWidget(self._add_plane_btn, 1)
        self._plane_atom_btn = QPushButton("Through selected atom")
        self._plane_atom_btn.setToolTip("Draw (hkl) through the one atom selected")
        self._plane_atom_btn.clicked.connect(self._add_plane_through_atom)
        row.addWidget(self._plane_atom_btn, 1)
        box.addLayout(row)

        # The other way to a plane: from atoms rather than from indices. Kept as
        # fitted — a molecule's plane is seldom a lattice plane, and snapping it
        # to the nearest one would tilt it off the atoms — and the row names the
        # (hkl) it is closest to.
        self._fit_plane_btn = QPushButton("Fit to selected atoms")
        self._fit_plane_btn.setToolTip(
            "Least-squares plane through 3 or more selected atoms, drawn exactly as "
            "fitted and listed with its rms deviation and the nearest (hkl)"
        )
        self._fit_plane_btn.clicked.connect(self._add_fitted_plane)
        box.addWidget(self._fit_plane_btn)

        self._plane_list = _CompactList()
        self._plane_list.setToolTip("Tick a plane to show it in the 3D view")
        self._plane_list.setSelectionMode(QListWidget.ExtendedSelection)
        self._plane_list.itemChanged.connect(lambda _item: self._emit_planes())
        self._plane_list.itemSelectionChanged.connect(self._sync_buttons)
        self._plane_list.itemSelectionChanged.connect(self._show_selected_opacity)
        box.addWidget(self._plane_list, 1)

        row = QHBoxLayout()
        self._plane_colour_btn = QPushButton("Colour…")
        self._plane_colour_btn.setToolTip("Set the colour of the selected plane(s)")
        self._plane_colour_btn.clicked.connect(self._set_plane_colour)
        row.addWidget(self._plane_colour_btn)
        self._plane_select_btn = QPushButton("Select atoms")
        self._plane_select_btn.setToolTip("Select the atoms lying on the selected plane(s)")
        self._plane_select_btn.clicked.connect(self._select_plane_atoms)
        row.addWidget(self._plane_select_btn)
        self._plane_remove_btn = QPushButton("Remove")
        self._plane_remove_btn.setToolTip("Remove the selected plane(s)")
        self._plane_remove_btn.clicked.connect(self._remove_selected_planes)
        row.addWidget(self._plane_remove_btn)
        row.addStretch(1)
        box.addLayout(row)

        # How see-through the sheets are: a plane that reads well over a sparse
        # cell can hide a dense one, or vanish against the background. Live, like
        # the Display panel's opacities; the outline stays solid at any value.
        row = QHBoxLayout()
        row.addWidget(QLabel("Opacity"))
        tip = ("Opacity of the selected plane(s) — of every plane when none is "
               "selected — and of the planes added next. At 0 only the outline is drawn.")
        self._plane_opacity_slider = QSlider(Qt.Horizontal)
        self._plane_opacity_slider.setRange(0, 100)
        self._plane_opacity_slider.setValue(int(round(lp.DEFAULT_OPACITY * 100)))
        self._plane_opacity_slider.setToolTip(tip)
        self._plane_opacity_slider.valueChanged.connect(
            lambda value: self._set_plane_opacity(value / 100.0)
        )
        row.addWidget(self._plane_opacity_slider, 1)
        self._plane_opacity_box = QDoubleSpinBox()
        self._plane_opacity_box.setRange(0.0, 1.0)
        self._plane_opacity_box.setSingleStep(0.05)
        self._plane_opacity_box.setDecimals(2)
        self._plane_opacity_box.setValue(lp.DEFAULT_OPACITY)
        self._plane_opacity_box.setToolTip(tip)
        self._plane_opacity_box.setFixedWidth(_OPACITY_BOX_WIDTH)
        self._plane_opacity_box.valueChanged.connect(self._set_plane_opacity)
        row.addWidget(self._plane_opacity_box)
        box.addLayout(row)
        return group

    def _build_atoms_group(self) -> QWidget:
        group = QWidget()
        box = QVBoxLayout(group)
        box.setContentsMargins(0, 0, 0, 0)

        # The gate the Edit menu applies invisibly — shown here so a greyed-out
        # button explains itself.
        self._edit_check = QCheckBox("Editing mode")
        self._edit_check.setToolTip(
            "Atom tools (and dragging atoms in 3D) need editing mode — same as Edit ▸ Editing mode (Ctrl+E)"
        )
        self._edit_check.toggled.connect(self._on_editing_toggled)
        box.addWidget(self._edit_check)

        row = QHBoxLayout()
        row.addWidget(QLabel("Element"))
        self._element = QComboBox()
        self._element.addItems(_ELEMENTS)
        self._element.setCurrentText("C")
        self._element.setEditable(True)  # any symbol, e.g. "Zr"
        row.addWidget(self._element, 1)
        # A drawn glyph rather than the "⊞" box-drawing character, whose shape and
        # weight are whatever font the platform happens to resolve it in.
        self._table_btn = QPushButton()
        self._table_btn.setIcon(_table_icon())
        self._table_btn.setToolTip("Pick an element from the periodic table")
        self._table_btn.setFixedWidth(34)
        self._table_btn.clicked.connect(self._pick_from_periodic_table)
        row.addWidget(self._table_btn)
        self._add_btn = QPushButton("Add")
        self._add_btn.setToolTip("Add an atom of this element at the centre of the cell")
        self._add_btn.clicked.connect(
            lambda: self.add_atom_requested.emit(self._element.currentText().strip())
        )
        row.addWidget(self._add_btn)
        self._set_element_btn = QPushButton("Set")
        self._set_element_btn.setToolTip("Change the selected atoms to this element")
        self._set_element_btn.clicked.connect(
            lambda: self.set_element_requested.emit(self._element.currentText().strip())
        )
        row.addWidget(self._set_element_btn)
        box.addLayout(row)

        row = QHBoxLayout()
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setToolTip("Delete the selected atoms (Del)")
        self._delete_btn.clicked.connect(self.delete_requested)
        row.addWidget(self._delete_btn)
        self._duplicate_btn = QPushButton("Duplicate")
        self._duplicate_btn.clicked.connect(self.duplicate_requested)
        row.addWidget(self._duplicate_btn)
        self._translate_btn = QPushButton("Translate…")
        self._translate_btn.clicked.connect(self.translate_requested)
        row.addWidget(self._translate_btn)
        box.addLayout(row)

        # ── exact position of a single atom ─────────────────────────────
        # Dragging is quick but never exact; a structure is usually specified by
        # coordinates, so one atom at a time can be placed by typing them.
        self._coord_label = QLabel()
        box.addWidget(self._coord_label)
        row = QHBoxLayout()
        self._coord_boxes = []
        for axis in ("x", "y", "z"):
            spin = QDoubleSpinBox()
            spin.setRange(-_COORD_RANGE, _COORD_RANGE)
            spin.setDecimals(4)
            spin.setPrefix(f"{axis} ")
            # A box wide enough for "-1000.0000" would add ~110px each to the
            # panel's preferred width, and a dock takes its width from that — three
            # of them pushed this dock wider than the 3D view beside it. Ignoring
            # the preferred width lets them simply share the row and shrink with
            # the dock, down to _COORD_BOX_MIN_WIDTH.
            spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            spin.setMinimumWidth(_COORD_BOX_MIN_WIDTH)
            # Applied as soon as the value is committed — Enter, tabbing out, or a
            # click on the spinner. Keyboard tracking stays off so that *typing*
            # "0.25" commits once, rather than moving the atom to 0, then 0.2,
            # then 0.25: three edits, three undo steps, three redraws.
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._emit_position)
            row.addWidget(spin, 1)
            self._coord_boxes.append(spin)
        box.addLayout(row)
        return group

    # ── coordinate frame ────────────────────────────────────────────────
    def _lattice(self):
        """``(cell, periodic)`` for the shown structure, or ``None`` without a cell.

        ``None`` means a molecule: there is no lattice for a fraction to be of,
        so the boxes fall back to plain Ångström.
        """
        cell = np.asarray(self._structure.cell, dtype=float)
        if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-8:
            return None
        return cell, np.asarray(self._structure.pbc, dtype=bool)

    def _to_display(self, position):
        """Cartesian position → the three numbers the boxes show.

        Fractional along each *periodic* lattice vector, Ångström along an
        aperiodic one — the convention CRYSTAL itself uses for slabs and polymers
        (and that this app's deck writer already follows), because a fraction of
        the formal 500 Å vacuum vector is not a coordinate anyone can read.
        """
        frame = self._lattice()
        if frame is None:
            return [float(v) for v in position]
        cell, periodic = frame
        fractional = np.asarray(position, dtype=float) @ np.linalg.inv(cell)
        return [
            float(fractional[k]) if periodic[k]
            else float(fractional[k] * np.linalg.norm(cell[k]))
            for k in range(3)
        ]

    def _from_display(self, values):
        """The inverse of :meth:`_to_display` — box values → a cartesian position."""
        frame = self._lattice()
        if frame is None:
            return np.asarray(values, dtype=float)
        cell, periodic = frame
        fractional = np.asarray(
            [
                values[k] if periodic[k]
                else values[k] / float(np.linalg.norm(cell[k]))
                for k in range(3)
            ],
            dtype=float,
        )
        return fractional @ cell

    def _current_position(self):
        """The selected atom's position, or ``None`` unless exactly one is selected."""
        if len(self._selection) != 1:
            return None
        index = self._selection[0]
        if not 0 <= index < len(self._structure):
            return None
        return self._structure.positions[index]

    def sync_position(self) -> None:
        """Show the selected atom's coordinates, in the frame that suits the cell.

        Called on every selection change *and* on every structure change, so the
        boxes still read true after the atom is dragged, nudged or undone —
        otherwise they would keep offering a stale position to move back to.

        Signals are blocked while filling: these boxes apply on ``valueChanged``,
        and writing the atom's own position back into them must not be mistaken
        for the user asking to move it there again.
        """
        frame = self._lattice()
        if frame is None:
            self._coord_label.setText("Position of the selected atom (Å)")
        elif bool(np.all(frame[1])):
            self._coord_label.setText("Position of the selected atom (fractional)")
        else:
            # A slab or polymer: fractional along the periodic axes, Å along the
            # vacuum one. Naming both keeps the mixed row honest.
            self._coord_label.setText("Position of the selected atom (fractional / Å)")

        position = self._current_position()
        shown = [0.0, 0.0, 0.0] if position is None else self._to_display(position)
        periodic = (True, True, True) if frame is None else frame[1]
        for k, spin in enumerate(self._coord_boxes):
            blocked = spin.blockSignals(True)  # programmatic fill is not an edit
            step = _FRACTIONAL_STEP if (frame is not None and periodic[k]) else _ANGSTROM_STEP
            spin.setSingleStep(step)
            spin.setValue(shown[k])
            spin.blockSignals(blocked)
        self._sync_buttons()

    def _emit_position(self, _value: float = 0.0) -> None:
        """Ask for the selected atom to be placed at the coordinates now shown.

        Silent when nothing would change: a box also commits on focus loss, so
        merely clicking away from an untouched one must not record an undo step
        for a move of nothing.
        """
        position = self._current_position()
        if position is None or not self._editing:
            return
        typed = [spin.value() for spin in self._coord_boxes]
        if all(
            abs(typed[k] - shown) < _COORD_EPS
            for k, shown in enumerate(self._to_display(position))
        ):
            return
        self.set_position_requested.emit(
            [float(v) for v in self._from_display(typed)]
        )

    def _pick_from_periodic_table(self) -> None:
        """Open the visual periodic table and load the chosen element into the box."""
        from crystalline.ui.periodic_table import PeriodicTableDialog

        symbol = PeriodicTableDialog.pick(self, current=self._element.currentText().strip())
        if symbol:
            self._element.setCurrentText(symbol)

    # ── external hooks ──────────────────────────────────────────────────
    def refresh_theme_icons(self) -> None:
        """Redraw the periodic-table glyph after a theme change."""
        self._table_btn.setIcon(_table_icon())

    def set_structure(self, structure: Structure) -> None:
        """Rebind after a load or a view change; measurements no longer apply.

        Lattice planes do: they are named by indices, not by atoms, so they are
        kept across a supercell or a change of view — and are cleared on a new
        file with :meth:`clear_lattice_planes`.
        """
        self._structure = structure
        self._selection = []
        self.clear_measurements()
        self.refresh_planes()
        self.sync_position()  # also calls _sync_buttons

    def set_selection(self, indices: Sequence[int]) -> None:
        """Track the shared selection (driven by 3D picking)."""
        self._selection = [int(i) for i in indices]
        self._hint.setText(measure_mod.selection_hint(len(self._selection)))
        self.sync_position()  # also calls _sync_buttons

    def set_editing_enabled(self, enabled: bool) -> None:
        """Reflect the Edit menu's editing mode without re-emitting it."""
        self._editing = bool(enabled)
        blocked = self._edit_check.blockSignals(True)
        self._edit_check.setChecked(self._editing)
        self._edit_check.blockSignals(blocked)
        self._sync_buttons()

    def measurements(self) -> List[measure_mod.Measurement]:
        """Every measurement in the list, shown or not (handy in tests)."""
        return list(self._measurements)

    def shown_annotations(self) -> List[measure_mod.Measurement]:
        """The measurements currently ticked for display in 3D."""
        return [
            m
            for m, item in zip(self._measurements, self._items())
            if item.checkState() == Qt.Checked
        ]

    def clear_measurements(self) -> None:
        self._measurements = []
        self._list.clear()
        self._emit_annotations()

    # ── measuring ───────────────────────────────────────────────────────
    def _measure_selection(self) -> None:
        self._add_measurement(
            measure_mod.measure(
                self._structure.positions, self._structure.symbols, self._selection
            )
        )

    def _add_measurement(self, result) -> None:
        if result is None:
            return
        result = dataclasses.replace(result, thickness=self.measurement_thickness())
        self._measurements.append(result)
        item = QListWidgetItem(result.summary())
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)  # new measurements are shown straight away
        self._list.addItem(item)
        self._emit_annotations()

    def _set_measurement_colour(self) -> None:
        """Recolour the selected measurement(s) — a per-item override of the
        type's default colour. Each item can carry its own colour."""
        rows = sorted(self._list.row(i) for i in self._list.selectedItems())
        if not rows:
            return
        current = self._measurements[rows[0]].color
        seed = QColor(current) if current else QColor("#ff7f0e")
        chosen = QColorDialog.getColor(seed, self, "Measurement colour")
        if not chosen.isValid():
            return
        hex_color = chosen.name()
        for row in rows:
            self._measurements[row] = dataclasses.replace(
                self._measurements[row], color=hex_color
            )
            self._list.item(row).setIcon(_colour_swatch(hex_color))
        self._emit_annotations()

    def measurement_thickness(self) -> float:
        """The thickness shown (Å) — what the next measurement is drawn with."""
        return float(self._thickness_box.value())

    def _show_thickness(self, value: float) -> None:
        """Put ``value`` in the slider and the box without it counting as a change."""
        for widget, shown in ((self._thickness_slider, int(round(value * 100))),
                              (self._thickness_box, value)):
            blocked = widget.blockSignals(True)
            widget.setValue(shown)
            widget.blockSignals(blocked)

    def _show_selected_thickness(self) -> None:
        """Picking a measurement in the list shows its thickness, ready to be changed."""
        rows = sorted(self._list.row(i) for i in self._list.selectedItems())
        if rows:
            self._show_thickness(self._measurements[rows[0]].thickness)

    def _set_measurement_thickness(self, value: float) -> None:
        """Apply the thickness to the selected measurements, or to all without a selection."""
        value = min(_THICKNESS_MAX, max(_THICKNESS_MIN, float(value)))
        self._show_thickness(value)     # the other of the slider/box pair follows
        rows = sorted(self._list.row(i) for i in self._list.selectedItems())
        for row in rows or range(len(self._measurements)):
            self._measurements[row] = dataclasses.replace(
                self._measurements[row], thickness=value
            )
        if self._measurements:
            self._emit_annotations()

    def _remove_selected_measurements(self) -> None:
        for row in sorted((self._list.row(i) for i in self._list.selectedItems()), reverse=True):
            self._list.takeItem(row)
            del self._measurements[row]
        self._emit_annotations()

    def _items(self) -> List[QListWidgetItem]:
        return [self._list.item(row) for row in range(self._list.count())]

    def _emit_annotations(self) -> None:
        self.annotations_changed.emit(self.shown_annotations())

    # ── lattice planes ──────────────────────────────────────────────────
    def set_miller_cell(self, cell) -> None:
        """The cell Miller indices are quoted in — the conventional cell.

        ``None`` for a molecule, which has no lattice planes. Re-sends the
        planes, since what (hkl) means has changed with the cell.
        """
        cell = None if cell is None else np.asarray(cell, dtype=float)
        if cell is not None and (cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-8):
            cell = None
        self._miller_cell = cell
        self._sync_plane_form()
        self.refresh_planes()
        self._emit_planes()

    def miller_cell(self) -> Optional[np.ndarray]:
        return None if self._miller_cell is None else self._miller_cell.copy()

    def lattice_planes(self) -> List[lp.LatticePlane]:
        """Every plane in the list, shown or not."""
        return list(self._planes)

    def shown_lattice_planes(self) -> List[lp.LatticePlane]:
        """The planes currently ticked for display in 3D."""
        return [
            plane
            for plane, item in zip(self._planes, self._plane_items())
            if item.checkState() == Qt.Checked
        ]

    def clear_lattice_planes(self) -> None:
        self._planes = []
        self._plane_list.clear()
        self._next_plane_colour = 0
        self._emit_planes()

    def refresh_planes(self) -> None:
        """Recount the atoms on each plane — after an edit or a new view.

        Nothing is re-sent: the renderer keeps the planes and redraws them on
        every rebuild itself. Only a new cell (:meth:`set_miller_cell`) changes
        what is drawn.
        """
        blocked = self._plane_list.blockSignals(True)  # relabelling is not a tick
        for plane, item in zip(self._planes, self._plane_items()):
            item.setText(self._plane_summary(plane))
            item.setToolTip(self._plane_tooltip(plane))
        self._plane_list.blockSignals(blocked)

    def add_lattice_plane(self, plane: lp.AnyPlane) -> None:
        """Append ``plane`` to the list, shown, in the next free colour if it has none."""
        if plane.color is None:
            plane = dataclasses.replace(
                plane, color=_PLANE_COLORS[self._next_plane_colour % len(_PLANE_COLORS)]
            )
            self._next_plane_colour += 1
        self._planes.append(plane)
        item = QListWidgetItem(self._plane_summary(plane))
        item.setToolTip(self._plane_tooltip(plane))
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)  # shown straight away, like a measurement
        item.setIcon(_colour_swatch(plane.color))
        blocked = self._plane_list.blockSignals(True)
        self._plane_list.addItem(item)
        self._plane_list.blockSignals(blocked)
        self._emit_planes()
        self._sync_buttons()

    def _typed_miller(self):
        """The indices typed, or ``None`` for (0 0 0), which is not a plane."""
        miller = self._miller.indices()
        return None if miller == (0, 0, 0) else miller

    def _add_typed_plane(self) -> None:
        miller = self._typed_miller()
        if miller is None or self._miller_cell is None:
            return
        self.add_lattice_plane(lp.LatticePlane(
            miller, self._plane_offset.value(), family=self._family_check.isChecked(),
            opacity=self.plane_opacity(),
        ))

    def _add_plane_through_atom(self) -> None:
        """(hkl) through the selected atom; the position box shows where that is."""
        miller = self._typed_miller()
        position = self._current_position()
        if miller is None or position is None or self._miller_cell is None:
            return
        offset = round(lp.offset_of(self._miller_cell, miller, position), 6)
        blocked = self._plane_offset.blockSignals(True)
        self._plane_offset.setValue(offset)
        self._plane_offset.blockSignals(blocked)
        self.add_lattice_plane(lp.LatticePlane(
            miller, offset, family=self._family_check.isChecked(),
            opacity=self.plane_opacity(),
        ))

    def _add_fitted_plane(self) -> None:
        """The least-squares plane through the selected atoms, as fitted."""
        if len(self._selection) < 3:
            return
        positions = np.asarray(self._structure.positions, dtype=float)[self._selection]
        self.add_lattice_plane(lp.fit_plane(positions, opacity=self.plane_opacity()))

    def _atoms_on(self, plane: lp.AnyPlane) -> np.ndarray:
        if not len(self._structure):
            return np.empty(0, dtype=int)
        if isinstance(plane, lp.FittedPlane):  # Cartesian: needs no lattice
            return lp.atoms_on(None, plane, self._structure.positions)
        if self._miller_cell is None:
            return np.empty(0, dtype=int)
        return lp.atoms_on(self._miller_cell, plane, self._structure.positions)

    def _plane_summary(self, plane: lp.AnyPlane) -> str:
        """``(1 1 1) at 0.50 d · d 3.256 Å · 4 atoms`` — or ``… family …``, or a fit."""
        count = len(self._atoms_on(plane))
        atoms = f"{count} atom" + ("" if count == 1 else "s")
        if isinstance(plane, lp.FittedPlane):
            return self._fitted_summary(plane, atoms)
        if self._miller_cell is None:
            return f"{lp.label(plane.miller)} (no lattice)"
        name = lp.label(plane.miller, self._miller_cell)
        d = lp.spacing(self._miller_cell, plane.miller)
        if plane.family:
            shift = plane.offset - np.floor(plane.offset)
            where = "family" if shift < 1e-6 or shift > 1 - 1e-6 else f"family +{shift:.2f} d"
        else:
            where = f"at {plane.offset:.2f} d"
        return f"{name} {where} · d {d:.3f} Å · {atoms}"

    def _fitted_summary(self, plane: lp.FittedPlane, atoms: str) -> str:
        """``Fit ≈ (1 2 0) 1.3° · rms 0.012 Å · 8 atoms`` — as long as an (hkl) row.

        The nearest (hkl) is quoted in the same cell as the typed planes', and
        left out for a molecule, which has none. How many atoms were fitted is
        in the row's tooltip (:meth:`_plane_tooltip`).
        """
        name = "Fit"
        if self._miller_cell is not None:
            miller, angle = lp.nearest_miller(self._miller_cell, plane.normal)
            name += f" ≈ {lp.label(miller, self._miller_cell)} {angle:.1f}°"
        return f"{name} · rms {plane.rms:.3f} Å · {atoms}"

    def _plane_tooltip(self, plane: lp.AnyPlane) -> str:
        """What a fitted plane's row leaves out; nothing for an (hkl) plane."""
        if not isinstance(plane, lp.FittedPlane):
            return ""
        text = (f"Least-squares plane through {len(plane.points)} atoms, which lie "
                f"{plane.rms:.3f} Å from it (rms).")
        if self._miller_cell is not None:
            miller, angle = lp.nearest_miller(self._miller_cell, plane.normal)
            text += (f"\nThe nearest lattice plane, with indices up to {lp.NEAREST_MAX_INDEX}, "
                     f"is {lp.label(miller, self._miller_cell)}, {angle:.1f}° away.")
        return text

    def _sync_plane_form(self) -> None:
        """Hint, i index and d(hkl) for the indices typed and the cell they are in."""
        cell = self._miller_cell
        miller = self._typed_miller()
        self._miller.set_cell(cell)
        hexagonal = self._miller.hexagonal
        if cell is None:
            self._plane_hint.setText(
                "A molecule has no lattice planes, but a plane can be fitted to its atoms."
                if len(self._structure)
                else "Open a crystal to draw its lattice planes."
            )
        else:
            a, b, c = (float(np.linalg.norm(v)) for v in cell)
            self._plane_hint.setText(
                f"Conventional cell (h k i l): a {a:.3f}, c {c:.3f} Å" if hexagonal
                else f"Conventional cell: a {a:.3f}, b {b:.3f}, c {c:.3f} Å"
            )
        if cell is None or miller is None:
            self._spacing_label.setText("d —")
        else:
            self._spacing_label.setText(f"d {lp.spacing(cell, miller):.3f} Å")
        for widget in (self._plane_offset, self._family_check, self._miller):
            widget.setEnabled(cell is not None)
        self._sync_buttons()

    def _set_plane_colour(self) -> None:
        rows = sorted(self._plane_list.row(i) for i in self._plane_list.selectedItems())
        if not rows:
            return
        seed = QColor(self._planes[rows[0]].color or _PLANE_COLORS[0])
        chosen = QColorDialog.getColor(seed, self, "Plane colour")
        if not chosen.isValid():
            return
        for row in rows:
            self._planes[row] = dataclasses.replace(self._planes[row], color=chosen.name())
            self._plane_list.item(row).setIcon(_colour_swatch(chosen.name()))
        self._emit_planes()

    def plane_opacity(self) -> float:
        """The opacity shown — what the next plane is drawn with."""
        return float(self._plane_opacity_box.value())

    def _show_opacity(self, value: float) -> None:
        """Put ``value`` in the slider and the box without it counting as a change."""
        for widget, shown in ((self._plane_opacity_slider, int(round(value * 100))),
                              (self._plane_opacity_box, value)):
            blocked = widget.blockSignals(True)
            widget.setValue(shown)
            widget.blockSignals(blocked)

    def _show_selected_opacity(self) -> None:
        """Picking a plane in the list shows its opacity, ready to be changed."""
        rows = sorted(self._plane_list.row(i) for i in self._plane_list.selectedItems())
        if rows:
            self._show_opacity(self._planes[rows[0]].opacity)

    def _set_plane_opacity(self, value: float) -> None:
        """Apply the opacity to the selected planes, or to all of them without a selection."""
        value = min(1.0, max(0.0, float(value)))
        self._show_opacity(value)     # the other of the slider/box pair follows
        rows = sorted(self._plane_list.row(i) for i in self._plane_list.selectedItems())
        for row in rows or range(len(self._planes)):
            self._planes[row] = dataclasses.replace(self._planes[row], opacity=value)
        if self._planes:
            self._emit_planes()

    def _select_plane_atoms(self) -> None:
        rows = sorted(self._plane_list.row(i) for i in self._plane_list.selectedItems())
        on = set()
        for row in rows:
            on.update(int(i) for i in self._atoms_on(self._planes[row]))
        if rows:
            self.select_atoms_requested.emit(sorted(on))

    def _remove_selected_planes(self) -> None:
        rows = sorted((self._plane_list.row(i) for i in self._plane_list.selectedItems()),
                      reverse=True)
        for row in rows:
            self._plane_list.takeItem(row)
            del self._planes[row]
        self._emit_planes()
        self._sync_buttons()

    def _plane_items(self) -> List[QListWidgetItem]:
        return [self._plane_list.item(row) for row in range(self._plane_list.count())]

    def _emit_planes(self) -> None:
        self.lattice_planes_changed.emit(self.shown_lattice_planes(), self.miller_cell())

    # ── editing ─────────────────────────────────────────────────────────
    def _on_editing_toggled(self, enabled: bool) -> None:
        self._editing = bool(enabled)
        self._sync_buttons()
        self.editing_toggled.emit(self._editing)

    def _sync_buttons(self) -> None:
        """Measuring needs a selection; editing needs editing mode as well."""
        count = len(self._selection)
        self._measure_btn.setEnabled(1 <= count <= 4)
        self._remove_btn.setEnabled(bool(self._measurements))
        self._clear_btn.setEnabled(bool(self._measurements))
        self._color_btn.setEnabled(bool(self._list.selectedItems()))

        # A plane needs indices that name one, and a cell for them to be in; the
        # atom it goes through, exactly one atom selected.
        can_place = self._miller_cell is not None and self._typed_miller() is not None
        self._add_plane_btn.setEnabled(can_place)
        self._plane_atom_btn.setEnabled(can_place and count == 1)
        # A fit needs only the atoms — a molecule's too, which has no lattice.
        self._fit_plane_btn.setEnabled(count >= 3)
        picked = bool(self._plane_list.selectedItems())
        self._plane_colour_btn.setEnabled(picked)
        self._plane_select_btn.setEnabled(picked)
        self._plane_remove_btn.setEnabled(picked)
        # Opacity is for planes there are, or will be: none in a molecule until
        # one is fitted.
        has_planes = self._miller_cell is not None or bool(self._planes)
        self._plane_opacity_slider.setEnabled(has_planes)
        self._plane_opacity_box.setEnabled(has_planes)

        self._add_btn.setEnabled(self._editing)
        on_selection = self._editing and count >= 1
        for button in (self._delete_btn, self._duplicate_btn,
                       self._translate_btn, self._set_element_btn):
            button.setEnabled(on_selection)

        # A typed position names one atom, so it needs exactly one selected —
        # with several, there is no single thing the coordinates could mean.
        one_atom = self._editing and count == 1
        for spin in self._coord_boxes:
            spin.setEnabled(one_atom)


class _CompactList(QListWidget):
    """A list that asks for a few rows and takes whatever height is spare."""

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt's name
        return QSize(super().sizeHint().width(), _LIST_HINT_HEIGHT)


def _table_icon() -> QIcon:
    """The periodic-table glyph, in the current theme's text colour."""
    from PySide6.QtWidgets import QApplication

    from crystalline.ui import theme

    palette = theme.active_palette(QApplication.instance())
    return theme.monochrome_icon("table.svg", palette.text)


def _colour_swatch(color: str, size: int = 12) -> QIcon:
    """A small solid-colour square, shown beside a recoloured measurement or a plane.

    The same square for a selected row: left to Qt, a selected item's icon is
    tinted with the highlight colour, and the swatch vanished into it.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color))
    icon = QIcon(pixmap)
    icon.addPixmap(pixmap, QIcon.Selected)
    return icon


__all__ = ["GeometryPanel"]
