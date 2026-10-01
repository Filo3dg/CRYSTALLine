"""The three boxes a lattice plane is named with: h, k and l.

Shared by everything that asks for a plane — the density slice and the
Geometry panel's lattice planes — so a plane is typed the same way wherever it
is asked for. For hexagonal axes the redundant fourth Miller–Bravais index
``i = −(h + k)`` is shown between k and l: it is fixed by the other two, so it
is displayed rather than typed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSpinBox, QWidget

from crystalline.core import lattice_planes as lp

# Indices beyond ±12 are planes a few hundredths of an Ångström apart, which
# nothing on screen can show.
MILLER_RANGE = 12
# How narrow a box may get. Their *preferred* width is ignored, so a row of
# them shares whatever width it is given instead of widening a dock.
_BOX_MIN_WIDTH = 44


class MillerIndices(QWidget):
    """``h k l`` boxes (and ``i`` for hexagonal axes); ``changed`` on any edit."""

    changed = Signal()

    def __init__(self, values: Sequence[int] = (0, 0, 1),
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.boxes: List[QSpinBox] = []
        for name, value in zip("hkl", values):
            if name == "l":
                self.i_label = QLabel()
                self.i_label.setToolTip("i = −(h + k): hexagonal axes are indexed (h k i l)")
                self.i_label.hide()
                row.addWidget(self.i_label)
            box = QSpinBox(self)
            box.setRange(-MILLER_RANGE, MILLER_RANGE)
            box.setValue(int(value))
            box.setPrefix(f"{name} ")
            box.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            box.setMinimumWidth(_BOX_MIN_WIDTH)
            box.valueChanged.connect(self._edited)
            row.addWidget(box, 1)
            self.boxes.append(box)
        self.hexagonal = False   # whether the cell given is hexagonal (then i is shown)
        self._show_i()

    def indices(self) -> Tuple[int, int, int]:
        return tuple(box.value() for box in self.boxes)

    def set_cell(self, cell) -> None:
        """The cell the indices are in: hexagonal axes also show ``i``."""
        self.hexagonal = cell is not None and lp.is_hexagonal(cell)
        self._show_i()

    def _edited(self, _value: int = 0) -> None:
        self._show_i()
        self.changed.emit()

    def _show_i(self) -> None:
        self.i_label.setVisible(self.hexagonal)
        if self.hexagonal:
            h, k, _l = self.indices()
            self.i_label.setText(f"i {-(h + k)}")


__all__ = ["MILLER_RANGE", "MillerIndices"]
