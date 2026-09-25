# Editing structures

Editing is enabled by **Edit → Editing mode** ({kbd}`Ctrl+E`). All operations
can be undone ({kbd}`Ctrl+Z`; redo with {kbd}`Ctrl+Shift+Z` or {kbd}`Ctrl+Y`),
and **Edit → Restore geometry** ({kbd}`Ctrl+R`) returns the structure to the
state in which it was read.

## Selection

An atom is selected by clicking it and added to the selection with
{kbd}`Ctrl` held down; clicking the background clears the selection.
**Edit → Select all** ({kbd}`Ctrl+A`), **Clear selection** and **Invert
selection** act on the whole structure.

## Moving atoms

A selected atom is moved by dragging it; its periodic images move with it, and
a selection of several atoms moves as a unit. The arrow keys displace the
selection by a small step, and **Edit → Translate selected…** applies a given
displacement.

## Adding and removing atoms

- **Delete selected** ({kbd}`Del`)
- **Duplicate selected** ({kbd}`Ctrl+D`)
- **Set element of selected…**, which opens a periodic table

## The cell

The **Cell** menu acts on the lattice rather than on the atoms:

- **Crystallographic cell** and **Primitive cell** select which of the two is
  displayed.
- **Lattice parameters…** edits *a*, *b*, *c*, α, β and γ.
- **Supercell…** expands the cell.
- **Complete molecules at cell boundary** completes the molecules cut by the
  edges of the cell. This affects the drawing; the deck written for CRYSTAL
  contains the cell itself.
- **Point symmetry analysis** and **Brillouin zone…** are described in
  [Symmetry](symmetry.md) and [Band paths](band-paths.md).

The Info panel is recomputed after each operation, so the space group shown is
that of the current structure.

## Measurements

The **Geometry** panel measures the current selection:

| Atoms selected | Quantity |
| --- | --- |
| 1 | the position of the atom |
| 2 | a distance |
| 3 | an angle |
| 4 | a dihedral angle |
| 3 or more | a least-squares plane |

Measurements remain drawn in the view and can be coloured individually or by
type.

## Lattice planes

A crystallographic plane is drawn from its Miller indices, with no atoms
selected, in the **Lattice planes (hkl)** part of the Geometry panel. The
indices are quoted in the conventional cell — the one named above the boxes,
the same as for a density slice — whatever cell, primitive view or supercell is
on screen; hexagonal and trigonal crystals are written with four indices
(*h k i l*), *i* = −(*h* + *k*) following from the other two.

The plane is placed either at a **position** along its normal, in units of the
interplanar spacing *d*(hkl) from the plane through the origin — 0 and 1 are
neighbouring planes, 0.5 lies halfway between — or **through the selected
atom**. **Whole family** draws every plane of the family across the cell on
screen, *d*(hkl) apart. Each plane is drawn where it cuts the displayed cell
(for a slab, the layer and 1 Å either side of it) and is listed with *d*(hkl)
and the number of atoms lying on it, within 0.1 Å; **Select atoms** selects
them, ready to be measured. The **Opacity** slider sets how see-through the
planes selected in the list are — every plane when none is selected — and the
opacity new planes are drawn with; at 0 only the outline of a plane is drawn.
Planes are kept through a change of view or supercell and cleared when another
file is opened.
