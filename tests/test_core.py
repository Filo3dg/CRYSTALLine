"""Unit tests for the Qt-free domain core (no display needed)."""

import numpy as np
import pytest

from crystalline.core.structure import Structure
from crystalline.core.phonons import (
    PhononMode,
    PhononModes,
    commensurate_repeats,
    displaced_positions,
    frame_displacement,
    phase_factors,
    qpoint_label,
)


def test_add_atoms_batch_appends_and_fires_once():
    s = Structure.empty()
    s.add_atom("C", [0.0, 0.0, 0.0])
    events = []
    s.add_listener(lambda st: events.append(len(st)))

    new = s.add_atoms(["O", "H", "H"], [[1, 0, 0], [2, 0, 0], [0, 1, 0]])
    assert new == [1, 2, 3]
    assert len(s) == 4
    assert s.symbols == ["C", "O", "H", "H"]
    assert events == [4]  # a single notification for the whole batch

    assert s.add_atoms([], []) == []  # empty is a no-op
    with pytest.raises(ValueError):
        s.add_atoms(["C"], [[0, 0, 0], [1, 1, 1]])  # length mismatch


def test_add_move_remove_atom_notifies():
    s = Structure.empty()
    events = []
    s.add_listener(lambda st: events.append(len(st)))

    i = s.add_atom("C", [0.0, 0.0, 0.0])
    j = s.add_atom("O", [1.2, 0.0, 0.0])
    assert (i, j) == (0, 1)
    assert len(s) == 2
    assert s.symbols == ["C", "O"]

    s.move_atom(1, [1.5, 0.0, 0.0])
    assert np.allclose(s.positions[1], [1.5, 0.0, 0.0])

    s.set_symbols([0], "N")
    assert s.symbols[0] == "N"

    s.remove_atoms([0])
    assert len(s) == 1 and s.symbols == ["O"]

    # each mutating call fired exactly one notification
    assert events == [1, 2, 2, 2, 1]


def test_invalid_symbol_and_index():
    s = Structure.empty()
    with pytest.raises(ValueError):
        s.add_atom("Xx", [0, 0, 0])
    s.add_atom("H", [0, 0, 0])
    with pytest.raises(IndexError):
        s.move_atom(5, [0, 0, 0])


def test_batch_edits_translate_duplicate_set_remove():
    s = Structure.empty()
    for i, el in enumerate(["C", "O", "N", "H"]):
        s.add_atom(el, [i, 0, 0])
    events = []
    s.add_listener(lambda st: events.append(len(st)))

    s.translate_atoms([0, 2], [0.0, 1.0, 0.0])
    assert np.allclose(s.positions[0], [0, 1, 0])
    assert np.allclose(s.positions[2], [2, 1, 0])
    assert np.allclose(s.positions[1], [1, 0, 0])  # untouched

    new = s.duplicate_atoms([1, 3], offset=[0.0, 0.0, 5.0])
    assert new == [4, 5]
    assert s.symbols[4] == "O" and s.symbols[5] == "H"
    assert np.allclose(s.positions[4], [1, 0, 5])

    s.set_symbols([0, 4], "S")
    assert s.symbols[0] == "S" and s.symbols[4] == "S"

    s.remove_atoms([5, 0])  # order-independent, high-to-low internally
    assert len(s) == 4
    assert s.symbols == ["O", "N", "H", "S"]

    # one notification per batch action (4 actions)
    assert events == [4, 6, 6, 4]


def test_batch_edits_validate_and_ignore_empty():
    s = Structure.empty()
    s.add_atom("C", [0, 0, 0])
    calls = []
    s.add_listener(lambda st: calls.append(1))

    with pytest.raises(IndexError):
        s.remove_atoms([0, 9])  # 9 out of range -> nothing removed
    assert len(s) == 1
    with pytest.raises(ValueError):
        s.set_symbols([0], "Zz")

    # empty selection is a no-op that does not notify
    s.translate_atoms([], [1, 2, 3])
    assert s.duplicate_atoms([]) == []
    assert calls == []


def test_phonon_mode_shape_validation():
    with pytest.raises(ValueError):
        PhononMode(frequency=100.0, eigenvector=np.zeros((4,)))  # not (N, 3)
    m = PhononMode(frequency=-2.0, eigenvector=np.zeros((3, 3)))
    assert m.is_imaginary and m.n_atoms == 3


def test_set_lattice_parameters_scales_atoms_and_notifies():
    s = Structure.empty()
    s.set_cell(np.diag([4.0, 4.0, 4.0]), periodic=True)
    s.add_atom("Na", [0.0, 0.0, 0.0])
    s.add_atom("Cl", [2.0, 2.0, 2.0])  # fractional (0.5, 0.5, 0.5)
    frac_before = np.linalg.solve(s.cell.T, s.positions.T).T

    events = []
    s.add_listener(lambda st: events.append(1))
    s.set_lattice_parameters(6.0, 6.0, 6.0, 90.0, 90.0, 90.0)

    assert np.allclose(s.cellpar, [6.0, 6.0, 6.0, 90.0, 90.0, 90.0])
    # atoms moved with the cell: fractional coordinates preserved
    frac_after = np.linalg.solve(s.cell.T, s.positions.T).T
    assert np.allclose(frac_before, frac_after)
    assert np.allclose(s.positions[1], [3.0, 3.0, 3.0])  # 0.5 * 6.0
    assert events == [1]  # exactly one notification

    # a non-orthogonal angle is applied faithfully
    s.set_lattice_parameters(6.0, 6.0, 6.0, 90.0, 90.0, 120.0)
    assert np.isclose(s.cellpar[5], 120.0)


def test_displaced_positions_at_key_phases():
    eq = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    evec = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    mode = PhononMode(frequency=50.0, eigenvector=evec)

    # phase 0 -> cos(0)=1 -> full displacement
    peak = displaced_positions(eq, mode, amplitude=0.5, phase=0.0)
    assert np.allclose(peak, eq + 0.5 * evec)
    # a quarter cycle on -> cos=0 -> passing through the equilibrium
    assert np.allclose(displaced_positions(eq, mode, amplitude=1.0, phase=np.pi / 2), eq)


def test_amplitude_is_the_peak_atomic_displacement_whatever_the_cell_size():
    """Eigenvectors come back normalised over all 3N components, so the motion
    of any one atom faded as 1/sqrt(N) and the default amplitude that suited a
    molecule was invisible for a large cell. Amplitude now means the peak
    displacement of the most-displaced atom, in Angstrom."""
    for natom in (2, 50, 500):
        eq = np.zeros((natom, 3))
        evec = np.zeros((natom, 3))
        evec[:, 0] = 1.0
        evec /= np.linalg.norm(evec)  # as CRYSTALClear normalises it

        peak = displaced_positions(eq, PhononMode(100.0, evec), amplitude=0.4, phase=0.0)
        assert np.isclose(np.max(np.linalg.norm(peak - eq, axis=1)), 0.4)


def test_amplitude_keeps_the_relative_motion_within_a_mode():
    # One atom moving twice as far as another must still do so after scaling.
    eq = np.zeros((3, 3))
    evec = np.array([[2.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0]])
    peak = displaced_positions(eq, PhononMode(100.0, evec), amplitude=0.5, phase=0.0)
    assert np.allclose(peak[:, 0], [0.5, 0.25, 0.0])


def test_a_null_eigenvector_does_not_move_or_blow_up():
    eq = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mode = PhononMode(frequency=0.0, eigenvector=np.zeros((2, 3)))
    assert np.allclose(displaced_positions(eq, mode, amplitude=1.0, phase=0.0), eq)


def test_activity_labels_ride_along_with_the_mode():
    mode = PhononMode(
        frequency=1050.0,
        eigenvector=np.eye(3)[:2],
        ir_active=True,
        raman_active=False,
        ir_intensity=12.5,
    )
    assert mode.has_activity

    # Remapping onto a supercell keeps frequency and labels, swaps the vectors.
    tiled = mode.with_eigenvector(np.zeros((4, 3)))
    assert tiled.n_atoms == 4
    assert tiled.frequency == 1050.0
    assert tiled.ir_active is True and tiled.raman_active is False
    assert tiled.ir_intensity == 12.5

    # An output without the analysis leaves the labels unknown, not False.
    plain = PhononMode(frequency=1050.0, eigenvector=np.eye(3)[:2])
    assert plain.ir_active is None and not plain.has_activity
    assert not PhononModes([plain]).has_activity
    assert PhononModes([plain, mode]).has_activity


def test_phonon_modes_collection():
    modes = PhononModes(
        [PhononMode(10.0, np.zeros((2, 3))), PhononMode(-5.0, np.zeros((2, 3)))]
    )
    assert len(modes) == 2
    assert np.allclose(modes.frequencies, [10.0, -5.0])
    assert modes[1].is_imaginary


# ── modes away from Gamma (a SCELPHONO run's other q-points) ──────────────
# Such a mode is a travelling wave: its eigenvector is complex, each cell lags
# the last by q·n, and the animation has to evaluate the real displacement at
# every frame instead of scaling one fixed pattern.
def test_a_mode_defaults_to_gamma_and_keeps_a_real_eigenvector():
    mode = PhononMode(frequency=100.0, eigenvector=np.eye(3)[:2])
    assert mode.is_gamma and mode.qpoint is None
    assert mode.qpoint_label == "Γ"
    assert not np.iscomplexobj(mode.eigenvector)
    # An explicit zero q is Gamma too — the loader labels the first set that way.
    assert PhononMode(100.0, np.eye(3)[:2], qpoint=[0, 0, 0]).is_gamma


def test_a_complex_eigenvector_survives_and_carries_its_qpoint():
    evec = np.array([[1.0 + 0.0j, 0, 0], [0, 0.5j, 0]])
    mode = PhononMode(frequency=100.0, eigenvector=evec, qpoint=[0.0, 0.0, 0.5])

    assert np.iscomplexobj(mode.eigenvector)
    assert not mode.is_gamma
    assert mode.qpoint_label == "(0, 0, 1/2)"
    with pytest.raises(ValueError):
        PhononMode(frequency=100.0, eigenvector=evec, qpoint=[0.0, 0.5])


def test_a_complex_mode_moves_as_a_wave_over_the_cycle():
    """Re(e) and Im(e) are CRYSTAL's in-phase and anti-phase blocks; the atom's
    displacement runs through both over a cycle, which is what makes one cell of
    a tiled wave lag another rather than merely scaling it."""
    eq = np.zeros((2, 3))
    evec = np.array([[1.0, 0, 0], [1.0j, 0, 0]])  # two atoms a quarter cycle apart
    mode = PhononMode(frequency=100.0, eigenvector=evec, qpoint=[0.0, 0.0, 0.5])

    at_zero = displaced_positions(eq, mode, amplitude=1.0, phase=0.0)
    quarter = displaced_positions(eq, mode, amplitude=1.0, phase=np.pi / 2)

    # phase 0: Re[e] — the real atom at full stretch, the imaginary one at rest
    assert np.allclose(at_zero[0], [1.0, 0, 0]) and np.allclose(at_zero[1], [0.0, 0, 0])
    # a quarter cycle later they have swapped roles
    assert np.allclose(quarter[0], [0.0, 0, 0]) and np.allclose(quarter[1], [1.0, 0, 0])
    # a real (Gamma) eigenvector moves as e*cos(phase)
    real = PhononMode(frequency=100.0, eigenvector=np.real(evec))
    assert np.allclose(frame_displacement(real.eigenvector, 0.3), np.real(evec) * np.cos(0.3))


def test_the_wave_travels_along_plus_q():
    """The pattern of a travelling wave moves the way the mode's q says it does.

    The displacement is ``Re[e exp(i(2 pi q·n - w t))]``, so as time runs the
    pattern advances along +q, by one wavelength per cycle. Evaluating the
    *imaginary* part instead flips the sign of the time term and runs the wave
    backwards — which is what this code did, invisibly, because at Gamma the two
    agree exactly and nothing away from Gamma ever looked at the direction.

    Checked as a shift of the whole field rather than by chasing one crest: a
    crest that leaves one end of the sampled range and re-enters at the other
    says nothing about which way it went.
    """
    q = 0.25                                  # wavelength: four cells
    per_cell = 100
    n = np.arange(0, 8 * per_cell + 1) / per_cell
    evec = np.zeros((n.size, 3), complex)
    evec[:, 2] = np.exp(2j * np.pi * q * n)   # a real mode, Bloch phase folded in

    now = frame_displacement(evec, 0.2)[:, 2]
    later = frame_displacement(evec, 0.2 + np.pi / 2)[:, 2]  # a quarter cycle on

    # A quarter cycle later the field is the same shape, moved one cell along +n.
    assert np.allclose(later[per_cell:], now[:-per_cell], atol=1e-9)
    # ...and emphatically not one cell back the other way.
    assert not np.allclose(later[:-per_cell], now[per_cell:], atol=1e-3)
    # A whole cycle is the same picture again — it has moved one wavelength on.
    assert np.allclose(frame_displacement(evec, 0.2 + 2 * np.pi)[:, 2], now)


def test_a_still_picture_of_a_mode_shows_the_in_phase_block():
    """The arrow field is drawn at phase 0 (``viz.phonon_animator``).
    There it must come out as Re(e) — what CRYSTAL prints as MODES IN PHASE —
    and not as the anti-phase block, which is a quarter cycle away from what the
    frequency table's own companion picture shows."""
    evec = np.array([[0.3 + 0.4j, 0.0, 0.0], [-0.1 + 0.2j, 0.0, 0.0]])

    assert np.allclose(frame_displacement(evec, 0.0), np.real(evec))


def test_amplitude_still_means_the_peak_atomic_displacement_when_complex():
    eq = np.zeros((2, 3))
    evec = np.array([[0.3 + 0.4j, 0, 0], [0.1, 0, 0]])  # |e| = 0.5 for atom 0
    mode = PhononMode(frequency=100.0, eigenvector=evec, qpoint=[0.5, 0, 0])

    frames = [
        displaced_positions(eq, mode, amplitude=0.2, phase=p)
        for p in np.linspace(0, 2 * np.pi, 64)
    ]
    excursion = max(np.max(np.linalg.norm(f - eq, axis=1)) for f in frames)
    assert np.isclose(excursion, 0.2, atol=1e-3)


def test_phase_factors_lag_each_cell_by_q_dot_n():
    offsets = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]], float)
    factors = phase_factors([0.0, 0.0, 0.25], offsets)

    assert np.allclose(factors, [1, 1j, -1, -1j])
    # Gamma (and a missing q) means "no phase at all", so callers can skip it
    assert phase_factors(None, offsets) is None
    assert phase_factors([0.0, 0.0, 0.0], offsets) is None


def test_qpoint_labels_read_as_the_fractions_crystal_sampled():
    assert qpoint_label(None) == "Γ"
    assert qpoint_label([0.0, 0.0, 0.0]) == "Γ"
    assert qpoint_label([0.5, 0.0, 1 / 3]) == "(1/2, 0, 1/3)"


def test_commensurate_repeats_is_the_smallest_whole_period():
    assert commensurate_repeats(None) == (1, 1, 1)
    assert commensurate_repeats([0.0, 0.0, 0.25]) == (1, 1, 4)
    assert commensurate_repeats([0.5, 1 / 3, 0.0]) == (2, 3, 1)
    # A fine q grid must not ask for a structure too big to draw.
    assert commensurate_repeats([1 / 16, 0.0, 0.0], limit=8) == (8, 1, 1)


def test_phonon_modes_report_the_qpoint_they_share():
    gamma = PhononModes([PhononMode(10.0, np.zeros((2, 3)))])
    assert gamma.is_gamma and gamma.qpoint_label == "Γ"

    away = PhononModes(
        [PhononMode(10.0, np.zeros((2, 3), complex), qpoint=[0.0, 0.5, 0.0])]
    )
    assert not away.is_gamma and away.qpoint_label == "(0, 1/2, 0)"
    assert PhononModes([]).is_gamma  # no modes: nothing to be away from Gamma


# ── the middle of a structure that is not a crystal ─────────────────────

def _crystal_slab_polymer():
    """One of each periodicity, with CRYSTAL's formal 500 Å across the vacuum."""
    from ase import Atoms
    from ase.build import bulk, fcc111

    crystal = Structure.from_ase(bulk("MgO", "rocksalt", a=4.21))

    slab = fcc111("Pt", size=(2, 2, 3), vacuum=0.0)
    cell = np.asarray(slab.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, 500.0]
    slab.set_cell(cell)
    slab.pbc = [True, True, False]

    chain = Atoms(
        "OMg", positions=[[1.7835, 0.0, 0.0], [0.0, 0.0, 0.0]],
        cell=[[3.567, 0, 0], [0, 500.0, 0], [0, 0, 500.0]],
        pbc=[True, False, False],
    )
    return crystal, Structure.from_ase(slab), Structure.from_ase(chain)


def test_the_middle_of_a_slab_or_a_polymer_is_not_the_middle_of_its_cell():
    """Where a new atom is put. Half of CRYSTAL's 500 Å is 250 Å of vacuum: an
    atom added there is added, and selected, where nothing can be seen or
    picked, which reads as not being added at all."""
    _crystal, slab, polymer = _crystal_slab_polymer()

    for structure in (slab, polymer):
        centre = structure.centre()
        atoms = np.asarray(structure.positions)
        aperiodic = ~np.asarray(structure.pbc, dtype=bool)
        # No component of it is out in the vacuum...
        assert np.all(np.abs(centre[aperiodic]) < 20.0)
        # ...and it lands among the atoms rather than a quarter of a micron off.
        assert np.linalg.norm(centre - atoms.mean(axis=0)) < 5.0

    # Along the directions that do repeat it is still the middle of the cell.
    assert slab.centre()[:2] == pytest.approx(0.5 * np.asarray(slab.cell)[:2].sum(axis=0)[:2])
    assert polymer.centre()[0] == pytest.approx(0.5 * np.asarray(polymer.cell)[0][0])


def test_the_middle_of_a_crystal_is_the_centre_of_its_cell():
    """The 3D case is what it always was, and must not move."""
    crystal, _slab, _polymer = _crystal_slab_polymer()

    assert crystal.centre() == pytest.approx(0.5 * np.asarray(crystal.cell).sum(axis=0))


def test_the_middle_of_a_molecule_is_its_centroid():
    """No direction repeats, so there is no cell to take the middle of — and an
    imported molecule sitting far from the origin must not send the atom back
    to it."""
    s = Structure.empty()
    s.add_atom("C", [10.0, 10.0, 10.0])
    s.add_atom("O", [12.0, 10.0, 10.0])

    assert s.centre() == pytest.approx([11.0, 10.0, 10.0])
    assert Structure.empty().centre() == pytest.approx([0.0, 0.0, 0.0])
