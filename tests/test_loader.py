"""Loading structures from the file formats the GUI accepts."""

import numpy as np
import pytest

pytest.importorskip("pymatgen")  # the CIF reader (CifParser) needs pymatgen

from crystalline.crystalio.loader import load, load_structure  # noqa: E402

# A minimal, self-contained CIF: rock-salt NaCl in its cubic conventional cell.
_NACL_CIF = """
data_NaCl
_cell_length_a     5.64
_cell_length_b     5.64
_cell_length_c     5.64
_cell_angle_alpha  90
_cell_angle_beta   90
_cell_angle_gamma  90
_symmetry_space_group_name_H-M   'F m -3 m'
_symmetry_Int_Tables_number      225
loop_
_symmetry_equiv_pos_as_xyz
  'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na Na 0.0 0.0 0.0
Cl Cl 0.5 0.5 0.5
"""


def _write_cif(tmp_path) -> str:
    path = tmp_path / "nacl.cif"
    path.write_text(_NACL_CIF)
    return str(path)


def test_load_structure_reads_a_cif(tmp_path):
    structure = load_structure(_write_cif(tmp_path))
    assert structure.is_periodic
    assert set(structure.symbols) == {"Na", "Cl"}
    # the CIF's cubic cell survives the round-trip through pymatgen/ase
    a, b, c, alpha, beta, gamma = structure.cellpar
    assert np.allclose([a, b, c], 5.64, atol=1e-3)
    assert np.allclose([alpha, beta, gamma], 90.0, atol=1e-3)


def test_load_of_a_cif_carries_no_phonon_modes(tmp_path):
    result = load(_write_cif(tmp_path))
    assert result.structure.is_periodic
    assert result.modes is None  # a CIF is geometry only — never scanned as a CRYSTAL out
    assert not result.has_phonons


class _StubGeometry:
    """Stands in for CRYSTALClear's ``Crystal_output`` geometry reader.

    ``get_geometry`` returns a pymatgen ``Structure`` — always 3D-periodic,
    exactly as CRYSTALClear's does — alongside the dimensionality CRYSTAL
    printed for the run.
    """

    def __init__(self, dimensionality, raises=False):
        self._dimensionality = dimensionality
        self._raises = raises

    def get_dimensionality(self):
        if self._raises:
            raise Exception("Invalid file. Dimension information not found.")
        return self._dimensionality

    def get_geometry(self, initial=False):
        from pymatgen.core.structure import Structure as PmgStructure

        # A slab as CRYSTAL writes one: a real a/b plane and a formal 500 Å c.
        return PmgStructure(
            [[5.4, 0.0, 0.0], [0.0, 5.6, 0.0], [0.0, 0.0, 500.0]],
            ["Ca", "O"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.002]],
        )


@pytest.mark.parametrize(
    "dimensionality, expected",
    [(3, [True, True, True]), (2, [True, True, False]), (1, [True, False, False])],
)
def test_out_geometry_carries_crystals_dimensionality(dimensionality, expected):
    """A 2D slab must not come back periodic along c.

    CRYSTAL fills an aperiodic direction with a formal 500 Å vacuum vector, and
    pymatgen's Structure flags every direction periodic regardless. Anything
    trusting that then treats the vacuum as a lattice direction — the cell is
    drawn 500 Å tall, and CrystalNN's Voronoi tessellation hangs on it.
    """
    from crystalline.crystalio.loader import _out_geometry_to_ase

    atoms = _out_geometry_to_ase(_StubGeometry(dimensionality), initial=False)

    assert list(atoms.get_pbc()) == expected


def test_out_geometry_stays_periodic_when_dimensionality_is_unreadable():
    from crystalline.crystalio.loader import _out_geometry_to_ase

    atoms = _out_geometry_to_ase(_StubGeometry(None, raises=True), initial=False)

    assert list(atoms.get_pbc()) == [True, True, True]


def test_slab_connectivity_falls_back_instead_of_hanging():
    """The guard that keeps CrystalNN off a vacuum axis keys on ``pbc``, so it
    only works if the loader set it — this pins the two together."""
    from crystalline.core.bonds import connectivity
    from crystalline.core.structure import Structure
    from crystalline.crystalio.loader import _out_geometry_to_ase

    slab = Structure.from_ase(_out_geometry_to_ase(_StubGeometry(2), initial=False))

    assert connectivity(slab) is None  # -> caller uses the bounded distance search


def test_per_mode_normalises_crystalclear_activity_arrays():
    """CRYSTALClear leaves IR/Raman/intens empty when the output has no
    selection-rule analysis; the modes must then read "unknown" rather than
    "inactive", which would let the panel's filter hide every mode."""
    from crystalline.crystalio.loader import _per_mode

    assert _per_mode([True, False, True], 3) == [True, False, True]
    assert _per_mode([], 3) == [None, None, None]  # no analysis in the output
    assert _per_mode(None, 2) == [None, None]
    assert _per_mode([True, False], 3) == [None, None, None]  # misaligned -> unlabelled
    assert _per_mode([1.5, 0.0], 2, cast=float) == [1.5, 0.0]
    assert _per_mode([np.nan, 2.0], 2, cast=float) == [None, 2.0]  # NaN'd imaginary mode


class _StubOutput:
    """Stands in for CRYSTALClear's Crystal_output — the row builder only ever
    calls ``get_calculation_info``."""

    def __init__(self, info):
        self._info = info

    def get_calculation_info(self):
        return self._info


_PBE_D3 = {
    "code": "CRYSTAL17",
    "run_type": ["CPHF", "FREQCALC"],
    "terminated": True,
    "method": "DFT",
    "exchange": "PERDEW-BURKE-ERNZERHOF",
    "correlation": "PERDEW-BURKE-ERNZERHOF",
    "hybrid_exchange": None,
    "dispersion": "DFT-D3(BJ)",
    "shell_type": "RESTRICTED CLOSED SHELL",
    "shrink": [8, 8, 8],
    "n_kpoints_ibz": 125,
    "n_ao": 444,
    "n_shells": 236,
    "n_electrons": 160,
    "n_core_electrons": 64,
    "n_symmops": 4,
    "toldee": 9,
    "tolinteg": [8, 8, 20, 20, 20],
    "grid_points": 399573,
}


def test_setup_rows_describe_the_calculation():
    from crystalline.crystalio.loader import _setup_rows

    rows = _setup_rows(_StubOutput(_PBE_D3))

    assert rows["Code"] == "CRYSTAL17"
    assert rows["Task"] == "CPHF + FREQCALC"
    assert rows["Method"] == "DFT"
    # CRYSTAL's full spelling is shortened to the name people use
    assert rows["Exchange"] == "PBE"
    assert rows["Correlation"] == "PBE"
    # the "DFT-" prefix is dropped: the row is already labelled Dispersion
    assert rows["Dispersion"] == "D3(BJ)"
    assert rows["Shell type"] == "Restricted closed shell"
    # mesh and count are separate rows — together they overflowed the dock
    mesh = "\u00a0\u00d7\u00a0".join("888")  # non-breaking spaces: the mesh never wraps mid-value
    assert rows["k-point mesh"] == mesh
    assert rows["k points (IBZ)"] == "125"
    assert rows["Basis functions"] == "444"
    assert rows["Electrons/cell"] == "160 (64 core)"
    assert rows["SCF tolerance"] == "10⁻⁹ Ha"
    assert rows["TOLINTEG"] == "8 8 20 20 20"
    assert rows["DFT grid"] == "399,573"
    assert "Termination" not in rows  # only flagged when the run ended badly


def test_setup_rows_report_a_hybrid_and_an_unlisted_functional():
    from crystalline.crystalio.loader import _setup_rows

    info = dict(_PBE_D3, hybrid_exchange=25.0, exchange="SOME-NEW-FUNCTIONAL")
    rows = _setup_rows(_StubOutput(info))

    assert rows["Method"] == "DFT (hybrid, 25% exact exchange)"
    # an acronym is only substituted when it's known; otherwise CRYSTAL's wording
    assert rows["Exchange"] == "SOME-NEW-FUNCTIONAL"


def test_setup_rows_omit_what_the_run_did_not_report():
    """A Hartree-Fock run has no functional, and a molecule no k points; those
    rows must be absent rather than blank or wrong."""
    from crystalline.crystalio.loader import _setup_rows

    rows = _setup_rows(
        _StubOutput(
            {
                "code": "CRYSTAL23",
                "method": "Hartree-Fock",
                "run_type": [],
                "terminated": True,
                "exchange": None,
                "correlation": None,
                "shrink": None,
                "grid_points": None,
            }
        )
    )

    assert rows["Method"] == "Hartree-Fock"
    for absent in ("Exchange", "Correlation", "k-point mesh", "k points (IBZ)",
                   "DFT grid", "Task"):
        assert absent not in rows


def test_setup_rows_flag_a_run_that_did_not_finish():
    from crystalline.crystalio.loader import _setup_rows

    rows = _setup_rows(_StubOutput(dict(_PBE_D3, terminated=False)))
    assert rows["Termination"] == "did not terminate normally"


def test_setup_rows_survive_an_older_crystalclear():
    """``get_calculation_info`` landed in CRYSTALClear 0.2.16; against an older
    one the panel must fall back to the computed-property rows, not error."""
    from crystalline.crystalio.loader import _setup_rows

    class _Old:
        pass

    assert _setup_rows(_Old()) == {}


def test_dispersion_drops_the_dft_prefix():
    from crystalline.crystalio.loader import _dispersion

    assert _dispersion("DFT-D3(BJ)") == "D3(BJ)"
    assert _dispersion("DFT-D3") == "D3"
    assert _dispersion("DFT-D2") == "D2"
    # older outputs name only the author, so there is no version to report
    assert _dispersion("GRIMME DISPERSION") == "yes"
    assert _dispersion(None) is None


# ── atomic displacement parameters ────────────────────────────────────────
# A minimal CRYSTAL ADP block: one temperature, one atom, a diagonal tensor of
# 0.01/0.02/0.04 a.u.² so the bohr²→Å² conversion is checkable by hand. The
# integer columns are CRYSTAL's own 10⁻⁴ Å² values, and the principal-values
# line is in Å² — as the printed layout really is.
_ADP_OUTPUT = """ <ADPS><ADPS><ADPS>

                       ATOMIC DISPLACEMENT PARAMETERS (ADP)

              COMPUTED VIA AN UNCORRELATED HARMONIC METROPOLIS MODEL

 <ADPS><ADPS><ADPS>


                               TEMPERATURE =  10.0000 K
                    NUMBER OF ACTIVE MODES =     3


 ATOM:     1

               ADP TENSOR (a.u.^2)                          (10^-4 ang^2)

    1.00000E-02    0.00000E+00    0.00000E+00             28      0      0
    0.00000E+00    2.00000E-02    0.00000E+00              0     56      0
    0.00000E+00    0.00000E+00    4.00000E-02              0      0    112

         PRINCIPAL AXES OF THE ELLIPSOID

    2.80028E-03    5.60056E-03    1.12011E-02             28     56    112

                 ROTATION TENSOR

    1.00000E+00    0.00000E+00    0.00000E+00
    0.00000E+00    1.00000E+00    0.00000E+00
    0.00000E+00    0.00000E+00    1.00000E+00

 *******************************************************************************
"""

_BOHR_SQUARED = 0.529177210903 ** 2


def _write_adp_out(tmp_path, text=_ADP_OUTPUT) -> str:
    path = tmp_path / "adp.out"
    path.write_text(text)
    return str(path)


def test_load_adp_returns_tensors_in_angstrom_squared(tmp_path):
    pytest.importorskip("CRYSTALClear")
    from crystalline.crystalio.loader import load_adp

    adps = load_adp(_write_adp_out(tmp_path))

    assert adps is not None
    assert list(adps.temperatures) == [10.0]
    assert adps.n_atoms == 1
    # CRYSTAL prints the tensor in bohr²; a renderer and a CIF both want Å².
    assert np.allclose(
        np.diag(adps.tensors[0, 0]), np.array([0.01, 0.02, 0.04]) * _BOHR_SQUARED
    )


def test_load_adp_agrees_with_crystals_own_principal_values(tmp_path):
    """The printed principal values sit under an ``(a.u.^2)`` header but are
    already in Å² — the converted tensor's eigenvalues must reproduce them."""
    pytest.importorskip("CRYSTALClear")
    from crystalline.crystalio.loader import load_adp

    adps = load_adp(_write_adp_out(tmp_path))

    printed = [2.80028e-03, 5.60056e-03, 1.12011e-02]
    assert np.allclose(np.sort(np.linalg.eigvalsh(adps.tensors[0, 0])), printed, rtol=1e-5)


def test_load_adp_is_none_for_a_run_without_them(tmp_path):
    """Most frequency runs have no ADP section; that is not an error."""
    pytest.importorskip("CRYSTALClear")
    from crystalline.crystalio.loader import load_adp

    plain = tmp_path / "plain.out"
    plain.write_text(" SOME OUTPUT\n EEEEEEEEEE TERMINATION\n")

    assert load_adp(str(plain)) is None


def test_load_adp_is_none_for_geometry_only_files(tmp_path):
    from crystalline.crystalio.loader import load_adp

    assert load_adp(_write_cif(tmp_path)) is None
    gui = tmp_path / "structure.gui"
    gui.write_text("3 1 1\n")
    assert load_adp(str(gui)) is None


# ── which cell the phonon modes belong to ───────────────────────────────
# A SCELPHONO run computes its force constants in a supercell and then reports
# the modes of the cell it expanded: the geometry sections describe 540 atoms
# while the eigenvectors span 20. Reshaping against the wrong one used to make
# such a file refuse to open at all.
class _FakeOutput:
    """The three calls ``_structure_for_modes`` makes on a Crystal_output."""

    def __init__(self, natom_cell, natom_primitive=None):
        self._cell = self._structure(natom_cell)
        self._primitive = (None if natom_primitive is None
                           else self._structure(natom_primitive))

    @staticmethod
    def _structure(natom):
        from pymatgen.core import Lattice, Structure as PmgStructure

        return PmgStructure(
            Lattice.cubic(4.0 * natom),
            ["Si"] * natom,
            [[i / natom, 0.0, 0.0] for i in range(natom)],
        )

    def get_geometry(self, initial=True, **kwargs):
        return self._cell

    def get_primitive_geometry(self, initial=True, **kwargs):
        if self._primitive is None:
            raise AttributeError("no primitive section in this output")
        return self._primitive

    def get_dimensionality(self):
        return 3


def test_modes_that_span_the_geometry_use_it():
    pytest.importorskip("pymatgen")
    from crystalline.crystalio.loader import _structure_for_modes

    structure = _structure_for_modes(_FakeOutput(8, natom_primitive=2), 8)

    assert len(structure) == 8


def test_modes_of_an_expanded_cell_fall_back_to_the_primitive_one():
    """The supercell is what the geometry sections describe; the modes belong
    to the cell it was built from."""
    pytest.importorskip("pymatgen")
    from crystalline.crystalio.loader import _structure_for_modes

    structure = _structure_for_modes(_FakeOutput(540, natom_primitive=20), 20)

    assert len(structure) == 20


def test_modes_matching_neither_cell_say_so_in_the_message():
    """Better than numpy's 'cannot reshape array of size 60 into shape (540,3)',
    which is what the user sees when a file will not open."""
    pytest.importorskip("pymatgen")
    from crystalline.crystalio.loader import _structure_for_modes

    with pytest.raises(ValueError, match="7 atoms.*540.*primitive cell 20"):
        _structure_for_modes(_FakeOutput(540, natom_primitive=20), 7)


def test_an_output_with_no_primitive_section_still_reports_the_mismatch():
    pytest.importorskip("pymatgen")
    from crystalline.crystalio.loader import _structure_for_modes

    with pytest.raises(ValueError, match="4 atoms"):
        _structure_for_modes(_FakeOutput(8), 4)


# ── which q-points a run sampled ──────────────────────────────────────────
# A SCELPHONO run reports modes at every q its supercell makes commensurate.
# Their coordinates are read from the output text rather than from
# CRYSTALClear's ``qpoint`` list, which omits the Gamma block that heads the
# run (so it can't be lined up with the frequencies) and divides all three
# components by the first shrinking factor (wrong for a non-cubic supercell).
_DISPERSION_HEADERS = """
 * WITH SHRINKING FACTORS: IS1 =     2 IS2 =     1 IS3 =     4                 *
  DISPERSION K POINT NUMBER     1 COORD:  R(  0  0  0 )    WEIGHT:    1.
  DISPERSION K POINT NUMBER     2 COORD:  C(  0  0  1 )    WEIGHT:    1.
  DISPERSION K POINT NUMBER     3 COORD:  R(  1  0  2 )    WEIGHT:    1.
"""


def test_qpoints_are_read_with_one_denominator_per_axis():
    from crystalline.crystalio.loader import _qpoints_from_lines

    qpoints = _qpoints_from_lines(_DISPERSION_HEADERS.splitlines())

    assert len(qpoints) == 3  # Gamma included: it is the run's first k point
    assert np.allclose(qpoints[0], [0.0, 0.0, 0.0])
    assert np.allclose(qpoints[1], [0.0, 0.0, 0.25])
    assert np.allclose(qpoints[2], [0.5, 0.0, 0.5])


def test_qpoints_also_read_the_single_denominator_spelling():
    from crystalline.crystalio.loader import _qpoints_from_lines

    lines = [
        " K POINTS EXPRESSED IN UNITS  OF DENOMINATOR    3",
        "  DISPERSION K POINT NUMBER     1 COORD:  R(  0  0  0 )    WEIGHT:    1.",
        "  DISPERSION K POINT NUMBER     2 COORD:  C(  0  1  0 )    WEIGHT:    2.",
    ]

    qpoints = _qpoints_from_lines(lines)

    assert np.allclose(qpoints[1], [0.0, 1 / 3, 0.0])


def test_a_plain_frequency_run_reports_no_qpoints():
    """No dispersion headers, or headers without their denominators, means
    there is nothing to place in reciprocal space — and Gamma is the answer."""
    from crystalline.crystalio.loader import _qpoints_from_lines

    assert _qpoints_from_lines([" MODES         EIGV          FREQUENCIES     IRREP"]) == []
    assert _qpoints_from_lines(
        ["  DISPERSION K POINT NUMBER     2 COORD:  C(  0  0  1 )    WEIGHT:    1."]
    ) == []


class _FakeDispersionOutput:
    """The attributes ``_sampled_qpoints`` reads off a ``Crystal_output``."""

    def __init__(self, text):
        self.data = text.splitlines()
        self.eoo = len(self.data)


def test_qpoints_that_do_not_line_up_with_the_modes_are_dropped():
    """A QHA run's "q-point" axis counts volumes, not points in reciprocal
    space, and any other mismatch would misphase every image atom. Unlabelled
    (i.e. Gamma) beats mislabelled."""
    from crystalline.crystalio.loader import _sampled_qpoints

    out = _FakeDispersionOutput(_DISPERSION_HEADERS)

    assert _sampled_qpoints(out, 3)[1] is not None  # three headers, three sets
    assert _sampled_qpoints(out, 5) == [None] * 5  # five sets: not this run's q


def test_an_output_whose_modes_cannot_be_read_still_opens(monkeypatch, tmp_path):
    """The geometry is not lost with the frequencies.

    Reading modes goes through CRYSTALClear's eigenvector parser, which gives
    up on an output carrying anything it does not expect — the extra output of
    a patched CRYSTAL, a job script's own listing. That used to take the whole
    file down: the structure was there to be read, and the window showed an
    error instead of it.
    """
    from ase.build import bulk

    from crystalline.crystalio import loader
    from crystalline.core.structure import Structure

    output = tmp_path / "run.out"
    output.write_text("not read: every step below is stubbed\n")
    geometry = Structure.from_ase(bulk("MgO", "rocksalt", a=4.21))

    monkeypatch.setattr(loader, "has_phonons", lambda _path: True)
    monkeypatch.setattr(loader, "load_structure",
                        lambda _path, initial=False: geometry)

    def explode(_path, **_kwargs):
        raise ValueError("inhomogeneous shape after 1 dimensions")

    monkeypatch.setattr(loader, "load_dispersion", explode)

    result = loader.load(str(output))

    assert result.structure is geometry        # the structure came through
    assert result.modes is None
    assert result.qpoints == []
    assert result.note and "could not be read" in result.note
    assert "inhomogeneous" in result.note      # and it says what went wrong


# One eigenvector block as CRYSTAL prints it: the frequencies, then a row per
# axis per atom. The blank line after the last row is what tells a reader the
# block has ended — the point of the two tests below.
_EIGENVECTOR_BLOCK = """
 FREQ(CM**-1)     48.10     48.10    188.99

 AT.   1 O  X     0.0000    0.0000   -0.0000
            Y     0.0000   -0.0010    0.0001
            Z     0.0006   -0.0019   -0.0002
 AT.   2 MG X    -0.0000   -0.0000   -0.0000
            Y     0.0848    0.0745   -0.1091
            Z    -0.0746    0.0841    0.0812
"""


def test_a_well_formed_eigenvector_block_is_left_alone():
    """Nothing is repaired in an output CRYSTAL wrote on its own."""
    from crystalline.crystalio.loader import _repair_eigenvector_blocks

    data = (_EIGENVECTOR_BLOCK + "\n VIBRATIONAL TEMPERATURES (K)\n").splitlines(True)
    original = list(data)

    assert _repair_eigenvector_blocks(data) == 0
    assert data == original


def test_lines_glued_to_an_eigenvector_block_are_blanked():
    """A foreign line touching the last row is read as another row of it.

    Its numbers do not count out to the block's modes, so the whole frequency
    section fails on the mismatched shape. The lines are blanked instead — and
    only they: the numbering the caller parses against has to survive, and so
    do the rows themselves.
    """
    from crystalline.crystalio.loader import _repair_eigenvector_blocks

    debug = " *** metroFRA_disp ***\n xa_g -7.7669E-07  2.5775E-05\n"
    data = (_EIGENVECTOR_BLOCK.rstrip("\n") + "\n" + debug + "\n MORE CRYSTAL OUTPUT\n")
    data = data.splitlines(True)
    rows = [line for line in data if " AT. " in line or line[:13].strip() in ("Y", "Z")]
    length = len(data)

    assert _repair_eigenvector_blocks(data) == 2  # the two foreign lines, no more
    assert len(data) == length  # blanked, not removed: the numbering has to hold
    assert [line for line in data if " AT. " in line or line[:13].strip() in ("Y", "Z")] == rows
    assert data[-1] == " MORE CRYSTAL OUTPUT\n"  # what came after the blank line
    last_row = max(i for i, line in enumerate(data) if line in rows)
    assert data[last_row + 1].strip() == ""  # the block ends where its rows end


def test_a_qpoint_header_is_never_blanked():
    """Whatever else a block ran into, the next q-point has to stay readable."""
    from crystalline.crystalio.loader import _repair_eigenvector_blocks

    header = "  DISPERSION K POINT NUMBER     2 COORD:  C(  1  0  0 )    WEIGHT:    1.\n"
    data = (_EIGENVECTOR_BLOCK.rstrip("\n") + "\n" + header).splitlines(True)

    assert _repair_eigenvector_blocks(data) == 0
    assert data[-1] == header


def test_lines_landing_inside_a_block_leave_its_rows_together():
    """Text written into the middle of a block cuts the rows after it adrift.

    The reader stops at the first line that is not a row, so those rows are lost
    from the block and its eigenvectors come out short — the same failure as
    text glued to the end, further from its cause. They are gathered back here.
    """
    from crystalline.crystalio.loader import _repair_eigenvector_blocks

    block = _EIGENVECTOR_BLOCK.rstrip("\n").splitlines(True)
    interruption = [" Sono dentro scanpes\n"] * 3
    data = block[:-2] + interruption + block[-2:] + ["\n"]
    length = len(data)

    assert _repair_eigenvector_blocks(data) == 3
    assert len(data) == length
    rows = [line for line in data if line.strip()][1:]  # past the FREQ header
    assert len(rows) == 6  # two atoms, three axes: every row of the block
    assert rows == [line for line in block if " AT. " in line or line[:13].strip() in ("Y", "Z")]
    assert all(line.strip() == "" for line in data[-4:])  # the block ends after them


# ── the vacuum CRYSTAL writes across a slab or a chain ──────────────────

def _chain(y_of_second_atom):
    """A two-atom MgO chain along a, with 500 Å of formal vacuum across it."""
    from ase import Atoms

    return Atoms(
        "OMg",
        cell=[[3.567, 0, 0], [0, 500.0, 0], [0, 0, 500.0]],
        scaled_positions=[[0.5, 2.7e-08, 0.0], [1.1e-07, y_of_second_atom, 0.0]],
        pbc=(True, False, False),
    )


def test_an_atom_wrapped_across_the_formal_vacuum_comes_back():
    """pymatgen wraps fractional coordinates; CRYSTAL's 500 Å has no other image.

    An atom a hair below the origin comes back at 0.99999997, which against a
    formal 500 Å cell vector puts it half a kilometre from the chain. Along a
    real lattice vector the same wrap would be a different image of the same
    atom and would mean nothing.
    """
    from crystalline.crystalio.loader import _unwrap_aperiodic

    wrapped = _unwrap_aperiodic(_chain(0.999999973))

    assert wrapped.get_positions()[1][1] == pytest.approx(-1.36e-05, abs=1e-7)
    # The Mg ends up where the chain is: one half of a = 3.567 from the O.
    assert wrapped.get_distance(0, 1) == pytest.approx(3.567 / 2, abs=1e-4)


def test_coordinates_that_were_not_wrapped_are_left_exactly_alone():
    """The unwrap runs on every output read, so it has to be a no-op otherwise."""
    from crystalline.crystalio.loader import _unwrap_aperiodic

    chain = _chain(-2.7e-08)
    before = chain.get_positions().copy()

    assert np.array_equal(_unwrap_aperiodic(chain).get_positions(), before)


def test_a_periodic_axis_is_never_unwrapped():
    """Along a real lattice vector, which image an atom is given is immaterial —
    and rewinding one would move it out of the cell it was reported in."""
    from ase.build import bulk

    from crystalline.crystalio.loader import _unwrap_aperiodic

    crystal = bulk("MgO", "rocksalt", a=4.21)
    crystal.set_scaled_positions([[0.0, 0.0, 0.0], [0.99, 0.5, 0.5]])
    before = crystal.get_positions().copy()

    assert np.array_equal(_unwrap_aperiodic(crystal).get_positions(), before)
