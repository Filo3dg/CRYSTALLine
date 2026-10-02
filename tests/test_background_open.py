"""Opening a file reads it off the main thread, then builds its tab from what was read.

The window itself needs a real display (VTK), so these exercise the two halves
apart: :func:`_read_file`, which is Qt-free and does all the reading, and the
queue in :meth:`MainWindow._load_path`, on a stand-in carrying the window's
own methods with the tab-building stubbed out.
"""

import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import sample_data as sample  # noqa: E402
from crystalline.ui import main_window as mw  # noqa: E402
from crystalline.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ── reading ───────────────────────────────────────────────────────────────
def test_an_output_is_read_whole_for_its_tab():
    """The structure, and everything the window used to read again afterwards:
    the Info panel's rows and what each menu needs to know about the output."""
    path = sample.path("coesite/coesite_ela.out")
    if path is None:
        pytest.skip("sample output not available")

    read = mw._read_file(path)

    assert len(read.loaded.structure) > 0
    assert read.output_path == path
    assert read.output_props                                 # the run's own rows
    assert set(read.capabilities) == set(mw._PROBES)
    for key, found in read.capabilities.items():             # the window's own answer
        assert found == mw._probe(key, path), key


def test_a_geometry_file_has_no_output_behind_it(tmp_path):
    from ase.build import bulk

    from crystalline.core.structure import Structure
    from crystalline.crystalio import save_structure_gui

    path = str(tmp_path / "mgo.gui")
    save_structure_gui(Structure.from_ase(bulk("MgO", "rocksalt", a=4.21)), path)

    read = mw._read_file(path)

    assert set(read.loaded.structure.symbols) == {"Mg", "O"}
    assert read.output_path is None                          # nothing to plot from
    assert read.adps is None


def test_a_file_that_will_not_read_raises_rather_than_opening_a_tab(tmp_path):
    path = tmp_path / "nothing.out"
    path.write_text("not a CRYSTAL output\n")
    with pytest.raises(Exception):
        mw._read_file(str(path))


def test_what_was_found_off_the_thread_answers_the_window(monkeypatch):
    """Found once, while reading: the window must not walk the output again."""

    class _Tab:
        capabilities = {("vci", "/runs/a.out"): True}

    class _Stub:
        _capability = MainWindow._capability
        _tab = _Tab()
        _output_path = "/runs/a.out"

    asked = []
    monkeypatch.setattr(mw, "_probe", lambda key, path: asked.append(key) or False)
    assert _Stub()._capability("vci") is True
    assert asked == []
    assert _Stub()._capability("pes") is False               # not found yet: asked now
    assert asked == ["pes"]


# ── the queue ─────────────────────────────────────────────────────────────
class _Busy:
    def __init__(self):
        self.messages = []

    def start(self, message):
        self.messages.append(message)

    def stop(self):
        pass


class _TabBar:
    enabled = True

    def setEnabled(self, enabled):
        self.enabled = enabled


class _Tabs:
    def __init__(self):
        self.bar = _TabBar()

    def tabBar(self):
        return self.bar


def _window():
    """The window's queue and worker plumbing, with the tab-building recorded instead."""

    class _Stub:
        _load_path = MainWindow._load_path
        _read_next = MainWindow._read_next
        _run_busy = MainWindow._run_busy

        def __init__(self):
            self._busy = _Busy()
            self._workers = []
            self._file_tabs = _Tabs()
            self._pending_reads = []
            self._reading = False
            self.shown = []

        def _show_read_file(self, path, read):
            self.shown.append((path, read, threading.current_thread()))

    return _Stub()


def test_files_are_read_off_the_main_thread_one_at_a_time_in_order(qapp, qtbot, monkeypatch):
    read_on = {}

    def fake_read(path):
        read_on[path] = threading.current_thread()
        if path.endswith("broken.out"):
            raise ValueError("Geometry information not found.")
        return f"read {path}"

    reported = []
    monkeypatch.setattr(mw, "_read_file", fake_read)
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda _parent, title, text: reported.append((title, text))))
    window = _window()

    for path in ("/runs/a.out", "/runs/broken.out", "/runs/c.gui"):
        window._load_path(path)
    qtbot.waitUntil(lambda: not window._reading, timeout=5000)

    main = threading.main_thread()
    assert all(thread is not main for thread in read_on.values())     # read off the main thread
    assert [(p, r) for p, r, _t in window.shown] == [                 # tabs built in order...
        ("/runs/a.out", "read /runs/a.out"), ("/runs/c.gui", "read /runs/c.gui")]
    assert all(thread is main for _p, _r, thread in window.shown)     # ...on the main thread
    assert reported == [("Load failed", "broken.out:\nGeometry information not found.")]
    assert window._busy.messages == ["Reading a.out…", "Reading broken.out…", "Reading c.gui…"]
    assert window._file_tabs.bar.enabled and window._workers == []


def test_a_file_opened_while_another_is_read_waits_its_turn(qapp, qtbot, monkeypatch):
    started = []
    release = threading.Event()

    def slow_read(path):
        started.append(path)
        release.wait(5)
        return path

    monkeypatch.setattr(mw, "_read_file", slow_read)
    window = _window()
    window._load_path("/runs/first.out")
    qtbot.waitUntil(lambda: started == ["/runs/first.out"], timeout=5000)
    window._load_path("/runs/second.out")                    # arrives mid-read
    assert started == ["/runs/first.out"]                    # not read alongside it
    release.set()
    qtbot.waitUntil(lambda: not window._reading, timeout=5000)
    assert [p for p, _r, _t in window.shown] == ["/runs/first.out", "/runs/second.out"]
