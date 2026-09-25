"""One tab per open file: naming, routing to the tab on screen, opening and closing.

A real ``MainWindow`` needs a live ``QtInteractor`` per tab, which cannot be
built here, so the window's tab handling is bound onto a stand-in whose tabs
are made of plain widgets. What is under test is the bookkeeping — which tab is
current, which widgets each dock shows, what a new file goes into, what closing
leaves behind — and that does not need VTK.
"""

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDockWidget,
    QMainWindow,
    QStackedWidget,
    QTabWidget,
    QWidget,
)
from PySide6.QtCore import Qt  # noqa: E402

from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui import file_tabs  # noqa: E402
from crystalline.ui.file_tabs import (  # noqa: E402
    PER_TAB_PANELS,
    UNTITLED,
    FileTab,
    openable,
    route_to_active_tab,
    tab_labels,
    window_title,
)
from crystalline.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ── names ─────────────────────────────────────────────────────────────────
def test_a_tab_is_named_after_its_file():
    assert tab_labels(["/runs/urea/urea.out", None]) == ["urea.out", UNTITLED]


def test_files_of_the_same_name_are_told_apart_by_their_folder():
    # a CRYSTAL series: the same name in every folder
    paths = ["/p/0GPa/fort.34", "/p/1GPa/fort.34", "/p/urea.out"]
    assert tab_labels(paths) == ["0GPa/fort.34", "1GPa/fort.34", "urea.out"]


def test_as_much_of_the_path_as_it_takes():
    paths = ["/a/x/opt/run.out", "/b/x/opt/run.out"]
    assert tab_labels(paths) == ["a/x/opt/run.out", "b/x/opt/run.out"]


def test_untitled_tabs_are_not_told_apart():
    assert tab_labels([None, None]) == [UNTITLED, UNTITLED]


def test_the_window_is_titled_after_the_file_on_screen():
    app = "CRYSTALLine — CRYSTAL structure & phonon viewer"
    assert window_title("1GPa/fort.34", app) == "1GPa/fort.34 — CRYSTALLine"
    assert window_title(None, app) == app


# ── what a drop opens ─────────────────────────────────────────────────────
def _action(path):
    return {".out": "open", ".cif": "open", ".xyz": "import"}.get(os.path.splitext(path)[1])


def test_every_structure_dropped_opens():
    assert openable(["a.out", "b.cif", "c.png"], _action) == (["a.out", "b.cif"], [], ["c.png"])


def test_atoms_are_imported_only_when_nothing_is_opened():
    assert openable(["x.xyz", "a.out"], _action) == (["a.out"], [], ["x.xyz"])
    assert openable(["x.xyz", "y.xyz"], _action) == ([], ["x.xyz"], ["y.xyz"])


# ── a tab ─────────────────────────────────────────────────────────────────
def test_a_tab_without_a_file_or_atoms_is_blank():
    tab = FileTab()
    assert tab.is_blank()
    tab._source = Structure.empty()
    tab.structure = Structure.empty()
    assert tab.is_blank()
    tab.structure.add_atom("C", [0.0, 0.0, 0.0])  # built by hand: not blank any more
    assert not tab.is_blank()
    other = FileTab()
    other.path = "/runs/urea.out"
    assert not other.is_blank()


def test_the_window_reads_and_writes_the_tab_on_screen(qapp):
    @route_to_active_tab
    class Window(QMainWindow):
        pass

    window = Window()
    assert not hasattr(window, "structure")      # no tab yet: as if never set
    first, second = FileTab(), FileTab()
    window._tab = first
    window._supercell = (2, 2, 2)
    assert first._supercell == (2, 2, 2)
    window._tab = second
    window._supercell = (1, 1, 1)
    assert window._supercell == (1, 1, 1) and first._supercell == (2, 2, 2)


# ── the window's tab handling, on a stand-in ──────────────────────────────
class _Part(QWidget):
    """Stands for any of a tab's widgets: records what the window asks of it."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.editing = None
        self.stopped = 0
        self.figures = 0
        self.cleared = 0
        self.closed = 0
        self.renderer = type("R", (), {"settings": object()})()
        self.interactor = self   # the viewport's VTK widget, closed with the tab

    def set_editing_enabled(self, enabled):
        self.editing = enabled

    def stop(self):
        self.stopped += 1

    def count(self):
        return self.figures

    def clear(self):
        self.cleared += 1
        self.figures = 0

    def close(self):
        self.closed += 1
        return super().close()


@route_to_active_tab
class _Window(QMainWindow):
    _add_tab = MainWindow._add_tab
    _activate_tab = MainWindow._activate_tab
    _close_tab = MainWindow._close_tab
    _close_tab_at = MainWindow._close_tab_at
    _close_current_tab = MainWindow._close_current_tab
    _take_tab_for_file = MainWindow._take_tab_for_file
    _tab_for_page = MainWindow._tab_for_page
    _on_file_tab_changed = MainWindow._on_file_tab_changed
    _on_plot_dock_visibility = MainWindow._on_plot_dock_visibility
    _update_tab_labels = MainWindow._update_tab_labels
    _update_window_title = MainWindow._update_window_title
    _next_tab = MainWindow._next_tab
    _previous_tab = MainWindow._previous_tab
    _step_tab = MainWindow._step_tab
    _routed = MainWindow._routed
    _acting_on = MainWindow._acting_on

    def __init__(self):
        super().__init__()
        self._tab = None
        self._tabs = []
        self._switching_tabs = False
        self._editing = False
        self._workers = []
        self.chrome = 0
        self.made = []
        self._file_tabs = QTabWidget(self)
        self._file_tabs.setTabsClosable(True)
        self.setCentralWidget(self._file_tabs)
        self._panel_stacks = {name: QStackedWidget(self) for name in PER_TAB_PANELS}
        self._plot_dock = QDockWidget("Plots", self)
        self._plot_dock.setWidget(self._panel_stacks["plot_panel"])
        self.addDockWidget(Qt.BottomDockWidgetArea, self._plot_dock)
        self._plot_dock.hide()
        self._plot_dock.visibilityChanged.connect(self._on_plot_dock_visibility)
        self._file_tabs.currentChanged.connect(self._on_file_tab_changed)
        self._add_tab()

    def _create_tab(self, structure=None, settings=None):
        tab = FileTab()
        tab._source = structure if structure is not None else Structure.empty()
        tab.structure = tab._source
        for name in file_tabs.PER_TAB_WIDGETS:
            setattr(tab, name, _Part(name))
        for name, stack in self._panel_stacks.items():
            stack.addWidget(getattr(tab, name))
        self.made.append((tab, settings))
        return tab

    def _refresh_chrome(self):
        self.chrome += 1
        self._update_window_title()

    def _follow_theme_background(self):
        pass

    def open(self, path):
        """What ``_load_path`` does with tabs, without reading anything."""
        tab = self._take_tab_for_file()
        tab.path = path
        self._update_tab_labels()
        return tab


def _titles(window):
    tabs = window._file_tabs
    return [tabs.tabText(i) for i in range(tabs.count())]


def test_the_window_starts_on_one_empty_tab_that_the_first_file_takes(qapp):
    window = _Window()
    assert _titles(window) == [UNTITLED]
    first = window._tab
    assert window.open("/runs/urea.out") is first    # no second tab for the first file
    assert _titles(window) == ["urea.out"]
    assert window.windowTitle() == "urea.out — CRYSTALLine"


def test_each_further_file_opens_in_a_tab_of_its_own_and_is_shown(qapp):
    window = _Window()
    urea = window.open("/runs/urea.out")
    ice = window.open("/runs/ice.out")
    assert ice is not urea and window._tab is ice
    assert _titles(window) == ["urea.out", "ice.out"]
    # the new tab starts from the look of the one it was opened from
    assert window.made[-1][1] is urea.viewport.renderer.settings
    # and every dock shows the new tab's panel
    for name, stack in window._panel_stacks.items():
        assert stack.currentWidget() is getattr(ice, name)
    assert window._file_tabs.currentWidget() is ice.viewport


def test_switching_tab_shows_that_tabs_panels_and_stops_the_other_animation(qapp):
    window = _Window()
    urea = window.open("/runs/urea.out")
    ice = window.open("/runs/ice.out")
    window._file_tabs.setCurrentIndex(0)                 # a click on the first tab
    assert window._tab is urea
    assert ice.phonon_panel.stopped == 1                 # nothing plays unseen
    assert window._panel_stacks["info_panel"].currentWidget() is urea.info_panel
    assert window.windowTitle() == "urea.out — CRYSTALLine"
    window._next_tab()
    assert window._tab is ice
    window._next_tab()                                   # and round again
    assert window._tab is urea
    window._previous_tab()
    assert window._tab is ice


def test_the_editing_mode_follows_to_the_tab_arrived_at(qapp):
    window = _Window()
    window.open("/runs/urea.out")
    ice = window.open("/runs/ice.out")
    window._editing = True
    window._file_tabs.setCurrentIndex(0)
    window._file_tabs.setCurrentIndex(1)
    assert ice.viewport.editing is True
    assert ice.geometry_panel.editing is True and ice.structure_panel.editing is True


def test_plots_come_and_go_with_their_tab(qapp):
    window = _Window()
    window.show()                                        # a dock is only visible in a shown window
    urea = window.open("/runs/urea.out")
    urea.plot_panel.figures = 1
    window._plot_dock.show()                             # the user opens urea's plots
    assert urea.plots_open
    ice = window.open("/runs/ice.out")
    assert window._plot_dock.isHidden()                  # ice has none
    assert urea.plots_open                               # hidden by the switch, not closed
    window._file_tabs.setCurrentIndex(0)
    assert not window._plot_dock.isHidden()              # urea's come back
    window._plot_dock.hide()                             # the user shuts them
    window._file_tabs.setCurrentIndex(1)
    window._file_tabs.setCurrentIndex(0)
    assert window._plot_dock.isHidden() and not urea.plots_open
    assert not ice.plots_open


def test_closing_a_tab_lets_its_file_go(qapp):
    window = _Window()
    urea = window.open("/runs/urea.out")
    ice = window.open("/runs/ice.out")
    ice.plot_panel.figures = 2
    window._close_current_tab()
    assert window._tabs == [urea] and window._tab is urea
    assert _titles(window) == ["urea.out"]
    assert ice.plot_panel.cleared == 1                   # figures released
    assert ice.viewport.closed >= 1                      # the render window too
    for name, stack in window._panel_stacks.items():
        assert stack.indexOf(getattr(ice, name)) == -1


def test_closing_the_last_tab_leaves_an_empty_one(qapp):
    window = _Window()
    urea = window.open("/runs/urea.out")
    window._close_tab(urea)
    assert len(window._tabs) == 1 and window._tab.is_blank()
    assert _titles(window) == [UNTITLED]
    assert window.windowTitle().startswith("CRYSTALLine")
    blank = window._tab
    window._close_tab(blank)                             # nothing to close
    assert window._tabs == [blank]


def test_nothing_closes_or_switches_while_a_task_runs(qapp):
    window = _Window()
    urea = window.open("/runs/urea.out")
    window.open("/runs/ice.out")
    window._workers.append(object())                     # an orbital being built
    window._close_current_tab()
    assert len(window._tabs) == 2
    window._next_tab()
    assert window._tab is not urea
    window._workers.clear()


def test_the_same_name_twice_is_shown_with_its_folder(qapp):
    window = _Window()
    window.open("/p/0GPa/fort.34")
    window.open("/p/1GPa/fort.34")
    assert _titles(window) == ["0GPa/fort.34", "1GPa/fort.34"]
    assert window.windowTitle() == "1GPa/fort.34 — CRYSTALLine"
    window._close_current_tab()
    assert _titles(window) == ["fort.34"]                # no clash left


def test_a_late_signal_acts_on_the_tab_it_came_from(qapp):
    """A display change applied after a delay, or the camera settling after a
    wheel zoom, may arrive once another tab is on screen."""
    window = _Window()
    urea = window.open("/runs/urea.out")
    ice = window.open("/runs/ice.out")
    seen = []

    def slot(value):
        seen.append((window._tab, window._source, value))

    late = window._routed(urea, slot)
    before = window.chrome
    late(7)
    assert seen == [(urea, urea._source, 7)]
    assert window._tab is ice                            # still ice on screen
    assert window.chrome > before                        # menus set back to ice's
    window._close_tab(urea)
    late(8)
    assert len(seen) == 1                                # nothing for a closed tab


def test_the_lone_empty_tab_offers_no_close_button(qapp):
    from PySide6.QtWidgets import QTabBar

    window = _Window()
    window.show()

    def closable(index):
        bar = window._file_tabs.tabBar()
        buttons = [bar.tabButton(index, side) for side in (QTabBar.LeftSide, QTabBar.RightSide)]
        return any(button is not None and button.isVisible() for button in buttons)

    assert not closable(0)                               # it would do nothing
    window.open("/runs/urea.out")
    assert closable(0)
    window.open("/runs/ice.out")
    assert closable(0) and closable(1)


def test_a_hidden_view_is_not_drawn_into():
    """With several files open only one 3D view is on screen. Rendering into a
    hidden one — which opening the second file did, while building its view —
    corrupts the OpenGL state the visible one shares, and the app aborted on
    shader errors. The viewport holds such renders until it is shown.

    Checked on the source rather than by building a Viewport, which needs a
    live QtInteractor.
    """
    import inspect

    from crystalline.ui.viewport import Viewport

    init = inspect.getsource(Viewport.__init__)
    assert "self.interactor.render = render_if_shown" in init
    assert "isVisible()" in init
    assert "self._render_held" in inspect.getsource(inspect.unwrap(Viewport.eventFilter))
