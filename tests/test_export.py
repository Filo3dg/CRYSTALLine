"""Image and phonon-animation export (off-screen PyVista)."""

import os

import numpy as np
import pytest

pytest.importorskip("pyvista")
pytest.importorskip("ase")

import pyvista as pv  # noqa: E402
from ase.build import bulk  # noqa: E402

from crystalline.core.phonons import PhononMode  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.viz import export  # noqa: E402
from crystalline.viz.render_settings import RenderSettings  # noqa: E402
from crystalline.viz.renderer import StructureRenderer  # noqa: E402


@pytest.fixture
def nacl():
    return Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))


def _plotter_with(structure):
    p = pv.Plotter(off_screen=True)
    StructureRenderer(p).set_structure(structure)
    p.reset_camera()
    return p


def test_save_view_image_raster_and_vector(nacl, tmp_path):
    p = _plotter_with(nacl)
    png = export.save_view_image(p, str(tmp_path / "view.png"))
    svg = export.save_view_image(p, str(tmp_path / "view.svg"))
    p.close()
    assert os.path.getsize(png) > 0
    assert os.path.getsize(svg) > 0


def test_save_view_image_rejects_unknown_format(nacl, tmp_path):
    p = _plotter_with(nacl)
    with pytest.raises(ValueError):
        export.save_view_image(p, str(tmp_path / "view.xyz"))
    p.close()


def test_render_animation_frames_shape(nacl):
    mode = PhononMode(120.0, np.array([[1, 0, 0], [-1, 0, 0]], float))
    frames = export.render_animation_frames(
        nacl, nacl.positions, mode, RenderSettings(), amplitude=0.6, n_frames=6
    )
    assert len(frames) == 6
    assert frames[0].ndim == 3 and frames[0].shape[2] in (3, 4)
    assert frames[0].dtype == np.uint8
    # the mode actually moves atoms, so not every frame is identical
    assert not all(np.array_equal(frames[0], f) for f in frames[1:])


def test_each_frame_is_reported_as_it_is_rendered(nacl):
    """Rendering is VTK's work and VTK's is the main thread's, so the window
    cannot turn its own indicator meanwhile. The caller is told after every
    frame instead, and turns it by hand — an export that shows nothing for the
    length of it is taken for an app that has died."""
    mode = PhononMode(120.0, np.array([[1, 0, 0], [-1, 0, 0]], float))
    seen = []

    frames = export.render_animation_frames(
        nacl, nacl.positions, mode, RenderSettings(), n_frames=5,
        on_frame=lambda done, total: seen.append((done, total)),
    )

    assert len(frames) == 5
    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


def test_nothing_is_held_while_an_animation_is_written(nacl, tmp_path):
    """The writers take frames one at a time, so the peak is a frame rather than
    a cycle. 240 of them at 1920x1440 is two gigabytes, and the GIF writer made
    its own copy of every one: the process was killed, not refused."""
    mode = PhononMode(120.0, np.array([[1, 0, 0], [-1, 0, 0]], float))
    alive = []

    def counted(frames):
        for frame in frames:
            alive.append(frame)
            yield frame
            alive.remove(frame)        # let go the moment the writer is done with it

    out = tmp_path / "cycle.gif"
    written = export.save_animation(
        counted(export.animation_frames(nacl, nacl.positions, mode, RenderSettings(),
                                        n_frames=6, window_size=(80, 60))),
        str(out),
    )

    assert written == [str(out)] and out.exists()
    assert alive == []                 # nothing of the cycle is still held
    # An empty cycle is still an error rather than an empty file.
    with pytest.raises(ValueError):
        export.save_animation(iter(()), str(tmp_path / "empty.gif"))


def test_apply_camera_preserves_parallel_zoom():
    """The export must reproduce the on-screen zoom. Under parallel projection that
    zoom is the camera's parallel scale, which a bare ``camera_position`` omits —
    so the full snapshot must be applied, or the render zooms in."""
    plotter = pv.Plotter(off_screen=True)
    try:
        plotter.add_mesh(pv.Sphere())
        plotter.enable_parallel_projection()
        snapshot = (plotter.camera_position, 0.37, plotter.camera.GetViewAngle())

        export._apply_camera(plotter, snapshot)
        assert np.isclose(plotter.camera.GetParallelScale(), 0.37)  # zoom applied

        # a bare camera_position must still work (placement only, no crash)
        export._apply_camera(plotter, plotter.camera_position)
    finally:
        plotter.close()


def test_save_animation_gif(tmp_path):
    frames = [np.full((20, 30, 3), i * 20, np.uint8) for i in range(5)]
    out = export.save_animation(frames, str(tmp_path / "anim.gif"), fps=10)
    assert out == [str(tmp_path / "anim.gif")]
    from PIL import Image

    with Image.open(out[0]) as im:
        assert getattr(im, "n_frames", 1) == 5


def test_save_animation_png_sequence(tmp_path):
    frames = [np.zeros((10, 10, 3), np.uint8) for _ in range(4)]
    written = export.save_animation(frames, str(tmp_path / "frame.png"))
    assert len(written) == 4
    assert all(os.path.exists(p) for p in written)
    assert os.path.basename(written[0]) == "frame_000.png"


def test_save_animation_rejects_unknown_and_empty(tmp_path):
    with pytest.raises(ValueError):
        export.save_animation([], str(tmp_path / "x.gif"))
    with pytest.raises(ValueError):
        export.save_animation([np.zeros((4, 4, 3), np.uint8)], str(tmp_path / "x.qqq"))


# ── video export ─────────────────────────────────────────────────────────
# Encoding needs imageio-ffmpeg, which a plain install doesn't pull in. The
# rules below have to hold whether or not it is present, so most of these test
# the decision rather than the encoding.
def test_a_missing_encoder_says_what_to_install(tmp_path, monkeypatch):
    """Without this the failure was `TypeError: write() got an unexpected keyword
    argument 'fps'` — imageio falling through to a plugin that cannot encode
    video at all, reported as a bug in our call rather than a missing package."""
    monkeypatch.setattr(export, "video_export_available", lambda: False)
    frames = [np.zeros((8, 8, 3), np.uint8)] * 2

    with pytest.raises(RuntimeError, match="imageio-ffmpeg"):
        export.save_animation(frames, str(tmp_path / "mode.mp4"))


def test_webm_is_encoded_with_a_codec_its_container_accepts():
    """Handing ffmpeg H.264 for a WebM container does not fail — it writes an
    empty file and reports success, so the export looked like it worked and
    produced 300 bytes."""
    assert export._codec_for("/tmp/mode.webm") == "libvpx-vp9"
    assert export._codec_for("/tmp/mode.mp4") == "libx264"
    assert export._codec_for("/tmp/mode.MOV") == "libx264"


def test_frames_are_trimmed_to_even_dimensions():
    """H.264 encodes even dimensions only. Trimming a row costs nothing visible;
    the alternative (imageio's default) rescales the whole movie to the next
    multiple of 16, quietly changing the resolution that was asked for."""
    assert export._even_sized(np.zeros((101, 103, 3), np.uint8)).shape == (100, 102, 3)
    assert export._even_sized(np.zeros((100, 102, 3), np.uint8)).shape == (100, 102, 3)


@pytest.mark.skipif(not export.video_export_available(), reason="needs imageio-ffmpeg")
@pytest.mark.parametrize("ext", [".mp4", ".webm"])
def test_a_written_video_decodes_back_to_the_frames_given(tmp_path, ext):
    """A file that exists is not a file that plays — this is what caught the
    empty WebM. Read back with imageio-ffmpeg's own reader rather than
    ``imageio.get_reader``: the latter parses the ffmpeg banner and mis-reads
    the one shipped with current imageio-ffmpeg, which is a fault in the check,
    not in the file."""
    import imageio_ffmpeg

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (64, 65, 3)).astype(np.uint8) for _ in range(8)]
    path = str(tmp_path / f"mode{ext}")

    export.save_animation(frames, path, fps=12)

    stream = imageio_ffmpeg.read_frames(path)
    meta = next(stream)
    decoded = sum(1 for _ in stream)
    assert decoded == len(frames)
    assert meta["size"] == (64, 64)  # odd width trimmed, not rescaled
    assert meta["fps"] == 12.0


# ── what the window shows while it waits ──────────────────────────────────
def test_the_overlay_can_be_turned_by_hand_while_the_loop_is_blocked(qapp):
    """``pulse`` paints inside the call, which is the only way the indicator
    moves while the main thread is rendering. It must not process events: the
    work it is covering is exactly what must not be re-entered."""
    from PySide6.QtWidgets import QWidget

    from crystalline.ui.widgets import BusyOverlay

    parent = QWidget()
    parent.resize(200, 160)
    overlay = BusyOverlay(parent)
    overlay.start("Rendering…")
    first = overlay._angle

    overlay.pulse("Rendering frame 2 of 9…")

    assert overlay._angle != first                       # the arc moved
    assert overlay._message == "Rendering frame 2 of 9…"  # and says where it is
    overlay.pulse()                                      # a message is optional
    assert overlay._message == "Rendering frame 2 of 9…"
    overlay.stop()


def test_an_export_is_drawn_by_the_view_on_screen():
    """A second VTK render window beside the live one is what killed the app:
    every frame came out and the first draw afterwards faulted in C++, with
    nothing in Python to catch. An image export does everything else an export
    does and only this differently, and has never done it — so the frames come
    from the view that is already there.
    """
    import inspect

    from crystalline.ui.main_window import MainWindow

    source = inspect.getsource(MainWindow._export_animation)
    assert "frames_from_view(" in source
    assert "animation_frames(" not in source        # no off-screen plotter of its own
    assert "render_animation_frames" not in source  # nor the list, which held them all
    assert "on_frame=self._exporting_frame" in source

    # The view moves while this runs, so it is put back where it stood.
    assert "standing_at = self.phonon_panel.current_phase()" in source
    assert "self.phonon_panel.show_phase(standing_at)" in source
    # A mode left playing would drive the same view from a timer, mid-export.
    assert source.index("self.phonon_panel.stop()") < source.index("frames_from_view(")

    # And the frame callback gives the window its turns, without letting a click
    # in: a second export started on top of this one would re-enter VTK.
    turn = inspect.getsource(MainWindow._exporting_frame)
    assert "self._busy.pulse(" in turn
    assert "QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)" in turn


def test_a_capture_reads_the_frame_it_drew_not_the_screen(nacl):
    """pyvista's own capture turns ``ReadFrontBuffer`` back on two lines after
    turning it off, so it takes the pixels as they stand on screen: anything
    over the window is captured with it, and what is not composited comes back
    black — which is what put black borders on an exported animation and made
    the a/b/c gizmo ragged in an exported image."""
    import inspect

    source = inspect.getsource(export.capture_view)
    assert "ReadFrontBufferOff()" in source
    assert "ReadFrontBufferOn" not in source
    assert "FixBoundaryOn()" in source          # no seams where the tiles meet

    plotter = _plotter_with(nacl)
    try:
        plain = export.capture_view(plotter)
        bigger = export.capture_view(plotter, scale=2)
        assert plain.ndim == 3 and plain.shape[2] == 3
        assert bigger.shape[0] == plain.shape[0] * 2
        assert bigger.shape[1] == plain.shape[1] * 2
        assert export.capture_view(plotter, transparent=True).shape[2] == 4
        # The view is drawn, not black: a capture of nothing is the bug above.
        assert plain.max() > 0
    finally:
        plotter.close()


def test_frames_come_from_the_live_view_one_cycle_at_a_time(nacl):
    """Driven by the animator and shot from the plotter it is already drawing
    into — no second render window anywhere in it."""
    from crystalline.viz.phonon_animator import PhononAnimator

    mode = PhononMode(120.0, np.array([[1, 0, 0], [-1, 0, 0]], float))
    plotter = _plotter_with(nacl)
    try:
        renderer = StructureRenderer(plotter)
        renderer.set_structure(nacl)
        animator = PhononAnimator(renderer)
        animator.set_mode(nacl.positions, mode)
        seen = []

        frames = list(export.frames_from_view(
            plotter, animator, n_frames=5, size=(320, 240),
            on_frame=lambda done, total: seen.append((done, total)),
        ))

        assert len(frames) == 5
        assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
        assert frames[0].dtype == np.uint8
        assert frames[0].shape[:2] == (240, 320)      # the size asked for, exactly
        assert not all(np.array_equal(frames[0], f) for f in frames[1:])  # it moves

        # And the view's own shape, when that is what is wanted.
        own = next(iter(export.frames_from_view(plotter, animator, n_frames=1)))
        assert own.shape[:2] == tuple(reversed(export._window_size(plotter)))
    finally:
        plotter.close()


def test_an_animation_leaves_each_frame_in_place_rather_than_clearing_it(nacl, tmp_path):
    """Only the first frame covers the whole canvas; every one after it is
    written as the rectangle that changed. ``disposal=2`` restores the rest to
    the *viewer's* idea of the background — black, in Preview and others,
    whatever the file says — so the first frame looked right and all the others
    came out framed in black. Left in place, there is nothing to restore."""
    mode = PhononMode(120.0, np.array([[1, 0, 0], [-1, 0, 0]], float))
    out = tmp_path / "cycle.gif"

    export.save_animation(
        export.render_animation_frames(nacl, nacl.positions, mode, RenderSettings(),
                                       n_frames=4, window_size=(80, 60)),
        str(out),
    )

    # Read the disposal straight out of each Graphic Control Extension: Pillow
    # does not report back what it wrote.
    data = out.read_bytes()
    found, at = [], 0
    while True:
        at = data.find(b"\x21\xf9\x04", at)
        if at < 0:
            break
        found.append((data[at + 3] >> 2) & 0x07)
        at += 4
    assert found and all(method == 1 for method in found), found


def test_a_frame_is_fitted_to_the_size_asked_for_not_cropped_to_it():
    """The frames are the view's shape, and a view with docks either side is
    taller than it is wide: asked for 4:3 it came out portrait, and every viewer
    letterboxes a portrait animation in black. Fitted inside the size and padded
    with the view's own background, it is the size asked for and looks like a
    wider view — which only works because that background is one flat colour."""
    frame = np.zeros((80, 40, 3), np.uint8)
    frame[:] = (10, 20, 30)

    out = export.fit_frame(frame, (200, 100), background=(255, 255, 255))

    assert out.shape == (100, 200, 3)
    assert out[0, 0].tolist() == [255, 255, 255]       # padded, not stretched
    assert out[50, 100].tolist() == [10, 20, 30]       # and the picture is still there
    # Taller than wide stays taller than wide: it is fitted, never squashed.
    inside = np.argwhere((out != 255).any(axis=2))
    height = int(inside[:, 0].max() - inside[:, 0].min()) + 1
    width = int(inside[:, 1].max() - inside[:, 1].min()) + 1
    assert height > width
    assert abs(width / height - 40 / 80) < 0.05


def test_a_native_crash_leaves_a_stack_behind():
    """VTK or Qt faulting kills the process without raising, and the log the
    launcher keeps then ends mid-sentence with nothing to say where. (A kill for
    memory is SIGKILL and catches nothing — the silence is the evidence.)"""
    import faulthandler
    import inspect

    from crystalline import __main__ as entry

    assert "faulthandler.enable()" in inspect.getsource(entry._leave_a_trace_on_a_native_crash)
    assert "_leave_a_trace_on_a_native_crash()" in inspect.getsource(entry.main)
    entry._leave_a_trace_on_a_native_crash()
    assert faulthandler.is_enabled()
