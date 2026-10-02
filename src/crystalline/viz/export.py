"""Export the 3D view: still images and phonon-mode animations.

* **Stills** are captured from a live plotter (WYSIWYG) — raster via
  ``screenshot`` (PNG/JPEG/…) or vector via ``save_graphic`` (SVG/PDF/EPS/…).
* **Animations** are rendered on a *fresh off-screen* plotter so the live view
  isn't disturbed and the whole thing stays testable headlessly. The same
  :class:`StructureRenderer` + :class:`PhononAnimator` are driven over one full
  vibration cycle; frames are written as an animated GIF (via Pillow, no extra
  deps), an MP4/video (via imageio-ffmpeg, if installed), or a PNG sequence.

Qt-free on purpose: the UI hands in the structure/mode/settings and a file path.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pyvista as pv

from crystalline.core.phonons import PhononMode
from crystalline.viz.phonon_animator import DEFAULT_AMPLITUDE, PhononAnimator
from crystalline.viz.renderer import StructureRenderer
from crystalline.viz.render_settings import RenderSettings

# Formats ``pyvista`` writes as a raster screenshot vs. a vector graphic.
RASTER_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VECTOR_IMAGE_EXTS = (".svg", ".eps", ".ps", ".pdf", ".tex")
# Animation containers.
GIF_EXTS = (".gif",)
MOVIE_EXTS = (".mp4", ".mov", ".avi", ".webm", ".mkv")

# Defaults for an exported loop: smooth enough, small enough.
DEFAULT_FRAMES = 36
DEFAULT_FPS = 18


# ── still image ─────────────────────────────────────────────────────────────
def capture_view(plotter, *, scale: int = 1, transparent: bool = False) -> np.ndarray:
    """The plotter's current view as an ``(h, w, 3|4)`` array, drawn fresh.

    Not ``plotter.screenshot``: pyvista configures its capture with
    ``ReadFrontBufferOff()`` and then turns it back on two lines later, so what
    comes out is the *front* buffer — the pixels as they stand on the screen.
    Anything over the window is captured with it, and whatever is not composited
    comes back black, which is where the black borders on an exported animation
    came from. Reading the back buffer takes the frame that was just drawn,
    whatever is in front of the window.

    ``FixBoundaryOn`` is what keeps a supersampled capture seamless: above
    ``scale=1`` VTK renders the view in tiles, and without it the seams show.
    """
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

    window = plotter.render_window if hasattr(plotter, "render_window") else plotter.ren_win
    filt = vtkWindowToImageFilter()
    filt.SetInput(window)
    filt.SetScale(max(1, int(scale)))
    filt.FixBoundaryOn()
    # Drawn here rather than by the filter: an off-screen window has no front
    # buffer to fall back on, and asking the filter to re-render one takes VTK
    # down. Render first, then read the back buffer it just filled.
    window.Render()
    filt.ShouldRerenderOff()
    filt.ReadFrontBufferOff()    # the frame just drawn, not the screen
    if transparent:
        filt.SetInputBufferTypeToRGBA()
    else:
        filt.SetInputBufferTypeToRGB()
    filt.Modified()
    filt.Update()

    image = filt.GetOutput()
    width, height, _ = image.GetDimensions()
    values = vtk_to_numpy(image.GetPointData().GetScalars())
    # VTK counts rows from the bottom; an image is written from the top.
    return values.reshape(height, width, -1)[::-1]


def save_view_image(
    plotter, path: str, *, scale: int = 1, transparent: bool = False
) -> str:
    """Save the plotter's current view to ``path`` (raster or vector by extension).

    ``scale`` supersamples a raster screenshot (2 → twice the pixels each way,
    the 3D equivalent of a higher DPI); ``transparent`` drops the background so
    the structure sits on an alpha channel. Both apply to raster formats only —
    vector output (``save_graphic``) has neither knob and ignores them.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in VECTOR_IMAGE_EXTS:
        plotter.save_graphic(path)
    elif ext in RASTER_IMAGE_EXTS:
        from PIL import Image

        # JPEG has no alpha channel to drop the background into; asking for one
        # fails at the write rather than at the dialog, so it is dropped here.
        alpha = transparent and ext not in (".jpg", ".jpeg")
        Image.fromarray(capture_view(plotter, scale=scale, transparent=alpha)).save(path)
    else:
        raise ValueError(
            f"unsupported image format '{ext}'. Use one of "
            f"{', '.join(RASTER_IMAGE_EXTS + VECTOR_IMAGE_EXTS)}."
        )
    return path


# ── animation ───────────────────────────────────────────────────────────────
def render_animation_frames(
    structure,
    equilibrium: np.ndarray,
    mode: PhononMode,
    settings: Optional[RenderSettings] = None,
    *,
    amplitude: float = DEFAULT_AMPLITUDE,
    n_frames: int = DEFAULT_FRAMES,
    reference_cell=None,
    bond_structure=None,
    camera=None,
    window_size=(800, 600),
    on_frame=None,
) -> List[np.ndarray]:
    """Render one full vibration cycle off-screen; return a list of RGB frames.

    A fresh off-screen plotter/renderer is used so nothing about the live view
    changes. ``camera`` reproduces the on-screen view when supplied; otherwise the
    structure is auto-framed. It may be either a plain pyvista ``camera_position``
    or a ``(camera_position, parallel_scale, view_angle)`` snapshot — the latter is
    needed under parallel (orthographic) projection, where the zoom lives in the
    parallel scale, *not* in ``camera_position`` (so passing only the placement
    would let the off-screen plotter reframe and zoom the view).

    ``on_frame(done, total)`` is called after each frame, for a caller that has
    something to show for the wait. It is called on this thread, between
    frames, so it must be quick and must not spin the event loop.
    """
    return list(animation_frames(
        structure, equilibrium, mode, settings, amplitude=amplitude,
        n_frames=n_frames, reference_cell=reference_cell,
        bond_structure=bond_structure, camera=camera, window_size=window_size,
        on_frame=on_frame,
    ))


def animation_frames(
    structure,
    equilibrium: np.ndarray,
    mode: PhononMode,
    settings: Optional[RenderSettings] = None,
    *,
    amplitude: float = DEFAULT_AMPLITUDE,
    n_frames: int = DEFAULT_FRAMES,
    reference_cell=None,
    bond_structure=None,
    camera=None,
    window_size=(800, 600),
    on_frame=None,
):
    """The same cycle, yielded one frame at a time as it is rendered.

    What :func:`render_animation_frames` returns in a list, which is where an
    export of any size went wrong: 240 frames at 1920x1440 is two gigabytes held
    at once, and the writer then made its own copy of every one — six and a half
    gigabytes for a single export, on a machine with sixteen. The system does
    not report that as an error; it kills the process, which looks exactly like
    a crash and leaves nothing behind to say otherwise.

    Taken one at a time by :func:`save_animation`, a frame is rendered, written
    and let go, and the peak is one frame.
    """
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    try:
        renderer = StructureRenderer(plotter, settings)
        renderer.set_reference_cell(reference_cell)
        renderer.set_structure(structure, bond_structure=bond_structure)
        if camera is not None:
            _apply_camera(plotter, camera)
        else:
            plotter.reset_camera()

        animator = PhononAnimator(renderer)
        animator.amplitude = amplitude
        animator.set_mode(np.asarray(equilibrium, dtype=float), mode)

        done = 0
        for phase in PhononAnimator.phase_sequence(n_frames):
            animator.set_frame(phase)
            frame = np.asarray(plotter.screenshot(return_img=True))
            done += 1
            if on_frame is not None:
                on_frame(done, n_frames)
            yield frame
    finally:
        plotter.close()


def fit_frame(frame: np.ndarray, size, background=(255, 255, 255)) -> np.ndarray:
    """``frame`` at exactly ``size``, as large as it goes, the rest background.

    The frames come from the view on screen, so their shape is the view's — and
    a view with the docks either side of it is taller than it is wide. Asked for
    a 4:3 animation, that comes out portrait, and every viewer letterboxes it in
    black. Scaled down to fit and padded with the view's own background, it
    comes out the size that was asked for and looks like a wider view, which is
    what the background being one flat colour buys.
    """
    from PIL import Image

    want_w, want_h = (max(1, int(v)) for v in size)
    image = Image.fromarray(frame)
    image.thumbnail((want_w, want_h), Image.LANCZOS)   # fit inside, keep the aspect
    out = Image.new(image.mode, (want_w, want_h), tuple(background)[:len(image.getbands())])
    out.paste(image, ((want_w - image.width) // 2, (want_h - image.height) // 2))
    return np.asarray(out)


def view_background(plotter) -> tuple:
    """The plotter's background as 8-bit RGB, for padding a frame out to size."""
    try:
        from pyvista import Color

        return tuple(Color(plotter.background_color).int_rgb)
    except Exception:  # noqa: BLE001 - a plotter that cannot say; white is the default
        return (255, 255, 255)


def frames_from_view(plotter, animator, *, n_frames: int = DEFAULT_FRAMES,
                     size=None, on_frame=None):
    """Yield one cycle, drawn by the view that is already on screen.

    The alternative — :func:`animation_frames`, which opens an off-screen
    plotter of its own — is a second VTK render window beside the live one, and
    on macOS the two are not independent: with both alive, the first draw after
    the export faults in C++, outside anything Python can catch. An image export
    has never done it, and the only thing it does differently is this.

    So the frames come from the live view, supersampled enough to cover ``size``
    and then fitted to it (:func:`fit_frame`) — the animation is the size that
    was asked for, whatever shape the window happens to be. ``size`` of ``None``
    takes the view as it is. The view really does move while this runs; the
    caller puts it back afterwards.
    """
    scale, background = 1, view_background(plotter)
    if size is not None:
        width, height = _window_size(plotter)
        # Enough to downsample from, never up: an animation blown up from fewer
        # pixels than it claims is a soft one.
        scale = max(1, int(np.ceil(max(size[0] / max(1, width), size[1] / max(1, height)))))
    for done, phase in enumerate(PhononAnimator.phase_sequence(n_frames), start=1):
        animator.set_frame(phase)
        frame = capture_view(plotter, scale=scale)
        yield frame if size is None else fit_frame(frame, size, background)
        if on_frame is not None:
            on_frame(done, n_frames)


def _window_size(plotter) -> tuple:
    """The render window's size in the pixels it actually draws."""
    window = plotter.render_window if hasattr(plotter, "render_window") else plotter.ren_win
    try:
        width, height = window.GetSize()
        if width and height:
            return int(width), int(height)
    except Exception:  # noqa: BLE001 - no window yet
        pass
    return (800, 600)


def _apply_camera(plotter, camera) -> None:
    """Reproduce ``camera`` on ``plotter`` — placement, and zoom if provided.

    ``camera`` is either a bare ``camera_position`` (placement only) or a
    ``(camera_position, parallel_scale, view_angle)`` snapshot. Applying the
    parallel scale is what keeps a parallel-projection render at the on-screen zoom.
    The two are told apart by the second element: a scalar (the parallel scale) for
    the snapshot, a focal-point vector for a bare ``camera_position``.
    """
    if np.isscalar(camera[1]):
        position, parallel_scale, view_angle = camera
        plotter.camera_position = position
        plotter.camera.SetParallelScale(parallel_scale)
        plotter.camera.SetViewAngle(view_angle)
    else:
        plotter.camera_position = camera


def save_animation(frames, path: str, *, fps: int = DEFAULT_FPS) -> List[str]:
    """Write ``frames`` to ``path``; format chosen by extension.

    ``.gif`` → animated GIF (Pillow); a video extension → MP4/… (imageio-ffmpeg);
    a raster image extension → a numbered PNG-style sequence (``stem_000.ext`` …).
    Returns the list of files written.

    ``frames`` may be a list or any iterable, and every writer below takes them
    one at a time: handed :func:`animation_frames` directly, an export of any
    length holds one frame rather than all of them.
    """
    frames = iter(frames)
    try:
        first = next(frames)
    except StopIteration:
        raise ValueError("no frames to write") from None
    frames = _chain(first, frames)
    ext = os.path.splitext(path)[1].lower()
    if ext in GIF_EXTS:
        _save_gif(frames, path, fps)
        return [path]
    if ext in MOVIE_EXTS:
        _save_movie(frames, path, fps)
        return [path]
    if ext in RASTER_IMAGE_EXTS:
        return _save_frame_sequence(frames, path)
    raise ValueError(
        f"unsupported animation format '{ext}'. Use .gif, a video "
        f"({', '.join(MOVIE_EXTS)}), or an image extension for a frame sequence."
    )


def _chain(first, rest):
    """``first`` and then ``rest`` — the first frame put back after the look."""
    yield first
    for item in rest:
        yield item


def _save_gif(frames, path, fps) -> None:
    """Write the GIF, converting each frame as Pillow asks for it.

    ``append_images`` is walked during the save, so a generator here is a
    generator all the way down: the whole cycle was converted to PIL images
    first, which doubled an already large pile of arrays.
    """
    from PIL import Image

    first = Image.fromarray(next(frames))
    duration = max(1, int(round(1000.0 / max(1, fps))))  # ms per frame
    first.save(
        path,
        save_all=True,
        append_images=(Image.fromarray(frame) for frame in frames),
        duration=duration,
        loop=0,  # loop forever
        # Leave each frame on the canvas for the next one to paint over, rather
        # than restoring the area around it to the background ("disposal=2").
        # Only the first frame covers the whole canvas; every frame after it is
        # written as the rectangle that changed, and what disposal=2 restores
        # the rest to is the *viewer's* idea of the background — black, in
        # Preview and others, whatever the file says. Nothing is left showing
        # through here: the area outside a delta is identical by construction.
        disposal=1,
    )


VIDEO_MISSING_MESSAGE = (
    "Video export needs the 'imageio-ffmpeg' package, which ships the ffmpeg "
    "encoder:\n\n    pip install imageio-ffmpeg\n\n"
    "GIF and frame-sequence export need no extras."
)


def video_export_available() -> bool:
    """Whether an ffmpeg backend is installed, so a video can actually be written.

    Checked *before* a mode is rendered rather than after: an animation is
    dozens of off-screen frames, and discovering the encoder is missing at the
    end means the whole wait was wasted.
    """
    try:
        import imageio_ffmpeg  # noqa: F401 - presence is the whole question
    except Exception:  # noqa: BLE001 - not installed, or installed but broken
        return False
    return True


def _save_movie(frames, path, fps) -> None:
    """Encode ``frames`` to a video container via imageio's ffmpeg plugin.

    The plugin is named explicitly. Left to itself, ``imageio.get_writer`` picks
    whatever plugin claims the extension, and with imageio-ffmpeg absent an
    ``.mp4`` lands on one that cannot encode video at all — which surfaced as
    ``TypeError: write() got an unexpected keyword argument 'fps'`` instead of
    anything a user could act on.
    """
    if not video_export_available():
        raise RuntimeError(VIDEO_MISSING_MESSAGE)
    import imageio

    frames = (_even_sized(np.asarray(frame)) for frame in frames)
    # macro_block_size=1 keeps the resolution that was asked for: the default
    # (16) silently rescales the whole movie up to the next multiple of 16.
    # h264 still needs even dimensions, which _even_sized guarantees.
    writer = imageio.get_writer(
        path, format="FFMPEG", fps=fps, macro_block_size=1, codec=_codec_for(path)
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


# Containers that cannot carry the default H.264 stream. Handing ffmpeg a codec
# its container rejects does not fail: it writes an empty file and reports
# success, so a .webm export came out as a few hundred bytes of header.
_CONTAINER_CODECS = {".webm": "libvpx-vp9"}


def _codec_for(path: str) -> str:
    """The video codec to encode ``path`` with, from its container."""
    return _CONTAINER_CODECS.get(os.path.splitext(path)[1].lower(), "libx264")


def _even_sized(frame: np.ndarray) -> np.ndarray:
    """Trim a frame to even width and height (h264 encodes nothing else).

    A row or column at most, off the bottom/right — invisible, and preferable to
    the alternative of rescaling every frame.
    """
    height, width = frame.shape[:2]
    return frame[: height - (height % 2), : width - (width % 2)]


def _save_frame_sequence(frames, path) -> List[str]:
    from PIL import Image

    directory = os.path.dirname(path) or "."
    stem, ext = os.path.splitext(os.path.basename(path))
    written = []
    for i, frame in enumerate(frames):
        out = os.path.join(directory, f"{stem}_{i:03d}{ext}")
        Image.fromarray(frame).save(out)
        written.append(out)
    return written


__all__ = [
    "RASTER_IMAGE_EXTS",
    "VECTOR_IMAGE_EXTS",
    "GIF_EXTS",
    "MOVIE_EXTS",
    "VIDEO_MISSING_MESSAGE",
    "save_view_image",
    "animation_frames",
    "capture_view",
    "fit_frame",
    "frames_from_view",
    "view_background",
    "render_animation_frames",
    "save_animation",
    "video_export_available",
]
