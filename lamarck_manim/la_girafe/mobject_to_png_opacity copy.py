"""
Mobject rendered to PNG then shown at a single opacity.

Use when overlapping mobjects cause opacity to "bleed" and darken overlaps:
render the figure in a one-frame scene to PNG, then add that image with
set_opacity() so one opacity applies to the whole image.

Example (static giraffes in GiraffeMoveScene):
  path_low, path_high = ensure_static_giraffe_image_paths()
  scene.add(ImageMobject(str(path_low)).move_to(6 * LEFT).set_opacity(0.5))
  scene.add(ImageMobject(str(path_high)).move_to(2 * LEFT).set_opacity(0.5))

To (re)generate the static giraffe PNGs from project root:
  manim -ql -s la_girafe/mobject_to_png_opacity.py GiraffeStillLow GiraffeStillHigh
"""

import sys
import subprocess
from pathlib import Path

# Project root for media/images and subprocess cwd.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from manim import ORIGIN, GOLD, ImageMobject, Scene, WHITE, config

from la_girafe.giraffe import giraffe_line_mobjects

# Scene names used to render static giraffe PNGs (one frame each, giraffe at ORIGIN).
GIRAFFE_STILL_LOW = "GiraffeStillLow"
GIRAFFE_STILL_HIGH = "GiraffeStillHigh"


def find_png_under(media_images_root: Path, scene_name: str):
    """Return path to a PNG whose name starts with scene_name under media/images, or None."""
    if not media_images_root.exists():
        return None
    for p in media_images_root.rglob(f"{scene_name}*.png"):
        return p
    return None


def ensure_static_giraffe_image_paths():
    """Return (path_low, path_high) for static giraffe PNGs, creating them via subprocess if missing."""
    media_images = _project_root / "media" / "images"
    path_low = find_png_under(media_images, GIRAFFE_STILL_LOW)
    path_high = find_png_under(media_images, GIRAFFE_STILL_HIGH)
    if path_low is not None and path_high is not None:
        return path_low, path_high
    # Render this module's still scenes so callers can use them as half-opacity images.
    scene_file = Path(__file__).resolve()
    cmd = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "-s",
        str(scene_file),
        GIRAFFE_STILL_LOW,
        GIRAFFE_STILL_HIGH,
    ]
    subprocess.run(cmd, cwd=str(_project_root), check=True)
    path_low = find_png_under(media_images, GIRAFFE_STILL_LOW)
    path_high = find_png_under(media_images, GIRAFFE_STILL_HIGH)
    if path_low is None or path_high is None:
        raise FileNotFoundError(
            "Static giraffe PNGs were not created. Run: manim -ql -s "
            f"la_girafe/mobject_to_png_opacity.py {GIRAFFE_STILL_LOW} {GIRAFFE_STILL_HIGH}"
        )
    return path_low, path_high


def image_at_position_opacity(path, position, opacity):
    """Return an ImageMobject from path, placed at position, with the given opacity.
    Uses the current render resolution so the image is not scaled down (ImageMobject
    defaults to scale_to_resolution=1080, which makes -ql renders look half size)."""
    try:
        scale_to_resolution = config.pixel_height
    except (AttributeError, KeyError, TypeError):
        scale_to_resolution = 1080
    return (
        ImageMobject(str(path), scale_to_resolution=scale_to_resolution)
        .move_to(position)
        .set_opacity(opacity)
    )


class GiraffeStillLow(Scene):
    """Single-frame scene: giraffe at ORIGIN with neck/leg 0.2. Rendered to PNG for half-opacity use."""

    def construct(self):
        self.camera.background_color = WHITE
        giraffe = giraffe_line_mobjects(
            color=GOLD,
            location=ORIGIN,
            stroke_width=9,
            scale=1,
            giraffe_direction="right",
            neck_length=0.2,
            leg_length=0.2,
        )
        self.add(giraffe)


class GiraffeStillHigh(Scene):
    """Single-frame scene: giraffe at ORIGIN with neck/leg 1.0. Rendered to PNG for half-opacity use."""

    def construct(self):
        self.camera.background_color = WHITE
        giraffe = giraffe_line_mobjects(
            color=GOLD,
            location=ORIGIN,
            stroke_width=9,
            scale=1,
            giraffe_direction="right",
            neck_length=1.0,
            leg_length=1.0,
        )
        self.add(giraffe)
