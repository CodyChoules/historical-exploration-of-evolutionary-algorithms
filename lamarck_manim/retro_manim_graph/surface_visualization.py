"""
Surface-only visualization: Rastrigin in the same style as Lamarckian/Darwinian
scenes (black-on-white, X TRAIT, Y TRAIT, z-axis label, solid surface, dashed
optimum marker) with no organisms.
"""

from manim import *
import numpy as np
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from retro_manim_graph import retro_configuration as retro_configuration_module
sys.modules.setdefault("retro_configuration", retro_configuration_module)
from retro_manim_graph.retro_configuration import (
    get_default_class_config,
    scene_config_overrides,
)
from retro_manim_graph.retro_construction import construct_retro_style_scene

from lamarckian_functions.core import rastrigin_func


class SurfaceVisualization(ThreeDScene):
    """Retro-style Rastrigin surface only: no organisms, same look as Lamarckian/Darwinian scenes."""

    get_default_class_config()

    def construct(self):
        DISPLAY_X_SHIFT = -10.0
        DISPLAY_Y_SHIFT = 15.0

        DISPLAY_SURFACE_FUNC = lambda u, v: rastrigin_func(u, v, scale=0.03)

        svg_dir = _project_root / "media" / "svg"
        preferred_svg = svg_dir / "topology_a892e75fe0607600.svg"
        topology_svg_path = None
        if preferred_svg.is_file():
            topology_svg_path = str(preferred_svg)
        elif svg_dir.is_dir():
            svg_files = list(svg_dir.glob("topology_*.svg"))
            if svg_files:
                topology_svg_path = str(max(svg_files, key=lambda p: p.stat().st_mtime))

        scene_elements = construct_retro_style_scene(
            scene=self,
            surface_func=DISPLAY_SURFACE_FUNC,
            config_overrides=scene_config_overrides(
                CAMERA_PRESET="orthoxyz",
                color_scheme="bw",
                VIEW_SCALE=0.7,
                AXIS_RANGE_MIN=-12.0,
                AXIS_RANGE_MAX=12.0,
                Z_AXIS_RANGE_MIN=-20.0,
                Z_AXIS_RANGE_MAX=10.0,
                SHOW_TITLE=False,
                X_AXIS_TITLE="X TRAIT",
                Y_AXIS_TITLE="Y TRAIT",
                Z_AXIS_TITLE="ADAPTIVE SURFACE VALUE",
                SHOW_SURFACE=True,
                SURFACE_RESOLUTION=(100, 100),
                SURFACE_FILL_OPACITY=1.0,
                GAUSSIAN_AMPLITUDE=2.0,
                CONTOUR_ALWAYS_USE_LINE3D=False,
                NUM_CONTOURS=8,
                CONTOUR_STROKE_WIDTH=0.004,
                CONTOUR_RESOLUTION=220,
                CONTOUR_OPACITY_MAX=0.7,
                CONTOUR_OPACITY_MIN=0.05,
                TICK_LABEL_STRIDE=6,
                Z_TICK_LABEL_STRIDE=5,
                SHOW_MINOR_TICKS=False,
                MINOR_TICKS_PER_INTERVAL=4,
                LABEL_FONT_SIZE=96,
                AXIS_TITLE_FONT_SIZE=96,
                Z_AXIS_TITLE_OFFSET=3.2,
                TITLE_SIZE=108,
                TITLE_RUN_TIME=0.07,
                SHORT_WAIT=0.07,
                MEDIUM_WAIT=0.07,
                LONG_WAIT=0.07,
                ANIMATE_GRAPH=False,
            ),
            topology_svg_path=topology_svg_path,
            topology_svg_cache_dir="media/svg",
            topology_id="rastrigin",
        )

        for mob in self.mobjects:
            mob.shift(np.array([DISPLAY_X_SHIFT, DISPLAY_Y_SHIFT, 0.0]))

        contour_z = scene_elements["config"].Z_AXIS_RANGE_MIN
        cx, cy = float(DISPLAY_X_SHIFT), float(DISPLAY_Y_SHIFT)
        z_top, z_bottom = -0.5, -20.0
        num_dashes = 48
        dash_segments = VGroup()
        for i in range(num_dashes):
            z0 = z_top + (z_bottom - z_top) * i / num_dashes
            z1 = z_top + (z_bottom - z_top) * (i + 1) / num_dashes
            if i % 2 == 0:
                start = np.array([cx, cy, z0])
                end = np.array([cx, cy, z1])
                seg = Line(start=start, end=end, color=BLACK, stroke_width=2.0)
                dash_segments.add(seg)
        optima_marker = dash_segments
        self.add(optima_marker)

        self.wait(0.5)
