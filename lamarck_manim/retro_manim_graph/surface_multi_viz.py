"""
Surface multi-viz: four surfaces in the same retro style.
Left: Rastrigin. Right: Rosenbrock. Below right: Ackley. East of third: Himmelblau.
Black-on-white, X/Y TRAIT axes, solid surface, contours; z normalized to [0, 10].

Contour SVG naming (condensed, used for cache lookup/save):
  {fn}_{xmin}_{xmax}_r{res}_n{ncont}_m{method}.svg
  - fn: function name (rastrigin, rosenbrock, ackley, himmelblau)
  - xmin, xmax: AXIS_RANGE_MIN, AXIS_RANGE_MAX (int)
  - res: CONTOUR_RESOLUTION
  - ncont: NUM_CONTOURS
  - method: CONTOUR_METHOD (auto, scipy, skimage, ms for marching_squares)
  Example: rastrigin_-12_12_r220_n8_mauto.svg
  SVGs are searched for and saved under CONTOUR_SVG_DIR (see below).
"""

from manim import *
import numpy as np
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent

# Directory for contour SVGs: we look here before generating and save here when generating.
# Naming scheme: see module docstring (legend above).
CONTOUR_SVG_DIR = _project_root / "media" / "svg" / "surface_multi_viz"


def _contour_svg_basename(func_name: str, config) -> str:
    """Condensed filename for contour SVG: {fn}_{xmin}_{xmax}_r{res}_n{ncont}_m{method}.svg"""
    xmin = int(getattr(config, "AXIS_RANGE_MIN", -12))
    xmax = int(getattr(config, "AXIS_RANGE_MAX", 12))
    res = getattr(config, "CONTOUR_RESOLUTION", 220)
    ncont = getattr(config, "NUM_CONTOURS", 8)
    method = (getattr(config, "CONTOUR_METHOD", None) or "auto").strip().lower()
    if method == "marching_squares":
        method = "ms"
    return f"{func_name}_{xmin}_{xmax}_r{res}_n{ncont}_m{method}.svg"


def _contour_svg_path(func_name: str, config) -> Path:
    """Full path for contour SVG (lookup or save). Legend: see module docstring."""
    return CONTOUR_SVG_DIR / _contour_svg_basename(func_name, config)


def _print_contour_svg_status(surface_label: str, func_name: str, path: Path, config, creating: bool = False):
    """Print SVG lookup/create status for one surface."""
    method = (getattr(config, "CONTOUR_METHOD", None) or "auto").strip().lower()
    name = path.name
    location = str(path.resolve())
    found = path.is_file()
    print(f"[{surface_label}] {func_name} | SVG name: {name} | location: {location}")
    print(f"  -> Found: {'Yes' if found else 'No'}")
    if not found:
        if creating:
            print(f"  -> Creating contours (method={method}, resolution={getattr(config, 'CONTOUR_RESOLUTION', 220)}, num_contours={getattr(config, 'NUM_CONTOURS', 8)}) and saving to: {location}")
        else:
            print(f"  -> Will create contours (method={method}) and save to: {location}")


def _load_contour_svg_for_group(svg_path: Path, config):
    """Load contour SVG and style/position it for adding to a group (origin-centered, z_plane)."""
    contour_svg = SVGMobject(str(svg_path))
    contour_stroke_width = getattr(config, "CONTOUR_STROKE_WIDTH", 0.001)
    contour_color = getattr(config, "CONTOUR_COLOR", getattr(config, "FOREGROUND_COLOR", None))
    if contour_color is not None:
        contour_svg.set_color(contour_color)
        contour_svg.set_stroke(color=contour_color)
    op_max = getattr(config, "CONTOUR_OPACITY_MAX", 1.0)
    op_min = getattr(config, "CONTOUR_OPACITY_MIN", 0.2)
    n_paths = len(contour_svg)
    for idx, sub in enumerate(contour_svg):
        op = op_max - (op_max - op_min) * (idx / max(1, n_paths - 1))
        sub.set_fill(opacity=0)
        sub.set_stroke(opacity=op)
    axis_span = config.AXIS_RANGE_MAX - config.AXIS_RANGE_MIN
    z_plane = getattr(config, "Z_AXIS_RANGE_MIN", -10.0)
    contour_svg.set_width(axis_span)
    contour_svg.set_height(axis_span)
    contour_svg.set_stroke(width=max(0.1, contour_stroke_width * 100))
    contour_svg.move_to(np.array([0, 0, z_plane]))
    return contour_svg


def _make_surface_title(text: str, anchor: np.ndarray, config, font_family) -> "Text":
    """Create a surface title with same style/rotations as Rastrigin title (centered on anchor, above z)."""
    title = Text(
        text,
        font_size=config.AXIS_TITLE_FONT_SIZE * 3,
        color=config.FOREGROUND_COLOR,
        font=font_family,
    )
    title.move_to(anchor)
    title.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=title.get_center())
    title.rotate(90 * DEGREES, axis=np.array([0, 1, 0]), about_point=title.get_center())
    title.flip(axis=np.array([0, 1, 0]), about_point=title.get_center())
    title.rotate(90 * DEGREES, axis=np.array([0, 1, 0]), about_point=title.get_center())
    title.shift(anchor - title.get_center())
    return title


def _optimum_dashed_line(
    opt_x: float,
    opt_y: float,
    opt_z: float,
    shift_x: float,
    shift_y: float,
    z_min: float = -20.0,
    dash_length: float = 1.0,
    color=RED,
) -> "VGroup":
    """Vertical dashed line from (opt_x, opt_y, opt_z) to (opt_x, opt_y, z_min) in scene coords."""
    start = np.array([shift_x + opt_x, shift_y + opt_y, opt_z])
    end = np.array([shift_x + opt_x, shift_y + opt_y, z_min])
    total_len = opt_z - z_min
    gap_length = dash_length
    seg_len = dash_length
    group = VGroup()
    z = float(opt_z)
    while z > z_min:
        z_end = max(z_min, z - seg_len)
        seg_start = np.array([shift_x + opt_x, shift_y + opt_y, z])
        seg_end = np.array([shift_x + opt_x, shift_y + opt_y, z_end])
        group.add(Line3D(start=seg_start, end=seg_end, color=color, thickness=0.04))
        z = z_end - gap_length
        if z <= z_min:
            break
    return group


def _optimum_horizontal_dashed_line(
    opt_x: float,
    opt_y: float,
    shift_x: float,
    shift_y: float,
    z_min: float,
    max_x: float,
    dash_length: float = 1.0,
    color=RED,
    opacity: float = 0.5,
) -> "VGroup":
    """Horizontal dashed line at z_min from (opt_x, opt_y) to (max_x, opt_y) in scene coords; stops at axis max so only the label sits at max_x + 2."""
    start = np.array([shift_x + opt_x, shift_y + opt_y, z_min])
    end = np.array([shift_x + max_x, shift_y + opt_y, z_min])
    total_len = abs(max_x - opt_x)
    if total_len < 1e-6:
        return VGroup()
    gap_length = dash_length
    seg_len = dash_length
    direction = (end - start) / total_len
    group = VGroup()
    t = 0.0
    while t < total_len:
        t_end = min(total_len, t + seg_len)
        seg_start = start + t * direction
        seg_end = start + t_end * direction
        seg = Line3D(start=seg_start, end=seg_end, color=color, thickness=0.04)
        seg.set_opacity(opacity)
        group.add(seg)
        t = t_end + gap_length
    return group


def _format_opt_coord(x: float, y: float) -> str:
    """Format (x, y) for display: integers without decimals, else 2 decimals."""
    if abs(x - round(x)) < 1e-6 and abs(y - round(y)) < 1e-6:
        return f"({int(round(x))}, {int(round(y))})"
    return f"({x:.2f}, {y:.2f})"


def _make_optimum_coord_label(
    opt_x: float,
    opt_y: float,
    shift_x: float,
    shift_y: float,
    z_min: float,
    max_x: float,
    config,
    font_family: str,
) -> "Text":
    """Text label at max x + 3 (decoupled from dashed line end at max_x); larger and bold."""
    label = Text(
        _format_opt_coord(opt_x, opt_y),
        font_size=config.AXIS_TITLE_FONT_SIZE * 1.5,
        color=RED,
        font=font_family,
        weight="BOLD",
    )
    pos = np.array([shift_x + max_x + 3, shift_y + opt_y, z_min])
    label.move_to(pos)
    return label


if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from retro_manim_graph import retro_configuration as retro_configuration_module
sys.modules.setdefault("retro_configuration", retro_configuration_module)
from retro_manim_graph.retro_configuration import (
    get_default_class_config,
    scene_config_overrides,
    build_config_for_scene,
    create_back_style_graph,
    create_contour_lines,
)
from retro_manim_graph.retro_construction import construct_retro_style_scene

from lamarckian_functions.core import rastrigin_func


def _rosenbrock_func(u, v, a=1, b=100, scale=0.01):
    """Rosenbrock f(x,y) = (a-x)² + b(y-x²)². Returns [x, y, z] for retro surface."""
    x, y = u, v
    z = ((a - x) ** 2 + b * (y - x ** 2) ** 2) * scale
    return np.array([x, y, z])


def _ackley_func(u, v, a=20, b=0.2, c=2 * np.pi, scale=0.1):
    """Ackley: -a*exp(-b*sqrt((x²+y²)/2)) - exp((cos(cx)+cos(cy))/2) + a + e. Returns [x, y, z]."""
    x, y = u, v
    term1 = -a * np.exp(-b * np.sqrt((x ** 2 + y ** 2) / 2))
    term2 = -np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
    z = (term1 + term2 + a + np.e) * scale
    return np.array([x, y, z])


def _himmelblau_func(u, v, scale=0.01):
    """Himmelblau: (x² + y - 11)² + (x + y² - 7)². Four equal global minima. Returns [x, y, z]."""
    x, y = u, v
    z = ((x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2) * scale
    return np.array([x, y, z])


def _surface_z_to_0_10(surface_func, u_range, v_range, sample_res=32):
    """
    Return a surface function (u, v) -> [u, v, z] with z normalized to [0, 10]
    over the given domain. Samples the function on a grid to get min/max z.
    """
    u_min, u_max = u_range[0], u_range[1]
    v_min, v_max = v_range[0], v_range[1]
    us = np.linspace(u_min, u_max, sample_res)
    vs = np.linspace(v_min, v_max, sample_res)
    z_min, z_max = np.inf, -np.inf
    for u in us:
        for v in vs:
            pt = surface_func(u, v)
            z = pt[2] if len(pt) >= 3 else pt[1]
            z_min = min(z_min, z)
            z_max = max(z_max, z)
    span = z_max - z_min
    if span <= 0:
        span = 1.0

    def normalized(u, v):
        pt = surface_func(u, v)
        x, y = pt[0], pt[1]
        z = pt[2] if len(pt) >= 3 else pt[1]
        z_norm = 0.0 + 10.0 * (z - z_min) / span
        return np.array([x, y, z_norm])

    return normalized


class SurfaceMultiViz(ThreeDScene):
    """Four retro-style surfaces: Rastrigin (left), Rosenbrock (right), Ackley (below right), Himmelblau (east of third)."""

    get_default_class_config()

    def construct(self):
        # Spacing: gap between surfaces (equidistant horizontal and vertical)
        SURFACE_SPACING = 55.0  # horizontal gap between 1 and 2, and vertical gap from 2 to 3
        DISPLAY_X_OFFSET = -40.0  # move entire layout this much in x
        DISPLAY_X_SHIFT_LEFT = -SURFACE_SPACING / 2 + DISPLAY_X_OFFSET
        DISPLAY_X_SHIFT_RIGHT = SURFACE_SPACING / 2 + DISPLAY_X_OFFSET
        HORIZONTAL_GAP = SURFACE_SPACING
        DISPLAY_Y_SHIFT = SURFACE_SPACING / 2
        DISPLAY_Y_SHIFT_BELOW = DISPLAY_Y_SHIFT - HORIZONTAL_GAP

        # Z axis range for all surfaces
        Z_DISPLAY_MIN = -20.0
        Z_DISPLAY_MAX = 10.0
        axis_range = (-12.0, 12.0)

        shared_overrides = scene_config_overrides(
            CAMERA_PRESET="custom",
            CAMERA_PHI_CUSTOM=45,
            CAMERA_THETA_CUSTOM=225,
            CAMERA_ZOOM_CUSTOM=0.1,
            CAMERA_FOCAL_DISTANCE_CUSTOM=100000.0,
            color_scheme="bw",
            VIEW_SCALE=1.3,
            AXIS_RANGE_MIN=axis_range[0],
            AXIS_RANGE_MAX=axis_range[1],
            Z_AXIS_RANGE_MIN=Z_DISPLAY_MIN,
            Z_AXIS_RANGE_MAX=Z_DISPLAY_MAX,
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
        )

        config = build_config_for_scene(self, shared_overrides)

        # ----- Left: Rastrigin (full scene via construct_retro_style_scene) -----
        # Contour SVG: lookup by naming scheme (see legend at top of file and CONTOUR_SVG_DIR).
        contour_path_s1 = _contour_svg_path("rastrigin", config)
        CONTOUR_SVG_DIR.mkdir(parents=True, exist_ok=True)
        _print_contour_svg_status("Surface 1", "Rastrigin", contour_path_s1, config, creating=not contour_path_s1.is_file())
        topology_svg_path_s1 = str(contour_path_s1) if contour_path_s1.is_file() else None
        topology_svg_save_path_s1 = str(contour_path_s1)

        rastrigin_raw = lambda u, v: rastrigin_func(u, v, scale=0.03)
        rastrigin_surface_func = _surface_z_to_0_10(
            rastrigin_raw, u_range=axis_range, v_range=axis_range
        )
        scene_elements = construct_retro_style_scene(
            scene=self,
            surface_func=rastrigin_surface_func,
            config=config,
            config_overrides=None,
            topology_svg_path=topology_svg_path_s1,
            topology_svg_save_path=topology_svg_save_path_s1,
        )

        for mob in self.mobjects:
            mob.shift(np.array([DISPLAY_X_SHIFT_LEFT, DISPLAY_Y_SHIFT, 0.0]))

        config = scene_elements["config"]

        # Titles for all surfaces: centered on x-axis midpoint, above highest z (same style)
        x_mid = (config.AXIS_RANGE_MIN + config.AXIS_RANGE_MAX) / 2
        z_anchor_y = config.AXIS_RANGE_MAX
        title_z = Z_DISPLAY_MAX + 8
        title_s1_anchor = np.array([DISPLAY_X_SHIFT_LEFT + x_mid, DISPLAY_Y_SHIFT + z_anchor_y, title_z])
        title_s1 = _make_surface_title("Rastrigin", title_s1_anchor, config, self.FONT_FAMILY)
        self.add(title_s1)
        # Global optimum marker: (0, 0), z=0 on normalized surface → vertical dashed line to z=-20
        self.add(_optimum_dashed_line(0, 0, 0, DISPLAY_X_SHIFT_LEFT, DISPLAY_Y_SHIFT, z_min=Z_DISPLAY_MIN))
        self.add(_optimum_horizontal_dashed_line(0, 0, DISPLAY_X_SHIFT_LEFT, DISPLAY_Y_SHIFT, Z_DISPLAY_MIN, config.AXIS_RANGE_MAX, opacity=0.5))
        self.add(_make_optimum_coord_label(0, 0, DISPLAY_X_SHIFT_LEFT, DISPLAY_Y_SHIFT, Z_DISPLAY_MIN, config.AXIS_RANGE_MAX, config, self.FONT_FAMILY))

        # ----- Right: Rosenbrock (same style, built at origin then shifted) -----
        # Contour SVG: lookup by naming scheme (see CONTOUR_SVG_DIR and legend at top).
        contour_path_s2 = _contour_svg_path("rosenbrock", config)
        _print_contour_svg_status("Surface 2", "Rosenbrock", contour_path_s2, config, creating=not contour_path_s2.is_file())
        rosenbrock_raw = lambda u, v: _rosenbrock_func(u, v, scale=0.01)
        rosenbrock_surface_func = _surface_z_to_0_10(
            rosenbrock_raw, u_range=axis_range, v_range=axis_range
        )
        graph_right = create_back_style_graph(
            config=config,
            foreground_color=config.FOREGROUND_COLOR,
            font_family=self.FONT_FAMILY,
        )
        surface_right = Surface(
            rosenbrock_surface_func,
            u_range=[config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX],
            v_range=[config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX],
            resolution=config.SURFACE_RESOLUTION,
            fill_color=config.SURFACE_FILL_COLOR,
            fill_opacity=config.SURFACE_FILL_OPACITY,
            checkerboard_colors=[config.SURFACE_FILL_COLOR, config.SURFACE_FILL_COLOR],
            stroke_color=config.SURFACE_STROKE_COLOR,
            stroke_width=config.SURFACE_STROKE_WIDTH,
        )
        if contour_path_s2.is_file():
            contour_right = _load_contour_svg_for_group(contour_path_s2, config)
            contour_lines_right = [contour_right]
        else:
            contour_lines_right = create_contour_lines(
                surface_func=rosenbrock_surface_func,
                config=config,
                foreground_color=config.FOREGROUND_COLOR,
                save_svg_path=contour_path_s2,
            )

        right_group = Group()
        for val in graph_right.values():
            if val is None:
                continue
            if isinstance(val, list):
                for item in val:
                    right_group.add(item)
            else:
                right_group.add(val)
        if contour_path_s2.is_file():
            right_group.add(contour_lines_right[0])
        else:
            right_group.add(contour_lines_right)
        right_group.shift(np.array([DISPLAY_X_SHIFT_RIGHT, DISPLAY_Y_SHIFT, 0.0]))
        self.add(right_group)
        surface_right.shift(np.array([DISPLAY_X_SHIFT_RIGHT, DISPLAY_Y_SHIFT, 0.0]))
        self.add(surface_right)
        title_s2_anchor = np.array([DISPLAY_X_SHIFT_RIGHT + x_mid, DISPLAY_Y_SHIFT + z_anchor_y, title_z])
        self.add(_make_surface_title("Rosenbrock", title_s2_anchor, config, self.FONT_FAMILY))
        # Global optimum at (1, 1)
        self.add(_optimum_dashed_line(1, 1, 0, DISPLAY_X_SHIFT_RIGHT, DISPLAY_Y_SHIFT, z_min=Z_DISPLAY_MIN))
        self.add(_optimum_horizontal_dashed_line(1, 1, DISPLAY_X_SHIFT_RIGHT, DISPLAY_Y_SHIFT, Z_DISPLAY_MIN, config.AXIS_RANGE_MAX, opacity=0.5))
        self.add(_make_optimum_coord_label(1, 1, DISPLAY_X_SHIFT_RIGHT, DISPLAY_Y_SHIFT, Z_DISPLAY_MIN, config.AXIS_RANGE_MAX, config, self.FONT_FAMILY))

        # ----- Below right (negative y): Ackley (same style, same x as Rosenbrock) -----
        # Contour SVG: lookup by naming scheme (see CONTOUR_SVG_DIR and legend at top).
        contour_path_s3 = _contour_svg_path("ackley", config)
        _print_contour_svg_status("Surface 3", "Ackley", contour_path_s3, config, creating=not contour_path_s3.is_file())
        ackley_raw = lambda u, v: _ackley_func(u, v, scale=0.1)
        ackley_surface_func = _surface_z_to_0_10(
            ackley_raw, u_range=axis_range, v_range=axis_range
        )
        graph_below = create_back_style_graph(
            config=config,
            foreground_color=config.FOREGROUND_COLOR,
            font_family=self.FONT_FAMILY,
        )
        surface_below = Surface(
            ackley_surface_func,
            u_range=[config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX],
            v_range=[config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX],
            resolution=config.SURFACE_RESOLUTION,
            fill_color=config.SURFACE_FILL_COLOR,
            fill_opacity=config.SURFACE_FILL_OPACITY,
            checkerboard_colors=[config.SURFACE_FILL_COLOR, config.SURFACE_FILL_COLOR],
            stroke_color=config.SURFACE_STROKE_COLOR,
            stroke_width=config.SURFACE_STROKE_WIDTH,
        )
        if contour_path_s3.is_file():
            contour_below = _load_contour_svg_for_group(contour_path_s3, config)
            contour_lines_below = [contour_below]
        else:
            contour_lines_below = create_contour_lines(
                surface_func=ackley_surface_func,
                config=config,
                foreground_color=config.FOREGROUND_COLOR,
                save_svg_path=contour_path_s3,
            )

        below_group = Group()
        for val in graph_below.values():
            if val is None:
                continue
            if isinstance(val, list):
                for item in val:
                    below_group.add(item)
            else:
                below_group.add(val)
        if contour_path_s3.is_file():
            below_group.add(contour_lines_below[0])
        else:
            below_group.add(contour_lines_below)
        below_group.shift(np.array([DISPLAY_X_SHIFT_RIGHT, DISPLAY_Y_SHIFT_BELOW, 0.0]))
        self.add(below_group)
        surface_below.shift(np.array([DISPLAY_X_SHIFT_RIGHT, DISPLAY_Y_SHIFT_BELOW, 0.0]))
        self.add(surface_below)
        title_s3_anchor = np.array([DISPLAY_X_SHIFT_RIGHT + x_mid, DISPLAY_Y_SHIFT_BELOW + z_anchor_y, title_z])
        self.add(_make_surface_title("Ackley", title_s3_anchor, config, self.FONT_FAMILY))
        # Global optimum at (0, 0)
        self.add(_optimum_dashed_line(0, 0, 0, DISPLAY_X_SHIFT_RIGHT, DISPLAY_Y_SHIFT_BELOW, z_min=Z_DISPLAY_MIN))
        self.add(_optimum_horizontal_dashed_line(0, 0, DISPLAY_X_SHIFT_RIGHT, DISPLAY_Y_SHIFT_BELOW, Z_DISPLAY_MIN, config.AXIS_RANGE_MAX, opacity=0.5))
        self.add(_make_optimum_coord_label(0, 0, DISPLAY_X_SHIFT_RIGHT, DISPLAY_Y_SHIFT_BELOW, Z_DISPLAY_MIN, config.AXIS_RANGE_MAX, config, self.FONT_FAMILY))

        # ----- East of third (same y as Ackley): Himmelblau -----
        # Contour SVG: lookup by naming scheme (see CONTOUR_SVG_DIR and legend at top).
        contour_path_s4 = _contour_svg_path("himmelblau", config)
        _print_contour_svg_status("Surface 4", "Himmelblau", contour_path_s4, config, creating=not contour_path_s4.is_file())
        DISPLAY_X_SHIFT_EAST = DISPLAY_X_SHIFT_RIGHT + HORIZONTAL_GAP
        himmelblau_raw = lambda u, v: _himmelblau_func(u, v, scale=0.01)
        himmelblau_surface_func = _surface_z_to_0_10(
            himmelblau_raw, u_range=axis_range, v_range=axis_range
        )
        graph_east = create_back_style_graph(
            config=config,
            foreground_color=config.FOREGROUND_COLOR,
            font_family=self.FONT_FAMILY,
        )
        surface_east = Surface(
            himmelblau_surface_func,
            u_range=[config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX],
            v_range=[config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX],
            resolution=config.SURFACE_RESOLUTION,
            fill_color=config.SURFACE_FILL_COLOR,
            fill_opacity=config.SURFACE_FILL_OPACITY,
            checkerboard_colors=[config.SURFACE_FILL_COLOR, config.SURFACE_FILL_COLOR],
            stroke_color=config.SURFACE_STROKE_COLOR,
            stroke_width=config.SURFACE_STROKE_WIDTH,
        )
        if contour_path_s4.is_file():
            contour_east = _load_contour_svg_for_group(contour_path_s4, config)
            contour_lines_east = [contour_east]
        else:
            contour_lines_east = create_contour_lines(
                surface_func=himmelblau_surface_func,
                config=config,
                foreground_color=config.FOREGROUND_COLOR,
                save_svg_path=contour_path_s4,
            )

        east_group = Group()
        for val in graph_east.values():
            if val is None:
                continue
            if isinstance(val, list):
                for item in val:
                    east_group.add(item)
            else:
                east_group.add(val)
        if contour_path_s4.is_file():
            east_group.add(contour_lines_east[0])
        else:
            east_group.add(contour_lines_east)
        east_group.shift(np.array([DISPLAY_X_SHIFT_EAST, DISPLAY_Y_SHIFT_BELOW, 0.0]))
        self.add(east_group)
        surface_east.shift(np.array([DISPLAY_X_SHIFT_EAST, DISPLAY_Y_SHIFT_BELOW, 0.0]))
        self.add(surface_east)
        title_s4_anchor = np.array([DISPLAY_X_SHIFT_EAST + x_mid, DISPLAY_Y_SHIFT_BELOW + z_anchor_y, title_z])
        self.add(_make_surface_title("Himmelblau", title_s4_anchor, config, self.FONT_FAMILY))
        # Global optimum at (3, 2) (one of four equal minima)
        self.add(_optimum_dashed_line(3, 2, 0, DISPLAY_X_SHIFT_EAST, DISPLAY_Y_SHIFT_BELOW, z_min=Z_DISPLAY_MIN))
        self.add(_optimum_horizontal_dashed_line(3, 2, DISPLAY_X_SHIFT_EAST, DISPLAY_Y_SHIFT_BELOW, Z_DISPLAY_MIN, config.AXIS_RANGE_MAX, opacity=0.5))
        self.add(_make_optimum_coord_label(3, 2, DISPLAY_X_SHIFT_EAST, DISPLAY_Y_SHIFT_BELOW, Z_DISPLAY_MIN, config.AXIS_RANGE_MAX, config, self.FONT_FAMILY))

        self.wait(0.5)
