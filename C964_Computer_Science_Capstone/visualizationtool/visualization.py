from manim import *
import numpy as np
import sys
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional

# Ensure project root importability when Manim loads this file by path.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import retro modules in the same style as retro_tester_2.py
from visualizationtool.retrograph import retro_configuration as retro_configuration_module
sys.modules.setdefault("retro_configuration", retro_configuration_module)
from visualizationtool.retrograph.retro_configuration import (
    build_config_for_scene,
    create_vertical_line_markers,
    get_default_class_config,
    get_rastrigin_wb_high_res_config,
    scene_config_overrides,
)
from visualizationtool.retrograph.retro_construction import construct_retro_style_scene

# Contour SVG dir and naming (same approach as retro_manim_graph/surface_multi_viz.py)
CONTOUR_SVG_DIR = _project_root / "media" / "svg" / "experimental_analysis"


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
    """Full path for contour SVG (lookup or save)."""
    return CONTOUR_SVG_DIR / _contour_svg_basename(func_name, config)


def _surface_z_to_0_10(surface_func, u_range, v_range, sample_res=32):
    """
    Return a surface function (u, v) -> [u, v, z] with z normalized to [0, 10]
    over the given domain. Samples the function on a grid to get min/max z.
    (Same logic as retro_manim_graph/surface_multi_viz.py for display consistency.)
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


def _display_surface_func_from_class(surface_function, axis_range, display_scale=0.03):
    """
    Build (u, v) -> [x, y, z] for retro scene: optional scale for display, then z normalized to [0, 10].
    If surface_function accepts a 'scale' kwarg (e.g. rastrigin_func), uses display_scale; else calls as (u, v).
    """
    import inspect
    try:
        sig = inspect.signature(surface_function)
        if "scale" in sig.parameters:
            raw = lambda u, v: surface_function(u, v, scale=display_scale)
        else:
            raw = surface_function
    except Exception:
        raw = surface_function
    return _surface_z_to_0_10(raw, u_range=axis_range, v_range=axis_range)


from optimizationfunctions.evolutionalgorithms.lamarckianfunctions.core import (
    pure_lamarckian_function,
    rastrigin_func,
)
from optimizationfunctions.evolutionalgorithms.metaevolutionfunctions import CountedFunction

# For configurable 1L2D viz: params object (run_up1 helpers used when evolution/algorithm step is implemented)
from visualizationtool.viz_params import OneLTwoDVizParams
from visualizationtool.viz_organisms import (
    create_organism_mobjects_for_plane,
    darwinian_generations_to_dataset,
    final_mean_xy,
    get_contour_plane_z,
    lamarckian_generations_to_dataset,
)

# Known global optima (x, y) per surface name for the line marker. Override via VisualizationScene.global_optimum_xy.
SURFACE_GLOBAL_OPTIMA: dict[str, tuple[float, float]] = {
    "rastrigin": (0.0, 0.0),
    "rosenbrock": (1.0, 1.0),
    "ackley": (0.0, 0.0),
    "himmelblau": (3.0, 2.0),  # one of four equal minima
}

# Default Lamarckian evolution levers (tuned from meta-evolution). VisualizationScene uses these when levers is None; self.levers overrides keys.
DEFAULT_LAMARCKIAN_LEVERS = {
    "besoin_weight": 1.1427907844220597,
    "topology_gradient_scale": 0.11282000353881588,
    "magnitude_std_fraction": 0.1235121700374864,
    "magnitude_weight": 0.8912894658829036,
    "direction_std": 0.3212217824540174,
    "min_magnitude": 0.01,
    "num_offspring": 2,
    "num_generations": 1000,
    "history_generation_stride": 1,
    "parent1_start": np.array([-6.0, -5.5, 0.0]),
    "parent1_end": np.array([-5.0, -4.2, 0.0]),
    "parent2_start": np.array([5.8, -5.2, 0.0]),
    "parent2_end": np.array([6.9, -4.0, 0.0]),
}


# Default key for info panel: what markers mean (used when panel_marker_key is None)
DEFAULT_PANEL_MARKER_KEY = [
    "Plane 0 (front): Lamarckian (vectors)",
    "Plane 1 (back): Darwinian (points)",
    "Color: one per seed (same on both planes)",
    "Circle: initial position(s)",
    "Darker dots/lines: final generation",
    "Vertical line + dot: final mean (x,y)",
]


class VisualizationScene(ThreeDScene):
    """Retro-style overlay for visualization of the experimental analysis.

    Set these on the class (or a subclass) before Manim runs; construct() reads them.

    Attributes:
        surface_function: Topology (u,v)->[x,y,z]; None = no surface.
        surface_name: Name for contour SVG (e.g. 'rastrigin').
        evolutionary_algorithm: Algorithm callable; None = blank.
        experiment_name: Title/filename string.
        levers: Dict of evolution levers.
    """
    # Leave these attribute discriptions empty in the class docustring, instead add them to the attribute docstrings.
    surface_function: Optional[Callable[..., Any]] = None
    """Topology/fitness surface: callable (u, v) -> [x, y, z]. None = no surface. Set e.g. rastrigin_func for Rastrigin."""
    surface_name: Optional[str] = None
    """Name for contour SVG / topology (e.g. 'rastrigin'). Defaults from surface_function.__name__.replace('_func','')."""
    evolutionary_algorithm: Optional[Callable[..., Any]] = None
    """Evolutionary algorithm: callable (e.g. pure_lamarckian_function, pure_darwinian_function). None is blank"""
    experiment_name: str = "unamed_experiment"
    """Experiment name: string (e.g. "experimental_analysis"). Used for title and filename."""
    levers: Optional[dict[str, Any]] = None
    """Evolution levers: dictionary (e.g. besoin_weight, num_generations). None uses defaults."""
    global_optimum_xy: Optional[tuple[float, float]] = None
    """(x, y) of global optimum for the current surface. If None, inferred from surface_name via SURFACE_GLOBAL_OPTIMA else (0, 0)."""
    algorithm_runs: Optional[List[dict]] = None
    """List of runs to draw on contour planes. Each dict: dataset, dataset_type ('darwinian'|'lamarckian'|'normalized'), contour_plane_index, render_mode ('points'|'vectors'|'lines'), optional surface_func, color, generation_stride."""
    algorithm_run: Optional[dict] = None
    """Single run (one dict, same shape as algorithm_runs items). Used when algorithm_runs is not set."""
    panel_replication_lines: Optional[List[str]] = None
    """Lines for the 'Replication' section of the info panel (experiment, seeds, call budget, etc.)."""
    panel_experiment_id: Optional[str] = None
    """Experiment ID displayed as fixed in-frame watermark text."""
    panel_started_at: Optional[str] = None
    """Experiment start timestamp displayed as fixed in-frame watermark text."""
    panel_marker_key: Optional[List[str]] = None
    """Lines for the 'Key' section explaining markers. If None, DEFAULT_PANEL_MARKER_KEY is used."""
    panel_levers_data: Optional[List[dict]] = None
    """Per-seed best levers: list of {seed, lam_levers, dar_levers} for table (MD2/UP1)."""
    panel_performance_data: Optional[List[dict]] = None
    """Per-seed performance rows for info panel table."""
    panel_levers_colors: Optional[List[str]] = None
    """Hex color per seed for lever table rows (same order as panel_levers_data)."""
    panel_lam_total_organisms: Optional[int] = None
    """Total Lamarckian organisms generated across the run (all seeds and generations)."""
    panel_dar_total_organisms: Optional[int] = None
    """Total Darwinian organisms generated across the run (all seeds and generations)."""

    get_default_class_config()

    def construct(self):
        # Pseudo-code (keep for implementation order):
        # 1. Graph config: position, style, seeds & colors
        # 2. Topology: evolution_topology (counted), display_surface_func (dampened to fit 0–10 z axis range, display only)
        # 3. Topology SVG: resolve path (same method as retro_manim_graph/surface_multi_viz.py)
        # 4. Graph: construct_retro_style_scene(surface_func, config_overrides, topology_svg_path, ...)
        # 5. Apply display shift to all mobjects (on initialization of all objects)
        # 6. Build optima marker (dashed vertical at surface optimum down to z axis minimum)
        # 7. Target algorithm levers/config: levers = merge(DEFAULT_LAMARCKIAN_LEVERS, self.levers); specify available algorithms; option to take points as input only
        # 8. Algorithm(s) provide points or vectors per generation. Render as dots (points), lines (vectors), or option: points/lines/polygons/arrowed vectors
        # 9. Add all_seed_lines, optima_marker, final_lines, mean_lines, last_org_markers and other objects to scene in correct order
        # 10. Build info panel (header with levers/stats, per-seed "s{seed} final mean=(x,y)"); add_fixed_in_frame_mobjects(panel); add(panel); wait(0.1); small font

        # 1. Graph config (position only when no surface; with surface, same as surface_multi_viz)
        DISPLAY_X_SHIFT = -10.0
        DISPLAY_Y_SHIFT = 40.0

        # Resolve from class so we get the assigned callable (e.g. rastrigin_func), not a bound method
        surface_function = getattr(type(self), "surface_function", None)
        surface_name = getattr(type(self), "surface_name", None)
        if surface_name is None and surface_function is not None:
            surface_name = getattr(surface_function, "__name__", "surface").replace("_func", "")

        # 4. Graph: config + contour SVG strategy (same approach as surface_multi_viz.py)
        seed_display = getattr(self, "experiment_name", "graph") or "graph"
        shared_overrides = scene_config_overrides(
            CAMERA_PRESET="orthoxyz",
            color_scheme="bw",
            VIEW_SCALE= 0.9,
            NUM_ADDITIONAL_CONTOUR_PLANES=1,
            ADDITIONAL_CONTOUR_PLANE_Z_SPACING=25.0,
            AXIS_RANGE_MIN=-12.0,
            AXIS_RANGE_MAX=12.0,
            Z_AXIS_RANGE_MIN=-20.0,
            Z_AXIS_RANGE_MAX=10.0,
            SHOW_TITLE=False,
            X_AXIS_TITLE="X TRAIT",
            Y_AXIS_TITLE="Y TRAIT",
            Z_AXIS_TITLE="ADAPTIVE SURFACE VALUE",
            SHOW_SURFACE=(surface_function is not None),
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
        axis_range = (config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX)
        display_surface_func = None
        if surface_function is not None:
            display_surface_func = _display_surface_func_from_class(
                surface_function, axis_range, display_scale=0.03
            )

        contour_name = surface_name if surface_name else "rastrigin"
        contour_path = _contour_svg_path(contour_name, config).resolve()
        CONTOUR_SVG_DIR.mkdir(parents=True, exist_ok=True)
        topology_svg_path = str(contour_path) if contour_path.is_file() else None
        topology_svg_save_path = str(contour_path)

        scene_elements = construct_retro_style_scene(
            scene=self,
            surface_func=display_surface_func,
            config=config,
            config_overrides=None,
            topology_svg_path=topology_svg_path,
            topology_svg_save_path=topology_svg_save_path,
            display_seed=seed_display,
        )

        # 5. Apply display shift to all mobjects
        for mob in self.mobjects:
            mob.shift(np.array([DISPLAY_X_SHIFT, DISPLAY_Y_SHIFT, 0.0]))

        # 6. Optima marker(s): vertical dashed line(s) from lowest contour plane to surface (or z max)
        config = scene_elements["config"]
        optimum_xy = getattr(type(self), "global_optimum_xy", None)
        if optimum_xy is None and surface_name:
            optimum_xy = SURFACE_GLOBAL_OPTIMA.get(surface_name.lower() if isinstance(surface_name, str) else "", (0.0, 0.0))
        if optimum_xy is None:
            optimum_xy = (0.0, 0.0)
        line_markers = create_vertical_line_markers(
            config,
            [optimum_xy],  # current surface global optimum; add more (x, y) or (x, y, z_top) as needed
            shift_x=DISPLAY_X_SHIFT,
            shift_y=DISPLAY_Y_SHIFT,
            surface_func=display_surface_func,
            dash_length=1,
            thickness=0.02,
            color=BLACK,
            show_crosses_at_planes=True,
            cross_size=0.1,
            time_execution=True,
        )
        self.add(line_markers)

        # 7. Algorithm runs: organisms per contour plane (points / vectors / lines)
        runs: List[dict] = []
        if getattr(type(self), "algorithm_runs", None):
            runs = list(type(self).algorithm_runs)
        elif getattr(type(self), "algorithm_run", None):
            runs = [type(self).algorithm_run]
        mean_markers: List[Any] = []  # collect to add last so they render on top
        for run in runs:
            raw = run.get("dataset", [])
            if not raw:
                continue
            dataset_type = run.get("dataset_type", "normalized")
            if dataset_type == "darwinian":
                dataset = darwinian_generations_to_dataset(raw)
            elif dataset_type == "lamarckian":
                dataset = lamarckian_generations_to_dataset(raw)
            else:
                dataset = raw
            plane_index = int(run.get("contour_plane_index", 0))
            render_mode = run.get("render_mode", "points")
            if render_mode not in ("points", "vectors", "lines"):
                render_mode = "points"
            z_plane = get_contour_plane_z(config, plane_index)
            # Use surface_func only when run explicitly provides it; else organisms render on contour plane (z_plane)
            run_surface = run.get("surface_func")
            run_color = run.get("color", BLACK)
            generation_stride = int(run.get("generation_stride", 1))
            if generation_stride < 1:
                generation_stride = 1
            # Default: show history as fainter dots so earlier generations are visible (points mode)
            history_opacity = run.get("history_opacity_range")
            if history_opacity is None and render_mode == "points":
                history_opacity = (0.5, 1.0)
            # Match graph x-scale (retro graph uses X_AXIS_SCALE for x-axis)
            scale_x = float(getattr(config, "X_AXIS_SCALE", 1.0))
            mobjects = create_organism_mobjects_for_plane(
                dataset,
                z_plane,
                render_mode,
                shift_x=DISPLAY_X_SHIFT,
                shift_y=DISPLAY_Y_SHIFT,
                surface_func=run_surface,
                color=run_color,
                point_radius=float(run.get("point_radius", 0.04)),
                line_thickness=float(run.get("line_thickness", 0.02)),
                generation_stride=generation_stride,
                history_opacity_range=history_opacity,
                final_color=run.get("final_color"),
                final_point_radius=run.get("final_point_radius"),
                initial_marker_radius=run.get("initial_marker_radius"),
                initial_marker_color=run.get("initial_marker_color"),
                scale_x=scale_x,
            )
            # Debug: confirm history is rendered (generations in dataset vs mobjects drawn)
            print(
                f"  Run {dataset_type} plane {plane_index}: "
                f"{len(dataset)} generations -> {len(mobjects)} mobjects (stride={generation_stride})"
            )
            self.add(mobjects)

            # Final mean marker (vertical line + dot on top); collect to add last so they render on top
            mean_xy = final_mean_xy(dataset, dataset_type)
            if mean_xy is not None:
                mean_x, mean_y = mean_xy
                mx = DISPLAY_X_SHIFT + mean_x * scale_x
                my = DISPLAY_Y_SHIFT + mean_y
                mean_height = 1.5
                z_lo = z_plane
                z_hi = z_plane + mean_height
                mean_color = run.get("final_color") or run_color
                mean_line = Line(
                    start=np.array([mx, my, z_lo]),
                    end=np.array([mx, my, z_hi]),
                    color=mean_color,
                    stroke_width=1.0,
                )
                mean_dot = Dot(point=np.array([mx, my, z_hi]), color=mean_color, radius=0.14)
                mean_dot.set_fill(opacity=1.0)
                mean_dot.set_stroke(opacity=1.0)
                mean_markers.extend([mean_line, mean_dot])
        for mob in mean_markers:
            self.add(mob)

        # 8. Info panel (fixed in frame): title, key (markers), replication (experiment details)
        panel_font = "Courier New"
        font_size_title = 16
        font_size_section = 14
        font_size_body = 11
        font_size_repl_header = 12
        font_size_repl_body = 9
        panel_title = Text("Info Panel", font=panel_font, color=BLACK, font_size=font_size_title)
        panel = VGroup(panel_title)

        # Key: what the markers mean
        key_header = Text("Key", font=panel_font, color=BLACK, font_size=font_size_section)
        key_header.next_to(panel_title, DOWN, buff=0.08).align_to(panel_title, LEFT)
        panel.add(key_header)
        marker_key = getattr(type(self), "panel_marker_key", None)
        if marker_key is None:
            marker_key = DEFAULT_PANEL_MARKER_KEY
        for i, line in enumerate(marker_key):
            t = Text(line, font=panel_font, color=BLACK, font_size=font_size_body)
            t.next_to(key_header if i == 0 else panel[-1], DOWN, buff=0.04).align_to(panel_title, LEFT)
            panel.add(t)

        # Experimental reproducibility information
        repl_lines = getattr(type(self), "panel_replication_lines", None)
        if repl_lines:
            repl_header = Text(
                "Experimental Reproducibility Information",
                font=panel_font,
                color=BLACK,
                font_size=font_size_repl_header,
            )
            repl_header.next_to(panel[-1], DOWN, buff=0.1).align_to(panel_title, LEFT)
            panel.add(repl_header)
            for line in repl_lines:
                t = Text(line, font=panel_font, color=BLACK, font_size=font_size_repl_body)
                t.next_to(panel[-1], DOWN, buff=0.04).align_to(panel_title, LEFT)
                panel.add(t)

        # Best levers tables: separate Lamarckian and Darwinian sections
        levers_data = getattr(type(self), "panel_levers_data", None)
        levers_colors = getattr(type(self), "panel_levers_colors", None)
        if levers_data and levers_colors:
            font_size_table = 8
            font_size_subtle = 7
            lam_keys = [
                "besoin_weight", "topology_gradient_scale", "magnitude_std_fraction",
                "magnitude_weight", "direction_std", "min_magnitude", "max_magnitude",
                "num_offspring", "first_generation_random_besoin",
            ]
            dar_keys = ["elimination_rate", "selection_pressure", "mutation_std"]
            def _format_cell(v):
                if isinstance(v, bool):
                    return "True" if v else "False"
                if isinstance(v, float):
                    return f"{v:.3g}"
                if v is None:
                    return "—"
                return str(v)

            def _format_table_row(values, widths, aligns=None):
                if aligns is None:
                    aligns = ["left"] * len(values)
                cells = []
                for val, w, align in zip(values, widths, aligns):
                    if align == "right":
                        cells.append(f"{val:>{w}}")
                    else:
                        cells.append(f"{val:<{w}}")
                return "| " + " | ".join(cells)

            def _header_words(label):
                if label == "seed":
                    return ["Seed"]
                if label == "first_generation_random_besoin":
                    return ["First", "Random", "Besoin"]
                if label == "best_final_organism":
                    return ["Best final organism"]
                if label == "final_mean":
                    return ["Final mean"]
                return [part.capitalize() for part in label.split("_")]

            def _header_lines(headers, widths, aligns):
                words_by_col = [_header_words(h) for h in headers]
                max_lines = max((len(words) for words in words_by_col), default=1)
                lines = []
                for line_idx in range(max_lines):
                    row = []
                    for words in words_by_col:
                        if line_idx < len(words):
                            row.append(words[line_idx])
                            continue
                        row.append("")
                    lines.append(_format_table_row(row, widths, aligns))
                return lines

            # Lamarckian levers section
            lam_section_header = Text("Lamarckian levers", font=panel_font, color=BLACK, font_size=font_size_section)
            lam_section_header.next_to(panel[-1], DOWN, buff=0.1).align_to(panel_title, LEFT)
            panel.add(lam_section_header)
            lam_total_organisms = getattr(type(self), "panel_lam_total_organisms", None)
            lam_total_text = (
                f"Total organisms over run: {lam_total_organisms}"
                if lam_total_organisms is not None
                else "Total organisms over run: n/a"
            )
            lam_total_label = Text(
                lam_total_text,
                font=panel_font,
                color=BLACK,
                font_size=font_size_subtle,
            )
            lam_total_label.next_to(panel[-1], DOWN, buff=0.02).align_to(panel_title, LEFT)
            panel.add(lam_total_label)
            lam_headers = ["seed"] + lam_keys
            lam_rows = []
            for row in levers_data:
                lam = row.get("lam_levers") or {}
                vals = [str(row.get("seed", ""))] + [_format_cell(lam.get(k)) for k in lam_keys]
                lam_rows.append(vals)
            lam_widths = []
            for i in range(len(lam_headers)):
                content_width = max((len(r[i]) for r in lam_rows), default=0)
                header_word_width = max((len(w) for w in _header_words(lam_headers[i])), default=0)
                lam_widths.append(max(header_word_width, content_width))
            lam_aligns = ["right"] + ["left"] * (len(lam_headers) - 1)
            lam_header_lines = _header_lines(lam_headers, lam_widths, lam_aligns)
            for idx, line in enumerate(lam_header_lines):
                lam_table_header = Text(
                    line,
                    font=panel_font,
                    color=BLACK,
                    font_size=font_size_table,
                )
                lam_table_header.next_to(panel[-1], DOWN, buff=0.04 if idx == 0 else 0.02).align_to(panel_title, LEFT)
                panel.add(lam_table_header)
            for idx, vals in enumerate(lam_rows):
                line_str = _format_table_row(vals, lam_widths, lam_aligns)
                row_color = levers_colors[idx] if idx < len(levers_colors) else BLACK
                t = Text(line_str, font=panel_font, color=row_color, font_size=font_size_table)
                t.next_to(panel[-1], DOWN, buff=0.03).align_to(panel_title, LEFT)
                panel.add(t)

            # Darwinian levers section
            dar_section_header = Text("Darwinian levers", font=panel_font, color=BLACK, font_size=font_size_section)
            dar_section_header.next_to(panel[-1], DOWN, buff=0.1).align_to(panel_title, LEFT)
            panel.add(dar_section_header)
            dar_total_organisms = getattr(type(self), "panel_dar_total_organisms", None)
            dar_total_text = (
                f"Total organisms over run: {dar_total_organisms}"
                if dar_total_organisms is not None
                else "Total organisms over run: n/a"
            )
            dar_total_label = Text(
                dar_total_text,
                font=panel_font,
                color=BLACK,
                font_size=font_size_subtle,
            )
            dar_total_label.next_to(panel[-1], DOWN, buff=0.02).align_to(panel_title, LEFT)
            panel.add(dar_total_label)
            dar_headers = ["seed"] + dar_keys
            dar_rows = []
            for row in levers_data:
                dar = row.get("dar_levers") or {}
                vals = [str(row.get("seed", ""))] + [_format_cell(dar.get(k)) for k in dar_keys]
                dar_rows.append(vals)
            dar_widths = []
            for i in range(len(dar_headers)):
                content_width = max((len(r[i]) for r in dar_rows), default=0)
                header_word_width = max((len(w) for w in _header_words(dar_headers[i])), default=0)
                dar_widths.append(max(header_word_width, content_width))
            dar_aligns = ["right"] + ["left"] * (len(dar_headers) - 1)
            dar_header_lines = _header_lines(dar_headers, dar_widths, dar_aligns)
            for idx, line in enumerate(dar_header_lines):
                dar_table_header = Text(
                    line,
                    font=panel_font,
                    color=BLACK,
                    font_size=font_size_table,
                )
                dar_table_header.next_to(panel[-1], DOWN, buff=0.04 if idx == 0 else 0.02).align_to(panel_title, LEFT)
                panel.add(dar_table_header)
            for idx, vals in enumerate(dar_rows):
                line_str = _format_table_row(vals, dar_widths, dar_aligns)
                row_color = levers_colors[idx] if idx < len(levers_colors) else BLACK
                t = Text(line_str, font=panel_font, color=row_color, font_size=font_size_table)
                t.next_to(panel[-1], DOWN, buff=0.03).align_to(panel_title, LEFT)
                panel.add(t)

            # Performance table (per-seed, with L/D winner markers)
            perf_section_header = Text("Performance", font=panel_font, color=BLACK, font_size=font_size_section)
            perf_section_header.next_to(panel[-1], DOWN, buff=0.1).align_to(panel_title, LEFT)
            panel.add(perf_section_header)
            perf_headers = ["seed", "best_final_organism", "won", "final_mean", "won"]
            perf_data = getattr(type(self), "panel_performance_data", None) or []

            def _as_float(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None

            def _fmt_fixed(v: float) -> str:
                # Fixed-width numeric formatting keeps '=', ',' and 'd=' aligned across rows.
                return f"{v:>7.3f}"

            perf_rows = []
            for row in perf_data:
                seed = str(row.get("seed", ""))
                lam_best_xy = row.get("lam_best_final_organism_xy")
                dar_best_xy = row.get("dar_best_final_organism_xy")
                lam_best_dist = _as_float(row.get("lam_best_final_organism_distance"))
                dar_best_dist = _as_float(row.get("dar_best_final_organism_distance"))
                lam_mean = _as_float(row.get("lam_final_mean"))
                dar_mean = _as_float(row.get("dar_final_mean"))

                if (
                    isinstance(lam_best_xy, (list, tuple))
                    and len(lam_best_xy) >= 2
                    and isinstance(dar_best_xy, (list, tuple))
                    and len(dar_best_xy) >= 2
                    and lam_best_dist is not None
                    and dar_best_dist is not None
                ):
                    lx, ly = float(lam_best_xy[0]), float(lam_best_xy[1])
                    dx, dy = float(dar_best_xy[0]), float(dar_best_xy[1])
                    best_cell = (
                        f"L=({_fmt_fixed(lx)},{_fmt_fixed(ly)}) d={_fmt_fixed(lam_best_dist)} "
                        f"D=({_fmt_fixed(dx)},{_fmt_fixed(dy)}) d={_fmt_fixed(dar_best_dist)}"
                    )
                    best_winner = (
                        "L"
                        if lam_best_dist < dar_best_dist
                        else ("D" if dar_best_dist < lam_best_dist else "T")
                    )
                else:
                    best_cell, best_winner = "n/a", "-"

                if lam_mean is not None and dar_mean is not None:
                    mean_cell = f"L={_fmt_fixed(lam_mean)} D={_fmt_fixed(dar_mean)}"
                    mean_winner = "L" if lam_mean < dar_mean else ("D" if dar_mean < lam_mean else "T")
                else:
                    mean_cell, mean_winner = "n/a", "-"

                perf_rows.append([seed, best_cell, best_winner, mean_cell, mean_winner])

            perf_widths = []
            for i in range(len(perf_headers)):
                content_width = max((len(r[i]) for r in perf_rows), default=0)
                header_word_width = max((len(w) for w in _header_words(perf_headers[i])), default=0)
                perf_widths.append(max(header_word_width, content_width))
            perf_aligns = ["right", "left", "left", "left", "left"]
            perf_header_lines = _header_lines(perf_headers, perf_widths, perf_aligns)
            for idx, line in enumerate(perf_header_lines):
                perf_table_header = Text(
                    line,
                    font=panel_font,
                    color=BLACK,
                    font_size=font_size_table,
                )
                perf_table_header.next_to(panel[-1], DOWN, buff=0.04 if idx == 0 else 0.02).align_to(panel_title, LEFT)
                panel.add(perf_table_header)

            for idx, vals in enumerate(perf_rows):
                line_str = _format_table_row(vals, perf_widths, perf_aligns)
                row_color = levers_colors[idx] if idx < len(levers_colors) else BLACK
                t = Text(line_str, font=panel_font, color=row_color, font_size=font_size_table)
                t.next_to(panel[-1], DOWN, buff=0.03).align_to(panel_title, LEFT)
                panel.add(t)

        panel.add_background_rectangle(color=WHITE, opacity=0.9, buff=0.08)
        panel.to_edge(RIGHT).shift(UP * 3)
        panel_border = SurroundingRectangle(panel, color=BLACK, buff=0)
        panel.add(panel_border)
        self.add_fixed_in_frame_mobjects(panel)
        self.add(panel)

        # Always-visible run metadata watermark (guarantees ID/timestamp appear in output image).
        exp_id = getattr(type(self), "panel_experiment_id", None)
        started_at = getattr(type(self), "panel_started_at", None)
        if exp_id or started_at:
            wm_lines = []
            if exp_id:
                wm_lines.append(
                    Text(
                        f"Experiment ID: {exp_id}",
                        font=panel_font,
                        color=BLACK,
                        font_size=11,
                    )
                )
            if started_at:
                wm_lines.append(
                    Text(
                        f"Started at: {started_at}",
                        font=panel_font,
                        color=BLACK,
                        font_size=11,
                    )
                )
            wm = VGroup(*wm_lines).arrange(DOWN, aligned_edge=LEFT, buff=0.03)
            wm.add_background_rectangle(color=WHITE, opacity=0.9, buff=0.06)
            wm.next_to(panel, UP, buff=0.08).align_to(panel, LEFT)
            self.add_fixed_in_frame_mobjects(wm)
            self.add(wm)

        self.wait(0.1)


# Extended palettes for N seeds (N > 6)
SEED_COLORS_EXTENDED = [BLUE, GREEN, RED, PURPLE, TEAL, PINK, MAROON, GOLD, ORANGE, YELLOW_C]
MEAN_POINT_COLORS_EXTENDED = [
    "#0D47A1", "#1B5E20", "#B71C1C", "#4A148C", "#004D40", "#880E4F",
    "#6D4C41", "#F57F17", "#E65100", "#F9A825",
]


