from manim import *
import numpy as np
import sys
from pathlib import Path

# Ensure project root importability when Manim loads this file by path.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import retro modules in the same style as retro_tester_2.py
from visualizationtool.retrograph import retro_configuration as retro_configuration_module
sys.modules.setdefault("retro_configuration", retro_configuration_module)
from visualizationtool.retrograph.retro_configuration import (
    get_default_class_config,
    scene_config_overrides,
)
from visualizationtool.retrograph.retro_construction import construct_retro_style_scene

from optimizationfunctions.evolutionalgorithms.darwinianfunctions.core import (
    pure_darwinian_function,
    rastrigin_func,
)
from optimizationfunctions.evolutionalgorithms.metaevolutionfunctions import CountedFunction

from visualizationtool.viz_params import OneLTwoDVizParams
from optimizationlab.experimentalsetup.run_up1 import (
    generate_initial_points,
    random_darwinian_levers,
)


class SixOptimizedDarwinianFunctions(ThreeDScene):
    """Retro-style six-seed Darwinian overlay: points on fitness landscape, selection + mutation."""

    get_default_class_config()

    def construct(self):
        # Optimized levers from meta-tuning for seed 7 (best candidate); applied to all seeds.
        OPTIMIZED_LEVERS = {
            "elimination_rate": 0.2126,
            "selection_pressure": 5.495,
            "mutation_std": 0.368,
        }
        POPULATION_SIZE = 4
        NUM_GENERATIONS = 1000
        INITIAL_BOUNDS = (-10.0, 10.0, -10.0, 10.0)
        HISTORY_GENERATION_STRIDE = 10
        HISTORY_DOT_RADIUS = 0.1
        HISTORY_DOT_OPACITY = 0.35
        FINAL_DOT_RADIUS = 0.1
        MEAN_LINE_Z_BOTTOM = -20.0
        MEAN_LINE_HEIGHT_MIN = 1.0
        MEAN_LINE_HEIGHT_MAX = 2.0
        MEAN_LINE_STROKE_WIDTH = 1.0
        MEAN_LINE_DOT_RADIUS = 0.08
        FINAL_DOT_DARKEN = 0.5  # interpolate toward BLACK (0=seed color, 1=black)
        DISPLAY_X_SHIFT = -10.0
        DISPLAY_Y_SHIFT = 15.0

        RNG_SEEDS = [7, 27, 107, 207, 327, 507]
        SEED_COLORS = [BLUE, GREEN, RED, PURPLE, TEAL, PINK]
        MEAN_POINT_COLORS = ["#0D47A1", "#1B5E20", "#B71C1C", "#4A148C", "#004D40", "#880E4F"]
        # Evolution uses same topology as Lamarckian (Rastrigin); display uses dampened surface.
        counted_topology = CountedFunction(rastrigin_func)
        EVOLUTION_TOPOLOGY = counted_topology
        DISPLAY_SURFACE_FUNC = lambda u, v: rastrigin_func(u, v, scale=0.03)

        # Use requested contour SVG if present; otherwise fallback to latest topology_*.svg.
        svg_dir = _project_root / "media" / "svg"
        preferred_svg = svg_dir / "topology_a892e75fe0607600.svg"
        topology_svg_path = None
        if preferred_svg.is_file():
            topology_svg_path = str(preferred_svg)
        elif svg_dir.is_dir():
            svg_files = list(svg_dir.glob("topology_*.svg"))
            if svg_files:
                topology_svg_path = str(max(svg_files, key=lambda p: p.stat().st_mtime))

        seed_display = ",".join(str(s) for s in RNG_SEEDS)
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
                Z_AXIS_TITLE="SELECTION COEFFICIENT",
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
            ),
            topology_svg_path=topology_svg_path,
            topology_svg_cache_dir="media/svg",
            topology_id="rastrigin",
            display_seed=seed_display,
        )

        for mob in self.mobjects:
            mob.shift(np.array([DISPLAY_X_SHIFT, DISPLAY_Y_SHIFT, 0.0]))

        contour_z = scene_elements["config"].Z_AXIS_RANGE_MIN
        # Dashed vertical line at global optimum (0,0).
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

        all_seed_points = VGroup()
        final_dots = VGroup()  # rendered after seed_dots so final dots appear on top
        mean_lines = VGroup()  # mean vertical lines + dots, rendered last
        final_mean_points = []

        for seed_idx, seed in enumerate(RNG_SEEDS):
            generations = pure_darwinian_function(
                fitness_topology_function=EVOLUTION_TOPOLOGY,
                population_size=POPULATION_SIZE,
                num_generations=NUM_GENERATIONS,
                elimination_rate=OPTIMIZED_LEVERS["elimination_rate"],
                selection_pressure=OPTIMIZED_LEVERS["selection_pressure"],
                mutation_std=OPTIMIZED_LEVERS["mutation_std"],
                seed=seed,
                initial_bounds=INITIAL_BOUNDS,
            )

            seed_color = SEED_COLORS[seed_idx % len(SEED_COLORS)]
            final_color = interpolate_color(seed_color, BLACK, FINAL_DOT_DARKEN)
            num_gens = len(generations)
            seed_dots = VGroup()

            for gen_data in generations:
                generation = gen_data["generation"]
                if generation != (num_gens - 1) and generation % HISTORY_GENERATION_STRIDE != 0:
                    continue

                organisms = gen_data["organisms"]
                is_final = generation == (num_gens - 1)
                r = FINAL_DOT_RADIUS if is_final else HISTORY_DOT_RADIUS
                op = 1.0 if is_final else HISTORY_DOT_OPACITY
                color = final_color if is_final else seed_color

                for pt in organisms:
                    x, y = float(pt[0]), float(pt[1])
                    pt_3d = np.array([x + DISPLAY_X_SHIFT, y + DISPLAY_Y_SHIFT, contour_z])
                    dot_contour = Dot(point=pt_3d, color=color, radius=r)
                    dot_contour.set_fill(opacity=op)
                    dot_contour.set_stroke(opacity=op)
                    if is_final:
                        final_dots.add(dot_contour)
                    else:
                        seed_dots.add(dot_contour)
                    z_surf = DISPLAY_SURFACE_FUNC(x, y)[2]
                    pt_on_surface = np.array([x + DISPLAY_X_SHIFT, y + DISPLAY_Y_SHIFT, z_surf])
                    dot_surf = Dot(point=pt_on_surface, color=color, radius=r)
                    dot_surf.set_fill(opacity=op)
                    dot_surf.set_stroke(opacity=op)
                    if is_final:
                        final_dots.add(dot_surf)
                    else:
                        seed_dots.add(dot_surf)

            all_seed_points.add(seed_dots)

            last_orgs = np.array(generations[-1]["organisms"], dtype=float)
            mean_xy = np.mean(last_orgs[:, :2], axis=0)
            mx, my = float(mean_xy[0]) + DISPLAY_X_SHIFT, float(mean_xy[1]) + DISPLAY_Y_SHIFT
            mean_color = MEAN_POINT_COLORS[seed_idx % len(MEAN_POINT_COLORS)]
            # Height by render order: first-added = tallest (2), last-added = shortest (1), so later-drawn sit on top
            n_seeds = len(RNG_SEEDS)
            mean_height = MEAN_LINE_HEIGHT_MAX - (seed_idx / max(1, n_seeds - 1)) * (MEAN_LINE_HEIGHT_MAX - MEAN_LINE_HEIGHT_MIN)
            z_lo = MEAN_LINE_Z_BOTTOM
            z_hi = MEAN_LINE_Z_BOTTOM + mean_height
            mean_line = Line(
                start=np.array([mx, my, z_lo]),
                end=np.array([mx, my, z_hi]),
                color=mean_color,
                stroke_width=MEAN_LINE_STROKE_WIDTH,
            )
            mean_lines.add(mean_line)
            mean_lines.add(Dot(point=np.array([mx, my, z_hi]), color=mean_color, radius=MEAN_LINE_DOT_RADIUS))
            final_mean_points.append((seed, mean_xy))

        self.add(all_seed_points, optima_marker, final_dots, mean_lines)

        p = OPTIMIZED_LEVERS
        header_lines = [
            "6 optimized Darwinian runs",
            "(levers from seed 7, applied to all seeds)",
            f"seeds: {RNG_SEEDS}",
            f"pop={POPULATION_SIZE}, gens={NUM_GENERATIONS}",
            f"history_stride={HISTORY_GENERATION_STRIDE}",
            f"elim={p['elimination_rate']:.3f} sel={p['selection_pressure']:.2f} mut={p['mutation_std']:.3f}",
            f"bounds={INITIAL_BOUNDS}",
            f"topology_calls={counted_topology.n_calls}",
        ]
        header_text = Text("\n".join(header_lines), font_size=9, color=BLACK, line_spacing=0.75, font="Courier New")
        mean_line_texts = VGroup(
            *[
                Text(
                    f"s{seed} final mean=({mean_xy[0]:.2f},{mean_xy[1]:.2f})",
                    font_size=9,
                    color=MEAN_POINT_COLORS[i % len(MEAN_POINT_COLORS)],
                    line_spacing=0.75,
                    font="Courier New",
                )
                for i, (seed, mean_xy) in enumerate(final_mean_points)
            ]
        )
        mean_line_texts.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        panel = VGroup(header_text, mean_line_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        panel.to_edge(RIGHT).shift(DOWN * 0.3)
        panel.add_background_rectangle(color=WHITE, opacity=0.9, buff=0.08)
        self.add_fixed_in_frame_mobjects(panel)
        self.add(panel)
        self.wait(0.1)


SEED_COLORS_EXTENDED = [BLUE, GREEN, RED, PURPLE, TEAL, PINK, MAROON, GOLD, ORANGE, YELLOW_C]
MEAN_POINT_COLORS_EXTENDED = [
    "#0D47A1", "#1B5E20", "#B71C1C", "#4A148C", "#004D40", "#880E4F",
    "#6D4C41", "#F57F17", "#E65100", "#F9A825",
]


class ConfigurableDarwinianViz(ThreeDScene):
    """Retro-style Darwinian overlay driven by a params object. Renders any number of seeds (e.g. 1L2D: 2 organisms per seed)."""

    viz_params = None  # OneLTwoDVizParams or None → use default_10_seeds()

    get_default_class_config()

    def construct(self):
        params = self.viz_params or OneLTwoDVizParams.default_10_seeds()
        RNG_SEEDS = list(params.seeds)
        n_seeds = len(RNG_SEEDS)
        DISPLAY_X_SHIFT = params.display_x_shift
        DISPLAY_Y_SHIFT = params.display_y_shift
        # Show history every N generations; stride 1 can make 150+ gens with max_calls=300 and pop=2 — use 5 so trail is visible
        HISTORY_GENERATION_STRIDE = max(1, getattr(params, "history_generation_stride", 1))
        if params.max_calls and params.darwinian_pop <= 2:
            HISTORY_GENERATION_STRIDE = max(HISTORY_GENERATION_STRIDE, 5)
        HISTORY_DOT_RADIUS = 0.12
        FINAL_DOT_RADIUS = 0.1
        HISTORY_DOT_OPACITY = 0.65
        MEAN_LINE_Z_BOTTOM = -20.0
        MEAN_LINE_HEIGHT_MIN = 1.0
        MEAN_LINE_HEIGHT_MAX = 2.0
        MEAN_LINE_STROKE_WIDTH = 1.0
        MEAN_LINE_DOT_RADIUS = 0.08
        FINAL_DOT_DARKEN = params.final_darken
        SEED_COLORS = SEED_COLORS_EXTENDED
        MEAN_POINT_COLORS = MEAN_POINT_COLORS_EXTENDED

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

        seed_display = ",".join(str(s) for s in RNG_SEEDS)
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
                Z_AXIS_TITLE="SELECTION COEFFICIENT",
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
            ),
            topology_svg_path=topology_svg_path,
            topology_svg_cache_dir="media/svg",
            topology_id="rastrigin",
            display_seed=seed_display,
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

        all_seed_points = VGroup()
        final_dots = VGroup()
        mean_lines = VGroup()
        final_mean_points = []
        total_dar_calls = 0

        for seed_idx, seed in enumerate(RNG_SEEDS):
            initial_points = generate_initial_points(seed, 2)
            dar_levers = random_darwinian_levers(seed, population_size=params.darwinian_pop)
            counted_topology = CountedFunction(rastrigin_func)
            generations = pure_darwinian_function(
                fitness_topology_function=counted_topology,
                num_generations=100000,
                seed=seed + 100,
                initial_bounds=params.initial_bounds,
                max_calls=params.max_calls,
                initial_population=initial_points.copy(),
                **dar_levers,
            )
            total_dar_calls += counted_topology.n_calls

            seed_color = SEED_COLORS[seed_idx % len(SEED_COLORS)]
            final_color = interpolate_color(seed_color, BLACK, FINAL_DOT_DARKEN)
            num_gens = len(generations)
            seed_dots = VGroup()

            for gen_data in generations:
                generation = gen_data["generation"]
                if generation != (num_gens - 1) and generation % HISTORY_GENERATION_STRIDE != 0:
                    continue
                organisms = gen_data["organisms"]
                is_final = generation == (num_gens - 1)
                r = FINAL_DOT_RADIUS if is_final else HISTORY_DOT_RADIUS
                op = 1.0 if is_final else HISTORY_DOT_OPACITY
                color = final_color if is_final else seed_color
                for pt in organisms:
                    arr = np.asarray(pt).flat
                    x, y = float(arr[0]), float(arr[1])
                    pt_3d = np.array([x + DISPLAY_X_SHIFT, y + DISPLAY_Y_SHIFT, contour_z])
                    dot_contour = Dot(point=pt_3d, color=color, radius=r)
                    dot_contour.set_fill(opacity=op)
                    dot_contour.set_stroke(opacity=op)
                    if is_final:
                        final_dots.add(dot_contour)
                    else:
                        seed_dots.add(dot_contour)
                    z_surf = DISPLAY_SURFACE_FUNC(x, y)[2]
                    pt_on_surface = np.array([x + DISPLAY_X_SHIFT, y + DISPLAY_Y_SHIFT, z_surf])
                    dot_surf = Dot(point=pt_on_surface, color=color, radius=r)
                    dot_surf.set_fill(opacity=op)
                    dot_surf.set_stroke(opacity=op)
                    if is_final:
                        final_dots.add(dot_surf)
                    else:
                        seed_dots.add(dot_surf)
            all_seed_points.add(seed_dots)

            last_orgs = np.array(generations[-1]["organisms"], dtype=float)
            mean_xy = np.mean(last_orgs[:, :2], axis=0)
            mx, my = float(mean_xy[0]) + DISPLAY_X_SHIFT, float(mean_xy[1]) + DISPLAY_Y_SHIFT
            mean_color = MEAN_POINT_COLORS[seed_idx % len(MEAN_POINT_COLORS)]
            mean_height = MEAN_LINE_HEIGHT_MAX - (seed_idx / max(1, n_seeds - 1)) * (MEAN_LINE_HEIGHT_MAX - MEAN_LINE_HEIGHT_MIN)
            z_lo = MEAN_LINE_Z_BOTTOM
            z_hi = MEAN_LINE_Z_BOTTOM + mean_height
            mean_line = Line(start=np.array([mx, my, z_lo]), end=np.array([mx, my, z_hi]), color=mean_color, stroke_width=MEAN_LINE_STROKE_WIDTH)
            mean_lines.add(mean_line)
            mean_lines.add(Dot(point=np.array([mx, my, z_hi]), color=mean_color, radius=MEAN_LINE_DOT_RADIUS))
            final_mean_points.append((seed, mean_xy))

        self.add(all_seed_points, optima_marker, final_dots, mean_lines)

        header_lines = [
            f"{n_seeds} Darwinian runs (1L2D: pop=2)",
            f"max_calls={params.max_calls}, history_stride={HISTORY_GENERATION_STRIDE}",
            f"total_topology_calls={total_dar_calls}",
        ]
        header_text = Text("\n".join(header_lines), font_size=9, color=BLACK, line_spacing=0.75, font="Courier New")
        mean_line_texts = VGroup(
            *[
                Text(
                    f"s{seed} final mean=({mean_xy[0]:.2f},{mean_xy[1]:.2f})",
                    font_size=9,
                    color=MEAN_POINT_COLORS[i % len(MEAN_POINT_COLORS)],
                    line_spacing=0.75,
                    font="Courier New",
                )
                for i, (seed, mean_xy) in enumerate(final_mean_points)
            ]
        )
        mean_line_texts.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        panel = VGroup(header_text, mean_line_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        panel.to_edge(RIGHT).shift(DOWN * 0.3)
        panel.add_background_rectangle(color=WHITE, opacity=0.9, buff=0.08)
        self.add_fixed_in_frame_mobjects(panel)
        self.add(panel)
        self.wait(0.1)
