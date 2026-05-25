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
    get_rastrigin_wb_high_res_config,
    scene_config_overrides,
)
from visualizationtool.retrograph.retro_construction import construct_retro_style_scene

from optimizationfunctions.evolutionalgorithms.lamarckianfunctions.core import (
    pure_lamarckian_function,
    rastrigin_func,
)
from optimizationfunctions.evolutionalgorithms.metaevolutionfunctions import CountedFunction
from la_girafe import giraffe_line_mobjects


class SixOptimizedLamarckianFunctions(ThreeDScene):
    """Retro-style six-seed Lamarckian overlay, aligned to retro_tester_2 flow."""

    get_default_class_config()

    def construct(self):
        # Tuned levers from meta-evolution.
        BESOIN_WEIGHT = 1.1427907844220597
        TOPOLOGY_GRADIENT_SCALE = 0.11282000353881588
        MAGNITUDE_STD_FRACTION = 0.1235121700374864
        MAGNITUDE_WEIGHT = 0.8912894658829036
        DIRECTION_STD = 0.3212217824540174
        MIN_MAGNITUDE = 0.01
        NUM_OFFSPRING = 2
        NUM_GENERATIONS = 100
        HISTORY_GENERATION_STRIDE = 1
        LINE_STROKE_WIDTH = 0.8
        HISTORY_LINE_OPACITY = 0.5
        FINAL_LINE_DARKEN = 0.5  # interpolate toward BLACK for final generation
        MEAN_LINE_Z_BOTTOM = -20.0
        MEAN_LINE_HEIGHT_MIN = 1.0
        MEAN_LINE_HEIGHT_MAX = 2.0
        MEAN_LINE_STROKE_WIDTH = 1.0
        MEAN_LINE_DOT_RADIUS = 0.08
        LAST_ORG_ARROW_Z_BASE = -19.0
        LAST_ORG_ARROW_THICKNESS = 0.03
        DISPLAY_X_SHIFT = -10.0
        DISPLAY_Y_SHIFT = 15.0

        PARENT1_START = np.array([-6.0, -5.5, 0.0])
        PARENT1_END = np.array([-5.0, -4.2, 0.0])
        PARENT2_START = np.array([5.8, -5.2, 0.0])
        PARENT2_END = np.array([6.9, -4.0, 0.0])

        RNG_SEEDS = [7, 27]  # First two seeds only; giraffes use same colors as these runs
        SEED_COLORS = [BLUE, GREEN, RED, PURPLE, TEAL, PINK]
        MEAN_POINT_COLORS = ["#0D47A1", "#1B5E20", "#B71C1C", "#4A148C", "#004D40", "#880E4F"]  # Dark colors for mean markers
        # Evolution uses same topology as points test (default Rastrigin scale=0.1); dampening only for display.
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
                ANIMATE_GRAPH=False,  # Graph, surface, contours already there (no Create animation)
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
                Z_AXIS_TITLE="NEEDS COEFFICIENT",
                SHOW_SURFACE=True,
                SURFACE_RESOLUTION=(100, 100),
                SURFACE_FILL_OPACITY=1.0,
                GAUSSIAN_AMPLITUDE=2.0,
                SHOW_CONTOUR_LINES=False,
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
                LABEL_FONT_SIZE=96,        # 3x larger tick labels
                AXIS_TITLE_FONT_SIZE=96,   # 3x larger axis titles (x, y, z)
                Z_AXIS_TITLE_OFFSET=3.2,   # Push z title farther from axis labels.
                TITLE_SIZE=108,            # 3x larger scene title if enabled
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

        # Shift the retro graph layer so it aligns with shifted function overlays.
        for mob in self.mobjects:
            mob.shift(np.array([DISPLAY_X_SHIFT, DISPLAY_Y_SHIFT, 0.0]))

        contour_z = scene_elements["config"].Z_AXIS_RANGE_MIN
        # Dashed vertical line from just below (0,0,0) to (0,0,-20) at global optimum (0,0) in xy. Rendered last so it draws over the functions.
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

        all_seed_lines = VGroup()
        final_lines = VGroup()  # final generation lines/dots (dark, rendered on top)
        mean_lines = VGroup()   # mean vertical lines + dots, rendered last
        last_org_markers = VGroup()  # tip-at-z markers for last organism per seed
        final_mean_points = []

        for seed_idx, seed in enumerate(RNG_SEEDS):
            generations = pure_lamarckian_function(
                besoin_topology_function=EVOLUTION_TOPOLOGY,
                parent1_start=PARENT1_START,
                parent1_end=PARENT1_END,
                parent2_start=PARENT2_START,
                parent2_end=PARENT2_END,
                num_offspring=NUM_OFFSPRING,
                num_generations=NUM_GENERATIONS,
                besoin_weight=BESOIN_WEIGHT,
                topology_gradient_scale=TOPOLOGY_GRADIENT_SCALE,
                magnitude_std_fraction=MAGNITUDE_STD_FRACTION,
                magnitude_weight=MAGNITUDE_WEIGHT,
                direction_std=DIRECTION_STD,
                min_magnitude=MIN_MAGNITUDE,
                seed=seed,
            )

            seed_color = SEED_COLORS[seed_idx % len(SEED_COLORS)]
            final_color = interpolate_color(seed_color, BLACK, FINAL_LINE_DARKEN)
            num_gens = len(generations)
            seed_lines = VGroup()

            for gen_data in generations:
                generation = gen_data["generation"]
                if generation != (num_gens - 1) and generation % HISTORY_GENERATION_STRIDE != 0:
                    continue

                organisms = gen_data["organisms"]
                is_final = generation == (num_gens - 1)
                color = final_color if is_final else seed_color
                line_opacity = 1.0 if is_final else HISTORY_LINE_OPACITY

                for org_start, org_end in organisms:
                    start_3d = np.array([float(org_start[0]) + DISPLAY_X_SHIFT, float(org_start[1]) + DISPLAY_Y_SHIFT, contour_z])
                    end_3d = np.array([float(org_end[0]) + DISPLAY_X_SHIFT, float(org_end[1]) + DISPLAY_Y_SHIFT, contour_z])
                    line = Line(start=start_3d, end=end_3d, color=color, stroke_width=LINE_STROKE_WIDTH)
                    line.set_stroke(opacity=line_opacity)
                    if is_final:
                        final_lines.add(line)
                    else:
                        seed_lines.add(line)
                    # Project organism points onto the surface (z from display topology).
                    z_start = DISPLAY_SURFACE_FUNC(float(org_start[0]), float(org_start[1]))[2]
                    z_end = DISPLAY_SURFACE_FUNC(float(org_end[0]), float(org_end[1]))[2]
                    start_on_surface = np.array([float(org_start[0]) + DISPLAY_X_SHIFT, float(org_start[1]) + DISPLAY_Y_SHIFT, z_start])
                    end_on_surface = np.array([float(org_end[0]) + DISPLAY_X_SHIFT, float(org_end[1]) + DISPLAY_Y_SHIFT, z_end])
                    dot_s = Dot(point=start_on_surface, color=color, radius=0.06)
                    dot_e = Dot(point=end_on_surface, color=color, radius=0.06)
                    dot_s.set_fill(opacity=line_opacity)
                    dot_s.set_stroke(opacity=line_opacity)
                    dot_e.set_fill(opacity=line_opacity)
                    dot_e.set_stroke(opacity=line_opacity)
                    if is_final:
                        final_lines.add(dot_s, dot_e)
                    else:
                        seed_lines.add(dot_s, dot_e)

            all_seed_lines.add(seed_lines)

            last_orgs = generations[-1]["organisms"]
            endpoints = np.array([end for _, end in last_orgs], dtype=float)
            mean_xy = np.mean(endpoints[:, :2], axis=0)
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
            # Arrow pointing at last organism (from z=-19 to organism on contour plane)
            last_start, last_end = last_orgs[-1]
            tip_x = float(last_end[0]) + DISPLAY_X_SHIFT
            tip_y = float(last_end[1]) + DISPLAY_Y_SHIFT
            arrow_start = np.array([tip_x, tip_y, LAST_ORG_ARROW_Z_BASE])
            arrow_end = np.array([tip_x, tip_y, contour_z])
            last_org_markers.add(Arrow3D(start=arrow_start, end=arrow_end, color=final_color, thickness=LAST_ORG_ARROW_THICKNESS))
            final_mean_points.append((seed, mean_xy))

        self.add(all_seed_lines, optima_marker, final_lines, mean_lines, last_org_markers)

        # Two giraffes: neck length = Y trait, leg length = X trait (from each run's final mean).
        axis_min = scene_elements["config"].AXIS_RANGE_MIN
        axis_max = scene_elements["config"].AXIS_RANGE_MAX
        axis_span = axis_max - axis_min

        def trait_to_length(trait_val):
            """Map trait value in [axis_min, axis_max] to length in [0.2, 1.0]."""
            t = np.clip((trait_val - axis_min) / axis_span, 0.0, 1.0)
            return 0.2 + 0.8 * float(t)

        giraffe_scale = 0.9
        stroke = 6
        # Seed 0: final mean (x, y) -> leg_length from x, neck_length from y
        _, mean_xy0 = final_mean_points[0]
        leg_len0 = trait_to_length(mean_xy0[0])
        neck_len0 = trait_to_length(mean_xy0[1])
        giraffe1 = giraffe_line_mobjects(
            color=SEED_COLORS[0],
            location=RIGHT * 3.8 + DOWN * 1.5,
            stroke_width=stroke,
            scale=giraffe_scale,
            alignment="bot",
            giraffe_direction="right",
            leg_length=leg_len0,
            neck_length=neck_len0,
        )
        # Seed 1: same mapping
        _, mean_xy1 = final_mean_points[1]
        leg_len1 = trait_to_length(mean_xy1[0])
        neck_len1 = trait_to_length(mean_xy1[1])
        giraffe2 = giraffe_line_mobjects(
            color=SEED_COLORS[1],
            location=RIGHT * 5.2 + DOWN * 1.5,
            stroke_width=stroke,
            scale=giraffe_scale,
            alignment="bot",
            giraffe_direction="left",
            leg_length=leg_len1,
            neck_length=neck_len1,
        )
        giraffes = VGroup(giraffe1, giraffe2)
        self.add_fixed_in_frame_mobjects(giraffes)
        self.add(giraffes)
        self.wait(0.1)