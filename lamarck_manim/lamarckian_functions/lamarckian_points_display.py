"""
Point-only Lamarckian visualization.

This module renders Lamarckian evolution using point endpoints of organism vectors
plus a clearly marked global optimum, without drawing arrows.
"""

from manim import *
import numpy as np
from pathlib import Path

from core import pure_lamarckian_function, rastrigin_func


def fit_camera_to_points(scene, points, padding_ratio=0.2, min_frame_size=12.0):
    """Frame camera so all provided points are visible."""
    if len(points) == 0:
        return
    arr = np.array(points, dtype=float)
    xs = arr[:, 0]
    ys = arr[:, 1]
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    frame_size = max(width, height) * (1.0 + padding_ratio * 2.0)
    frame_size = max(frame_size, min_frame_size)
    center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, 0.0])
    scene.camera.frame.move_to(center)
    scene.camera.frame.set_width(frame_size)


def read_points_display_version():
    """Read render version from .version_points; return '?' if unavailable."""
    version_path = Path(__file__).with_name(".version_points")
    try:
        return version_path.read_text(encoding="utf-8").strip()
    except Exception:
        return "?"


def rastrigin_local_optima_in_bounds(x_min, x_max, y_min, y_max, include_global=False):
    """
    Return local minima points for Rastrigin within bounds.

    In 2D Rastrigin, minima occur at integer lattice points. The global minimum
    is at (0, 0) and can be excluded so it remains uniquely highlighted.
    """
    x_start = int(np.floor(x_min))
    x_end = int(np.ceil(x_max))
    y_start = int(np.floor(y_min))
    y_end = int(np.ceil(y_max))
    points = []
    for xi in range(x_start, x_end + 1):
        for yi in range(y_start, y_end + 1):
            if not include_global and xi == 0 and yi == 0:
                continue
            points.append(np.array([float(xi), float(yi), 0.0]))
    return points


def add_global_optima_markers(
    scene,
    optima_points,
    marker_color=GOLD,
    marker_radius=0.18,
    label_text_prefix="Global Optimum",
):
    """Draw visible global-optima markers and labels."""
    markers = VGroup()
    for idx, point in enumerate(optima_points):
        if len(point) == 2:
            p = np.array([point[0], point[1], 0.0])
        else:
            p = np.array([point[0], point[1], point[2]])

        ring = Circle(radius=marker_radius, color=marker_color, stroke_width=4).move_to(p)
        center_dot = Dot(point=p, radius=0.08, color=marker_color)
        cross_h = Line(p + LEFT * marker_radius, p + RIGHT * marker_radius, color=marker_color, stroke_width=3)
        cross_v = Line(p + DOWN * marker_radius, p + UP * marker_radius, color=marker_color, stroke_width=3)
        marker = VGroup(ring, center_dot, cross_h, cross_v)

        label = Text(
            f"{label_text_prefix} {idx + 1}: ({p[0]:.1f}, {p[1]:.1f})",
            font_size=20,
            color=marker_color,
        )
        label.next_to(marker, UR, buff=0.2)
        label.add_background_rectangle(color=BLACK, opacity=0.7)
        markers.add(marker, label)

    scene.add(markers)
    return markers


class TestPureLamarckianPoints(MovingCameraScene):
    """Render only vector endpoints and global optimum for Lamarckian evolution."""

    ANIMATION_SPEED = 5.0

    def play(self, *args, **kwargs):
        if "run_time" in kwargs:
            scaled_time = kwargs["run_time"] / self.ANIMATION_SPEED
            frame_rate = getattr(config, "frame_rate", 15.0)
            min_frame_time = 1.0 / frame_rate
            kwargs["run_time"] = max(scaled_time, min_frame_time)
        return super().play(*args, **kwargs)

    def wait(self, duration=1, **kwargs):
        scaled_duration = duration / self.ANIMATION_SPEED
        frame_rate = getattr(config, "frame_rate", 15.0)
        min_frame_time = 1.0 / frame_rate
        return super().wait(max(scaled_duration, min_frame_time), **kwargs)

    def construct(self):
        # Rendering mode toggle:
        # - True: render organism vectors as lines.
        # - False: render start/end as points.
        RENDER_WITH_LINES = True
        TOPOLOGY_FUNCTION = rastrigin_func
        # Start farther from origin for clearer separation from optimum area.
        PARENT1_START = np.array([-6.0, -5.5, 0.0])
        PARENT1_END = np.array([-5.0, -4.2, 0.0])
        PARENT2_START = np.array([5.8, -5.2, 0.0])
        PARENT2_END = np.array([6.9, -4.0, 0.0])
        NUM_OFFSPRING = 2
        NUM_GENERATIONS = 1000
        BESOIN_WEIGHT = 1.1427907844220597
        TOPOLOGY_GRADIENT_SCALE = 0.11282000353881588
        MAGNITUDE_STD_FRACTION = 0.1235121700374864
        MAGNITUDE_WEIGHT = 0.8912894658829036
        DIRECTION_STD = 0.3212217824540174
        MIN_MAGNITUDE = 0.01
        RNG_SEEDS = [7, 27, 107, 207, 327, 507]
        SEED_COLORS = [BLUE, GREEN, RED, PURPLE, TEAL, PINK]

        title = Text("Pure Lamarckian Evolution (Points + Optimum)", font_size=32, color=YELLOW)
        title.to_edge(UP)
        self.add(title)

        # Rastrigin global minimum is at (0, 0).
        optima_points = [np.array([0.0, 0.0, 0.0])]
        add_global_optima_markers(self, optima_points)

        # Run the same lever set across multiple seeds in one render.
        runs = []
        for seed_idx, seed in enumerate(RNG_SEEDS):
            generations = pure_lamarckian_function(
                besoin_topology_function=TOPOLOGY_FUNCTION,
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

            historical_organisms = []
            for gen_data in generations[:-1]:
                historical_organisms.extend(gen_data["organisms"])
            last_gen = generations[-1]
            organisms = last_gen["organisms"]

            runs.append(
                {
                    "seed": seed,
                    "color": SEED_COLORS[seed_idx % len(SEED_COLORS)],
                    "last_gen_idx": last_gen["generation"],
                    "history": historical_organisms,
                    "current": organisms,
                }
            )

        all_points_for_camera = optima_points.copy()
        for run in runs:
            all_points_for_camera.extend([s for s, _ in run["history"]])
            all_points_for_camera.extend([e for _, e in run["history"]])
            all_points_for_camera.extend([s for s, _ in run["current"]])
            all_points_for_camera.extend([e for _, e in run["current"]])
        if len(all_points_for_camera) > 0:
            arr = np.array(all_points_for_camera, dtype=float)
            x_min = float(np.min(arr[:, 0]))
            x_max = float(np.max(arr[:, 0]))
            y_min = float(np.min(arr[:, 1]))
            y_max = float(np.max(arr[:, 1]))
        else:
            x_min, x_max, y_min, y_max = -8.0, 8.0, -8.0, 8.0

        local_optima_points = rastrigin_local_optima_in_bounds(x_min, x_max, y_min, y_max, include_global=False)
        local_optima_dots = VGroup(
            *[Dot(point=p, radius=0.032, color=ORANGE, fill_opacity=0.9, stroke_opacity=0.9) for p in local_optima_points]
        )
        self.add(local_optima_dots)

        fit_camera_to_points(self, all_points_for_camera, padding_ratio=0.25, min_frame_size=16.0)

        # Small reproducibility panel on the right.
        config_panel = Text(
            "\n".join(
                [
                    f"fn={TOPOLOGY_FUNCTION.__name__}",
                    f"seeds={RNG_SEEDS}",
                    f"mode={'lines' if RENDER_WITH_LINES else 'points'}",
                    f"num_offspring={NUM_OFFSPRING}",
                    f"num_generations={NUM_GENERATIONS}",
                    f"besoin_weight={BESOIN_WEIGHT}",
                    f"topo_grad_scale={TOPOLOGY_GRADIENT_SCALE}",
                    f"magnitude_std_fraction={MAGNITUDE_STD_FRACTION}",
                    f"magnitude_weight={MAGNITUDE_WEIGHT}",
                    f"direction_std={DIRECTION_STD}",
                    f"min_magnitude={MIN_MAGNITUDE}",
                    f"p1s=({PARENT1_START[0]:.1f},{PARENT1_START[1]:.1f})",
                    f"p1e=({PARENT1_END[0]:.1f},{PARENT1_END[1]:.1f})",
                    f"p2s=({PARENT2_START[0]:.1f},{PARENT2_START[1]:.1f})",
                    f"p2e=({PARENT2_END[0]:.1f},{PARENT2_END[1]:.1f})",
                ]
            ),
            font_size=13,
            color=GRAY_B,
            line_spacing=0.75,
        )
        config_panel.to_edge(RIGHT).shift(DOWN * 0.5)
        config_panel.add_background_rectangle(color=BLACK, opacity=0.65, buff=0.08)
        self.add(config_panel)

        # Per-seed legend on the right for quick visual mapping.
        legend_lines = [f"seed {run['seed']} -> color" for run in runs]
        legend_text = Text(
            "\n".join(legend_lines),
            font_size=14,
            color=WHITE,
            line_spacing=0.75,
        )
        legend_text.to_edge(RIGHT).shift(UP * 2.5)
        legend_text.add_background_rectangle(color=BLACK, opacity=0.55, buff=0.08)
        self.add(legend_text)

        legend_markers = VGroup()
        for i, run in enumerate(runs):
            marker = Dot(color=run["color"], radius=0.06)
            marker.move_to(legend_text.get_left() + RIGHT * 0.35 + DOWN * (0.34 + i * 0.28))
            legend_markers.add(marker)
        self.add(legend_markers)

        if RENDER_WITH_LINES:
            historical_lines = VGroup()
            current_lines = VGroup()
            total_history = 0
            total_current = 0
            for run in runs:
                for s, e in run["history"]:
                    historical_lines.add(
                        Line(start=s, end=e, color=run["color"], stroke_width=1.0, stroke_opacity=0.20)
                    )
                for s, e in run["current"]:
                    current_lines.add(
                        Line(start=s, end=e, color=run["color"], stroke_width=2.0, stroke_opacity=0.92)
                    )
                total_history += len(run["history"])
                total_current += len(run["current"])

            if len(historical_lines) > 0:
                self.play(Create(historical_lines), run_time=1.6)
            if len(current_lines) > 0:
                self.play(Create(current_lines), run_time=1.1)
            dots_count = total_current * 2
            history_count = total_history
            current_count = total_current
            last_gen_idx = runs[0]["last_gen_idx"] if len(runs) > 0 else NUM_GENERATIONS - 1
            render_mode_text = "lines"
        else:
            history_dots = VGroup()
            current_dots = VGroup()
            total_history = 0
            total_current = 0
            for run in runs:
                for s, e in run["history"]:
                    history_dots.add(Dot(point=s, radius=0.014, color=run["color"], fill_opacity=0.25, stroke_opacity=0.25))
                    history_dots.add(Dot(point=e, radius=0.014, color=run["color"], fill_opacity=0.25, stroke_opacity=0.25))
                for s, e in run["current"]:
                    current_dots.add(Dot(point=s, radius=0.02, color=run["color"], fill_opacity=0.9, stroke_opacity=0.9))
                    current_dots.add(Dot(point=e, radius=0.02, color=run["color"], fill_opacity=0.9, stroke_opacity=0.9))
                total_history += len(run["history"])
                total_current += len(run["current"])

            if len(history_dots) > 0:
                self.play(Create(history_dots), run_time=1.4)
            if len(current_dots) > 0:
                self.play(Create(current_dots), run_time=1.0)
            dots_count = len(current_dots)
            history_count = total_history
            current_count = total_current
            last_gen_idx = runs[0]["last_gen_idx"] if len(runs) > 0 else NUM_GENERATIONS - 1
            render_mode_text = "points"

        version = read_points_display_version()
        summary = Text(
            (
                f"v{version} | Gen {last_gen_idx} | mode={render_mode_text}: "
                f"seeds={len(runs)} current={current_count} history={history_count} dots={dots_count}"
            ),
            font_size=22,
            color=WHITE,
        )
        summary.to_edge(DOWN)
        summary.add_background_rectangle(color=BLACK, opacity=0.7)

        # Mean of all current organism points (start + end points) at the end.
        current_all_points = []
        for run in runs:
            current_all_points.extend([s for s, _ in run["current"]])
            current_all_points.extend([e for _, e in run["current"]])
        if len(current_all_points) > 0:
            mean_point = np.mean(np.array(current_all_points, dtype=float), axis=0)
            # Print each seed's mean (mean of that run's current start+end points).
            for run in runs:
                pts = [s for s, _ in run["current"]] + [e for _, e in run["current"]]
                if pts:
                    m = np.mean(np.array(pts, dtype=float), axis=0)
                    print(f"FINAL_MEAN_POINT seed={run['seed']} x={m[0]:.6f} y={m[1]:.6f} gen={last_gen_idx}")
            print(f"FINAL_MEAN_POINT (overall) x={mean_point[0]:.6f} y={mean_point[1]:.6f} gen={last_gen_idx} seeds={RNG_SEEDS}")
            mean_marker = Dot(
                point=mean_point,
                radius=0.05,
                color=WHITE,
                fill_opacity=0.45,
                stroke_opacity=0.45,
            )
            mean_label = Text(
                f"Mean: ({mean_point[0]:.2f}, {mean_point[1]:.2f})",
                font_size=20,
                color=WHITE,
            )
            mean_label.next_to(mean_marker, UR, buff=0.15)
            mean_label.add_background_rectangle(color=BLACK, opacity=0.7)
            self.play(Create(mean_marker), run_time=0.5)
            self.play(Write(mean_label), run_time=0.5)

        self.play(Write(summary), run_time=0.6)
        self.wait(1.0)
