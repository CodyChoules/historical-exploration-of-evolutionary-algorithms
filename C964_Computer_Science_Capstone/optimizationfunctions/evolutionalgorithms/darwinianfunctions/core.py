"""
Darwinian Evolution Functions Module

This module implements a simple Darwinian process using organism points.
Each generation:
1) spawns candidate organisms as points,
2) evaluates fitness on a topology function,
3) probabilistically eliminates less-fit organisms,
4) repopulates from survivors with mutation.
"""

from manim import *
import numpy as np
import random

from problemspace.surfacefunctions import rastrigin_func

def _extract_fitness_value(function_result):
    """Extract scalar fitness from z-value or scalar function output."""
    if isinstance(function_result, np.ndarray):
        if len(function_result) >= 3:
            return float(function_result[2])
        if len(function_result) >= 1:
            return float(function_result[-1])
    return float(function_result)


def _sample_initial_population(population_size, initial_bounds):
    """Sample random 2D points and return as Nx3 points (z fixed at 0)."""
    x_min, x_max, y_min, y_max = initial_bounds
    xs = np.random.uniform(x_min, x_max, population_size)
    ys = np.random.uniform(y_min, y_max, population_size)
    zs = np.zeros(population_size)
    return np.column_stack((xs, ys, zs))


def _select_survivor_indices(fitness_values, survivor_count, selection_pressure):
    """
    Select survivors without replacement.

    Lower fitness => higher survival probability.
    """
    if survivor_count <= 0:
        return np.array([], dtype=int)
    if survivor_count >= len(fitness_values):
        return np.arange(len(fitness_values), dtype=int)

    f = np.asarray(fitness_values, dtype=float)
    f_min = float(np.min(f))
    f_max = float(np.max(f))
    if f_max > f_min:
        normalized = (f - f_min) / (f_max - f_min)
    else:
        normalized = np.zeros_like(f)

    weights = np.exp(-selection_pressure * normalized)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        probabilities = np.full(len(f), 1.0 / len(f))
    else:
        probabilities = weights / weight_sum

    return np.random.choice(len(f), size=survivor_count, replace=False, p=probabilities)


def pure_darwinian_function(
    fitness_topology_function,
    population_size=32,
    num_generations=10,
    elimination_rate=0.5,
    selection_pressure=4.0,
    mutation_std=0.8,
    seed=None,
    initial_bounds=(-10.0, 10.0, -10.0, 10.0),
    max_calls=None,
    initial_population=None,
):
    """
    Pure Darwinian evolution using points, fitness, and probabilistic elimination.

    Args:
        fitness_topology_function: function(x, y) -> z or [x, y, z]
        population_size: number of organisms per generation
        num_generations: number of generations to simulate
        elimination_rate: fraction eliminated each generation (0 to <1)
        selection_pressure: higher => stronger preference for low-fitness survivors
        mutation_std: std-dev for Gaussian offspring mutation in x/y
        seed: optional random seed
        initial_bounds: tuple (x_min, x_max, y_min, y_max)
        max_calls: int or None; if set, stop when fitness_topology_function.n_calls >= max_calls
                   (requires topology to be a CountedFunction or have n_calls attribute). Overrides num_generations as limit.
        initial_population: array of shape (N, 3) or None; if provided, use these points as the initial population
                           (population_size becomes N). Each row is (x, y, z) with z typically 0.

    Returns:
        list[dict]: generation records with organism points, fitness, survivors, and eliminated.
    """
    if not (0 <= elimination_rate < 1):
        raise ValueError("elimination_rate must satisfy 0 <= elimination_rate < 1.")

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    x_min, x_max, y_min, y_max = initial_bounds
    if initial_population is not None:
        current_population = np.asarray(initial_population, dtype=float)
        if current_population.ndim != 2 or current_population.shape[1] < 2:
            raise ValueError("initial_population must be array of shape (N, 2) or (N, 3).")
        if current_population.shape[0] < 2:
            raise ValueError("initial_population must have at least 2 points.")
        if current_population.shape[1] == 2:
            z = np.zeros((current_population.shape[0], 1))
            current_population = np.hstack([current_population, z])
        population_size = current_population.shape[0]
    else:
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        current_population = _sample_initial_population(population_size, initial_bounds)
    # Number of survivors per generation. At least 1 so we can repopulate (offspring from survivor(s)).
    # For population_size=2 (asexual 1L2D): we allow 1 survivor so one organism lives and one dies;
    # _select_survivor_indices chooses the survivor with probability favoring higher fitness (lower
    # fitness value) according to selection_pressure. Next generation = 1 survivor + 1 mutated offspring.
    survivors_target = max(1, int(round(population_size * (1.0 - elimination_rate))))
    generations = []

    for generation in range(num_generations):
        fitness_values = np.array(
            [_extract_fitness_value(fitness_topology_function(p[0], p[1])) for p in current_population],
            dtype=float,
        )
        # Selection: who survives. "Death" = not being chosen as a survivor. Lower fitness =>
        # higher survival probability (selection_pressure controls how strong the preference is).
        survivor_indices = _select_survivor_indices(fitness_values, survivors_target, selection_pressure)
        eliminated_mask = np.ones(population_size, dtype=bool)
        eliminated_mask[survivor_indices] = False
        eliminated_indices = np.where(eliminated_mask)[0]

        survivor_points = current_population[survivor_indices].copy()
        survivor_fitness = fitness_values[survivor_indices].copy()
        eliminated_points = current_population[eliminated_indices].copy()
        eliminated_fitness = fitness_values[eliminated_indices].copy()

        generations.append(
            {
                "generation": generation,
                "organisms": [p.copy() for p in current_population],
                "fitness_values": [float(v) for v in fitness_values],
                "survivor_indices": [int(i) for i in survivor_indices],
                "survivors": [p.copy() for p in survivor_points],
                "survivor_fitness": [float(v) for v in survivor_fitness],
                "eliminated_indices": [int(i) for i in eliminated_indices],
                "eliminated": [p.copy() for p in eliminated_points],
                "eliminated_fitness": [float(v) for v in eliminated_fitness],
            }
        )

        if generation == num_generations - 1:
            break
        if max_calls is not None and getattr(fitness_topology_function, "n_calls", 0) >= max_calls:
            break

        # Repopulate: copy survivors into next generation, then fill remaining slots with
        # mutated offspring (Gaussian perturbation in x, y from a random survivor).
        next_population = np.zeros((population_size, 3), dtype=float)
        next_population[: len(survivor_points)] = survivor_points
        for idx in range(len(survivor_points), population_size):
            parent = survivor_points[np.random.randint(0, len(survivor_points))]
            child_x = np.clip(parent[0] + np.random.normal(0, mutation_std), x_min, x_max)
            child_y = np.clip(parent[1] + np.random.normal(0, mutation_std), y_min, y_max)
            next_population[idx] = np.array([child_x, child_y, 0.0])

        current_population = next_population

    return generations


def add_global_optima_markers(
    scene,
    optima_points,
    marker_color=GOLD,
    marker_radius=0.18,
    label_text_prefix="Global Optimum",
):
    """
    Draw clearly visible global-optima markers and labels.

    Args:
        scene: Manim scene instance
        optima_points: iterable of points ([x, y] or [x, y, z])
        marker_color: color for marker and label
        marker_radius: radius for surrounding ring
        label_text_prefix: label prefix used for each optimum

    Returns:
        VGroup containing all markers and labels.
    """
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


class TestPureDarwinianFunction(MovingCameraScene):
    """Visual test scene for point-based Darwinian evolution."""

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
        # Style toggle:
        # - True: all plotted dots use same size and opacity.
        # - False: use previous differentiated style.
        USE_UNIFORM_DOT_STYLE = True
        UNIFORM_DOT_RADIUS = 0.04
        UNIFORM_DOT_OPACITY = 0.55
        # Blue dots (history + eliminated) can be tuned together.
        BLUE_DOT_RADIUS = 0.032
        BLUE_DOT_OPACITY = 0.35

        title = Text("Pure Darwinian Evolution (Points + Selection)", font_size=32, color=YELLOW)
        title.to_edge(UP)
        self.add(title)

        # Rastrigin global minimum is at (0, 0).
        add_global_optima_markers(self, [np.array([0.0, 0.0, 0.0])])

        generations = pure_darwinian_function(
            fitness_topology_function=rastrigin_func,
            population_size=100,
            num_generations=200,
            elimination_rate=0.5,
            selection_pressure=4.0,
            mutation_std=0.6,
            seed=7,
        )

        # Build faded history from prior generations.
        historical_points = []
        for gen_data in generations[:-1]:
            historical_points.extend(gen_data["organisms"])

        total_history_points = len(historical_points)
        max_history_points_for_render = 12000
        if total_history_points > max_history_points_for_render:
            sampled_indices = np.random.choice(
                total_history_points, size=max_history_points_for_render, replace=False
            )
            historical_points = [historical_points[i] for i in sampled_indices]

        if USE_UNIFORM_DOT_STYLE:
            historical_dots = VGroup(
                *[
                    Dot(
                        point=p,
                        radius=BLUE_DOT_RADIUS,
                        color=BLUE_C,
                        fill_opacity=BLUE_DOT_OPACITY,
                        stroke_opacity=BLUE_DOT_OPACITY,
                    )
                    for p in historical_points
                ]
            )
        else:
            historical_dots = VGroup(
                *[
                    Dot(point=p, radius=0.018, color=BLUE_C, fill_opacity=0.30, stroke_opacity=0.30)
                    for p in historical_points
                ]
            )

        # Display current generation: survivors (green) and eliminated (blue).
        last_gen = generations[-1]
        survivors = last_gen["survivors"]
        eliminated = last_gen["eliminated"]

        if USE_UNIFORM_DOT_STYLE:
            survivor_dots = VGroup(
                *[
                    Dot(
                        point=p,
                        radius=UNIFORM_DOT_RADIUS,
                        color=GREEN,
                        fill_opacity=UNIFORM_DOT_OPACITY,
                        stroke_opacity=UNIFORM_DOT_OPACITY,
                    )
                    for p in survivors
                ]
            )
            eliminated_dots = VGroup(
                *[
                    Dot(
                        point=p,
                        radius=BLUE_DOT_RADIUS,
                        color=BLUE,
                        fill_opacity=BLUE_DOT_OPACITY,
                        stroke_opacity=BLUE_DOT_OPACITY,
                    )
                    for p in eliminated
                ]
            )
        else:
            survivor_dots = VGroup(*[Dot(point=p, radius=0.06, color=GREEN) for p in survivors])
            eliminated_dots = VGroup(*[Dot(point=p, radius=0.04, color=BLUE) for p in eliminated])

        # Frame camera to include history, current population, and optimum marker.
        all_points_for_camera = historical_points + survivors + eliminated + [np.array([0.0, 0.0, 0.0])]
        if len(all_points_for_camera) > 0:
            arr = np.array(all_points_for_camera, dtype=float)
            xs = arr[:, 0]
            ys = arr[:, 1]
            min_x, max_x = float(np.min(xs)), float(np.max(xs))
            min_y, max_y = float(np.min(ys)), float(np.max(ys))
            width = max(max_x - min_x, 1.0)
            height = max(max_y - min_y, 1.0)
            frame_size = max(width, height) * 1.5
            frame_size = max(frame_size, 16.0)
            center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, 0.0])
            self.camera.frame.move_to(center)
            self.camera.frame.set_width(frame_size)
        else:
            min_x, max_x, min_y, max_y = -10.0, 10.0, -10.0, 10.0

        # Add local optima markers as small orange dots.
        local_optima_points = rastrigin_local_optima_in_bounds(
            min_x, max_x, min_y, max_y, include_global=False
        )
        if USE_UNIFORM_DOT_STYLE:
            local_optima_dots = VGroup(
                *[
                    Dot(
                        point=p,
                        radius=UNIFORM_DOT_RADIUS,
                        color=ORANGE,
                        fill_opacity=UNIFORM_DOT_OPACITY,
                        stroke_opacity=UNIFORM_DOT_OPACITY,
                    )
                    for p in local_optima_points
                ]
            )
        else:
            local_optima_dots = VGroup(
                *[
                    Dot(point=p, radius=0.028, color=ORANGE, fill_opacity=0.9, stroke_opacity=0.9)
                    for p in local_optima_points
                ]
            )
        self.add(local_optima_dots)

        self.play(FadeIn(historical_dots), run_time=1.0)
        self.play(Create(eliminated_dots), run_time=1.2)
        self.play(Create(survivor_dots), run_time=1.2)

        summary = Text(
            (
                f"Gen {last_gen['generation']}: survivors={len(survivors)} "
                f"eliminated={len(eliminated)} history={len(historical_points)}"
            ),
            font_size=22,
            color=WHITE,
        )
        summary.to_edge(DOWN)
        summary.add_background_rectangle(color=BLACK, opacity=0.7)
        self.play(Write(summary), run_time=0.6)
        self.wait(1.0)


def animate_darwinian_selection(
    scene,
    generations,
    dot_radius=0.06,
    population_color=BLUE,
    survivor_color=GREEN,
    eliminated_color=RED,
    cycle_run_time=0.8,
):
    """
    Animate Darwinian selection over generations on a Scene.

    This is intentionally reusable so other scenes can call it later.
    """
    if len(generations) == 0:
        return VGroup()

    # Start with generation 0 as a population cloud.
    current_dots = VGroup(
        *[Dot(point=p, radius=dot_radius, color=population_color) for p in generations[0]["organisms"]]
    )
    scene.play(Create(current_dots), run_time=cycle_run_time)

    for gen_data in generations:
        survivor_set = {tuple(np.round(p, 8)) for p in gen_data["survivors"]}
        survivors = VGroup()
        eliminated = VGroup()

        for dot in current_dots:
            key = tuple(np.round(dot.get_center(), 8))
            if key in survivor_set:
                survivors.add(dot)
            else:
                eliminated.add(dot)

        # Emphasize survivors and remove eliminated.
        animations = []
        if len(eliminated) > 0:
            animations.append(FadeOut(eliminated))
        if len(survivors) > 0:
            animations.append(survivors.animate.set_color(survivor_color))
        if len(animations) > 0:
            scene.play(*animations, run_time=cycle_run_time)

        # If this is the final generation, keep final survivor state.
        if gen_data["generation"] == generations[-1]["generation"]:
            current_dots = survivors
            break

        # Spawn next generation offspring from next generation data.
        next_population = generations[gen_data["generation"] + 1]["organisms"]
        new_points = []
        survivor_points = {tuple(np.round(p, 8)) for p in gen_data["survivors"]}
        for p in next_population:
            if tuple(np.round(p, 8)) not in survivor_points:
                new_points.append(p)

        offspring = VGroup(*[Dot(point=p, radius=dot_radius, color=population_color) for p in new_points])
        if len(offspring) > 0:
            scene.play(FadeIn(offspring), run_time=cycle_run_time)

        # Next cycle starts from survivors + offspring.
        current_dots = VGroup(*survivors, *offspring)

    return current_dots


class DarwinianSelectionAnimation(MovingCameraScene):
    """Reusable scene that animates generation-by-generation selection."""

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
        title = Text("Darwinian Selection Animation", font_size=32, color=YELLOW)
        title.to_edge(UP)
        self.add(title)

        # Rastrigin global minimum is at (0, 0).
        add_global_optima_markers(self, [np.array([0.0, 0.0, 0.0])])

        generations = pure_darwinian_function(
            fitness_topology_function=rastrigin_func,
            population_size=100,
            num_generations=500,
            elimination_rate=0.5,
            selection_pressure=4.0,
            mutation_std=0.55,
            seed=9,
        )
        animate_darwinian_selection(self, generations, cycle_run_time=0.9)
        self.wait(1.0)
