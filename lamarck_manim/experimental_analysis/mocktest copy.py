#!/usr/bin/env python3
"""
Run VisualizationScene with custom attributes (surface function, evolutionary algorithm, levers).
Sets class attributes, then builds the scene by instantiating and calling render().

Usage (from project root):
    python experimental_analysis/mocktest.py

Output: snapshot PNG to experimental_analysis/mockmedia/images/; opens after render (preview). Optional: MANIM_QUALITY=l|m|h (default: "l").
"""

import os
import sys
from pathlib import Path

# Project root so we can import experimental_analysis and its dependencies
_project_root = Path(__file__).resolve().parent.parent
_this_dir = Path(__file__).resolve().parent
MOCKMEDIA_DIR = _this_dir / "mockmedia"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import after path is set
from manim import BLACK, BLUE, RED, config
from lamarckian_functions.core import pure_lamarckian_function, rastrigin_func
from darwinian_functions.core import pure_darwinian_function
from experimental_analysis.vizualization import VisualizationScene


def main() -> None:
    # Run from project root so media/ and imports resolve
    os.chdir(_project_root)

    # Manim config: output to mockmedia next to this file; low quality; snapshot; preview
    config.media_dir = str(MOCKMEDIA_DIR)
    config.quality = "fourk_quality"
    config.save_last_frame = True  # render as snapshot (PNG of last frame) instead of video
    config.preview = True  # open output after render (-p)

    # --- Set attributes on the scene class (read by construct()) ---
    VisualizationScene.surface_function = rastrigin_func
    VisualizationScene.surface_name = "rastrigin"
    VisualizationScene.evolutionary_algorithm = pure_lamarckian_function
    VisualizationScene.experiment_name = "mocktest_run"

    # --- Run algorithm with optimized values and attach dataset for visualization ---
    # Termination condition: fixed number of generations (1000). No fitness threshold or early stop.
    generations = pure_lamarckian_function(
        besoin_topology_function=rastrigin_func,
        num_offspring=2,
        num_generations=1000,
        besoin_weight=1.1427907844220597,
        topology_gradient_scale=0.11282000353881588,
        magnitude_std_fraction=0.1235121700374864,
        magnitude_weight=0.8912894658829036,
        direction_std=0.3212217824540174,
        min_magnitude=0.01,
        seed=7,
        initial_bounds=(-12.0, 12.0, -12.0, 12.0),  # match viz axes
    )
    # Output entire dataset of generations
    print("pure_lamarckian_function full dataset:")
    print(f"  {len(generations)} generations\n")
    for gen_idx, gen in enumerate(generations):
        orgs = gen.get("organisms", [])
        print(f"  gen[{gen_idx}] ({len(orgs)} organisms):")
        for o_idx, (start, end) in enumerate(orgs):
            s = getattr(start, "tolist", lambda: list(start))()
            e = getattr(end, "tolist", lambda: list(end))()
            print(f"    organism[{o_idx}]: start {s}, end {e}")
        print()
    # Darwinian: optimized levers, on second contour plane (index 1).
    # Termination condition: same as Lamarckian — fixed num_generations=1000.
    darwinian_generations = pure_darwinian_function(
        fitness_topology_function=rastrigin_func,
        population_size=4,
        num_generations=1000,
        elimination_rate=0.2126,
        selection_pressure=5.495,
        mutation_std=0.368,
        seed=7,
        initial_bounds=(-12.0, 12.0, -12.0, 12.0),
    )

    # Two runs: Lamarckian on plane 0 (main), Darwinian on plane 1 (second contour surface)
    # Final points on both planes drawn in darker color for emphasis.
    VisualizationScene.algorithm_runs = [
        {
            "dataset": generations,
            "dataset_type": "lamarckian",
            "contour_plane_index": 0,
            "render_mode": "vectors",
            "color": RED,
            "final_color": "#8B0000",  # dark red for final endpoints
            "final_point_radius": 0.12,
            "initial_marker_radius": 0.5,
            "initial_marker_color": "#8B0000",  # dark red circle at initial
            "generation_stride": 1,
        },
        {
            "dataset": darwinian_generations,
            "dataset_type": "darwinian",
            "contour_plane_index": 1,
            "render_mode": "points",
            "color": BLUE,
            "final_color": "#00008B",  # dark blue for final points
            "point_radius": 0.14,
            "final_point_radius": 0.18,
            "initial_marker_radius": 0.5,
            "initial_marker_color": "#00008B",  # dark blue circle at initial
            "generation_stride": 1,
        },
    ]

    # --- Build and render the scene in this process ---
    scene = VisualizationScene()
    try:
        scene.render()
    except FileNotFoundError:
        pass  # preview tried to open video path but we only saved last frame


if __name__ == "__main__":
    main()
