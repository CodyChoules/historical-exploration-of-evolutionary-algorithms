#!/usr/bin/env python3
"""
Run VisualizationScene using outputs from UP1 or MD2 experiments.

- **UP1** (run_up1.py): Unoptimized Pure Lamarckian v Darwinian; random levers per seed.
- **MD2** (run_MD2.py): Meta-optimized (Darwinian-tuned) Lamarckian v Darwinian.
- **MDSINGLE** (run_MDSingle.py): UP1 with fixed MD2-optimized levers (seed 42) for all seeds.
- **LD4** (run_LD4.py): Sampling-based Lamarckian (random-sample besoin) v Darwinian (fixed MDSingle levers).
- **TSP3** (run_TSP3.py): MD2-style meta-optimized TSP (Lamarckian 2-opt + Darwinian permutation GA); no 3D viz, results only.
- Lamarckian runs → plane 0; Darwinian runs → plane 1. Color is consistent per seed.

Usage (from project root):
    python experimental_analysis/mocktest.py
        (default: experiment=md2, 6 seeds 7,27,107,207,327,507)
    python experimental_analysis/mocktest.py --experiment up1
    python experimental_analysis/mocktest.py --experiment md2 --seed N
    python experimental_analysis/mocktest.py --experiment mdsingle
    python experimental_analysis/mocktest.py --experiment ld4
    python experimental_analysis/mocktest.py --experiment tsp3 [--seed N] [--seeds 7,27,107] [--save tsp3_result.pkl]
    python experimental_analysis/mocktest.py --experiment up1 --seeds 7,27,107 --save up1_result.pkl
    python experimental_analysis/mocktest.py --experiment md2 --load md2_result.pkl

Output: snapshot PNG to experimental_analysis/mockmedia/images/ (except tsp3); tsp3 prints summary only. Use -p to preview.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

# Project root so we can import experimental_analysis and its dependencies
_project_root = Path(__file__).resolve().parent.parent
_this_dir = Path(__file__).resolve().parent
MOCKMEDIA_DIR = _this_dir / "mockmedia"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import after path is set
from manim import BLACK, BLUE, GREEN, RED, PURPLE, TEAL, PINK, MAROON, GOLD, ORANGE, YELLOW, config
from lamarckian_functions.core import rastrigin_func
from experimental_analysis.run_up1 import run_up1, run_up1_multi, UP1_CALL_BUDGET, UP1_INITIAL_BOUNDS, DEFAULT_10_SEEDS as UP1_DEFAULT_10_SEEDS
from experimental_analysis.run_MD2 import (
    run_md2,
    run_md2_multi,
    MD2_CALL_BUDGET,
    MD2_META_CALL_BUDGET,
    MD2_META_POPULATION,
    MD2_META_GENERATIONS,
    MD2_META_ELITE,
    DEFAULT_10_SEEDS as MD2_DEFAULT_10_SEEDS,
)
from experimental_analysis import run_MDSingle
from experimental_analysis.run_LD4 import run_ld4, run_ld4_multi, LD4_CALL_BUDGET
from experimental_analysis.run_TSP3 import run_tsp3, TSP3_CALL_BUDGET
from experimental_analysis.vizualization import VisualizationScene

# Same seed list in both modules; use one for --num-seeds
DEFAULT_10_SEEDS = UP1_DEFAULT_10_SEEDS

# Seeds used when no --seed, --seeds, or --num-seeds is given
DEFAULT_SEEDS = [7, 27, 107, 207, 327, 507]

# Per-seed colors: same index = same seed on plane 0 (Lamarckian) and plane 1 (Darwinian)
SEED_COLORS = [BLUE, GREEN, RED, PURPLE, TEAL, PINK, MAROON, GOLD, ORANGE, YELLOW]
FINAL_SEED_COLORS = [
    "#0D47A1", "#1B5E20", "#B71C1C", "#4A148C", "#004D40", "#880E4F",
    "#6D4C41", "#F57F17", "#E65100", "#F9A825",
]


def _build_algorithm_runs_from_results(results: list):
    """
    Build VisualizationScene.algorithm_runs from a list of experiment result dicts
    (run_up1 / run_MD2 shape: lam_generations, dar_generations, seed, ...).
    Each seed gets one color; Lamarckian (plane 0) and Darwinian (plane 1) for that seed
    use the same color.
    """
    runs = []
    for seed_idx, result in enumerate(results):
        lam_gen = result.get("lam_generations") or []
        dar_gen = result.get("dar_generations") or []
        color = SEED_COLORS[seed_idx % len(SEED_COLORS)]
        final_hex = FINAL_SEED_COLORS[seed_idx % len(FINAL_SEED_COLORS)]
        # Lamarckian → first plane (0)
        runs.append({
            "dataset": lam_gen,
            "dataset_type": "lamarckian",
            "contour_plane_index": 0,
            "render_mode": "vectors",
            "color": color,
            "final_color": final_hex,
            "final_point_radius": 0.12,
            "initial_marker_radius": 0.5,
            "initial_marker_color": final_hex,
            "generation_stride": 1,
        })
        # Darwinian → second plane (1)
        runs.append({
            "dataset": dar_gen,
            "dataset_type": "darwinian",
            "contour_plane_index": 1,
            "render_mode": "points",
            "color": color,
            "final_color": final_hex,
            "point_radius": 0.14,
            "final_point_radius": 0.18,
            "initial_marker_radius": 0.5,
            "initial_marker_color": final_hex,
            "generation_stride": 1,
        })
    return runs


EXPERIMENT_CHOICES = ("up1", "md2", "mdsingle", "ld4", "tsp3")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize UP1 or MD2 experiment (run_up1 / run_MD2 output)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=EXPERIMENT_CHOICES,
        default="md2",
        help="Experiment: up1, md2, mdsingle, ld4, tsp3 (TSP meta-optimized; no 3D viz). Default: md2",
    )
    parser.add_argument("--seed", type=int, default=None, help="Single run seed (ignored if --seeds/--num-seeds set)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 7,27,107")
    parser.add_argument("--num-seeds", type=int, default=None, help=f"Use first N seeds from default set (max {len(DEFAULT_10_SEEDS)})")
    parser.add_argument("--calls", type=int, default=None, help="Main run call budget (default: 300 for both UP1 and MD2)")
    parser.add_argument("--load", type=str, default=None, help="Load result(s) from .pkl (skip running experiment)")
    parser.add_argument("--save", type=str, default=None, help="Save result(s) to .pkl after run (for later --load)")
    parser.add_argument("--quiet", action="store_true", help="Less console output from experiment")
    parser.add_argument("--meta-verbose", action="store_true", help="Print per-meta-generation stats (MD2 only)")
    args = parser.parse_args()

    exp = args.experiment.lower()
    if args.calls is not None:
        call_budget = args.calls
    elif exp == "up1" or exp == "mdsingle":
        call_budget = UP1_CALL_BUDGET
    elif exp == "ld4":
        call_budget = LD4_CALL_BUDGET
    elif exp == "tsp3":
        call_budget = TSP3_CALL_BUDGET
    else:
        call_budget = MD2_CALL_BUDGET

    # Resolve seed list (default: DEFAULT_SEEDS when no flags given)
    if args.num_seeds is not None:
        n = min(args.num_seeds, len(DEFAULT_10_SEEDS))
        seeds = DEFAULT_10_SEEDS[:n] if n <= len(DEFAULT_10_SEEDS) else (DEFAULT_10_SEEDS + list(range(len(DEFAULT_10_SEEDS), n)))
    elif args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(DEFAULT_SEEDS)
    seeds = sorted(seeds)

    # Run from project root so media/ and imports resolve
    os.chdir(_project_root)

    # Manim config: output to mockmedia; snapshot; preview
    config.media_dir = str(MOCKMEDIA_DIR)
    config.quality = "fourk_quality"
    config.save_last_frame = True
    config.preview = True

    # --- Get experiment output (list of result dicts, one per seed) ---
    if args.load:
        path = Path(args.load)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        results = loaded if isinstance(loaded, list) else [loaded]
        print(f"Loaded {exp.upper()} result(s) from {path}: {len(results)} seed(s) {[r.get('seed') for r in results]}")
    else:
        if exp == "up1":
            if len(seeds) == 1:
                result = run_up1(seed=seeds[0], call_budget=call_budget, verbose=not args.quiet)
                results = [result]
                print(f"UP1 run seed={seeds[0]} -> {len(result.get('lam_generations') or [])} Lamarckian gens, "
                      f"{len(result.get('dar_generations') or [])} Darwinian gens")
            else:
                results = run_up1_multi(seeds, call_budget=call_budget, verbose=not args.quiet)
                print(f"UP1 runs: {len(results)} seeds -> {len(results) * 2} runs (Lamarckian plane 0, Darwinian plane 1)")
        elif exp == "mdsingle":
            if len(seeds) == 1:
                result = run_MDSingle.run_up1(seed=seeds[0], call_budget=call_budget, verbose=not args.quiet)
                results = [result]
                print(f"MDSingle run seed={seeds[0]} -> {len(result.get('lam_generations') or [])} Lamarckian gens, "
                      f"{len(result.get('dar_generations') or [])} Darwinian gens")
            else:
                results = run_MDSingle.run_up1_multi(seeds, call_budget=call_budget, verbose=not args.quiet)
                print(f"MDSingle runs: {len(results)} seeds -> {len(results) * 2} runs (Lamarckian plane 0, Darwinian plane 1)")
        elif exp == "ld4":
            if len(seeds) == 1:
                result = run_ld4(seed=seeds[0], call_budget=call_budget, verbose=not args.quiet)
                results = [result]
                print(f"LD4 run seed={seeds[0]} -> {len(result.get('lam_generations') or [])} Lamarckian(samp) gens, "
                      f"{len(result.get('dar_generations') or [])} Darwinian gens")
            else:
                results = run_ld4_multi(seeds, call_budget=call_budget, verbose=not args.quiet)
                print(f"LD4 runs: {len(results)} seeds -> {len(results) * 2} runs (Lamarckian sampling plane 0, Darwinian plane 1)")
        elif exp == "tsp3":
            results = []
            for s in seeds:
                result = run_tsp3(
                    seed=s,
                    call_budget=call_budget,
                    verbose=not args.quiet,
                    meta_verbose=args.meta_verbose,
                )
                results.append(result)
            print(f"TSP3 runs: {len(results)} seed(s) -> best tour length (Lam vs Dar) per seed (no 3D visualization)")
        else:
            if len(seeds) == 1:
                result = run_md2(
                    seed=seeds[0],
                    call_budget=call_budget,
                    verbose=not args.quiet,
                    meta_verbose=args.meta_verbose,
                )
                results = [result]
                print(f"MD2 run seed={seeds[0]} -> {len(result.get('lam_generations') or [])} Lamarckian gens, "
                      f"{len(result.get('dar_generations') or [])} Darwinian gens")
            else:
                results = run_md2_multi(
                    seeds,
                    call_budget=call_budget,
                    verbose=not args.quiet,
                    meta_verbose=args.meta_verbose,
                )
                print(f"MD2 runs: {len(results)} seeds -> {len(results) * 2} runs (Lamarckian plane 0, Darwinian plane 1)")
        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            to_save = results[0] if len(results) == 1 else results
            with open(save_path, "wb") as f:
                pickle.dump(to_save, f)
            print(f"Saved result(s) to {save_path}")

    # TSP3: no 3D Rastrigin visualization; print summary and exit
    if exp == "tsp3":
        print("\n--- TSP3 summary (best tour length, lower is better) ---")
        for r in results:
            nn = r.get("nn_length")
            lam_best = (r.get("lam_summary") or {}).get("best_fitness")
            dar_best = (r.get("dar_summary") or {}).get("best_fitness")
            nn_s = f"{nn:.2f}" if nn is not None else "—"
            lam_best = f"{lam_best:.2f}" if lam_best is not None else "—"
            dar_best = f"{dar_best:.2f}" if dar_best is not None else "—"
            print(f"  seed {r['seed']}: NN={nn_s}  Lam={lam_best}  Dar={dar_best}")
        print("(No 3D visualization for TSP; use --save <file.pkl> to store results.)")
        return

    # --- Set scene attributes ---
    VisualizationScene.surface_function = rastrigin_func
    VisualizationScene.surface_name = "rastrigin"
    n_seeds = len(results)
    experiment_name = f"{exp}_viz_{n_seeds}seed" + ("s" if n_seeds != 1 else "")
    VisualizationScene.experiment_name = experiment_name

    VisualizationScene.algorithm_runs = _build_algorithm_runs_from_results(results)

    # Replication info for the info panel (so the experiment can be reproduced)
    seeds_display = sorted(r["seed"] for r in results)
    call_display = call_budget
    if args.load and results:
        lam_calls = (results[0].get("lam_summary") or {}).get("function_calls")
        if lam_calls is not None:
            call_display = lam_calls
    xmin, xmax, ymin, ymax = UP1_INITIAL_BOUNDS
    repl_lines = [
        f"Experiment: {exp.upper()}",
        f"Seeds: {seeds_display}",
        f"Call budget: {call_display}",
        "Topology: Rastrigin",
        f"Initial bounds: [{xmin}, {xmax}] x [{ymin}, {ymax}]",
    ]
    if exp == "md2":
        repl_lines.extend([
            f"Meta: calls={MD2_META_CALL_BUDGET}, pop={MD2_META_POPULATION}, "
            f"gen={MD2_META_GENERATIONS}, elite={MD2_META_ELITE}",
        ])
    elif exp == "mdsingle":
        repl_lines.append("Levers: fixed MD2-optimized (seed 42) from run_MDSingle")
    elif exp == "ld4":
        repl_lines.append("Lamarckian: sampling-based besoin (run_LD4); Darwinian: fixed levers from MDSingle")
    VisualizationScene.panel_replication_lines = repl_lines
    VisualizationScene.panel_marker_key = None  # use default key in vizualization.py

    # Best levers per seed (for table: seed + lam_levers + dar_levers, row color = seed color)
    panel_levers_data = [
        {"seed": r["seed"], "lam_levers": r.get("lam_levers"), "dar_levers": r.get("dar_levers")}
        for r in results
    ]
    VisualizationScene.panel_levers_data = panel_levers_data
    VisualizationScene.panel_levers_colors = [
        FINAL_SEED_COLORS[i % len(FINAL_SEED_COLORS)] for i in range(len(results))
    ]

    # --- Build and render (dynamic class name so output image is named with experiment) ---
    SceneClass = type(experiment_name, (VisualizationScene,), {})
    scene = SceneClass()
    try:
        scene.render()
    except FileNotFoundError:
        pass  # preview tried to open video path but we only saved last frame


if __name__ == "__main__":
    main()
