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
    python optimizationlab/visualize_experiment.py
        (default: experiment=md2, 6 seeds 7,27,107,207,327,507)
    python optimizationlab/visualize_experiment.py --experiment up1
    python optimizationlab/visualize_experiment.py --experiment md2 --seed N
    python optimizationlab/visualize_experiment.py --experiment mdsingle
    python optimizationlab/visualize_experiment.py --experiment ld4
    python optimizationlab/visualize_experiment.py --experiment tsp3 [--seed N] [--seeds 7,27,107] [--save tsp3_result.pkl]
    python optimizationlab/visualize_experiment.py --experiment up1 --seeds 7,27,107 --save up1_result.pkl
    python optimizationlab/visualize_experiment.py --experiment md2 --load md2_result.pkl

Output: snapshot PNG to optimizationlab/_Experimental_Media/images/ (except tsp3); tsp3 prints summary only. Use -p to preview.
"""

import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# Project root so we can import experimental_analysis and its dependencies
_project_root = Path(__file__).resolve().parent.parent
_this_dir = Path(__file__).resolve().parent
MOCKMEDIA_DIR = _this_dir / "_Experimental_Media"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import after path is set
from manim import BLACK, BLUE, GREEN, RED, PURPLE, TEAL, PINK, MAROON, GOLD, ORANGE, YELLOW, config
from optimizationfunctions.evolutionalgorithms.lamarckianfunctions.core import rastrigin_func
from optimizationlab.experimentalsetup.run_up1 import run_up1, run_up1_multi, UP1_CALL_BUDGET, UP1_INITIAL_BOUNDS, DEFAULT_10_SEEDS as UP1_DEFAULT_10_SEEDS
from optimizationlab.experimentalsetup.run_MD2 import (
    run_md2,
    run_md2_multi,
    MD2_CALL_BUDGET,
    MD2_META_CALL_BUDGET,
    MD2_META_POPULATION,
    MD2_META_GENERATIONS,
    MD2_META_ELITE,
    DEFAULT_10_SEEDS as MD2_DEFAULT_10_SEEDS,
)
from optimizationlab.notready import run_MDSingle
from optimizationlab.evaluation import evaluate_results, format_evaluation_report
from monitormaintain import log_experiment_bundle
from visualizationtool.bar_charts import write_best_candidate_bar_chart_svg
from visualizationtool.meta_trends import write_meta_optimization_trends_svg
from visualizationtool.visualization import VisualizationScene

# Optional experiments may not be present in all project states.
run_ld4 = None
run_ld4_multi = None
LD4_CALL_BUDGET = None
run_tsp3 = None
TSP3_CALL_BUDGET = None

try:
    _ld4_module = importlib.import_module("optimizationlab.experimentalsetup.run_LD4")
    run_ld4 = _ld4_module.run_ld4
    run_ld4_multi = _ld4_module.run_ld4_multi
    LD4_CALL_BUDGET = _ld4_module.LD4_CALL_BUDGET
except ModuleNotFoundError:
    pass

try:
    _tsp3_module = importlib.import_module("optimizationlab.experimentalsetup.run_TSP3")
    run_tsp3 = _tsp3_module.run_tsp3
    TSP3_CALL_BUDGET = _tsp3_module.TSP3_CALL_BUDGET
except ModuleNotFoundError:
    pass

# Same seed list in both modules; use one for --num-seeds
DEFAULT_10_SEEDS = UP1_DEFAULT_10_SEEDS

# Seeds used when no --seed, --seeds, or --num-seeds is given
DEFAULT_SEEDS = [7, 27, 107, 207, 327, 507]

EXPERIMENT_LABELS = {
    "up1": "UP1 - Unoptimized Pure Lamarckian vs Darwinian",
    "md2": "MD2 - Meta-optimized (Darwinian-tuned) Lamarckian vs Darwinian",
    "mdsingle": "MDSingle - UP1 with fixed MD2-optimized levers",
    "ld4": "LD4 - Sampling-based Lamarckian vs fixed-lever Darwinian",
    "tsp3": "TSP3 - MD2-style meta-optimized Traveling Salesman Problem",
}

# Per-seed colors: same index = same seed on plane 0 (Lamarckian) and plane 1 (Darwinian)
SEED_COLORS = [RED, GREEN, BLUE, PURPLE, TEAL, PINK, MAROON, GOLD, ORANGE, YELLOW]
FINAL_SEED_COLORS = [
    "#B71C1C", "#1B5E20", "#0D47A1", "#4A148C", "#004D40", "#880E4F",
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
    parser.add_argument(
        "--meta-calls",
        type=int,
        default=None,
        help="MD2 meta-evaluation call budget per candidate (defaults to MD2 setting when omitted)",
    )
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
        if LD4_CALL_BUDGET is None:
            raise ModuleNotFoundError(
                "LD4 experiment is unavailable: missing optimizationlab.experimentalsetup.run_LD4"
            )
        call_budget = LD4_CALL_BUDGET
    elif exp == "tsp3":
        if TSP3_CALL_BUDGET is None:
            raise ModuleNotFoundError(
                "TSP3 experiment is unavailable: missing optimizationlab.experimentalsetup.run_TSP3"
            )
        call_budget = TSP3_CALL_BUDGET
    else:
        call_budget = MD2_CALL_BUDGET
    meta_call_budget = args.meta_calls if args.meta_calls is not None else MD2_META_CALL_BUDGET

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
    # Unique run metadata for naming and traceability
    experiment_started_at = datetime.now()
    experiment_started_iso = experiment_started_at.strftime("%Y-%m-%d %H:%M:%S")
    experiment_started_compact = experiment_started_at.strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{exp}_{experiment_started_compact}_{uuid4().hex[:8]}"

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
            if run_ld4 is None or run_ld4_multi is None:
                raise ModuleNotFoundError(
                    "LD4 experiment is unavailable: missing optimizationlab.experimentalsetup.run_LD4"
                )
            if len(seeds) == 1:
                result = run_ld4(seed=seeds[0], call_budget=call_budget, verbose=not args.quiet)
                results = [result]
                print(f"LD4 run seed={seeds[0]} -> {len(result.get('lam_generations') or [])} Lamarckian(samp) gens, "
                      f"{len(result.get('dar_generations') or [])} Darwinian gens")
            else:
                results = run_ld4_multi(seeds, call_budget=call_budget, verbose=not args.quiet)
                print(f"LD4 runs: {len(results)} seeds -> {len(results) * 2} runs (Lamarckian sampling plane 0, Darwinian plane 1)")
        elif exp == "tsp3":
            if run_tsp3 is None:
                raise ModuleNotFoundError(
                    "TSP3 experiment is unavailable: missing optimizationlab.experimentalsetup.run_TSP3"
                )
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
                    meta_call_budget=meta_call_budget,
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
                    meta_call_budget=meta_call_budget,
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

    # Accuracy report: aggregate metrics, confidence intervals, and win comparison
    eval_dir = MOCKMEDIA_DIR / "evaluation_reports"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_report = evaluate_results(results, confidence=0.95, primary_metric="best_fitness", lower_is_better=True)
    eval_text = format_evaluation_report(eval_report)
    eval_path = eval_dir / f"{experiment_id}_evaluation_report.txt"
    eval_path.write_text(eval_text, encoding="utf-8")
    print(f"Wrote evaluation report to {eval_path}")
    if not args.quiet:
        print("\n--- Evaluation report ---")
        print(eval_text.rstrip())

    # MD2: write meta-optimization trend-line SVG(s) for inspection/reporting
    if exp == "md2":
        trend_dir = MOCKMEDIA_DIR / "meta_trends"
        trend_dir.mkdir(parents=True, exist_ok=True)
        trend_path = trend_dir / f"{experiment_id}_meta_trend_all_seeds.svg"
        write_meta_optimization_trends_svg(
            results,
            trend_path,
            title="MD2 Meta Trend (all seeds)",
            subtitle=f"{experiment_id} | {experiment_started_iso}",
            seed_colors=FINAL_SEED_COLORS,
        )
        print(f"Wrote combined MD2 meta trend SVG to {trend_path}")

    # Best-candidate grouped bar chart (Lamarckian vs Darwinian best_fitness per seed)
    if exp != "tsp3":
        bar_dir = MOCKMEDIA_DIR / "bar_charts"
        bar_path = bar_dir / f"{experiment_id}_best_candidates_bar.svg"
        write_best_candidate_bar_chart_svg(
            results,
            bar_path,
            title=f"{exp.upper()} Best Candidates by Seed",
            subtitle=f"{experiment_id} | {experiment_started_iso}",
        )
        print(f"Wrote best-candidate bar chart to {bar_path}")

    # --- Set scene attributes ---
    VisualizationScene.surface_function = rastrigin_func
    VisualizationScene.surface_name = "rastrigin"
    n_seeds = len(results)
    experiment_name = f"{exp}_viz_{n_seeds}seed" + ("s" if n_seeds != 1 else "")
    experiment_name = f"{experiment_name}_{experiment_id}"
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
    lam_best = [
        float(v)
        for v in ((r.get("lam_summary") or {}).get("best_fitness") for r in results)
        if v is not None
    ]
    lam_dist = [
        float(v)
        for v in ((r.get("lam_summary") or {}).get("distance_to_optimum") for r in results)
        if v is not None
    ]
    dar_best = [
        float(v)
        for v in ((r.get("dar_summary") or {}).get("best_fitness") for r in results)
        if v is not None
    ]
    dar_dist = [
        float(v)
        for v in ((r.get("dar_summary") or {}).get("distance_to_optimum") for r in results)
        if v is not None
    ]

    def _metric_line(name: str, best_vals: list[float], dist_vals: list[float]) -> str:
        if not best_vals or not dist_vals:
            return f"{name}: n/a"
        if len(best_vals) == 1:
            return f"{name}: best_f={best_vals[0]:.4g}, dist={dist_vals[0]:.4g}"
        return (
            f"{name}: best_f avg={sum(best_vals)/len(best_vals):.4g}, min={min(best_vals):.4g}; "
            f"dist avg={sum(dist_vals)/len(dist_vals):.4g}, min={min(dist_vals):.4g}"
        )

    repl_lines = [
        "Experiment:",
        f"{EXPERIMENT_LABELS.get(exp, exp.upper())}",
        f"Experiment ID: {experiment_id}",
        f"Seeds: {seeds_display}",
        f"Call budget: {call_display}",
        "Topology: Rastrigin",
        f"Initial bounds: [{xmin}, {xmax}] x [{ymin}, {ymax}]",
        "Final performance metrics:",
        _metric_line("Lamarckian", lam_best, lam_dist),
        _metric_line("Darwinian", dar_best, dar_dist),
    ]
    if exp == "md2":
        repl_lines.extend([
            f"Meta: calls={meta_call_budget}, pop={MD2_META_POPULATION}, "
            f"gen={MD2_META_GENERATIONS}, elite={MD2_META_ELITE}",
        ])
    elif exp == "mdsingle":
        repl_lines.append("Levers: fixed MD2-optimized (seed 42) from run_MDSingle")
    elif exp == "ld4":
        repl_lines.append("Lamarckian: sampling-based besoin (run_LD4); Darwinian: fixed levers from MDSingle")
    VisualizationScene.panel_replication_lines = repl_lines
    VisualizationScene.panel_experiment_id = experiment_id
    VisualizationScene.panel_started_at = experiment_started_iso
    VisualizationScene.panel_marker_key = None  # use default key in vizualization.py
    first_result = results[0] if results else {}
    VisualizationScene.panel_lam_total_organisms = sum(
        len(gen.get("organisms", []))
        for gen in (first_result.get("lam_generations") or [])
    )
    VisualizationScene.panel_dar_total_organisms = sum(
        len(gen.get("organisms", []))
        for gen in (first_result.get("dar_generations") or [])
    )

    # Best levers per seed (for table: seed + lam_levers + dar_levers, row color = seed color)
    panel_levers_data = [
        {"seed": r["seed"], "lam_levers": r.get("lam_levers"), "dar_levers": r.get("dar_levers")}
        for r in results
    ]
    def _best_final_point_and_distance(points):
        if not points:
            return None, None
        best_xy = None
        best_f = None
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            x = float(pt[0])
            y = float(pt[1])
            z = rastrigin_func(x, y)[2]
            f_val = float(z)
            if best_f is None or f_val < best_f:
                best_f = f_val
                best_xy = (x, y)
        if best_xy is None:
            return None, None
        distance = (best_xy[0] ** 2 + best_xy[1] ** 2) ** 0.5
        return best_xy, float(distance)

    panel_performance_data = []
    for r in results:
        lam_summary = r.get("lam_summary") or {}
        dar_summary = r.get("dar_summary") or {}
        lam_best_xy, lam_best_dist = _best_final_point_and_distance(r.get("final_distribution_lam"))
        dar_best_xy, dar_best_dist = _best_final_point_and_distance(r.get("final_distribution_dar"))
        panel_performance_data.append(
            {
                "seed": r.get("seed"),
                "lam_best_final_organism_xy": lam_best_xy,
                "dar_best_final_organism_xy": dar_best_xy,
                "lam_best_final_organism_distance": lam_best_dist,
                "dar_best_final_organism_distance": dar_best_dist,
                "lam_final_mean": lam_summary.get("mean_fitness"),
                "dar_final_mean": dar_summary.get("mean_fitness"),
            }
        )
    VisualizationScene.panel_levers_data = panel_levers_data
    VisualizationScene.panel_performance_data = panel_performance_data
    VisualizationScene.panel_levers_colors = [
        FINAL_SEED_COLORS[i % len(FINAL_SEED_COLORS)] for i in range(len(results))
    ]

    # Mirror the on-image info panel content in CLI output for logging/auditing.
    info_panel_snapshot: list[str] = []
    print("\n=== Info Panel ===")
    print("[Experimental Reproducibility Information]")
    info_panel_snapshot.append("[Experimental Reproducibility Information]")
    for line in repl_lines:
        print(f"- {line}")
        info_panel_snapshot.append(line)
    lam_total_line = (
        f"Total organisms over run (Lamarckian, one seed): "
        f"{VisualizationScene.panel_lam_total_organisms}"
    )
    dar_total_line = (
        f"Total organisms over run (Darwinian, one seed): "
        f"{VisualizationScene.panel_dar_total_organisms}"
    )
    print(
        f"- {lam_total_line}"
    )
    print(
        f"- {dar_total_line}"
    )
    info_panel_snapshot.append(lam_total_line)
    info_panel_snapshot.append(dar_total_line)

    print("\n[Lamarckian levers]")
    info_panel_snapshot.append("[Lamarckian levers]")
    lam_keys = [
        "besoin_weight", "topology_gradient_scale", "magnitude_std_fraction",
        "magnitude_weight", "direction_std", "min_magnitude", "max_magnitude",
        "num_offspring", "first_generation_random_besoin",
    ]
    for row in panel_levers_data:
        lam = row.get("lam_levers") or {}
        vals = ", ".join(f"{k}={lam.get(k)}" for k in lam_keys)
        line = f"seed={row.get('seed')}: {vals}"
        print(line)
        info_panel_snapshot.append(line)

    print("\n[Darwinian levers]")
    info_panel_snapshot.append("[Darwinian levers]")
    dar_keys = ["elimination_rate", "selection_pressure", "mutation_std"]
    for row in panel_levers_data:
        dar = row.get("dar_levers") or {}
        vals = ", ".join(f"{k}={dar.get(k)}" for k in dar_keys)
        line = f"seed={row.get('seed')}: {vals}"
        print(line)
        info_panel_snapshot.append(line)

    print("\n[Performance]")
    info_panel_snapshot.append("[Performance]")
    for row in panel_performance_data:
        seed = row.get("seed")
        lbxy = row.get("lam_best_final_organism_xy")
        dbxy = row.get("dar_best_final_organism_xy")
        lbd = row.get("lam_best_final_organism_distance")
        dbd = row.get("dar_best_final_organism_distance")
        lm = row.get("lam_final_mean")
        dm = row.get("dar_final_mean")
        line = (
            f"seed={seed}: "
            f"best_final_organism "
            f"L={lbxy} d={lbd}, D={dbxy} d={dbd}; "
            f"final_mean L={lm}, D={dm}"
        )
        print(line)
        info_panel_snapshot.append(line)
    print("=== End Info Panel ===\n")

    # --- Build and render (dynamic class name so output image is named with experiment) ---
    SceneClass = type(experiment_name, (VisualizationScene,), {})
    scene = SceneClass()
    try:
        scene.render()
    except FileNotFoundError:
        pass  # preview tried to open video path but we only saved last frame

    # Archive artifacts + info panel snapshot under _Experimental_Results/<experiment_id>/.
    artifact_paths = []
    artifact_paths.extend((MOCKMEDIA_DIR / "images").glob(f"*{experiment_id}*.png"))
    artifact_paths.extend((MOCKMEDIA_DIR / "bar_charts").glob(f"{experiment_id}_best_candidates_bar.svg"))
    artifact_paths.extend((MOCKMEDIA_DIR / "meta_trends").glob(f"{experiment_id}_meta_trend_*.svg"))
    artifact_paths.extend((MOCKMEDIA_DIR / "evaluation_reports").glob(f"{experiment_id}_evaluation_report.txt"))
    try:
        logged_dir = log_experiment_bundle(
            experiment_id=experiment_id,
            experiment_name=EXPERIMENT_LABELS.get(exp, exp.upper()),
            info_panel_lines=info_panel_snapshot,
            artifact_paths=artifact_paths,
        )
        print(f"Logged experiment bundle to {logged_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not log experiment bundle: {exc}")


if __name__ == "__main__":
    main()
