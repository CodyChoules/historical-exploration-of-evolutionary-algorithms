"""
Run Lamarckian and Darwinian evolution under shared conditions and compare results.

Usage:
    python -m comparative_testing.run_comparison
    python -m comparative_testing.run_comparison --generations 30 --seeds 7,27,107
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Project root for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from lamarckian_functions.core import pure_lamarckian_function, rastrigin_func
from darwinian_functions.core import pure_darwinian_function, _extract_fitness_value
from meta_evolution_functions import CountedFunction


# -----------------------------------------------------------------------------
# Shared topology and metrics
# -----------------------------------------------------------------------------

def fitness_at(x, y, topology_func=rastrigin_func):
    """Scalar fitness (z) at (x, y). Lower is better (minimization)."""
    out = topology_func(x, y)
    if isinstance(out, np.ndarray) and len(out) >= 3:
        return float(out[2])
    return float(out)


def distance_to_optimum(x, y, optimum=(0.0, 0.0)):
    """Euclidean distance from (x, y) to optimum (e.g. Rastrigin at (0,0))."""
    return float(np.sqrt((x - optimum[0]) ** 2 + (y - optimum[1]) ** 2))


def summarize_lamarckian_run(generations, topology_func=rastrigin_func):
    """
    From Lamarckian generations (list of dicts with 'organisms' as (start, end) tuples),
    return dict with final mean (x,y), best fitness, mean fitness, distance to optimum.
    """
    if not generations:
        return None
    last = generations[-1]
    organisms = last["organisms"]
    if not organisms:
        return None
    # Use end points as trait positions
    endpoints = np.array([end for _, end in organisms], dtype=float)
    mean_xy = np.mean(endpoints[:, :2], axis=0)
    mx, my = float(mean_xy[0]), float(mean_xy[1])
    fitnesses = [fitness_at(e[0], e[1], topology_func) for _, e in organisms]
    return {
        "mean_x": mx,
        "mean_y": my,
        "best_fitness": min(fitnesses),
        "mean_fitness": float(np.mean(fitnesses)),
        "distance_to_optimum": distance_to_optimum(mx, my),
        "n_organisms": len(organisms),
        "function_calls": None,  # set by run_comparison when using CountedFunction
    }


def summarize_darwinian_run(generations, topology_func=rastrigin_func):
    """
    From Darwinian generations (list of dicts with 'organisms' and 'fitness_values'),
    return dict with final mean (x,y), best fitness, mean fitness, distance to optimum.
    """
    if not generations:
        return None
    last = generations[-1]
    organisms = last["organisms"]
    fitness_values = last["fitness_values"]
    if not organisms:
        return None
    points = np.array(organisms, dtype=float)
    mean_xy = np.mean(points[:, :2], axis=0)
    mx, my = float(mean_xy[0]), float(mean_xy[1])
    return {
        "mean_x": mx,
        "mean_y": my,
        "best_fitness": min(fitness_values),
        "mean_fitness": float(np.mean(fitness_values)),
        "distance_to_optimum": distance_to_optimum(mx, my),
        "n_organisms": len(organisms),
        "function_calls": None,  # set by run_comparison when using CountedFunction
    }


# -----------------------------------------------------------------------------
# Default config (aligned with project conventions)
# -----------------------------------------------------------------------------

DEFAULT_SEEDS = [7, 27, 107]
DEFAULT_NUM_GENERATIONS = 30
DEFAULT_CALL_BUDGET = 300  # when set, both runs stop at this many topology evaluations (overrides generations)
DEFAULT_INITIAL_BOUNDS = (-12.0, 12.0, -12.0, 12.0)

# Lamarckian defaults (from testing_la_girafe / 6optimized)
LAMARCKIAN_DEFAULTS = dict(
    besoin_weight=1.14,
    topology_gradient_scale=0.11,
    magnitude_std_fraction=0.12,
    magnitude_weight=0.89,
    direction_std=0.32,
    min_magnitude=0.01,
    num_offspring=2,
    first_generation_random_besoin=False,
)

# Darwinian defaults (reasonable for comparison)
DARWINIAN_DEFAULTS = dict(
    population_size=32,
    elimination_rate=0.5,
    selection_pressure=4.0,
    mutation_std=0.8,
)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def run_comparison(
    seeds=None,
    num_generations=None,
    call_budget=None,
    initial_bounds=None,
    topology_func=None,
    lamarckian_kwargs=None,
    darwinian_kwargs=None,
    verbose=True,
):
    """
    Run Lamarckian and Darwinian evolution for each seed and return comparison.

    When call_budget is set (e.g. 300), both runs stop when topology evaluations reach that limit;
    num_generations is then used only as an upper bound (set high internally). When call_budget is None,
    num_generations is the limit as before.

    Returns:
        list of dicts, one per seed, with keys:
          seed, lamarckian_summary, darwinian_summary, lamarckian_generations, darwinian_generations
    """
    seeds = seeds or DEFAULT_SEEDS
    initial_bounds = initial_bounds or DEFAULT_INITIAL_BOUNDS
    topology_func = topology_func or rastrigin_func
    lamarckian_kwargs = dict(LAMARCKIAN_DEFAULTS, **(lamarckian_kwargs or {}))
    darwinian_kwargs = dict(DARWINIAN_DEFAULTS, **(darwinian_kwargs or {}))
    use_call_budget = call_budget is not None
    if use_call_budget:
        max_calls = int(call_budget)
        num_generations = 100000  # high cap so limit is by calls
    else:
        max_calls = None
        num_generations = num_generations or DEFAULT_NUM_GENERATIONS

    results = []
    for seed in seeds:
        # Lamarckian: wrap topology to count function calls
        counted_lam = CountedFunction(topology_func)
        lam_gen = pure_lamarckian_function(
            besoin_topology_function=counted_lam,
            parent1_start=None,
            parent1_end=None,
            parent2_start=None,
            parent2_end=None,
            num_generations=num_generations,
            seed=seed,
            initial_bounds=initial_bounds,
            max_calls=max_calls,
            **lamarckian_kwargs,
        )
        lam_sum = summarize_lamarckian_run(lam_gen, topology_func)
        lam_sum["function_calls"] = counted_lam.n_calls

        # Darwinian: wrap topology to count function calls
        counted_dar = CountedFunction(topology_func)
        dar_gen = pure_darwinian_function(
            fitness_topology_function=counted_dar,
            num_generations=num_generations,
            seed=seed,
            initial_bounds=initial_bounds,
            max_calls=max_calls,
            **darwinian_kwargs,
        )
        dar_sum = summarize_darwinian_run(dar_gen, topology_func)
        dar_sum["function_calls"] = counted_dar.n_calls

        results.append({
            "seed": seed,
            "lamarckian_summary": lam_sum,
            "darwinian_summary": dar_sum,
            "lamarckian_generations": lam_gen,
            "darwinian_generations": dar_gen,
        })
        if verbose:
            print(f"Seed {seed}: Lamarckian mean=({lam_sum['mean_x']:.3f}, {lam_sum['mean_y']:.3f}) "
                  f"best_f={lam_sum['best_fitness']:.4f} dist={lam_sum['distance_to_optimum']:.3f} calls={lam_sum['function_calls']} | "
                  f"Darwinian mean=({dar_sum['mean_x']:.3f}, {dar_sum['mean_y']:.3f}) "
                  f"best_f={dar_sum['best_fitness']:.4f} dist={dar_sum['distance_to_optimum']:.3f} calls={dar_sum['function_calls']}")
    return results


def print_summary_table(results):
    """Print a simple comparison table."""
    if not results:
        return
    print("\n" + "=" * 92)
    print("Comparative summary (lower fitness & distance = better; function_calls = topology evaluations)")
    print("=" * 92)
    print(f"{'Seed':>6} | {'Lam mean (x,y)':>18} | {'Lam best f':>10} | {'Lam dist':>8} | {'Lam calls':>9} | "
          f"{'Dar mean (x,y)':>18} | {'Dar best f':>10} | {'Dar dist':>8} | {'Dar calls':>9}")
    print("-" * 92)
    for r in results:
        ls = r["lamarckian_summary"]
        ds = r["darwinian_summary"]
        lam_calls = ls.get("function_calls")
        dar_calls = ds.get("function_calls")
        lam_c_str = str(lam_calls) if lam_calls is not None else "—"
        dar_c_str = str(dar_calls) if dar_calls is not None else "—"
        print(f"{r['seed']:>6} | ({ls['mean_x']:>6.2f}, {ls['mean_y']:>6.2f}) | {ls['best_fitness']:>10.4f} | {ls['distance_to_optimum']:>8.3f} | {lam_c_str:>9} | "
              f"({ds['mean_x']:>6.2f}, {ds['mean_y']:>6.2f}) | {ds['best_fitness']:>10.4f} | {ds['distance_to_optimum']:>8.3f} | {dar_c_str:>9}")
    print("=" * 92)


def main():
    parser = argparse.ArgumentParser(description="Compare Lamarckian vs Darwinian evolution under shared conditions.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 7,27,107")
    parser.add_argument("--generations", type=int, default=None, help="Number of generations (default: 30); ignored when --calls is set")
    parser.add_argument("--calls", type=int, default=None, help="Cap both runs at this many topology evaluations (e.g. 300)")
    parser.add_argument("--quiet", action="store_true", help="Only print final table")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
    # Default: cap at 300 calls. Use generations only when --generations is set and --calls is not.
    if args.calls is not None:
        call_budget = args.calls
    elif args.generations is not None:
        call_budget = None
    else:
        call_budget = DEFAULT_CALL_BUDGET
    num_generations = args.generations
    verbose = not args.quiet

    results = run_comparison(seeds=seeds, num_generations=num_generations, call_budget=call_budget, verbose=verbose)
    print_summary_table(results)
    return results


if __name__ == "__main__":
    main()
