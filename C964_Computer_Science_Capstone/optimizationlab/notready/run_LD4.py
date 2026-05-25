"""
LD4: Lamarckian (sampling-based besoin) v Darwinian.

- Same setup as UP1/MDSingle: 4 initial points (fixed or random), 2 Lamarckian vectors, 4 Darwinian organisms.
- Lamarckian: pure_lamarckian_function_sampling (random sampling of points in a range, best point → besoin vector).
- Darwinian: same fixed levers as MDSingle (OPTIMIZED_DAR_LEVERS seed 42) for comparable comparison.
- Termination: 300 topology calls per run.
- Output shape matches run_up1 / run_MDSingle for use with mocktest.
"""

import sys
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from optimizationfunctions.evolutionalgorithms.lamarckianfunctions.core import (
    pure_lamarckian_function_sampling,
    rastrigin_func,
)
from optimizationfunctions.evolutionalgorithms.darwinianfunctions.core import (
    pure_darwinian_function,
)
from optimizationfunctions.evolutionalgorithms.metaevolutionfunctions import CountedFunction

from optimizationlab.run_comparison import (
    summarize_lamarckian_run,
    summarize_darwinian_run,
)
from optimizationlab.experimentalsetup.run_up1 import (
    get_initial_points,
    lamarckian_parents_from_points,
    UP1_INITIAL_BOUNDS,
)
from optimizationlab.experimentalsetup.run_MDSingle import OPTIMIZED_DAR_LEVERS

LD4_CALL_BUDGET = 300

# Fixed levers for sampling-based Lamarckian (no gradient; sampling radius, count, scale).
LD4_LAM_LEVERS = {
    "besoin_weight": 1.0,
    "sampling_radius": 2.0,
    "num_sampling_points": 10,
    "sampling_scale": 0.1,
    "magnitude_std_fraction": 0.1,
    "magnitude_weight": 1.0,
    "direction_std": 0.1,
    "min_magnitude": 0.01,
    "max_magnitude": None,
    "num_offspring": 2,
    "first_generation_random_besoin": False,
}

# Reuse Darwinian levers from MDSingle for comparable comparison
DEFAULT_10_SEEDS = [7, 27, 107, 207, 327, 507, 42, 123, 456, 789]


def run_ld4(
    seed=42,
    call_budget=LD4_CALL_BUDGET,
    verbose=True,
    lamarckian_vectors=2,
    darwinian_pop=4,
):
    """
    Run LD4: sampling-based Lamarckian v Darwinian.
    Returns dict compatible with run_up1 (lam_generations, dar_generations, summaries, levers, etc.).
    """
    n_points = 4 if (lamarckian_vectors == 2 and darwinian_pop == 4) else 2
    topology_func = rastrigin_func
    initial_points = get_initial_points(seed, n_points, use_fixed_4=(n_points == 4))
    parent1_start, parent1_end, parent2_start, parent2_end = lamarckian_parents_from_points(
        initial_points, num_vectors=lamarckian_vectors
    )
    lam_levers = dict(LD4_LAM_LEVERS)
    dar_levers = dict(OPTIMIZED_DAR_LEVERS)
    dar_levers["population_size"] = darwinian_pop
    if darwinian_pop == 2:
        max_er = (darwinian_pop - 1) / darwinian_pop
        dar_levers["elimination_rate"] = min(dar_levers["elimination_rate"], max_er)
    if lamarckian_vectors == 1 and darwinian_pop == 2:
        lam_levers["num_offspring"] = 1

    initial_pop = np.array([parent1_start, parent1_end, parent2_start, parent2_end])

    # Lamarckian run (sampling-based besoin)
    counted_lam = CountedFunction(topology_func)
    lam_gen = pure_lamarckian_function_sampling(
        besoin_topology_function=counted_lam,
        parent1_start=parent1_start,
        parent1_end=parent1_end,
        parent2_start=parent2_start,
        parent2_end=parent2_end,
        num_generations=100000,
        seed=seed + 50,
        initial_bounds=UP1_INITIAL_BOUNDS,
        max_calls=call_budget,
        **lam_levers,
    )
    lam_sum = summarize_lamarckian_run(lam_gen, topology_func)
    if lam_sum:
        lam_sum["function_calls"] = counted_lam.n_calls
    final_lam = []
    if lam_gen:
        for start, end in lam_gen[-1]["organisms"]:
            final_lam.append(end[:2].tolist())
    final_distribution_lam = np.array(final_lam) if final_lam else None

    # Darwinian run
    counted_dar = CountedFunction(topology_func)
    dar_gen = pure_darwinian_function(
        fitness_topology_function=counted_dar,
        num_generations=100000,
        seed=seed + 100,
        initial_bounds=UP1_INITIAL_BOUNDS,
        max_calls=call_budget,
        initial_population=initial_pop,
        **dar_levers,
    )
    dar_sum = summarize_darwinian_run(dar_gen, topology_func)
    if dar_sum:
        dar_sum["function_calls"] = counted_dar.n_calls
    final_dar = None
    if dar_gen:
        final_dar = np.array([p[:2] for p in dar_gen[-1]["organisms"]])

    n_init = initial_pop.shape[0]
    out = {
        "seed": seed,
        "lamarckian_vectors": lamarckian_vectors,
        "darwinian_pop": darwinian_pop,
        "initial_points": initial_pop,
        "initial_distribution": [initial_pop[i, :2].tolist() for i in range(n_init)],
        "lam_levers": lam_levers,
        "dar_levers": dar_levers,
        "lam_generations": lam_gen,
        "dar_generations": dar_gen,
        "lam_summary": lam_sum,
        "dar_summary": dar_sum,
        "final_distribution_lam": final_distribution_lam.tolist() if final_distribution_lam is not None else None,
        "final_distribution_dar": final_dar.tolist() if final_dar is not None else None,
    }
    if verbose:
        print(f"LD4 seed={seed} | Lam(samp): best_f={lam_sum['best_fitness']:.4f} dist={lam_sum['distance_to_optimum']:.3f} calls={lam_sum['function_calls']} | "
              f"Dar: best_f={dar_sum['best_fitness']:.4f} dist={dar_sum['distance_to_optimum']:.3f} calls={dar_sum['function_calls']}")
    return out


def run_ld4_multi(seeds, call_budget=LD4_CALL_BUDGET, verbose=True, lamarckian_vectors=2, darwinian_pop=4):
    """Run LD4 for each seed; return list of result dicts."""
    seeds = sorted(seeds)
    results = []
    for seed in seeds:
        r = run_ld4(seed=seed, call_budget=call_budget, verbose=False, lamarckian_vectors=lamarckian_vectors, darwinian_pop=darwinian_pop)
        results.append(r)
        if verbose:
            lam = r["lam_summary"] or {}
            dar = r["dar_summary"] or {}
            print(f"  seed={seed}: Lam(samp) best_f={lam.get('best_fitness', float('nan')):.4f} dist={lam.get('distance_to_optimum', float('nan')):.3f} | "
                  f"Dar best_f={dar.get('best_fitness', float('nan')):.4f} dist={dar.get('distance_to_optimum', float('nan')):.3f}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LD4: Sampling-based Lamarckian v Darwinian")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds")
    p.add_argument("--num-seeds", type=int, default=None, help="Use first N of default seeds")
    p.add_argument("--calls", type=int, default=LD4_CALL_BUDGET)
    p.add_argument("--lamarckian-vectors", type=int, default=2)
    p.add_argument("--darwinian-pop", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.num_seeds is not None:
        seeds = DEFAULT_10_SEEDS[: args.num_seeds] if args.num_seeds <= len(DEFAULT_10_SEEDS) else (
            DEFAULT_10_SEEDS + list(range(len(DEFAULT_10_SEEDS), args.num_seeds))
        )
    elif args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed if args.seed is not None else 42]

    lam_vecs = getattr(args, "lamarckian_vectors", 2)
    dar_pop = getattr(args, "darwinian_pop", 4)

    if len(seeds) == 1:
        run_ld4(seed=seeds[0], call_budget=args.calls, verbose=not args.quiet, lamarckian_vectors=lam_vecs, darwinian_pop=dar_pop)
    else:
        run_ld4_multi(seeds, call_budget=args.calls, verbose=not args.quiet, lamarckian_vectors=lam_vecs, darwinian_pop=dar_pop)
