"""
Test MD2: Meta-optimized (Darwinian-tuned) Lamarckian v Darwinian.

For each seed:
  1. Generate shared initial points (same as run_up1).
  2. Meta-optimize Lamarckian levers: run a Darwinian meta-algorithm over lever
     space; each candidate is evaluated by running Lamarckian with those levers
     (same initial parents, fixed call budget); score = distance of final mean to (0,0).
  3. Meta-optimize Darwinian levers: same idea over Darwinian lever space.
  4. Main run: Lamarckian with best Lamarckian levers, Darwinian with best Darwinian
     levers (same initial conditions, same call budget). Compare results.

Termination: meta-evaluation uses a per-candidate call budget; main run uses 300 calls.
"""

import random
import sys
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from optimizationfunctions.evolutionalgorithms.lamarckianfunctions.core import (
    pure_lamarckian_function,
    rastrigin_func,
)
from optimizationfunctions.evolutionalgorithms.darwinianfunctions.core import (
    pure_darwinian_function,
)
from optimizationfunctions.evolutionalgorithms.metaevolutionfunctions import CountedFunction

from optimizationlab.run_comparison import (
    distance_to_optimum,
    summarize_lamarckian_run,
    summarize_darwinian_run,
)
from optimizationlab.experimentalsetup.run_up1 import (
    UP1_INITIAL_BOUNDS,
    get_initial_points,
    lamarckian_parents_from_points,
)

# -----------------------------------------------------------------------------
# Constants (aligned with run_up1 where applicable)
# -----------------------------------------------------------------------------
# initial_bounds: MD2 uses UP1_INITIAL_BOUNDS from run_up1 (-12, 12, -12, 12) so
# Darwinian offspring are not clipped to a smaller box and match the visualization axis range.
MD2_CALL_BUDGET = 300
MD2_META_CALL_BUDGET = 300
MD2_META_POPULATION = 24
MD2_META_GENERATIONS = 6
MD2_META_ELITE = 6
NUM_INITIAL_POINTS = 4

MD2_RESULTS_DIR = _project_root / "experimental_results" / "complexity 1" / "2MD2" / "generated results"

# Lamarckian lever bounds and mutation std (for meta-evolution)
LAM_LEVER_BOUNDS = {
    "besoin_weight": (0.2, 2.0),
    "topology_gradient_scale": (0.02, 0.3),
    "magnitude_std_fraction": (0.0, 0.4),
    "magnitude_weight": (0.0, 1.0),
    "direction_std": (0.0, 0.5),
    "min_magnitude": (0.005, 0.05),
    "num_offspring": (2, 4),
}
LAM_LEVER_MUTATION_STD = {
    "besoin_weight": 0.18,
    "topology_gradient_scale": 0.02,
    "magnitude_std_fraction": 0.05,
    "magnitude_weight": 0.08,
    "direction_std": 0.08,
    "min_magnitude": 0.005,
    "num_offspring": 0.3,
}

# Darwinian lever bounds and mutation std
DAR_LEVER_BOUNDS = {
    "elimination_rate": (0.1, 0.9),
    "selection_pressure": (0.5, 15.0),
    "mutation_std": (0.1, 2.0),
}
DAR_LEVER_MUTATION_STD = {
    "elimination_rate": 0.08,
    "selection_pressure": 1.2,
    "mutation_std": 0.15,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sample_lamarckian_levers(rng: random.Random) -> dict:
    """One random Lamarckian lever set for meta-evolution."""
    return {
        "besoin_weight": rng.uniform(*LAM_LEVER_BOUNDS["besoin_weight"]),
        "topology_gradient_scale": rng.uniform(*LAM_LEVER_BOUNDS["topology_gradient_scale"]),
        "magnitude_std_fraction": rng.uniform(*LAM_LEVER_BOUNDS["magnitude_std_fraction"]),
        "magnitude_weight": rng.uniform(*LAM_LEVER_BOUNDS["magnitude_weight"]),
        "direction_std": rng.uniform(*LAM_LEVER_BOUNDS["direction_std"]),
        "min_magnitude": rng.uniform(*LAM_LEVER_BOUNDS["min_magnitude"]),
        "num_offspring": int(rng.randint(2, 4)),
        "first_generation_random_besoin": bool(rng.randint(0, 2)),
    }


def mutate_lamarckian_levers(levers: dict, rng: random.Random) -> dict:
    """Gaussian mutation, clamped to bounds."""
    out = {}
    for name, (low, high) in LAM_LEVER_BOUNDS.items():
        std = LAM_LEVER_MUTATION_STD[name]
        if name == "num_offspring":
            child = int(clamp(round(rng.gauss(levers[name], std)), int(low), int(high)))
        else:
            child = clamp(rng.gauss(levers[name], std), low, high)
        out[name] = child
    out["first_generation_random_besoin"] = levers["first_generation_random_besoin"]
    return out


def sample_darwinian_levers(rng: random.Random, population_size: int = 4) -> dict:
    """One random Darwinian lever set for meta-evolution."""
    max_er = (population_size - 1) / population_size
    return {
        "population_size": population_size,
        "elimination_rate": rng.uniform(0.1, min(0.9, max_er)),
        "selection_pressure": rng.uniform(*DAR_LEVER_BOUNDS["selection_pressure"]),
        "mutation_std": rng.uniform(*DAR_LEVER_BOUNDS["mutation_std"]),
    }


def mutate_darwinian_levers(levers: dict, rng: random.Random) -> dict:
    out = {
        "population_size": levers["population_size"],
        "elimination_rate": clamp(
            rng.gauss(levers["elimination_rate"], DAR_LEVER_MUTATION_STD["elimination_rate"]),
            *DAR_LEVER_BOUNDS["elimination_rate"],
        ),
        "selection_pressure": clamp(
            rng.gauss(levers["selection_pressure"], DAR_LEVER_MUTATION_STD["selection_pressure"]),
            *DAR_LEVER_BOUNDS["selection_pressure"],
        ),
        "mutation_std": clamp(
            rng.gauss(levers["mutation_std"], DAR_LEVER_MUTATION_STD["mutation_std"]),
            *DAR_LEVER_BOUNDS["mutation_std"],
        ),
    }
    max_er = (out["population_size"] - 1) / out["population_size"]
    out["elimination_rate"] = min(out["elimination_rate"], max_er)
    return out


def evaluate_lamarckian_candidate(
    levers: dict,
    parent1_start: np.ndarray,
    parent1_end: np.ndarray,
    parent2_start: np.ndarray,
    parent2_end: np.ndarray,
    seed: int,
    call_budget: int,
) -> float:
    """Run Lamarckian with levers; return distance of final mean to (0,0). Lower is better."""
    counted = CountedFunction(rastrigin_func)
    gen = pure_lamarckian_function(
        besoin_topology_function=counted,
        parent1_start=parent1_start,
        parent1_end=parent1_end,
        parent2_start=parent2_start,
        parent2_end=parent2_end,
        num_generations=100000,
        seed=seed,
        initial_bounds=UP1_INITIAL_BOUNDS,
        max_calls=call_budget,
        **levers,
    )
    if not gen:
        return float("inf")
    summary = summarize_lamarckian_run(gen, rastrigin_func)
    if summary is None:
        return float("inf")
    return summary["distance_to_optimum"]


def evaluate_darwinian_candidate(
    levers: dict,
    initial_population: np.ndarray,
    seed: int,
    call_budget: int,
) -> float:
    """Run Darwinian with levers; return distance of final mean to (0,0). Lower is better."""
    counted = CountedFunction(rastrigin_func)
    gen = pure_darwinian_function(
        fitness_topology_function=counted,
        num_generations=100000,
        seed=seed,
        initial_bounds=UP1_INITIAL_BOUNDS,
        max_calls=call_budget,
        initial_population=initial_population,
        **levers,
    )
    if not gen:
        return float("inf")
    summary = summarize_darwinian_run(gen, rastrigin_func)
    if summary is None:
        return float("inf")
    return summary["distance_to_optimum"]


def meta_optimize_lamarckian(
    parent1_start: np.ndarray,
    parent1_end: np.ndarray,
    parent2_start: np.ndarray,
    parent2_end: np.ndarray,
    seed: int,
    rng: random.Random,
    call_budget: int = MD2_META_CALL_BUDGET,
    pop_size: int = MD2_META_POPULATION,
    num_gens: int = MD2_META_GENERATIONS,
    elite: int = MD2_META_ELITE,
    verbose: bool = False,
) -> tuple[dict, float]:
    """Return (best_lamarckian_levers, best_score)."""
    population = [sample_lamarckian_levers(rng) for _ in range(pop_size)]
    best_levers = None
    best_score = float("inf")

    for g in range(num_gens):
        scores = []
        for levers in population:
            s = evaluate_lamarckian_candidate(
                levers,
                parent1_start, parent1_end, parent2_start, parent2_end,
                seed=seed + 1000,
                call_budget=call_budget,
            )
            scores.append((s, levers))
            if s < best_score:
                best_score = s
                best_levers = levers.copy()

        if verbose:
            avg = sum(x[0] for x in scores) / len(scores)
            print(f"  meta_lam gen={g} best={best_score:.4f} avg={avg:.4f}")

        if g == num_gens - 1:
            break

        ranked = sorted(scores, key=lambda x: x[0])
        elites_levers = [x[1] for x in ranked[:elite]]
        next_pop = list(elites_levers)
        while len(next_pop) < pop_size:
            parent = rng.choice(elites_levers)
            next_pop.append(mutate_lamarckian_levers(parent, rng))
        population = next_pop

    return best_levers or sample_lamarckian_levers(rng), best_score


def meta_optimize_darwinian(
    initial_population: np.ndarray,
    seed: int,
    rng: random.Random,
    call_budget: int = MD2_META_CALL_BUDGET,
    pop_size: int = MD2_META_POPULATION,
    num_gens: int = MD2_META_GENERATIONS,
    elite: int = MD2_META_ELITE,
    verbose: bool = False,
) -> tuple[dict, float]:
    """Return (best_darwinian_levers, best_score)."""
    n = initial_population.shape[0]
    population = [sample_darwinian_levers(rng, population_size=n) for _ in range(pop_size)]
    best_levers = None
    best_score = float("inf")

    for g in range(num_gens):
        scores = []
        for levers in population:
            s = evaluate_darwinian_candidate(
                levers,
                initial_population,
                seed=seed + 2000,
                call_budget=call_budget,
            )
            scores.append((s, levers))
            if s < best_score:
                best_score = s
                best_levers = levers.copy()

        if verbose:
            avg = sum(x[0] for x in scores) / len(scores)
            print(f"  meta_dar gen={g} best={best_score:.4f} avg={avg:.4f}")

        if g == num_gens - 1:
            break

        ranked = sorted(scores, key=lambda x: x[0])
        elites_levers = [x[1] for x in ranked[:elite]]
        next_pop = list(elites_levers)
        while len(next_pop) < pop_size:
            parent = rng.choice(elites_levers)
            next_pop.append(mutate_darwinian_levers(parent, rng))
        population = next_pop

    return best_levers or sample_darwinian_levers(rng, n), best_score


def run_md2(
    seed: int = 42,
    call_budget: int = MD2_CALL_BUDGET,
    verbose: bool = True,
    lamarckian_vectors: int = 2,
    darwinian_pop: int = 4,
    meta_verbose: bool = False,
) -> dict:
    """
    Run MD2 for one seed: meta-optimize both algorithms, then main run with best levers.
    Returns dict compatible with run_up1 (plus meta_lam_levers, meta_dar_levers, meta_lam_score, meta_dar_score).
    """
    n_points = 4 if (lamarckian_vectors == 2 and darwinian_pop == 4) else 2
    initial_points = get_initial_points(seed, n_points, use_fixed_4=(n_points == 4))
    parent1_start, parent1_end, parent2_start, parent2_end = lamarckian_parents_from_points(
        initial_points, num_vectors=lamarckian_vectors
    )
    if lamarckian_vectors == 1 and darwinian_pop == 2:
        darwinian_pop = 2

    rng = random.Random(seed + 99999)

    # Meta-optimize Lamarckian levers
    if verbose:
        print(f"MD2 seed={seed}: meta-optimizing Lamarckian levers...")
    lam_levers, meta_lam_score = meta_optimize_lamarckian(
        parent1_start, parent1_end, parent2_start, parent2_end,
        seed=seed,
        rng=rng,
        call_budget=MD2_META_CALL_BUDGET,
        verbose=meta_verbose,
    )

    # Meta-optimize Darwinian levers
    if verbose:
        print(f"MD2 seed={seed}: meta-optimizing Darwinian levers...")
    initial_pop = initial_points.copy()
    dar_levers, meta_dar_score = meta_optimize_darwinian(
        initial_pop,
        seed=seed,
        rng=rng,
        call_budget=MD2_META_CALL_BUDGET,
        verbose=meta_verbose,
    )

    # Main run: Lamarckian with best levers
    counted_lam = CountedFunction(rastrigin_func)
    lam_gen = pure_lamarckian_function(
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
    lam_sum = summarize_lamarckian_run(lam_gen, rastrigin_func)
    if lam_sum:
        lam_sum["function_calls"] = counted_lam.n_calls
    final_lam = []
    if lam_gen:
        for start, end in lam_gen[-1]["organisms"]:
            final_lam.append(end[:2].tolist())
    final_distribution_lam = np.array(final_lam) if final_lam else None

    # Main run: Darwinian with best levers
    counted_dar = CountedFunction(rastrigin_func)
    dar_gen = pure_darwinian_function(
        fitness_topology_function=counted_dar,
        num_generations=100000,
        seed=seed + 100,
        initial_bounds=UP1_INITIAL_BOUNDS,
        max_calls=call_budget,
        initial_population=initial_pop,
        **dar_levers,
    )
    dar_sum = summarize_darwinian_run(dar_gen, rastrigin_func)
    if dar_sum:
        dar_sum["function_calls"] = counted_dar.n_calls
    final_dar = None
    if dar_gen:
        final_dar = np.array([p[:2] for p in dar_gen[-1]["organisms"]])

    n_init = initial_points.shape[0]
    out = {
        "seed": seed,
        "lamarckian_vectors": lamarckian_vectors,
        "darwinian_pop": darwinian_pop,
        "initial_points": initial_points,
        "initial_distribution": [initial_points[i, :2].tolist() for i in range(n_init)],
        "lam_levers": lam_levers,
        "dar_levers": dar_levers,
        "meta_lam_score": meta_lam_score,
        "meta_dar_score": meta_dar_score,
        "lam_generations": lam_gen,
        "dar_generations": dar_gen,
        "lam_summary": lam_sum,
        "dar_summary": dar_sum,
        "final_distribution_lam": final_distribution_lam.tolist() if final_distribution_lam is not None else None,
        "final_distribution_dar": final_dar.tolist() if final_dar is not None else None,
    }
    if verbose:
        print(
            f"MD2 seed={seed} | Lam: best_f={lam_sum['best_fitness']:.4f} dist={lam_sum['distance_to_optimum']:.3f} calls={lam_sum['function_calls']} "
            f"(meta_score={meta_lam_score:.4f}) | "
            f"Dar: best_f={dar_sum['best_fitness']:.4f} dist={dar_sum['distance_to_optimum']:.3f} calls={dar_sum['function_calls']} "
            f"(meta_score={meta_dar_score:.4f})"
        )
    return out


DEFAULT_10_SEEDS = [7, 27, 107, 207, 327, 507, 42, 123, 456, 789]


def run_md2_multi(
    seeds: list[int],
    call_budget: int = MD2_CALL_BUDGET,
    verbose: bool = True,
    lamarckian_vectors: int = 2,
    darwinian_pop: int = 4,
    meta_verbose: bool = False,
) -> list[dict]:
    """Run MD2 for each seed; return list of result dicts and print summary table."""
    seeds = sorted(seeds)
    results = []
    for seed in seeds:
        r = run_md2(
            seed=seed,
            call_budget=call_budget,
            verbose=False,
            lamarckian_vectors=lamarckian_vectors,
            darwinian_pop=darwinian_pop,
            meta_verbose=meta_verbose,
        )
        results.append(r)
        if verbose:
            lam = r["lam_summary"] or {}
            dar = r["dar_summary"] or {}
            print(
                f"  seed={seed}: Lam best_f={lam.get('best_fitness', float('nan')):.4f} dist={lam.get('distance_to_optimum', float('nan')):.3f} | "
                f"Dar best_f={dar.get('best_fitness', float('nan')):.4f} dist={dar.get('distance_to_optimum', float('nan')):.3f}"
            )
    if verbose and results:
        print("")
        print("Summary (seed order):")
        print("| Seed | Lam best_f | Dar best_f | Lam dist | Dar dist | Lam wins |")
        print("|------|------------|------------|----------|----------|----------|")
        lam_wins = 0
        for r in sorted(results, key=lambda x: x["seed"]):
            seed = r["seed"]
            lam = r["lam_summary"] or {}
            dar = r["dar_summary"] or {}
            lb, db = lam.get("best_fitness", float("nan")), dar.get("best_fitness", float("nan"))
            ld, dd = lam.get("distance_to_optimum", float("nan")), dar.get("distance_to_optimum", float("nan"))
            win = "Dar" if db < lb else "Lam"
            if lb < db:
                lam_wins += 1
            print(f"| {seed} | {lb:.4f} | {db:.4f} | {ld:.3f} | {dd:.3f} | {win} |")
        print("| (Better best_f = lower value) |")
        print(f"Lamarckian better best_fitness: {lam_wins}/{len(seeds)} seeds; Darwinian: {len(seeds) - lam_wins}/{len(seeds)}.")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Test MD2: Meta-optimized (Darwinian-tuned) Lamarckian v Darwinian")
    p.add_argument("--seed", type=int, default=None, help="Single seed")
    p.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds")
    p.add_argument("--num-seeds", type=int, default=None, help="Use first N of default 10 seeds")
    p.add_argument("--calls", type=int, default=MD2_CALL_BUDGET, help="Main run call budget")
    p.add_argument("--meta-calls", type=int, default=MD2_META_CALL_BUDGET, help="Meta-evaluation call budget per candidate")
    p.add_argument("--lamarckian-vectors", type=int, default=2)
    p.add_argument("--darwinian-pop", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--meta-verbose", action="store_true", help="Print per-meta-generation stats")
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
        run_md2(
            seed=seeds[0],
            call_budget=args.calls,
            verbose=not args.quiet,
            lamarckian_vectors=lam_vecs,
            darwinian_pop=dar_pop,
            meta_verbose=args.meta_verbose,
        )
    else:
        run_md2_multi(
            seeds,
            call_budget=args.calls,
            verbose=not args.quiet,
            lamarckian_vectors=lam_vecs,
            darwinian_pop=dar_pop,
            meta_verbose=args.meta_verbose,
        )
