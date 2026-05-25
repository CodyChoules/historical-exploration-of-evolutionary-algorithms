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

Termination: meta-evaluation uses a per-candidate call budget; main run uses 2000 calls.
"""

import random
import sys
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from lamarckian_functions.core import pure_lamarckian_function, rastrigin_func
from darwinian_functions.core import pure_darwinian_function
from meta_evolution_functions import CountedFunction

from experimental_analysis.run_comparison import (
    distance_to_optimum,
    summarize_lamarckian_run,
    summarize_darwinian_run,
)
from experimental_analysis.run_up1 import (
    UP1_INITIAL_BOUNDS,
    get_initial_points,
    lamarckian_parents_from_points,
)

# -----------------------------------------------------------------------------
# Constants (aligned with run_up1 where applicable)
# -----------------------------------------------------------------------------
# initial_bounds: MD2 uses UP1_INITIAL_BOUNDS from run_up1 (-12, 12, -12, 12) so
# Darwinian offspring are not clipped to a smaller box and match the visualization axis range.
# Main run: max topology (fitness) evaluations per algorithm for the final comparison run.
MD2_CALL_BUDGET = 100
# Meta-evaluation: max topology calls per candidate when scoring lever sets (shorter = faster, noisier).
MD2_META_CALL_BUDGET = 100
# Meta-evolution: number of lever candidates per generation.
MD2_META_POPULATION = 100
# Meta-evolution: number of generations (each generation scores all candidates, then selects/mutates).
MD2_META_GENERATIONS = 100
# Meta-evolution: number of best candidates kept as elites for reproduction each generation.
MD2_META_ELITE = 10
# Number of initial points: 4 → 2 Lamarckian parent vectors, 4 Darwinian organisms.
NUM_INITIAL_POINTS = 4

MD2_RESULTS_DIR = _project_root / "experimental_results" / "complexity 1" / "2MD2" / "generated results"

# Lamarckian lever bounds and mutation std (for meta-evolution)
LAM_LEVER_MUTATION_STD = {
    # Weight of gradient-based besoin vs parent displacements; >1 = more pull from topology.
    "besoin_weight": 0.18,
    # Scale factor for gradient-based besoin vector; larger = stronger gradient influence on step size.
    "topology_gradient_scale": 0.02,
    # Random variation in child length: std = mean_magnitude * this (0 = deterministic; 1 = high).
    "magnitude_std_fraction": 0.05,
    # Blend for base child length: 1.0 = parent mean length, 0.0 = length from mean displacement; in [0,1].
    "magnitude_weight": 0.08,
    # Std of random noise added to child direction (radians); larger = more exploration.
    "direction_std": 0.08,
    # Lower bound on child displacement length (avoids vanishing steps).
    "min_magnitude": 0.005,
    # Upper bound on child displacement length (caps step size).
    "max_magnitude": 1.5,
    # Number of child organisms per generation (int).
    "num_offspring": 0.3,
    # Length of initial parent vector 1 (direction unchanged).
    "initial_magnitude_1": 1.0,
    # Length of initial parent vector 2 (direction unchanged).
    "initial_magnitude_2": 1.0,
}

LAM_LEVER_BOUNDS = {
    "besoin_weight": (0.2, 2.0),
    "topology_gradient_scale": (0.02, 0.3),
    "magnitude_std_fraction": (0.0, 1),
    "magnitude_weight": (0.0, 1.0),
    "direction_std": (0.2, 0.6),
    "min_magnitude": (0.005, 0.1),
    "max_magnitude": (0.5, 10.0),
    "num_offspring": (2, 4),
    "initial_magnitude_1": (0.5, 20.0),
    "initial_magnitude_2": (0.5, 20.0),
}


# Darwinian lever bounds and mutation std (point2/point4 = second and fourth initial organism positions)
_xmin, _xmax, _ymin, _ymax = UP1_INITIAL_BOUNDS
DAR_LEVER_BOUNDS = {
    "elimination_rate": (0.1, 0.9),
    "selection_pressure": (0.5, 15.0),
    "mutation_std": (0.1, 2.0),
    "point2_x": (_xmin, _xmax),
    "point2_y": (_ymin, _ymax),
    "point4_x": (_xmin, _xmax),
    "point4_y": (_ymin, _ymax),
}
DAR_LEVER_MUTATION_STD = {
    "elimination_rate": 0.08,
    "selection_pressure": 1.2,
    "mutation_std": 0.15,
    "point2_x": 0.5,
    "point2_y": 0.5,
    "point4_x": 0.5,
    "point4_y": 0.5,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _rescale_parent_vectors_by_magnitude(
    parent1_start: np.ndarray,
    parent1_end: np.ndarray,
    parent2_start: np.ndarray,
    parent2_end: np.ndarray,
    magnitude_1: float,
    magnitude_2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rescale parent vector end points to the given magnitudes without changing direction.
    Returns (parent1_start, new_parent1_end, parent2_start, new_parent2_end).
    """
    p1_start = parent1_start.copy()
    p2_start = parent2_start.copy()
    d1 = (parent1_end - parent1_start)[:2]
    d2 = (parent2_end - parent2_start)[:2]
    len1 = np.linalg.norm(d1)
    len2 = np.linalg.norm(d2)
    if len1 < 1e-12:
        unit1 = np.array([1.0, 0.0])
    else:
        unit1 = d1 / len1
    if len2 < 1e-12:
        unit2 = np.array([1.0, 0.0])
    else:
        unit2 = d2 / len2
    new_p1_end = np.array([p1_start[0] + unit1[0] * magnitude_1, p1_start[1] + unit1[1] * magnitude_1, 0.0])
    new_p2_end = np.array([p2_start[0] + unit2[0] * magnitude_2, p2_start[1] + unit2[1] * magnitude_2, 0.0])
    return p1_start, new_p1_end, p2_start, new_p2_end


# ---------------------------------------------------------------------------
# Initial Lamarckian lever candidates for meta-evolution
# ---------------------------------------------------------------------------
# The initial population of lever sets is generated by sample_lamarckian_levers(rng):
# each lever is drawn uniformly at random from LAM_LEVER_BOUNDS. The rng is seeded
# in run_md2 as random.Random(seed + 99999), so the same run seed yields the same
# initial candidates. There are no hand-picked or formula-based starting levers—
# only random sampling within bounds. Later generations are created by selecting
# the best (elite) candidates and mutating them via mutate_lamarckian_levers.
# ---------------------------------------------------------------------------


def sample_lamarckian_levers(rng: random.Random) -> dict:
    """One random Lamarckian lever set for meta-evolution (used for initial population)."""
    lo, hi = LAM_LEVER_BOUNDS["num_offspring"]
    return {
        "besoin_weight": rng.uniform(*LAM_LEVER_BOUNDS["besoin_weight"]),
        "topology_gradient_scale": rng.uniform(*LAM_LEVER_BOUNDS["topology_gradient_scale"]),
        "magnitude_std_fraction": rng.uniform(*LAM_LEVER_BOUNDS["magnitude_std_fraction"]),
        "magnitude_weight": rng.uniform(*LAM_LEVER_BOUNDS["magnitude_weight"]),
        "direction_std": rng.uniform(*LAM_LEVER_BOUNDS["direction_std"]),
        "min_magnitude": rng.uniform(*LAM_LEVER_BOUNDS["min_magnitude"]),
        "max_magnitude": rng.uniform(*LAM_LEVER_BOUNDS["max_magnitude"]),
        "num_offspring": int(rng.randint(int(lo), int(hi))),
        "initial_magnitude_1": rng.uniform(*LAM_LEVER_BOUNDS["initial_magnitude_1"]),
        "initial_magnitude_2": rng.uniform(*LAM_LEVER_BOUNDS["initial_magnitude_2"]),
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
    """One random Darwinian lever set for meta-evolution. When population_size==4, includes optimizable point2 and point4."""
    max_er = (population_size - 1) / population_size
    out = {
        "population_size": population_size,
        "elimination_rate": rng.uniform(0.1, min(0.9, max_er)),
        "selection_pressure": rng.uniform(*DAR_LEVER_BOUNDS["selection_pressure"]),
        "mutation_std": rng.uniform(*DAR_LEVER_BOUNDS["mutation_std"]),
    }
    if population_size == 4:
        out["point2_x"] = rng.uniform(*DAR_LEVER_BOUNDS["point2_x"])
        out["point2_y"] = rng.uniform(*DAR_LEVER_BOUNDS["point2_y"])
        out["point4_x"] = rng.uniform(*DAR_LEVER_BOUNDS["point4_x"])
        out["point4_y"] = rng.uniform(*DAR_LEVER_BOUNDS["point4_y"])
    return out


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
    if "point2_x" in levers:
        out["point2_x"] = clamp(
            rng.gauss(levers["point2_x"], DAR_LEVER_MUTATION_STD["point2_x"]),
            *DAR_LEVER_BOUNDS["point2_x"],
        )
        out["point2_y"] = clamp(
            rng.gauss(levers["point2_y"], DAR_LEVER_MUTATION_STD["point2_y"]),
            *DAR_LEVER_BOUNDS["point2_y"],
        )
        out["point4_x"] = clamp(
            rng.gauss(levers["point4_x"], DAR_LEVER_MUTATION_STD["point4_x"]),
            *DAR_LEVER_BOUNDS["point4_x"],
        )
        out["point4_y"] = clamp(
            rng.gauss(levers["point4_y"], DAR_LEVER_MUTATION_STD["point4_y"]),
            *DAR_LEVER_BOUNDS["point4_y"],
        )
    return out


# Keys passed to pure_lamarckian_function (excludes initial_magnitude_1/2 which only rescale parents)
_LAM_PURE_KEYS = {
    "besoin_weight", "topology_gradient_scale", "magnitude_std_fraction", "magnitude_weight",
    "direction_std", "min_magnitude", "max_magnitude", "num_offspring", "first_generation_random_besoin",
}
# Keys passed to pure_darwinian_function (excludes point2_*, point4_* which only build initial_population)
_DAR_PURE_KEYS = {"population_size", "elimination_rate", "selection_pressure", "mutation_std"}


def build_initial_population_from_levers(levers: dict, base_points: np.ndarray) -> np.ndarray:
    """Build 4-point initial population from base (points 1 and 3 fixed) and levers (point2 and point4)."""
    pop = np.asarray(base_points, dtype=float).copy()
    if pop.shape[0] < 4:
        return pop
    if "point2_x" in levers and "point2_y" in levers:
        pop[1, 0] = levers["point2_x"]
        pop[1, 1] = levers["point2_y"]
        pop[1, 2] = 0.0
    if "point4_x" in levers and "point4_y" in levers:
        pop[3, 0] = levers["point4_x"]
        pop[3, 1] = levers["point4_y"]
        pop[3, 2] = 0.0
    return pop


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
    mag1 = levers.get("initial_magnitude_1")
    mag2 = levers.get("initial_magnitude_2")
    if mag1 is not None and mag2 is not None:
        parent1_start, parent1_end, parent2_start, parent2_end = _rescale_parent_vectors_by_magnitude(
            parent1_start, parent1_end, parent2_start, parent2_end, mag1, mag2
        )
    lam_kw = {k: v for k, v in levers.items() if k in _LAM_PURE_KEYS}
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
        **lam_kw,
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
    pop = build_initial_population_from_levers(levers, initial_population)
    dar_kw = {k: levers[k] for k in _DAR_PURE_KEYS if k in levers}
    counted = CountedFunction(rastrigin_func)
    gen = pure_darwinian_function(
        fitness_topology_function=counted,
        num_generations=100000,
        seed=seed,
        initial_bounds=UP1_INITIAL_BOUNDS,
        max_calls=call_budget,
        initial_population=pop,
        **dar_kw,
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

    # Rescale parent vectors to optimized initial magnitudes (direction unchanged)
    mag1 = lam_levers.get("initial_magnitude_1")
    mag2 = lam_levers.get("initial_magnitude_2")
    if mag1 is not None and mag2 is not None:
        parent1_start, parent1_end, parent2_start, parent2_end = _rescale_parent_vectors_by_magnitude(
            parent1_start, parent1_end, parent2_start, parent2_end, mag1, mag2
        )
    lam_kw = {k: v for k, v in lam_levers.items() if k in _LAM_PURE_KEYS}

    # Meta-optimize Darwinian levers (uses original initial points)
    if verbose:
        print(f"MD2 seed={seed}: meta-optimizing Darwinian levers...")
    initial_pop_meta = initial_points.copy()
    dar_levers, meta_dar_score = meta_optimize_darwinian(
        initial_pop_meta,
        seed=seed,
        rng=rng,
        call_budget=MD2_META_CALL_BUDGET,
        verbose=meta_verbose,
    )

    # Main run: Lamarckian with best levers (rescaled parents; no magnitude levers passed)
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
        **lam_kw,
    )
    lam_sum = summarize_lamarckian_run(lam_gen, rastrigin_func)
    if lam_sum:
        lam_sum["function_calls"] = counted_lam.n_calls
    final_lam = []
    if lam_gen:
        for start, end in lam_gen[-1]["organisms"]:
            final_lam.append(end[:2].tolist())
    final_distribution_lam = np.array(final_lam) if final_lam else None

    # Main run: Darwinian with best levers (initial pop: points 1 and 3 fixed, 2 and 4 from optimized levers)
    darwinian_initial_pop = build_initial_population_from_levers(dar_levers, initial_points.copy())
    dar_kw = {k: dar_levers[k] for k in _DAR_PURE_KEYS if k in dar_levers}
    counted_dar = CountedFunction(rastrigin_func)
    dar_gen = pure_darwinian_function(
        fitness_topology_function=counted_dar,
        num_generations=100000,
        seed=seed + 100,
        initial_bounds=UP1_INITIAL_BOUNDS,
        max_calls=call_budget,
        initial_population=darwinian_initial_pop,
        **dar_kw,
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
        "darwinian_initial_points": darwinian_initial_pop,
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
    p.add_argument("--print-levers", action="store_true", help="Print optimized lam_levers and dar_levers (single seed only)")
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
        result = run_md2(
            seed=seeds[0],
            call_budget=args.calls,
            verbose=not args.quiet,
            lamarckian_vectors=lam_vecs,
            darwinian_pop=dar_pop,
            meta_verbose=args.meta_verbose,
        )
        if getattr(args, "print_levers", False):
            print("\n=== LAMARCKIAN (optimized for seed {}) ===".format(seeds[0]))
            for k, v in sorted(result["lam_levers"].items()):
                print(f"  {k}: {v}")
            print("\n=== DARWINIAN (optimized for seed {}) ===".format(seeds[0]))
            for k, v in sorted(result["dar_levers"].items()):
                print(f"  {k}: {v}")
    else:
        run_md2_multi(
            seeds,
            call_budget=args.calls,
            verbose=not args.quiet,
            lamarckian_vectors=lam_vecs,
            darwinian_pop=dar_pop,
            meta_verbose=args.meta_verbose,
        )
