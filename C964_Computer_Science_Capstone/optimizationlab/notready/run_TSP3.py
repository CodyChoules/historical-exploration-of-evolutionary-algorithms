"""
Test TSP3: MD2-style meta-optimization applied to the Traveling Salesman Problem.

Same flow as MD2 but with TSP instead of Rastrigin:
  1. Generate shared initial tours (random permutations).
  2. Meta-optimize TSP-LGA (Lamarckian Genetic Algorithm) levers (2-opt depth, mutation, selection, etc.).
  3. Meta-optimize TSP-Darwinian levers (elimination, selection, mutation).
  4. Main run: LGA and Darwinian with best levers; compare best tour length.

LGA = Lamarckian Genetic Algorithm: 2-opt improvement (acquired character) then selection and
mutation. Distinct from the continuous Lamarckian (besoin vectors) used in other experiments.
Fitness = tour length (lower is better). Call budget limits total fitness evaluations.
"""

import random
import sys
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from optimizationfunctions.evolutionalgorithms.metaevolutionfunctions import CountedFunction


# -----------------------------------------------------------------------------
# TSP problem: cities, distance matrix, tour length
# -----------------------------------------------------------------------------

def make_cities(n_cities: int, seed: int) -> np.ndarray:
    """Return (n_cities, 2) coordinates in [0, 1]^2."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (n_cities, 2)).astype(np.float64)


def distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Return (n, n) Euclidean distance matrix."""
    n = cities.shape[0]
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(cities[i] - cities[j])
            d[i, j] = d[j, i] = dist
    return d


def tour_length(perm: list, dist: np.ndarray) -> float:
    """Closed tour length for permutation of city indices (0..n-1)."""
    n = len(perm)
    if n < 2:
        return 0.0
    total = 0.0
    for k in range(n):
        i, j = perm[k], perm[(k + 1) % n]
        total += dist[i, j]
    return total


def make_tour_fitness(dist: np.ndarray):
    """Return a callable perm -> tour_length(perm) for use with CountedFunction."""
    def fitness(perm):
        return tour_length(perm, dist)
    return fitness


def nearest_neighbor_tour(dist: np.ndarray, start: int = 0) -> tuple[list, float]:
    """
    Build a TSP tour from city `start` by repeatedly going to the nearest unvisited city.
    Returns (tour as permutation of indices, closed tour length).
    """
    n = dist.shape[0]
    if n < 2:
        return ([start] if n == 1 else [], 0.0)
    tour = [start]
    unvisited = set(range(n)) - {start}
    while unvisited:
        i = tour[-1]
        j = min(unvisited, key=lambda k: dist[i, k])
        tour.append(j)
        unvisited.discard(j)
    length = tour_length(tour, dist)
    return tour, length


def best_nearest_neighbor(dist: np.ndarray) -> tuple[list, float]:
    """Run nearest neighbor from each city as start; return best (tour, length)."""
    n = dist.shape[0]
    best_tour, best_len = None, float("inf")
    for start in range(n):
        tour, length = nearest_neighbor_tour(dist, start)
        if length < best_len:
            best_tour, best_len = tour, length
    return best_tour, best_len


# -----------------------------------------------------------------------------
# Permutation mutation and 2-opt
# -----------------------------------------------------------------------------

def mutate_permutation_swaps(perm: list, rng: random.Random, num_swaps: int = 1) -> list:
    """Return a new permutation with num_swaps random pairwise swaps."""
    perm = list(perm)
    n = len(perm)
    for _ in range(num_swaps):
        i, j = rng.randrange(n), rng.randrange(n)
        if i != j:
            perm[i], perm[j] = perm[j], perm[i]
    return perm


def two_opt_improve(perm: list, dist: np.ndarray, max_steps: int, rng: random.Random) -> list:
    """Apply up to max_steps 2-opt moves; return improved permutation (copy)."""
    perm = list(perm)
    n = len(perm)
    if n < 4 or max_steps <= 0:
        return perm
    current_len = tour_length(perm, dist)
    for _ in range(max_steps):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        # Reverse segment [i+1, j]
        new_perm = perm[: i + 1] + list(reversed(perm[i + 1 : j + 1])) + perm[j + 1 :]
        new_len = tour_length(new_perm, dist)
        if new_len < current_len:
            perm = new_perm
            current_len = new_len
    return perm


def two_opt_single_move(tour: list, i: int, j: int) -> list:
    """One 2-opt move: reverse segment [i+1, j]. Requires 0 <= i < j < len(tour)."""
    return tour[: i + 1] + list(reversed(tour[i + 1 : j + 1])) + tour[j + 1 :]


# -----------------------------------------------------------------------------
# Trait space for TSP Pure Lamarckian (Schwefel-style: trait = priority per city)
# -----------------------------------------------------------------------------

TSP_TRAIT_LOW = -12.0
TSP_TRAIT_HIGH = 12.0


def perm_to_trait(perm: list) -> np.ndarray:
    """Map permutation to trait vector: trait[c] = position of city c in tour, scaled to [TRAIT_LOW, TRAIT_LOW+1]."""
    n = len(perm)
    trait = np.zeros(n, dtype=np.float64)
    for pos, city in enumerate(perm):
        trait[city] = float(pos)
    if n > 1:
        trait = TSP_TRAIT_LOW + (trait / (n - 1))  # scale to [TSP_TRAIT_LOW, TSP_TRAIT_LOW + 1]
    else:
        trait[:] = TSP_TRAIT_LOW
    return trait


def trait_to_perm(trait_vec: np.ndarray) -> list:
    """Decode trait vector to tour: visit cities in order of increasing trait (argsort)."""
    return np.argsort(trait_vec).tolist()


def random_point_in_quadrilateral_nd(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """Random point inside quadrilateral (convex combination) in R^n."""
    w = np.random.random(4)
    w /= w.sum()
    return w[0] * p1 + w[1] * p2 + w[2] * p3 + w[3] * p4


# -----------------------------------------------------------------------------
# TSP Darwinian: population of tours, selection, swap mutation
# -----------------------------------------------------------------------------

def _select_survivor_indices_tsp(fitness_values: list, survivors_target: int, selection_pressure: float, rng: random.Random) -> list:
    """Fitness = tour length (lower better). Prob proportional to exp(-selection_pressure * f)."""
    n = len(fitness_values)
    f = np.array(fitness_values, dtype=float)
    f_max = np.max(f)
    weights = np.exp(-selection_pressure * (f - f_max))
    weights /= np.sum(weights)
    # Weighted sampling without replacement: remaining (index, weight) list
    remaining = [(i, float(weights[i])) for i in range(n)]
    chosen = []
    for _ in range(min(survivors_target, n)):
        total = sum(w for _, w in remaining)
        if total <= 0:
            break
        r = rng.uniform(0, total)
        for j, (i, w) in enumerate(remaining):
            r -= w
            if r <= 0:
                chosen.append(i)
                remaining.pop(j)
                break
    return chosen


def pure_tsp_darwinian(
    tour_fitness_fn,
    num_generations: int = 10000,
    seed: int = None,
    max_calls: int = None,
    initial_tours: list = None,
    population_size: int = 32,
    elimination_rate: float = 0.5,
    selection_pressure: float = 4.0,
    mutation_swaps: float = 1.0,
) -> list:
    """
    Pure Darwinian TSP: population of permutations, selection by tour length, mutation by swaps.
    tour_fitness_fn(perm) -> float (lower better). If CountedFunction, max_calls is respected.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    if initial_tours is None or len(initial_tours) < 2:
        raise ValueError("initial_tours must be a list of at least 2 permutations")
    n_cities = len(initial_tours[0])
    current_pop = [list(p) for p in initial_tours]
    population_size = len(current_pop)
    survivors_target = max(1, int(round(population_size * (1.0 - elimination_rate))))
    generations = []

    for gen in range(num_generations):
        fitness_values = [tour_fitness_fn(p) for p in current_pop]
        survivor_indices = _select_survivor_indices_tsp(
            fitness_values, survivors_target, selection_pressure, rng
        )
        survivors = [current_pop[i] for i in survivor_indices]
        generations.append({
            "generation": gen,
            "organisms": [list(p) for p in current_pop],
            "fitness_values": list(fitness_values),
            "survivor_indices": survivor_indices,
        })

        if max_calls is not None and getattr(tour_fitness_fn, "n_calls", 0) >= max_calls:
            break
        if gen == num_generations - 1:
            break

        # Repopulate: survivors + mutated offspring
        next_pop = list(survivors)
        num_swaps = max(1, int(round(mutation_swaps)))
        while len(next_pop) < population_size:
            parent = rng.choice(survivors)
            child = mutate_permutation_swaps(parent, rng, num_swaps=num_swaps)
            next_pop.append(child)
        current_pop = next_pop

    return generations


# -----------------------------------------------------------------------------
# TSP LGA (Lamarckian Genetic Algorithm): 2-opt improvement then selection and mutation
# -----------------------------------------------------------------------------

def pure_tsp_lga(
    tour_fitness_fn,
    initial_parent_tours: list,
    num_generations: int = 10000,
    seed: int = None,
    max_calls: int = None,
    num_offspring: int = 2,
    two_opt_steps: int = 5,
    mutation_swaps: float = 1.0,
    elimination_rate: float = 0.5,
    selection_pressure: float = 4.0,
) -> list:
    """
    TSP LGA (Lamarckian Genetic Algorithm): each gen apply 2-opt to each organism (acquired
    improvement), then select and mutate. initial_parent_tours: list of 2 (or more) starting
    permutations. Population = 2 + num_offspring.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    dist = getattr(tour_fitness_fn, "_dist", None)
    if dist is None:
        raise ValueError("tour_fitness_fn must have ._dist for 2-opt (pass a wrapper that sets it)")
    if len(initial_parent_tours) < 2:
        raise ValueError("initial_parent_tours must have at least 2 tours")
    population_size = 2 + num_offspring
    # LAMARCKIAN: Start from given parent tours (genotype = permutation). No deviation.
    current_pop = [list(p) for p in initial_parent_tours[:2]]
    # DEVIATION: We pad to population_size by mutating from current pop; Lamarckian does not require
    # a fixed "2 parents + offspring" structure, but we mirror a small population for comparability.
    while len(current_pop) < population_size:
        current_pop.append(mutate_permutation_swaps(rng.choice(current_pop), rng, num_swaps=1))
    survivors_target = max(1, int(round(population_size * (1.0 - elimination_rate))))
    num_swaps = max(1, int(round(mutation_swaps)))
    generations = []

    for gen in range(num_generations):
        # ---------- LAMARCKIAN CORE: Acquired character ----------
        # Each organism is improved during its "lifetime" (2-opt local search). The improved tour
        # is what gets evaluated and passed on—this is inheritance of acquired characteristics.
        improved = [two_opt_improve(p, dist, two_opt_steps, rng) for p in current_pop]
        # Fitness is evaluated on the *improved* phenotype (Lamarckian: selection sees acquired gain).
        fitness_values = [tour_fitness_fn(p) for p in improved]
        # Selection acts on improved individuals (Lamarckian: fitter = those who acquired improvement).
        survivor_indices = _select_survivor_indices_tsp(
            fitness_values, survivors_target, selection_pressure, rng
        )
        # Survivors are the improved tours; they become the gene pool for the next generation
        # (Lamarckian: the acquired improvement is inherited because we use improved as parents).
        survivors = [improved[i] for i in survivor_indices]
        generations.append({
            "generation": gen,
            "organisms": [list(p) for p in improved],
            "fitness_values": list(fitness_values),
            "survivor_indices": survivor_indices,
        })

        if max_calls is not None and getattr(tour_fitness_fn, "n_calls", 0) >= max_calls:
            break
        if gen == num_generations - 1:
            break

        # ---------- Reproduction for next generation ----------
        # LAMARCKIAN: Next gen is built from survivors (improved tours). Offspring are mutations
        # of those improved tours, so the "acquired" genotype is what gets mutated and passed on.
        next_pop = list(survivors)
        # DEVIATION: Offspring are created by mutation only (no crossover). Pure Lamarckian does
        # not forbid crossover; we use mutation-only here for simplicity and to match common EA.
        while len(next_pop) < population_size:
            parent = rng.choice(survivors)
            next_pop.append(mutate_permutation_swaps(parent, rng, num_swaps=num_swaps))
        current_pop = next_pop

    return generations


# -----------------------------------------------------------------------------
# TSP Pure Lamarckian (skeleton): besoin-style, spawn from parents, to be developed
# -----------------------------------------------------------------------------
#
# This function will follow the following approach:
#
# One: Map each trait-value dimension to a single city. The trait value represents the priority preference for that city, giving a continuous space of trait values following Schwefel's idea. These values represent the organism's preference for the cities; on each function call we order the cities to visit based on this priority preference.
#
# Two: Use these values to define a multidimensional space of trait values. As in UP1, construct initial n (default 2) parent vectors by giving -12 to each city at the start point and m (default: start point plus a random value in [0, 1]) at the end point, with m different for each parent vector and each trait. This defines the habit and spawn region.
#
# Besoin at spawn points: We cannot use gradient descent as in UP1 because we optimize a tour, not a function. We still use a besoin vector to guide search. To get a besoin that represents a reduction in "need" (shorter tour): randomly remove two edges from the current tour; there is only one alternative reconnection. Translate that change into a Schwefel (trait) vector by setting trait values so argsort yields the new tour. If the new tour is better, besoin points toward it; if worse, opposite direction.
#
# For future work: consider a UMAP of cities to define a continuous trait space.
#
def _compute_besoin_tsp(
    spawn_trait: np.ndarray,
    tour_fitness_fn,
    besoin_scale: float,
    rng: random.Random,
) -> np.ndarray:
    """
    One random 2-opt move at spawn_trait; besoin points toward better tour or opposite if worse.
    Returns besoin vector in trait space (same shape as spawn_trait).
    """
    n = len(spawn_trait)
    if n < 4:
        return np.zeros_like(spawn_trait)
    tour = trait_to_perm(spawn_trait)
    current_fit = tour_fitness_fn(tour)
    i, j = rng.randrange(n), rng.randrange(n)
    if i == j:
        return np.zeros_like(spawn_trait)
    if i > j:
        i, j = j, i
    if j - i < 2:  # need at least 2 between i and j for a non-trivial 2-opt
        return np.zeros_like(spawn_trait)
    new_tour = two_opt_single_move(tour, i, j)
    new_fit = tour_fitness_fn(new_tour)
    new_trait = perm_to_trait(new_tour)
    diff = new_trait - spawn_trait
    if new_fit < current_fit:
        return diff * besoin_scale
    return -diff * besoin_scale


def pure_tsp_lamarckian(
    tour_fitness_fn,
    initial_parent_tours: list,
    num_generations: int = 10000,
    seed: int = None,
    max_calls: int = None,
    num_offspring: int = 2,
    besoin_weight: float = 1.0,
    besoin_scale: float = 0.2,
    direction_std: float = 0.0,
    magnitude_std_fraction: float = 0.1,
    magnitude_weight: float = 1.0,
    min_magnitude: float = 0.01,
    max_magnitude: float = None,
    **kwargs,
) -> list:
    """
    Pure Lamarckian TSP: trait space (Schwefel-style), spawn region from parent vectors,
    besoin from 2-opt direction in trait space. Same structure as continuous pure_lamarckian_function.

    Returns list of generations, each dict with:
        generation: int
        organisms: list of permutations (tours)
        fitness_values: list of float (tour lengths)
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    if len(initial_parent_tours) < 2:
        raise ValueError("initial_parent_tours must have at least 2 tours")
    n_cities = len(initial_parent_tours[0])
    generations = []

    # Initial parent vectors in trait space: start = (-12,...,-12), end = perm_to_trait(perm)
    current_parent1_start = np.full(n_cities, TSP_TRAIT_LOW, dtype=np.float64)
    current_parent1_end = perm_to_trait(initial_parent_tours[0])
    current_parent2_start = np.full(n_cities, TSP_TRAIT_LOW, dtype=np.float64)
    current_parent2_end = perm_to_trait(initial_parent_tours[1])

    # Generation 0: record initial parent tours (as permutations)
    gen0_organisms = [list(initial_parent_tours[0]), list(initial_parent_tours[1])]
    gen0_fitness = [tour_fitness_fn(gen0_organisms[0]), tour_fitness_fn(gen0_organisms[1])]
    generations.append({
        "generation": 0,
        "organisms": gen0_organisms,
        "fitness_values": gen0_fitness,
    })

    for gen in range(1, num_generations + 1):
        if max_calls is not None and getattr(tour_fitness_fn, "n_calls", 0) >= max_calls:
            break

        spawn_corners = [
            current_parent1_start.copy(),
            current_parent1_end.copy(),
            current_parent2_end.copy(),
            current_parent2_start.copy(),
        ]
        parent1_disp = current_parent1_end - current_parent1_start
        parent2_disp = current_parent2_end - current_parent2_start
        parent_magnitudes = [np.linalg.norm(parent1_disp), np.linalg.norm(parent2_disp)]
        mean_magnitude = float(np.mean(parent_magnitudes)) if parent_magnitudes else 1.0

        organisms_this_gen = []
        fitness_this_gen = []
        offspring_spawn_and_child = []  # (spawn_trait, child_trait) for next parent vectors

        for _ in range(num_offspring):
            spawn_trait = random_point_in_quadrilateral_nd(
                spawn_corners[0], spawn_corners[1], spawn_corners[2], spawn_corners[3]
            )
            besoin = _compute_besoin_tsp(spawn_trait, tour_fitness_fn, besoin_scale, rng)
            sum_vec = parent1_disp.copy()
            sum_vec += parent2_disp
            if besoin_weight > 0:
                sum_vec += besoin * besoin_weight
            total_weight = 2.0 + besoin_weight
            mean_disp = sum_vec / total_weight
            mean_norm = np.linalg.norm(mean_disp)
            if mean_norm > 1e-12:
                direction = mean_disp / mean_norm
            else:
                direction = np.ones(n_cities) / np.sqrt(n_cities)
            if direction_std > 0:
                direction = direction + np.random.normal(0, direction_std, n_cities)
                nrm = np.linalg.norm(direction)
                if nrm > 1e-12:
                    direction = direction / nrm
            mean_disp_mag = np.linalg.norm(mean_disp)
            base_mag = magnitude_weight * mean_magnitude + (1.0 - magnitude_weight) * mean_disp_mag
            if magnitude_std_fraction > 0:
                child_mag = rng.gauss(base_mag, base_mag * magnitude_std_fraction)
                child_mag = max(min_magnitude, child_mag)
            else:
                child_mag = max(min_magnitude, base_mag)
            if max_magnitude is not None:
                child_mag = min(max_magnitude, child_mag)
            child_trait = spawn_trait + direction * child_mag
            child_trait = np.clip(child_trait, TSP_TRAIT_LOW, TSP_TRAIT_HIGH)
            child_perm = trait_to_perm(child_trait)
            organisms_this_gen.append(child_perm)
            fitness_this_gen.append(tour_fitness_fn(child_perm))
            offspring_spawn_and_child.append((spawn_trait.copy(), child_trait.copy()))

        generations.append({
            "generation": gen,
            "organisms": organisms_this_gen,
            "fitness_values": fitness_this_gen,
        })

        # Next parents: first two offspring (spawn = start, child_trait = end), as in continuous Lamarckian
        if len(offspring_spawn_and_child) >= 2:
            current_parent1_start, current_parent1_end = offspring_spawn_and_child[0]
            current_parent2_start, current_parent2_end = offspring_spawn_and_child[1]

    return generations


# -----------------------------------------------------------------------------
# Summaries for meta-objective (best tour length)
# -----------------------------------------------------------------------------

def summarize_tsp_run(generations: list) -> dict | None:
    """Return best_fitness (min tour length), mean_fitness, n_organisms from last gen."""
    if not generations:
        return None
    last = generations[-1]
    f = last["fitness_values"]
    if not f:
        return None
    return {
        "best_fitness": min(f),
        "mean_fitness": float(np.mean(f)),
        "n_organisms": len(last["organisms"]),
    }


# -----------------------------------------------------------------------------
# TSP3 constants and lever bounds (MD2-style)
# -----------------------------------------------------------------------------

TSP3_N_CITIES = 10
TSP3_CALL_BUDGET = 500
TSP3_META_CALL_BUDGET = 200
TSP3_META_POPULATION = 80
TSP3_META_GENERATIONS = 60
TSP3_META_ELITE = 8

# LGA (Lamarckian Genetic Algorithm) TSP levers
TSP_LGA_LEVER_BOUNDS = {
    "two_opt_steps": (0, 20),
    "num_offspring": (1, 6),
    "mutation_swaps": (0.5, 4.0),
    "elimination_rate": (0.1, 0.8),
    "selection_pressure": (0.5, 15.0),
}
TSP_LGA_LEVER_MUTATION_STD = {
    "two_opt_steps": 2.0,
    "num_offspring": 0.8,
    "mutation_swaps": 0.3,
    "elimination_rate": 0.08,
    "selection_pressure": 1.2,
}

# Pure Lamarckian TSP levers (trait space + besoin from 2-opt)
TSP_PURE_LAM_LEVER_BOUNDS = {
    "besoin_weight": (0.2, 2.0),
    "besoin_scale": (0.05, 0.5),
    "direction_std": (0.0, 0.5),
    "magnitude_std_fraction": (0.0, 0.5),
    "magnitude_weight": (0.0, 1.0),
    "min_magnitude": (0.001, 0.1),
    "max_magnitude": (0.5, 5.0),
    "num_offspring": (1, 6),
}
TSP_PURE_LAM_LEVER_MUTATION_STD = {
    "besoin_weight": 0.2,
    "besoin_scale": 0.05,
    "direction_std": 0.05,
    "magnitude_std_fraction": 0.05,
    "magnitude_weight": 0.1,
    "min_magnitude": 0.01,
    "max_magnitude": 0.5,
    "num_offspring": 0.8,
}

# Darwinian TSP levers
TSP_DAR_LEVER_BOUNDS = {
    "elimination_rate": (0.1, 0.8),
    "selection_pressure": (0.5, 15.0),
    "mutation_swaps": (0.5, 4.0),
}
TSP_DAR_LEVER_MUTATION_STD = {
    "elimination_rate": 0.08,
    "selection_pressure": 1.2,
    "mutation_swaps": 0.3,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sample_tsp_lga_levers(rng: random.Random) -> dict:
    out = {}
    for k, (lo, hi) in TSP_LGA_LEVER_BOUNDS.items():
        if k == "num_offspring":
            out[k] = int(rng.randint(int(lo), int(hi)))
        elif k == "two_opt_steps":
            out[k] = int(rng.randint(int(lo), int(hi)))
        else:
            out[k] = rng.uniform(lo, hi)
    return out


def mutate_tsp_lga_levers(levers: dict, rng: random.Random) -> dict:
    out = {}
    for k in TSP_LGA_LEVER_BOUNDS:
        lo, hi = TSP_LGA_LEVER_BOUNDS[k]
        std = TSP_LGA_LEVER_MUTATION_STD[k]
        if k == "num_offspring":
            out[k] = int(clamp(round(rng.gauss(levers[k], std)), int(lo), int(hi)))
        elif k == "two_opt_steps":
            out[k] = int(clamp(round(rng.gauss(levers[k], std)), int(lo), int(hi)))
        else:
            out[k] = clamp(rng.gauss(levers[k], std), lo, hi)
    return out


def sample_tsp_pure_lamarckian_levers(rng: random.Random) -> dict:
    out = {}
    for k, (lo, hi) in TSP_PURE_LAM_LEVER_BOUNDS.items():
        if k == "num_offspring":
            out[k] = int(rng.randint(int(lo), int(hi)))
        else:
            out[k] = rng.uniform(lo, hi)
    return out


def mutate_tsp_pure_lamarckian_levers(levers: dict, rng: random.Random) -> dict:
    out = {}
    for k in TSP_PURE_LAM_LEVER_BOUNDS:
        lo, hi = TSP_PURE_LAM_LEVER_BOUNDS[k]
        std = TSP_PURE_LAM_LEVER_MUTATION_STD[k]
        if k == "num_offspring":
            out[k] = int(clamp(round(rng.gauss(levers[k], std)), int(lo), int(hi)))
        else:
            out[k] = clamp(rng.gauss(levers[k], std), lo, hi)
    return out


def sample_tsp_darwinian_levers(rng: random.Random) -> dict:
    return {
        k: rng.uniform(*TSP_DAR_LEVER_BOUNDS[k])
        for k in TSP_DAR_LEVER_BOUNDS
    }


def mutate_tsp_darwinian_levers(levers: dict, rng: random.Random) -> dict:
    return {
        k: clamp(
            rng.gauss(levers[k], TSP_DAR_LEVER_MUTATION_STD[k]),
            *TSP_DAR_LEVER_BOUNDS[k],
        )
        for k in TSP_DAR_LEVER_BOUNDS
    }


# -----------------------------------------------------------------------------
# Wrapper so 2-opt has access to distance matrix
# -----------------------------------------------------------------------------

def counted_tour_fitness(dist: np.ndarray):
    """Return a CountedFunction that evaluates tour length and exposes ._dist for 2-opt."""
    raw = make_tour_fitness(dist)
    counted = CountedFunction(raw)
    counted._dist = dist
    return counted


# -----------------------------------------------------------------------------
# Meta-optimize and main run
# -----------------------------------------------------------------------------

def evaluate_tsp_lga_candidate(
    levers: dict,
    parent_tours: list,
    cities: np.ndarray,
    dist: np.ndarray,
    seed: int,
    call_budget: int,
) -> float:
    """Run TSP LGA (Lamarckian Genetic Algorithm); return best tour length (lower better)."""
    fitness_fn = counted_tour_fitness(dist)
    pop_size = 2 + levers.get("num_offspring", 2)
    initial = [list(parent_tours[0]), list(parent_tours[1])]
    while len(initial) < pop_size:
        initial.append(mutate_permutation_swaps(initial[-1], random.Random(seed + 1), 1))
    gen = pure_tsp_lga(
        tour_fitness_fn=fitness_fn,
        initial_parent_tours=initial[:2],
        num_generations=100000,
        seed=seed + 1000,
        max_calls=call_budget,
        num_offspring=levers.get("num_offspring", 2),
        two_opt_steps=int(levers.get("two_opt_steps", 5)),
        mutation_swaps=levers.get("mutation_swaps", 1.0),
        elimination_rate=levers.get("elimination_rate", 0.5),
        selection_pressure=levers.get("selection_pressure", 4.0),
    )
    s = summarize_tsp_run(gen)
    return s["best_fitness"] if s else float("inf")


def evaluate_tsp_darwinian_candidate(
    levers: dict,
    initial_tours: list,
    dist: np.ndarray,
    seed: int,
    call_budget: int,
) -> float:
    """Run TSP Darwinian; return best tour length (lower better)."""
    fitness_fn = counted_tour_fitness(dist)
    gen = pure_tsp_darwinian(
        tour_fitness_fn=fitness_fn,
        num_generations=100000,
        seed=seed + 2000,
        max_calls=call_budget,
        initial_tours=initial_tours,
        elimination_rate=levers["elimination_rate"],
        selection_pressure=levers["selection_pressure"],
        mutation_swaps=levers["mutation_swaps"],
    )
    s = summarize_tsp_run(gen)
    return s["best_fitness"] if s else float("inf")


def evaluate_tsp_pure_lamarckian_candidate(
    levers: dict,
    parent_tours: list,
    cities: np.ndarray,
    dist: np.ndarray,
    seed: int,
    call_budget: int,
) -> float:
    """Run TSP Pure Lamarckian (trait space + besoin); return best tour length (lower better)."""
    fitness_fn = counted_tour_fitness(dist)
    gen = pure_tsp_lamarckian(
        tour_fitness_fn=fitness_fn,
        initial_parent_tours=[list(parent_tours[0]), list(parent_tours[1])],
        num_generations=100000,
        seed=seed + 3000,
        max_calls=call_budget,
        num_offspring=levers.get("num_offspring", 2),
        besoin_weight=levers.get("besoin_weight", 1.0),
        besoin_scale=levers.get("besoin_scale", 0.2),
        direction_std=levers.get("direction_std", 0.0),
        magnitude_std_fraction=levers.get("magnitude_std_fraction", 0.1),
        magnitude_weight=levers.get("magnitude_weight", 1.0),
        min_magnitude=levers.get("min_magnitude", 0.01),
        max_magnitude=levers.get("max_magnitude"),
    )
    s = summarize_tsp_run(gen)
    return s["best_fitness"] if s else float("inf")


def meta_optimize_tsp_lga(
    parent_tours: list,
    cities: np.ndarray,
    dist: np.ndarray,
    seed: int,
    rng: random.Random,
    call_budget: int = TSP3_META_CALL_BUDGET,
    pop_size: int = TSP3_META_POPULATION,
    num_gens: int = TSP3_META_GENERATIONS,
    elite: int = TSP3_META_ELITE,
    verbose: bool = False,
) -> tuple[dict, float]:
    population = [sample_tsp_lga_levers(rng) for _ in range(pop_size)]
    best_levers = None
    best_score = float("inf")
    for g in range(num_gens):
        scores = []
        for levers in population:
            s = evaluate_tsp_lga_candidate(levers, parent_tours, cities, dist, seed, call_budget)
            scores.append((s, levers))
            if s < best_score:
                best_score = s
                best_levers = levers.copy()
        if verbose:
            avg = sum(x[0] for x in scores) / len(scores)
            print(f"  meta_lga gen={g} best={best_score:.2f} avg={avg:.2f}")
        if g == num_gens - 1:
            break
        ranked = sorted(scores, key=lambda x: x[0])
        elites = [x[1] for x in ranked[:elite]]
        next_pop = list(elites)
        while len(next_pop) < pop_size:
            next_pop.append(mutate_tsp_lga_levers(rng.choice(elites), rng))
        population = next_pop
    return best_levers or sample_tsp_lga_levers(rng), best_score


def meta_optimize_tsp_darwinian(
    initial_tours: list,
    dist: np.ndarray,
    seed: int,
    rng: random.Random,
    call_budget: int = TSP3_META_CALL_BUDGET,
    pop_size: int = TSP3_META_POPULATION,
    num_gens: int = TSP3_META_GENERATIONS,
    elite: int = TSP3_META_ELITE,
    verbose: bool = False,
) -> tuple[dict, float]:
    population = [sample_tsp_darwinian_levers(rng) for _ in range(pop_size)]
    best_levers = None
    best_score = float("inf")
    for g in range(num_gens):
        scores = []
        for levers in population:
            s = evaluate_tsp_darwinian_candidate(levers, initial_tours, dist, seed, call_budget)
            scores.append((s, levers))
            if s < best_score:
                best_score = s
                best_levers = levers.copy()
        if verbose:
            avg = sum(x[0] for x in scores) / len(scores)
            print(f"  meta_dar gen={g} best={best_score:.2f} avg={avg:.2f}")
        if g == num_gens - 1:
            break
        ranked = sorted(scores, key=lambda x: x[0])
        elites = [x[1] for x in ranked[:elite]]
        next_pop = list(elites)
        while len(next_pop) < pop_size:
            next_pop.append(mutate_tsp_darwinian_levers(rng.choice(elites), rng))
        population = next_pop
    return best_levers or sample_tsp_darwinian_levers(rng), best_score


def meta_optimize_tsp_pure_lamarckian(
    parent_tours: list,
    cities: np.ndarray,
    dist: np.ndarray,
    seed: int,
    rng: random.Random,
    call_budget: int = TSP3_META_CALL_BUDGET,
    pop_size: int = TSP3_META_POPULATION,
    num_gens: int = TSP3_META_GENERATIONS,
    elite: int = TSP3_META_ELITE,
    verbose: bool = False,
) -> tuple[dict, float]:
    population = [sample_tsp_pure_lamarckian_levers(rng) for _ in range(pop_size)]
    best_levers = None
    best_score = float("inf")
    for g in range(num_gens):
        scores = []
        for levers in population:
            s = evaluate_tsp_pure_lamarckian_candidate(levers, parent_tours, cities, dist, seed, call_budget)
            scores.append((s, levers))
            if s < best_score:
                best_score = s
                best_levers = levers.copy()
        if verbose:
            avg = sum(x[0] for x in scores) / len(scores)
            print(f"  meta_pure_lam gen={g} best={best_score:.2f} avg={avg:.2f}")
        if g == num_gens - 1:
            break
        ranked = sorted(scores, key=lambda x: x[0])
        elites = [x[1] for x in ranked[:elite]]
        next_pop = list(elites)
        while len(next_pop) < pop_size:
            next_pop.append(mutate_tsp_pure_lamarckian_levers(rng.choice(elites), rng))
        population = next_pop
    return best_levers or sample_tsp_pure_lamarckian_levers(rng), best_score


def run_tsp3(
    seed: int = 42,
    n_cities: int = TSP3_N_CITIES,
    call_budget: int = TSP3_CALL_BUDGET,
    verbose: bool = True,
    meta_verbose: bool = False,
) -> dict:
    """
    MD2-style run for TSP: shared cities and initial tours, meta-optimize LGA, Pure Lamarckian, and Darwinian, then main run.
    """
    rng = random.Random(seed + 99999)
    cities = make_cities(n_cities, seed)
    dist = distance_matrix(cities)
    # Nearest neighbor baseline (deterministic, no call budget)
    nn_tour, nn_length = best_nearest_neighbor(dist)
    # Shared initial tours: 2 for LGA parents, 4 for Darwinian population
    rng_perm = random.Random(seed)
    def random_perm():
        p = list(range(n_cities))
        rng_perm.shuffle(p)
        return p
    parent_tours = [random_perm() for _ in range(2)]
    initial_tours_dar = [random_perm() for _ in range(4)]

    if verbose:
        print(f"TSP3 seed={seed} n_cities={n_cities}: meta-optimizing TSP-LGA...")
    lga_levers, meta_lga_score = meta_optimize_tsp_lga(
        parent_tours, cities, dist, seed, rng, call_budget=TSP3_META_CALL_BUDGET, verbose=meta_verbose
    )
    if verbose:
        print(f"TSP3 seed={seed}: meta-optimizing TSP-Pure-Lamarckian...")
    pure_lam_levers, meta_pure_lam_score = meta_optimize_tsp_pure_lamarckian(
        parent_tours, cities, dist, seed, rng, call_budget=TSP3_META_CALL_BUDGET, verbose=meta_verbose
    )
    if verbose:
        print(f"TSP3 seed={seed}: meta-optimizing TSP-Darwinian...")
    dar_levers, meta_dar_score = meta_optimize_tsp_darwinian(
        initial_tours_dar, dist, seed, rng, call_budget=TSP3_META_CALL_BUDGET, verbose=meta_verbose
    )

    # Main run: LGA (Lamarckian Genetic Algorithm)
    fitness_lga = counted_tour_fitness(dist)
    pop_size_lga = 2 + lga_levers.get("num_offspring", 2)
    initial_lga = [list(parent_tours[0]), list(parent_tours[1])]
    while len(initial_lga) < pop_size_lga:
        initial_lga.append(mutate_permutation_swaps(initial_lga[-1], random.Random(seed + 2), 1))
    lga_gen = pure_tsp_lga(
        tour_fitness_fn=fitness_lga,
        initial_parent_tours=initial_lga[:2],
        num_generations=100000,
        seed=seed + 50,
        max_calls=call_budget,
        num_offspring=lga_levers.get("num_offspring", 2),
        two_opt_steps=int(lga_levers.get("two_opt_steps", 5)),
        mutation_swaps=lga_levers.get("mutation_swaps", 1.0),
        elimination_rate=lga_levers.get("elimination_rate", 0.5),
        selection_pressure=lga_levers.get("selection_pressure", 4.0),
    )
    lga_sum = summarize_tsp_run(lga_gen)

    # Main run: Pure Lamarckian (trait space + besoin)
    fitness_pure_lam = counted_tour_fitness(dist)
    pure_lam_gen = pure_tsp_lamarckian(
        tour_fitness_fn=fitness_pure_lam,
        initial_parent_tours=[list(parent_tours[0]), list(parent_tours[1])],
        num_generations=100000,
        seed=seed + 75,
        max_calls=call_budget,
        num_offspring=pure_lam_levers.get("num_offspring", 2),
        besoin_weight=pure_lam_levers.get("besoin_weight", 1.0),
        besoin_scale=pure_lam_levers.get("besoin_scale", 0.2),
        direction_std=pure_lam_levers.get("direction_std", 0.0),
        magnitude_std_fraction=pure_lam_levers.get("magnitude_std_fraction", 0.1),
        magnitude_weight=pure_lam_levers.get("magnitude_weight", 1.0),
        min_magnitude=pure_lam_levers.get("min_magnitude", 0.01),
        max_magnitude=pure_lam_levers.get("max_magnitude"),
    )
    pure_lam_sum = summarize_tsp_run(pure_lam_gen)

    # Main run: Darwinian
    fitness_dar = counted_tour_fitness(dist)
    dar_gen = pure_tsp_darwinian(
        tour_fitness_fn=fitness_dar,
        num_generations=100000,
        seed=seed + 100,
        max_calls=call_budget,
        initial_tours=initial_tours_dar,
        elimination_rate=dar_levers["elimination_rate"],
        selection_pressure=dar_levers["selection_pressure"],
        mutation_swaps=dar_levers["mutation_swaps"],
    )
    dar_sum = summarize_tsp_run(dar_gen)

    out = {
        "seed": seed,
        "n_cities": n_cities,
        "cities": cities,
        "nn_tour": nn_tour,
        "nn_length": nn_length,
        "lamarckian_genetic_algorithm": "LGA",
        "lga_levers": lga_levers,
        "pure_lam_levers": pure_lam_levers,
        "dar_levers": dar_levers,
        "meta_lga_score": meta_lga_score,
        "meta_pure_lam_score": meta_pure_lam_score,
        "meta_dar_score": meta_dar_score,
        "lga_generations": lga_gen,
        "pure_lam_generations": pure_lam_gen,
        "dar_generations": dar_gen,
        "lga_summary": lga_sum,
        "pure_lam_summary": pure_lam_sum,
        "dar_summary": dar_sum,
        "lga_calls": fitness_lga.n_calls if hasattr(fitness_lga, "n_calls") else None,
        "pure_lam_calls": fitness_pure_lam.n_calls if hasattr(fitness_pure_lam, "n_calls") else None,
        "dar_calls": fitness_dar.n_calls if hasattr(fitness_dar, "n_calls") else None,
        "lam_levers": lga_levers,
        "lam_generations": lga_gen,
        "lam_summary": lga_sum,
        "meta_lam_score": meta_lga_score,
        "lam_calls": fitness_lga.n_calls if hasattr(fitness_lga, "n_calls") else None,
    }
    if verbose:
        print(
            f"TSP3 seed={seed} | NN={nn_length:.2f} | LGA: best={lga_sum['best_fitness']:.2f} (meta={meta_lga_score:.2f}) | "
            f"PureLam: best={pure_lam_sum['best_fitness']:.2f} (meta={meta_pure_lam_score:.2f}) | "
            f"Dar: best={dar_sum['best_fitness']:.2f} (meta={meta_dar_score:.2f})"
        )
    return out


if __name__ == "__main__":
    result = run_tsp3(seed=7, call_budget=TSP3_CALL_BUDGET, meta_verbose=False)
    print("LGA levers:", result["lga_levers"])
    print("Pure Lamarckian levers:", result["pure_lam_levers"])
    print("Dar levers:", result["dar_levers"])
