"""
Test 1 UP1: Unoptimized Pure Lamarckian v Darwinian.

- Initial: 4 points (fixed or random in [-12, 12]^2). Same for both: 2 vectors for Lamarckian, 4 organisms for Darwinian.
- Levers: random from seed (no optimized/tuned values); all recorded for later optimization.
- Termination: 2000 topology calls per run.
- Records: initial distribution, final distribution, all lever values, metrics.
"""

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
    fitness_at,
    distance_to_optimum,
    summarize_lamarckian_run,
    summarize_darwinian_run,
)


# Match visualization axis range (-12, 12) so Darwinian offspring are not clipped to a smaller box
UP1_INITIAL_BOUNDS = (-12.0, 12.0, -12.0, 12.0)
UP1_CALL_BUDGET = 2000
NUM_INITIAL_POINTS = 4  # default: 4 points = 2 Lamarckian vectors = 4 Darwinian organisms
UP1_LAM_INITIAL_MAGNITUDE_RANGE = (2.0, 12.0)

# 4-point setup: both Lamarckian parent starts at (-12, -12); one end at (-11, -12), other at (-12, -11).
# Gives 2 parent vectors and 4 Darwinian organism points.
# Order: parent1_start, parent1_end, parent2_start, parent2_end.
FIXED_INITIAL_POINTS_4 = np.array([
    [-12.0, -12.0, 0.0],   # parent1_start, Darwinian organism 1
    [-11.0, -12.0, 0.0],   # parent1_end,   Darwinian organism 2
    [-12.0, -12.0, 0.0],   # parent2_start, Darwinian organism 3
    [-12.0, -11.0, 0.0],   # parent2_end,   Darwinian organism 4
], dtype=float)

# Generated results folder: experimental_results/complexity 1/1UP/generated results
UP1_RESULTS_DIR = _project_root / "experimental_results" / "complexity 1" / "1UP" / "generated results"

# TODO Set up docustrings on these parameters and fix discrepancies such as levers being partially tunable or random.
def run_up1(
    seed=42,
    call_budget=UP1_CALL_BUDGET,
    verbose=True,
    lamarckian_vectors=2,
    darwinian_population=4,
    use_fixed_initial_points_4=True,
    lamarckian_lever_parameters=None,
    dar_levers=None,
    lam_initial_magnitude_1=None,
    lam_initial_magnitude_2=None,
    lam_initial_magnitude_range=UP1_LAM_INITIAL_MAGNITUDE_RANGE,
    lam_lever_seed_offset=0,
    dar_lever_seed_offset=1,
    lam_magnitude_seed_offset=17,
    lam_run_seed_offset=50,
    dar_run_seed_offset=100,
    force_1l2d_single_offspring=True,
    dar_selection_pressure=None,
    dar_mutation_std=None,
):
    """
    Run Test 1 UP1: shared initial points, random levers from seed, call budget.
    lamarckian_vectors=1, darwinian_population=2 → 2 points: one Lamarckian vector (same twice), 2 Darwinian organisms.
    lamarckian_vectors=2, darwinian_population=4 (default) → 4 points: two vectors, 4 organisms.

    Lever dictionaries:
    - lamarckian_lever_parameters: optional partial override dict with keys
      {besoin_weight, topology_gradient_scale, magnitude_std_fraction, magnitude_weight,
      direction_std, min_magnitude, num_offspring, first_generation_random_besoin}.
      Missing keys keep seeded-random defaults. Note: UP1 always forces besoin_weight=0.0.
    - dar_levers: optional partial override dict with keys
      {population_size, elimination_rate, selection_pressure, mutation_std}.
      Missing keys keep seeded-random defaults.

    Returns dict with initial_points, lam_levers, dar_levers, generations, summaries, initial_distribution, final_distributions.
    """
    n_points = 4 if (lamarckian_vectors == 2 and darwinian_population == 4) else 2
    topology_func = rastrigin_func
    initial_points = get_initial_points(
        seed,
        n_points,
        use_fixed_4=(n_points == 4 and use_fixed_initial_points_4),
    )
    parent1_start, parent1_end, parent2_start, parent2_end = lamarckian_parents_from_points(
        initial_points, num_vectors=lamarckian_vectors
    )

    # Set Lamarckian levers with override behavior:
    # start from seeded random defaults, then replace only user-provided keys.
    lam_levers = random_lamarckian_levers(seed + lam_lever_seed_offset)
    if lamarckian_lever_parameters is not None:
        lam_levers.update(dict(lamarckian_lever_parameters))
    # UP1 experiment requirement: disable gradient besoin weighting.

    # Set Darwinian levers with override behavior:
    # start from seeded random defaults, then replace only user-provided keys.
    dar_lever_parameters = dar_levers
    dar_levers = random_darwinian_levers(
        seed + dar_lever_seed_offset, population_size=darwinian_population
    )
    if dar_lever_parameters is not None:
        dar_levers.update(dict(dar_lever_parameters))
    dar_levers.setdefault("population_size", darwinian_population)
    if dar_selection_pressure is not None:
        dar_levers["selection_pressure"] = float(dar_selection_pressure)
    if dar_mutation_std is not None:
        dar_levers["mutation_std"] = float(dar_mutation_std)

    # Lamarckian-only: parent-vector magnitudes (random by default).
    if lam_initial_magnitude_1 is None or lam_initial_magnitude_2 is None:
        mag_rng = np.random.default_rng(seed + lam_magnitude_seed_offset)
        min_mag, max_mag = lam_initial_magnitude_range
        if lam_initial_magnitude_1 is None:
            lam_initial_magnitude_1 = float(mag_rng.uniform(min_mag, max_mag))
        if lam_initial_magnitude_2 is None:
            lam_initial_magnitude_2 = float(mag_rng.uniform(min_mag, max_mag))
    lam_initial_magnitude_1 = float(lam_initial_magnitude_1)
    lam_initial_magnitude_2 = float(lam_initial_magnitude_2)
    if lamarckian_vectors == 1:
        lam_initial_magnitude_2 = lam_initial_magnitude_1
    parent1_start, parent1_end, parent2_start, parent2_end = _rescale_parent_vectors_by_magnitude(
        parent1_start,
        parent1_end,
        parent2_start,
        parent2_end,
        lam_initial_magnitude_1,
        lam_initial_magnitude_2,
    )
    # 1L2D variant: fix Lamarckian num_offspring=1, Darwinian pop=2 (already set)
    if force_1l2d_single_offspring and lamarckian_vectors == 1 and darwinian_population == 2:
        lam_levers["num_offspring"] = 1

    # Lamarckian run
    counted_lam = CountedFunction(topology_func)
    lam_gen = pure_lamarckian_function(
        besoin_topology_function=counted_lam,
        parent1_start=parent1_start,
        parent1_end=parent1_end,
        parent2_start=parent2_start,
        parent2_end=parent2_end,
        num_generations=100000,
        seed=seed + lam_run_seed_offset,  # fixed seed for reproducible evolution (offspring placement, etc.)
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

    # Darwinian run: same seed for reproducibility of run (selection/mutation)
    counted_dar = CountedFunction(topology_func)
    initial_pop = initial_points.copy()  # shape (n_points, 3)
    dar_gen = pure_darwinian_function(
        fitness_topology_function=counted_dar,
        num_generations=100000,
        seed=seed + dar_run_seed_offset,  # fixed seed for run reproducibility
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

    n_init = initial_points.shape[0]
    out = {
        "seed": seed,
        "lamarckian_vectors": lamarckian_vectors,
        "darwinian_population": darwinian_population,
        "initial_points": initial_points,
        "initial_distribution": [initial_points[i, :2].tolist() for i in range(n_init)],
        "lam_levers": lam_levers,
        "lam_initial_magnitudes": {
            "parent1": lam_initial_magnitude_1,
            "parent2": lam_initial_magnitude_2,
        },
        "dar_levers": dar_levers,
        "lam_generations": lam_gen,
        "dar_generations": dar_gen,
        "lam_summary": lam_sum,
        "dar_summary": dar_sum,
        "final_distribution_lam": final_distribution_lam.tolist() if final_distribution_lam is not None else None,
        "final_distribution_dar": final_dar.tolist() if final_dar is not None else None,
    }
    if verbose:
        print(f"UP1 seed={seed} | Lam: best_f={lam_sum['best_fitness']:.4f} dist={lam_sum['distance_to_optimum']:.3f} calls={lam_sum['function_calls']} | "
              f"Dar: best_f={dar_sum['best_fitness']:.4f} dist={dar_sum['distance_to_optimum']:.3f} calls={dar_sum['function_calls']}")
    return out



def generate_initial_points(seed, n_points=4):
    """Return n_points (x, y, 0) in initial_bounds (random). Uses UP1_INITIAL_BOUNDS when called from run_up1."""
    np.random.seed(seed)
    x_min, x_max, y_min, y_max = UP1_INITIAL_BOUNDS
    xs = np.random.uniform(x_min, x_max, n_points)
    ys = np.random.uniform(y_min, y_max, n_points)
    points = np.column_stack([xs, ys, np.zeros(n_points)])
    return points


def get_initial_points(seed, n_points=4, use_fixed_4=True):
    """
    Return n_points for experiments. For n_points==4 and use_fixed_4=True, return
    FIXED_INITIAL_POINTS_4 (both starts at (-12,-12), ends at (-11,-12) and (-12,-11)).
    Otherwise return random points from generate_initial_points(seed, n_points).
    """
    if n_points == 4 and use_fixed_4:
        return FIXED_INITIAL_POINTS_4.copy()
    return generate_initial_points(seed, n_points)


def generate_initial_four_points(seed):
    """Return 4 points (fixed configuration; use get_initial_points(seed, 4) for explicit control)."""
    return get_initial_points(seed, 4)


def lamarckian_parents_from_points(points, num_vectors=2):
    """
    Map points to (parent1_start, parent1_end, parent2_start, parent2_end).
    - num_vectors=2: 4 points → p1,p2 and p3,p4 (two vectors).
    - num_vectors=1: 2 points → one vector p1→p2 used as both parents (degenerate quadrilateral).
    """
    if num_vectors == 1:
        p1, p2 = points[0], points[1]
        return p1.copy(), p2.copy(), p1.copy(), p2.copy()
    p1, p2, p3, p4 = points[0], points[1], points[2], points[3]
    return p1.copy(), p2.copy(), p3.copy(), p4.copy()


def lamarckian_parents_from_four_points(points):
    """Map 4 points to (parent1_start, parent1_end, parent2_start, parent2_end)."""
    return lamarckian_parents_from_points(points, num_vectors=2)


def _rescale_parent_vectors_by_magnitude(
    parent1_start: np.ndarray,
    parent1_end: np.ndarray,
    parent2_start: np.ndarray,
    parent2_end: np.ndarray,
    magnitude_1: float,
    magnitude_2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rescale parent vector lengths while preserving each vector direction."""
    p1_start = parent1_start.copy()
    p2_start = parent2_start.copy()
    d1 = (parent1_end - parent1_start)[:2]
    d2 = (parent2_end - parent2_start)[:2]
    len1 = np.linalg.norm(d1)
    len2 = np.linalg.norm(d2)
    unit1 = d1 / len1 if len1 >= 1e-12 else np.array([1.0, 0.0])
    unit2 = d2 / len2 if len2 >= 1e-12 else np.array([1.0, 0.0])
    new_p1_end = np.array([p1_start[0] + unit1[0] * magnitude_1, p1_start[1] + unit1[1] * magnitude_1, 0.0])
    new_p2_end = np.array([p2_start[0] + unit2[0] * magnitude_2, p2_start[1] + unit2[1] * magnitude_2, 0.0])
    return p1_start, new_p1_end, p2_start, new_p2_end


def random_lamarckian_levers(
    seed,
    besoin_weight_range=(0.2, 2.0),
    topology_gradient_scale_range=(0.02, 0.3),
    magnitude_std_fraction_range=(0.0, 0.4),
    magnitude_weight_range=(0.0, 1.0),
    direction_std_range=(0.0, 0.5),
    min_magnitude_range=(0.005, 0.05),
    num_offspring_range=(2, 4),
    random_besoin_int_range=(0, 2),
):
    """Generate random Lamarckian levers from seed with configurable ranges."""
    rng = np.random.default_rng(seed)
    offspring_min, offspring_max = num_offspring_range
    rb_min, rb_max = random_besoin_int_range
    return {
        "besoin_weight": float(rng.uniform(*besoin_weight_range)),
        "topology_gradient_scale": float(rng.uniform(*topology_gradient_scale_range)),
        "magnitude_std_fraction": float(rng.uniform(*magnitude_std_fraction_range)),
        "magnitude_weight": float(rng.uniform(*magnitude_weight_range)),
        "direction_std": float(rng.uniform(*direction_std_range)),
        "min_magnitude": float(rng.uniform(*min_magnitude_range)),
        "num_offspring": int(rng.integers(int(offspring_min), int(offspring_max) + 1)),
        "first_generation_random_besoin": bool(rng.integers(int(rb_min), int(rb_max))),
    }


def random_darwinian_levers(
    seed,
    population_size=4,
    elimination_rate_range=(0.0, 0.5),
    selection_pressure_range=(1.0, 8.0),
    mutation_std_range=(0.2, 1.5),
    pop2_elimination_rate=0.5,
):
    """Generate random Darwinian levers from seed. population_size matches initial points (e.g. 4 or 2)."""
    rng = np.random.default_rng(seed + 1)  # different stream from Lamarckian
    # elimination_rate: leave at least 1 survivor (N*(1-er) >= 1 => er <= (N-1)/N).
    # For N=2 (asexual 1L2D): use 0.5 so one lives, one dies; selection_pressure then favors fitter.
    if population_size == 2:
        elimination_rate = float(pop2_elimination_rate)  # round(2*(1-0.5))=1 survivor
    else:
        max_er = (population_size - 1) / population_size
        er_min, er_max = elimination_rate_range
        elimination_rate = float(rng.uniform(er_min, min(er_max, max_er)))
    return {
        "population_size": population_size,
        "elimination_rate": elimination_rate,
        "selection_pressure": float(rng.uniform(*selection_pressure_range)),
        "mutation_std": float(rng.uniform(*mutation_std_range)),
    }


def write_up1_md(result, path=None):
    """Write Test 1 UP1 markdown document (parameter table, levers, initial/final distribution, results)."""
    if path is None:
        path = UP1_RESULTS_DIR / "Test_1_UP1.md"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = result["seed"]
    lam = result["lam_levers"]
    dar = result["dar_levers"]
    lam_sum = result["lam_summary"] or {}
    dar_sum = result["dar_summary"] or {}
    init_dist = result["initial_distribution"]
    final_lam = result["final_distribution_lam"]
    final_dar = result["final_distribution_dar"]
    n_init = len(init_dist)
    lam_vecs = result.get("lamarckian_vectors", 2)
    darwinian_population = result.get("darwinian_population", 4)

    md = []
    md.append("# Test 1 Unoptimized Pure Lamarckian v Darwinian (UP1)")
    if n_init == 2:
        md.append("")
        md.append("*(Variant: 1 Lamarckian vector, 2 Darwinian organisms — 2 initial points.)*")
    md.append("")
    md.append("## Setup")
    md.append("")
    md.append(f"- **Initial conditions:** {n_init} points (fixed or random in \\(x,y \\in [-12, 12]\\)) (same for both algorithms).")
    if n_init == 4:
        md.append("- **Lamarckian:** these 4 points define 2 parent vectors: \\(p_1\\to p_2\\), \\(p_3\\to p_4\\).")
        md.append("- **Darwinian:** the same 4 points are the 4 initial organism points (population size 4).")
    else:
        md.append("- **Lamarckian:** 1 parent vector \\(p_1\\to p_2\\) (same vector used as both parents).")
        md.append("- **Darwinian:** the same 2 points are the 2 initial organism points (population size 2).")
    md.append("- **Termination:** 2000 topology (fitness) evaluations per run (`max_calls=2000`).")
    md.append("- **Topology:** Rastrigin (minimization; global minimum at \\((0,0)\\)).")
    md.append("- **Levers:** All algorithm levers are drawn at random from the experiment seed (no optimized/tuned values).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Parameters: Lever or Not")
    md.append("")
    md.append("Every parameter is listed. **Lever** = tuned in later optimization; **Not a lever** = fixed by experiment design or excluded for the stated reason.")
    md.append("")
    md.append("### Lamarckian")
    md.append("")
    md.append("| Parameter | Lever? | Reason |")
    md.append("|-----------|--------|--------|")
    md.append("| `besoin_topology_function` | No | Fixed: Rastrigin (shared topology). |")
    md.append(f"| `parent1_start`, `parent1_end`, `parent2_start`, `parent2_end` | No | Fixed by UP1: from shared {n_init} random points. |")
    md.append("| `num_offspring` | **Yes** | Controls offspring per generation; affects diversity and call usage. |")
    md.append("| `num_generations` | No | Overridden by `max_calls`; effectively infinite. |")
    md.append("| `besoin_weight` | **Yes** | Weight of gradient-based besoin vs parent displacement. |")
    md.append("| `topology_gradient_scale` | **Yes** | Scale of gradient-based besoin magnitude. |")
    md.append("| `magnitude_std_fraction` | **Yes** | Random variation in offspring magnitude. |")
    md.append("| `magnitude_weight` | **Yes** | Blend of parent mean magnitude vs vector-average magnitude. |")
    md.append("| `direction_std` | **Yes** | Random variation in offspring direction. |")
    md.append("| `min_magnitude` | **Yes** | Lower bound on displacement magnitude. |")
    md.append("| `seed` | No | Set for reproducibility; not tuned. |")
    md.append("| `initial_bounds` | No | Fixed: \\([-12,12]^2\\) for UP1 (matches viz axis range). |")
    md.append("| `first_generation_random_besoin` | **Yes** | Whether gen 0 uses random besoin instead of gradient. |")
    md.append("| `max_calls` | No | Fixed: 2000 (experiment budget). |")
    md.append("")
    md.append("### Darwinian")
    md.append("")
    md.append("| Parameter | Lever? | Reason |")
    md.append("|-----------|--------|--------|")
    md.append("| `fitness_topology_function` | No | Fixed: Rastrigin (shared topology). |")
    md.append(f"| `population_size` | No | Fixed to {darwinian_population} in this run to match initial points. |")
    md.append("| `num_generations` | No | Overridden by `max_calls`. |")
    md.append("| `elimination_rate` | **Yes** | Fraction eliminated each generation; survivor count = max(2, round(N×(1−er))). |")
    md.append("| `selection_pressure` | **Yes** | Strength of preference for fitter survivors. |")
    md.append("| `mutation_std` | **Yes** | Std of Gaussian mutation for offspring. |")
    md.append("| `seed` | No | Set for reproducibility. |")
    md.append("| `initial_bounds` | No | Fixed: \\([-12,12]^2\\). |")
    md.append("| `max_calls` | No | Fixed: 2000. |")
    md.append(f"| `initial_population` | No | Fixed by UP1: same {n_init} points as Lamarckian. |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Recorded Levers (This Run)")
    md.append("")
    md.append("**Seed:** " + str(seed))
    md.append("")
    md.append("### Lamarckian")
    md.append("")
    for k, v in lam.items():
        md.append(f"- `{k}`: {repr(v)}")
    md.append("")
    md.append("### Darwinian")
    md.append("")
    for k, v in dar.items():
        md.append(f"- `{k}`: {repr(v)}")
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"## Initial Distribution ({n_init} points)")
    md.append("")
    md.append("| Point | x | y | Role (Lamarckian) | Role (Darwinian) |")
    md.append("|-------|------|------|--------------------|------------------|")
    if n_init == 4:
        labels_lam = ["parent1_start", "parent1_end", "parent2_start", "parent2_end"]
    else:
        labels_lam = ["parent1_start (vector 1)", "parent1_end (vector 1)"]
    for i, (pt, role) in enumerate(zip(init_dist, labels_lam)):
        x, y = pt[0], pt[1]
        md.append(f"| \\(p_{i+1}\\) | {x:.6g} | {y:.6g} | {role} | organism {i+1} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Final Distribution")
    md.append("")
    md.append("### Lamarckian (endpoints of last generation)")
    md.append("")
    if final_lam:
        md.append("| Endpoint | x | y |")
        md.append("|----------|------|------|")
        for i, pt in enumerate(final_lam):
            x, y = pt[0], pt[1]
            md.append(f"| {i+1} | {x:.6g} | {y:.6g} |")
    else:
        md.append("(No organisms in last generation.)")
    md.append("")
    md.append("### Darwinian (population of last generation)")
    md.append("")
    if final_dar:
        md.append("| Organism | x | y |")
        md.append("|----------|------|------|")
        for i, pt in enumerate(final_dar):
            x, y = pt[0], pt[1]
            md.append(f"| {i+1} | {x:.6g} | {y:.6g} |")
    else:
        md.append("(No organisms.)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Results (Summary)")
    md.append("")
    md.append("| Metric | Lamarckian | Darwinian |")
    md.append("|--------|------------|-----------|")
    md.append(f"| Best fitness (lower is better) | {lam_sum.get('best_fitness', float('nan')):.4f} | {dar_sum.get('best_fitness', float('nan')):.4f} |")
    md.append(f"| Mean fitness | {lam_sum.get('mean_fitness', float('nan')):.4f} | {dar_sum.get('mean_fitness', float('nan')):.4f} |")
    md.append(f"| Distance of mean to (0,0) | {lam_sum.get('distance_to_optimum', float('nan')):.3f} | {dar_sum.get('distance_to_optimum', float('nan')):.3f} |")
    md.append(f"| Topology calls | {lam_sum.get('function_calls', '—')} | {dar_sum.get('function_calls', '—')} |")
    md.append("")
    md.append("### Interpretation")
    md.append("")
    md.append(f"UP1 uses **unoptimized** levers (random from seed) and a shared {n_init}-point initial condition. ")
    md.append("Results are baseline only; levers will be optimized in later experiments. ")
    md.append("Same call budget (2000) allows direct comparison of best/mean fitness and distance to optimum.")
    md.append("")
    path.write_text("\n".join(md), encoding="utf-8")
    return path


# Default 10 seeds for multi-seed runs (diverse spread)
DEFAULT_10_SEEDS = [7, 27, 107, 207, 327, 507, 42, 123, 456, 789]


def run_up1_multi(seeds, call_budget=UP1_CALL_BUDGET, verbose=True, lamarckian_vectors=2, darwinian_population=4):
    """Run UP1 for each seed; return list of result dicts and print summary table (seed order)."""
    seeds = sorted(seeds)
    results = []
    for seed in seeds:
        r = run_up1(
            seed=seed,
            call_budget=call_budget,
            verbose=False,
            lamarckian_vectors=lamarckian_vectors,
            darwinian_population=darwinian_population,
        )
        results.append(r)
        if verbose:
            lam = r["lam_summary"] or {}
            dar = r["dar_summary"] or {}
            print(f"  seed={seed}: Lam best_f={lam.get('best_fitness', float('nan')):.4f} dist={lam.get('distance_to_optimum', float('nan')):.3f} | "
                  f"Dar best_f={dar.get('best_fitness', float('nan')):.4f} dist={dar.get('distance_to_optimum', float('nan')):.3f}")
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
                lam_wins += 1  # Lam has better (lower) best_fitness
            print(f"| {seed} | {lb:.4f} | {db:.4f} | {ld:.3f} | {dd:.3f} | {win} |")
        print("| (Better best_f = lower value) |")
        print(f"Lamarckian better best_fitness: {lam_wins}/{len(seeds)} seeds; Darwinian: {len(seeds) - lam_wins}/{len(seeds)}.")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Test 1 UP1: Unoptimized Pure Lamarckian v Darwinian")
    p.add_argument("--seed", type=int, default=None, help="Single seed (ignored if --seeds or --num-seeds set)")
    p.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 7,27,107")
    p.add_argument("--num-seeds", type=int, default=None, help="Run this many seeds (default set: 7,27,107,207,327,507,42,123,456,789)")
    p.add_argument("--calls", type=int, default=UP1_CALL_BUDGET)
    p.add_argument("--lamarckian-vectors", type=int, default=2, help="Number of Lamarckian parent vectors (1 or 2). 1 => 2 initial points.")
    p.add_argument("--darwinian-pop", dest="darwinian_population", type=int, default=4, help="Darwinian population size (2 or 4). Must match initial points: use 2 with --lamarckian-vectors 1.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--write-md", action="store_true", help="Write Test_1_UP1.md (or Test_1_UP1_1L2D.md for 1 vector / 2 pop)")
    args = p.parse_args()

    if args.num_seeds is not None:
        seeds = DEFAULT_10_SEEDS[: args.num_seeds] if args.num_seeds <= len(DEFAULT_10_SEEDS) else (DEFAULT_10_SEEDS + list(range(len(DEFAULT_10_SEEDS), args.num_seeds)))
    elif args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed if args.seed is not None else 42]

    lam_vecs = getattr(args, "lamarckian_vectors", 2)
    darwinian_population = getattr(args, "darwinian_population", 4)
    out_basename = "Test_1_UP1_1L2D.md" if (lam_vecs == 1 and darwinian_population == 2) else "Test_1_UP1.md"

    if len(seeds) == 1:
        result = run_up1(
            seed=seeds[0],
            call_budget=args.calls,
            verbose=not args.quiet,
            lamarckian_vectors=lam_vecs,
            darwinian_population=darwinian_population,
        )
        if args.write_md:
            write_up1_md(result, path=UP1_RESULTS_DIR / out_basename)
            print("Wrote", UP1_RESULTS_DIR / out_basename)
    else:
        results = run_up1_multi(
            seeds,
            call_budget=args.calls,
            verbose=not args.quiet,
            lamarckian_vectors=lam_vecs,
            darwinian_population=darwinian_population,
        )
        if args.write_md:
            md_path = UP1_RESULTS_DIR / out_basename
            UP1_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            write_up1_md(results[0], path=md_path)
            # Append multi-seed summary to MD (table in seed order)
            results_sorted = sorted(results, key=lambda r: r["seed"])
            extra = []
            extra.append("")
            extra.append("---")
            extra.append("")
            extra.append(f"## Multi-seed runs ({len(seeds)} seeds, seed order)")
            extra.append("")
            extra.append("| Seed | Lam best_f | Dar best_f | Lam dist | Dar dist | Better best_f |")
            extra.append("|------|------------|------------|----------|----------|----------------|")
            for r in results_sorted:
                lam = r["lam_summary"] or {}
                dar = r["dar_summary"] or {}
                lb, db = lam.get("best_fitness", float("nan")), dar.get("best_fitness", float("nan"))
                ld, dd = lam.get("distance_to_optimum", float("nan")), dar.get("distance_to_optimum", float("nan"))
                better = "Darwinian" if db < lb else "Lamarckian"
                extra.append(f"| {r['seed']} | {lb:.4f} | {db:.4f} | {ld:.3f} | {dd:.3f} | {better} |")
            extra.append("")
            with open(md_path, "a", encoding="utf-8") as f:
                f.write("\n".join(extra))
            print("Wrote", md_path, "(single-seed detail + multi-seed table).")
