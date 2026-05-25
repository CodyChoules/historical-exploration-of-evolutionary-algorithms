#!/usr/bin/env python3
"""
Quick Darwinian-only meta-tuning smoke test.

Samples random Darwinian lever candidates, while keeping seed and
num_generations fixed across all candidates. Initial population is
drawn from the same initial_bounds; with fixed seed, all candidates
see the same starting points. Judges by the mean of the final organisms
(full last generation); scores by distance of that mean to the known
optimum (0, 0). Prints the top results.

Usage:
    python meta_evolution_functions/darwin_tunes_darwin/run_meta_tuning_test.py
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from pathlib import Path
import sys

# Ensure project root is importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimizationfunctions.evolutionalgorithms.metaevolutionfunctions import (
    DarwinianLevers,
    MetaCandidate,
    run_darwinian_with_levers,
    summarize_final_state,
)
from optimizationfunctions.evolutionalgorithms.darwinianfunctions.core import rastrigin_func

FIXED_SEED = 7
FIXED_NUM_GENERATIONS = 1000
FIXED_POPULATION_SIZE = 4
# Same initial_bounds for all candidates => with fixed seed, identical initial population.
INITIAL_BOUNDS = (-10.0, 10.0, -10.0, 10.0)

POPULATION_SIZE = 24
NUM_META_GENERATIONS = 6
ELITE_COUNT = 6
TOP_K = 5

LEVER_BOUNDS = {
    "elimination_rate": (0.1, 0.9),
    "selection_pressure": (0.5, 15.0),
    "mutation_std": (0.1, 2.0),
}

MUTATION_STD = {
    "elimination_rate": 0.08,
    "selection_pressure": 1.2,
    "mutation_std": 0.15,
}


def distance_to_origin(x: float, y: float) -> float:
    """Euclidean distance to the global optimum at (0, 0)."""
    return math.sqrt(x * x + y * y)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value to [low, high]."""
    return max(low, min(high, value))


def sample_candidate(rng: random.Random) -> MetaCandidate:
    """Generate one random Darwinian candidate (with fixed seed, num_generations, population_size)."""
    d = DarwinianLevers(
        elimination_rate=rng.uniform(*LEVER_BOUNDS["elimination_rate"]),
        selection_pressure=rng.uniform(*LEVER_BOUNDS["selection_pressure"]),
        mutation_std=rng.uniform(*LEVER_BOUNDS["mutation_std"]),
        population_size=FIXED_POPULATION_SIZE,
        num_generations=FIXED_NUM_GENERATIONS,
    )
    return MetaCandidate(lamarckian=None, darwinian=d, seed=FIXED_SEED)


def mutate_from_elite(elite: MetaCandidate, rng: random.Random) -> MetaCandidate:
    """Create one mutated candidate from an elite parent."""
    parent = elite.darwinian
    if parent is None:
        return sample_candidate(rng)

    mutated = {}
    for name, (low, high) in LEVER_BOUNDS.items():
        parent_value = getattr(parent, name)
        std = MUTATION_STD[name]
        child_value = clamp(rng.gauss(parent_value, std), low, high)
        mutated[name] = child_value

    d = DarwinianLevers(
        elimination_rate=mutated["elimination_rate"],
        selection_pressure=mutated["selection_pressure"],
        mutation_std=mutated["mutation_std"],
        population_size=FIXED_POPULATION_SIZE,
        num_generations=FIXED_NUM_GENERATIONS,
    )
    return MetaCandidate(lamarckian=None, darwinian=d, seed=FIXED_SEED)


def evaluate_candidate(candidate: MetaCandidate) -> dict:
    """Run Darwinian evolution for one candidate and compute score."""
    results = {}
    if candidate.darwinian is not None:
        results["darwinian"] = run_darwinian_with_levers(
            topology_function=rastrigin_func,
            levers=candidate.darwinian,
            seed=candidate.seed,
            initial_bounds=INITIAL_BOUNDS,
        )

    summary = summarize_final_state(results)

    darw_dist = None
    if "darwinian_mean_x" in summary and "darwinian_mean_y" in summary:
        darw_dist = distance_to_origin(
            summary["darwinian_mean_x"], summary["darwinian_mean_y"]
        )
    score = float(darw_dist) if darw_dist is not None else float("inf")

    return {
        "score": score,
        "darw_dist": darw_dist,
        "summary": summary,
        "candidate": candidate,
    }


def main() -> None:
    rng = random.Random(20260217)

    print(
        "Darwinian-only tuning with fixed settings: "
        f"seed={FIXED_SEED}, num_generations={FIXED_NUM_GENERATIONS}, "
        f"population_size={FIXED_POPULATION_SIZE}, initial_bounds={INITIAL_BOUNDS}"
    )

    # Initial random population of lever settings
    population = [sample_candidate(rng) for _ in range(POPULATION_SIZE)]

    for generation_idx in range(NUM_META_GENERATIONS):
        evaluations = [evaluate_candidate(c) for c in population]
        ranked = sorted(evaluations, key=lambda e: e["score"])
        best = ranked[0]
        avg_score = sum(item["score"] for item in ranked) / len(ranked)
        print(
            f"meta_gen={generation_idx} best={best['score']:.4f} "
            f"avg={avg_score:.4f} darw_mean=("
            f"{best['summary'].get('darwinian_mean_x', float('nan')):.3f}, "
            f"{best['summary'].get('darwinian_mean_y', float('nan')):.3f})"
        )

        if generation_idx == NUM_META_GENERATIONS - 1:
            final_ranked = ranked
            break

        elites = [item["candidate"] for item in ranked[:ELITE_COUNT]]

        # Next generation = elites + mutated children from elites
        next_population = elites.copy()
        while len(next_population) < POPULATION_SIZE:
            parent = rng.choice(elites)
            next_population.append(mutate_from_elite(parent, rng))
        population = next_population

    print(f"Top {TOP_K} final candidates:")
    for idx, item in enumerate(final_ranked[:TOP_K], start=1):
        candidate = item["candidate"]
        darw_txt = f"{item['darw_dist']:.4f}" if item["darw_dist"] is not None else "n/a"
        print("-" * 80)
        print(f"#{idx} score={item['score']:.4f} darw_dist={darw_txt}")
        print(f"seed={candidate.seed}")
        print(
            f"Darwinian levers: {asdict(candidate.darwinian) if candidate.darwinian else None}"
        )
        print(f"summary: {item['summary']}")


if __name__ == "__main__":
    main()
