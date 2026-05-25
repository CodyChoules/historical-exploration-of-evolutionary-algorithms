#!/usr/bin/env python3
"""
Quick Lamarckian-only meta-tuning smoke test.

Samples random Lamarckian lever candidates, while keeping seed,
num_generations, and num_offspring fixed across all candidates.
Judges by the mean of the final organisms (mean of endpoints);
scores by distance of that mean to the known optimum (0, 0).
Prints the top results.

Usage:
    python meta_evolution_functions/run_meta_tuning_test.py
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np

# Ensure project root is importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meta_evolution_functions import (
    LamarckianLevers,
    MetaCandidate,
    run_lamarckian_with_levers,
    summarize_final_state,
)
from lamarckian_functions.core import rastrigin_func

FIXED_SEED = 7
FIXED_NUM_OFFSPRING = 2
FIXED_NUM_GENERATIONS = 1000
FIXED_PARENT1_START = np.array([-6.0, -5.5, 0.0])
FIXED_PARENT1_END = np.array([-5.0, -4.2, 0.0])
FIXED_PARENT2_START = np.array([5.8, -5.2, 0.0])
FIXED_PARENT2_END = np.array([6.9, -4.0, 0.0])
POPULATION_SIZE = 24
NUM_META_GENERATIONS = 6
ELITE_COUNT = 6
TOP_K = 5

LEVER_BOUNDS = {
    "besoin_weight": (0.2, 2.0),
    "topology_gradient_scale": (0.02, 0.25),
    "magnitude_std_fraction": (0.0, 0.5),
    "magnitude_weight": (0.0, 1.0),
    "direction_std": (0.0, 1.0),
}

MUTATION_STD = {
    "besoin_weight": 0.18,
    "topology_gradient_scale": 0.02,
    "magnitude_std_fraction": 0.05,
    "magnitude_weight": 0.08,
    "direction_std": 0.08,
}


def distance_to_origin(x: float, y: float) -> float:
    """Euclidean distance to the global optimum at (0, 0)."""
    return math.sqrt(x * x + y * y)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value to [low, high]."""
    return max(low, min(high, value))


def sample_candidate(rng: random.Random) -> MetaCandidate:
    """Generate one random Lamarckian candidate (with fixed seed/counts)."""
    l = LamarckianLevers(
        besoin_weight=rng.uniform(0.2, 2.0),
        topology_gradient_scale=rng.uniform(0.02, 0.25),
        magnitude_std_fraction=rng.uniform(0.0, 0.5),
        magnitude_weight=rng.uniform(0.0, 1.0),
        direction_std=rng.uniform(0.0, 1.0),
        min_magnitude=0.01,
        num_offspring=FIXED_NUM_OFFSPRING,
        num_generations=FIXED_NUM_GENERATIONS,
    )
    return MetaCandidate(lamarckian=l, darwinian=None, seed=FIXED_SEED)


def mutate_from_elite(elite: MetaCandidate, rng: random.Random) -> MetaCandidate:
    """Create one mutated candidate from an elite parent."""
    parent = elite.lamarckian
    if parent is None:
        return sample_candidate(rng)

    mutated = {}
    for name, (low, high) in LEVER_BOUNDS.items():
        parent_value = getattr(parent, name)
        std = MUTATION_STD[name]
        child_value = clamp(rng.gauss(parent_value, std), low, high)
        mutated[name] = child_value

    l = LamarckianLevers(
        besoin_weight=mutated["besoin_weight"],
        topology_gradient_scale=mutated["topology_gradient_scale"],
        magnitude_std_fraction=mutated["magnitude_std_fraction"],
        magnitude_weight=mutated["magnitude_weight"],
        direction_std=mutated["direction_std"],
        min_magnitude=0.01,
        num_offspring=FIXED_NUM_OFFSPRING,
        num_generations=FIXED_NUM_GENERATIONS,
    )
    return MetaCandidate(lamarckian=l, darwinian=None, seed=FIXED_SEED)


def evaluate_candidate(candidate: MetaCandidate) -> dict:
    """Run Lamarckian mode for one candidate and compute score."""
    results = {}
    if candidate.lamarckian is not None:
        results["lamarckian"] = run_lamarckian_with_levers(
            topology_function=rastrigin_func,
            levers=candidate.lamarckian,
            seed=candidate.seed,
            parent1_start=FIXED_PARENT1_START,
            parent1_end=FIXED_PARENT1_END,
            parent2_start=FIXED_PARENT2_START,
            parent2_end=FIXED_PARENT2_END,
        )

    summary = summarize_final_state(results)

    lam_dist = None
    if "lamarckian_mean_x" in summary and "lamarckian_mean_y" in summary:
        lam_dist = distance_to_origin(summary["lamarckian_mean_x"], summary["lamarckian_mean_y"])
    score = float(lam_dist) if lam_dist is not None else float("inf")

    return {
        "score": score,
        "lam_dist": lam_dist,
        "summary": summary,
        "candidate": candidate,
    }


def main() -> None:
    rng = random.Random(20260217)

    print(
        "Lamarckian-only tuning with fixed settings: "
        f"seed={FIXED_SEED}, num_offspring={FIXED_NUM_OFFSPRING}, num_generations={FIXED_NUM_GENERATIONS}"
    )
    print(
        "fixed parents: "
        f"p1s=({FIXED_PARENT1_START[0]:.1f},{FIXED_PARENT1_START[1]:.1f}) "
        f"p1e=({FIXED_PARENT1_END[0]:.1f},{FIXED_PARENT1_END[1]:.1f}) "
        f"p2s=({FIXED_PARENT2_START[0]:.1f},{FIXED_PARENT2_START[1]:.1f}) "
        f"p2e=({FIXED_PARENT2_END[0]:.1f},{FIXED_PARENT2_END[1]:.1f})"
    )

    # Initial random population
    population = [sample_candidate(rng) for _ in range(POPULATION_SIZE)]

    for generation_idx in range(NUM_META_GENERATIONS):
        evaluations = [evaluate_candidate(c) for c in population]
        ranked = sorted(evaluations, key=lambda e: e["score"])
        best = ranked[0]
        avg_score = sum(item["score"] for item in ranked) / len(ranked)
        print(
            f"meta_gen={generation_idx} best={best['score']:.4f} "
            f"avg={avg_score:.4f} lam_mean=({best['summary'].get('lamarckian_mean_x', float('nan')):.3f},"
            f" {best['summary'].get('lamarckian_mean_y', float('nan')):.3f})"
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
        lam_txt = f"{item['lam_dist']:.4f}" if item["lam_dist"] is not None else "n/a"
        print("-" * 80)
        print(f"#{idx} score={item['score']:.4f} lam_dist={lam_txt}")
        print(f"seed={candidate.seed}")
        print(f"Lamarckian levers: {asdict(candidate.lamarckian) if candidate.lamarckian else None}")
        print(f"summary: {item['summary']}")


if __name__ == "__main__":
    main()

