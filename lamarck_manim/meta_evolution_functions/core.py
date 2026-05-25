"""
Meta evolution orchestration utilities.

This module provides a lightweight bridge between Lamarckian and Darwinian
evolution functions so meta-evolution logic can tune their levers.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any
import numpy as np

from lamarckian_functions import pure_lamarckian_function
from darwinian_functions import pure_darwinian_function


class CountedFunction:
    """
    Wraps a callable and counts how many times it is called.
    Use this to measure topology/fitness evaluations during evolution.

    Example:
        from meta_evolution_functions import CountedFunction, run_darwinian_with_levers
        from darwinian_functions.core import rastrigin_func

        counted = CountedFunction(rastrigin_func)
        result = run_darwinian_with_levers(counted, levers, seed=7)
        print("Topology evaluations:", counted.n_calls)
    """

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        self.n_calls: int = 0

    def __call__(self, *args, **kwargs) -> Any:
        self.n_calls += 1
        return self.func(*args, **kwargs)

    def reset(self) -> None:
        """Reset the call counter to zero."""
        self.n_calls = 0


@dataclass(frozen=True)
class LamarckianLevers:
    """Tunable lever set for pure_lamarckian_function."""

    besoin_weight: float = 1.0
    topology_gradient_scale: float = 0.1
    magnitude_std_fraction: float = 0.1
    magnitude_weight: float = 1.0
    direction_std: float = 0.1
    min_magnitude: float = 0.01
    num_offspring: int = 2
    num_generations: int = 10


@dataclass(frozen=True)
class DarwinianLevers:
    """Tunable lever set for pure_darwinian_function."""

    elimination_rate: float = 0.5
    selection_pressure: float = 4.0
    mutation_std: float = 0.8
    population_size: int = 32
    num_generations: int = 10


@dataclass(frozen=True)
class MetaCandidate:
    """
    One candidate setting in meta-evolution search.

    A candidate can tune Lamarckian, Darwinian, or both.
    """

    lamarckian: Optional[LamarckianLevers] = None
    darwinian: Optional[DarwinianLevers] = None
    seed: Optional[int] = None


def run_lamarckian_with_levers(
    topology_function: Callable[[float, float], Any],
    levers: LamarckianLevers,
    seed: Optional[int] = None,
    parent1_start: Optional[np.ndarray] = None,
    parent1_end: Optional[np.ndarray] = None,
    parent2_start: Optional[np.ndarray] = None,
    parent2_end: Optional[np.ndarray] = None,
    initial_bounds: tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0),
):
    """Run Lamarckian evolution with a lever bundle."""
    return pure_lamarckian_function(
        besoin_topology_function=topology_function,
        parent1_start=parent1_start,
        parent1_end=parent1_end,
        parent2_start=parent2_start,
        parent2_end=parent2_end,
        num_offspring=levers.num_offspring,
        num_generations=levers.num_generations,
        besoin_weight=levers.besoin_weight,
        topology_gradient_scale=levers.topology_gradient_scale,
        magnitude_std_fraction=levers.magnitude_std_fraction,
        magnitude_weight=levers.magnitude_weight,
        direction_std=levers.direction_std,
        min_magnitude=levers.min_magnitude,
        seed=seed,
        initial_bounds=initial_bounds,
    )


def run_darwinian_with_levers(
    topology_function: Callable[[float, float], Any],
    levers: DarwinianLevers,
    seed: Optional[int] = None,
    initial_bounds: tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0),
):
    """Run Darwinian evolution with a lever bundle."""
    return pure_darwinian_function(
        fitness_topology_function=topology_function,
        population_size=levers.population_size,
        num_generations=levers.num_generations,
        elimination_rate=levers.elimination_rate,
        selection_pressure=levers.selection_pressure,
        mutation_std=levers.mutation_std,
        seed=seed,
        initial_bounds=initial_bounds,
    )


def summarize_final_state(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Produce minimal numeric summary from combined evolution results.
    Meta-optimization judges by the mean of the final generation's organisms
    (Lamarckian: mean of endpoints; Darwinian: mean of all organism points).

    Expected keys:
    - 'lamarckian': return value from pure_lamarckian_function (optional)
    - 'darwinian': return value from pure_darwinian_function (optional)
    """
    summary: Dict[str, float] = {}

    lamarckian = results.get("lamarckian")
    if lamarckian:
        last_generation = lamarckian[-1]
        organisms = last_generation.get("organisms", [])
        if organisms:
            # Mean of all final organisms' endpoints
            endpoints = np.array([end for _, end in organisms], dtype=float)
            endpoint_mean = np.mean(endpoints[:, :2], axis=0)
            summary["lamarckian_mean_x"] = float(endpoint_mean[0])
            summary["lamarckian_mean_y"] = float(endpoint_mean[1])
            summary["lamarckian_count"] = float(len(organisms))

    darwinian = results.get("darwinian")
    if darwinian:
        last_generation = darwinian[-1]
        # Mean of all final organisms (full population), not just survivors
        organisms = np.array(last_generation.get("organisms", []), dtype=float)
        if len(organisms) > 0:
            organism_mean = np.mean(organisms[:, :2], axis=0)
            summary["darwinian_mean_x"] = float(organism_mean[0])
            summary["darwinian_mean_y"] = float(organism_mean[1])
            summary["darwinian_count"] = float(len(organisms))
            summary["darwinian_survivor_count"] = float(len(last_generation.get("survivors", [])))

    return summary

