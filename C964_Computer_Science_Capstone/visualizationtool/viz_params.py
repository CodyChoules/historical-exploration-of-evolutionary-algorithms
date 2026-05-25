"""
Shared visualization parameters for 1L2D (and other) experiment viz.
Scenes accept a params object to render any number of seeds with consistent settings.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class OneLTwoDVizParams:
    """Parameters for 1 Lamarckian vector / 2 Darwinian organisms viz (2 initial points per seed)."""

    seeds: List[int] = field(default_factory=lambda: [7, 27, 107, 207, 327, 507, 42, 123, 456, 789])
    max_calls: int = 300
    initial_bounds: Tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0)
    lamarckian_vectors: int = 1
    darwinian_pop: int = 2
    num_offspring: int = 1
    display_x_shift: float = -10.0
    display_y_shift: float = 15.0
    history_generation_stride: int = 1
    history_line_opacity: float = 0.5
    history_dot_opacity: float = 0.35
    final_darken: float = 0.5

    @classmethod
    def default_10_seeds(cls) -> "OneLTwoDVizParams":
        """Default 10 candidates (seeds) from the 1L2D experiment."""
        return cls(seeds=[7, 27, 107, 207, 327, 507, 42, 123, 456, 789])
