"""
Lamarckian evolution functions package.

Provides Lamarckian evolution processes: organism vector generation,
spawn region calculation, gradient-based besoin vectors, and visualization helpers.
"""

from .core import (
    pure_lamarckian_function,
    pure_lamarckian_function_sampling,
    calculate_besoin_by_sampling,
    generate_organism_vectors,
    generate_and_visualize_child_organisms,
    calculate_spawn_quadrilateral,
    calculate_gradient,
    mean_displacement_vector,
    mean_magnitude,
    bound_to_displacement_vector,
    random_point_in_quadrilateral,
    remove_mobjects,
    rastrigin_func,
    TestOrganismFunctions,
    TestPureLamarckianFunction,
)

__all__ = [
    "pure_lamarckian_function",
    "pure_lamarckian_function_sampling",
    "calculate_besoin_by_sampling",
    "generate_organism_vectors",
    "generate_and_visualize_child_organisms",
    "calculate_spawn_quadrilateral",
    "calculate_gradient",
    "mean_displacement_vector",
    "mean_magnitude",
    "bound_to_displacement_vector",
    "random_point_in_quadrilateral",
    "remove_mobjects",
    "rastrigin_func",
    "TestOrganismFunctions",
    "TestPureLamarckianFunction",
]
