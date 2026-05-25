"""
Darwinian evolution functions package.

Provides Darwinian evolution processes and test helpers.
"""

from .core import (
    add_global_optima_markers,
    animate_darwinian_selection,
    pure_darwinian_function,
    rastrigin_func,
    DarwinianSelectionAnimation,
    TestPureDarwinianFunction,
)

__all__ = [
    "add_global_optima_markers",
    "animate_darwinian_selection",
    "pure_darwinian_function",
    "rastrigin_func",
    "DarwinianSelectionAnimation",
    "TestPureDarwinianFunction",
]
