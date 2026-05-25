"""
Comparative testing module.

Used to run and compare Lamarckian vs Darwinian evolution (or other configurations)
under shared conditions (topology, seeds, generations) for analysis and visualization.
"""

from .data_pipeline import (
    clean_dataset,
    featurize_dataset,
    load_dataset,
    normalize_fields,
    prepare_for_experiment,
    write_dataset,
)
from .evaluation import (
    evaluate_results,
    format_evaluation_report,
    reproducibility_check,
)

__all__ = [
    "load_dataset",
    "write_dataset",
    "clean_dataset",
    "featurize_dataset",
    "normalize_fields",
    "prepare_for_experiment",
    "evaluate_results",
    "reproducibility_check",
    "format_evaluation_report",
]
