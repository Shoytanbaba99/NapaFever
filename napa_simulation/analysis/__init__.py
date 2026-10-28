"""Statistical hypothesis testing and inferential analytics module."""

from napa_simulation.analysis.hypothesis_testing import (
    HypothesisTestResult,
    run_experiment_1_acute_vs_staggered,
    run_experiment_2_malnutrition,
    run_statistical_test_suite,
    calculate_cohens_d,
)

__all__ = [
    "HypothesisTestResult",
    "run_experiment_1_acute_vs_staggered",
    "run_experiment_2_malnutrition",
    "run_statistical_test_suite",
    "calculate_cohens_d",
]
