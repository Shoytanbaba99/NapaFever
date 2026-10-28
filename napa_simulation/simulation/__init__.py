"""Simulation engine and clinical scenario runner."""

from napa_simulation.simulation.engine import (
    SimulationResult,
    simulate_patient,
    simulate_cohort,
    classify_clinical_risk,
)
from napa_simulation.simulation.scenarios import (
    ClinicalScenario,
    get_acute_overdose_scenario,
    get_staggered_overdose_scenario,
    get_malnutrition_scenario,
    get_nac_timing_scenario,
    run_all_scenarios,
)

__all__ = [
    "SimulationResult",
    "simulate_patient",
    "simulate_cohort",
    "classify_clinical_risk",
    "ClinicalScenario",
    "get_acute_overdose_scenario",
    "get_staggered_overdose_scenario",
    "get_malnutrition_scenario",
    "get_nac_timing_scenario",
    "run_all_scenarios",
]
