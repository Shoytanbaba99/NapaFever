"""
Preconfigured Clinical Scenarios for Laboratory and Defense Demonstration.

Provides standardized, literature-grounded clinical scenarios:
1. Acute Overdose (10,000 mg single bolus)
2. Staggered Overdose (2,000 mg every 2.5 h x 5 intakes = 10,000 mg)
3. Nutritional Vulnerability (5,000 mg borderline dose in Normal vs Malnourished)
4. Therapeutic Antidote Window (12,000 mg overdose with Early NAC 8h vs Delayed NAC 16h vs Untreated)
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import pandas as pd

from napa_simulation.models.formulations import Formulation, FormulationType
from napa_simulation.models.pkpd import NACIntervention
from napa_simulation.simulation.engine import (
    SimulationResult,
    simulate_patient,
    simulate_cohort,
)


@dataclass
class ClinicalScenario:
    """Wrapper for clinical scenario definition and execution."""
    name: str
    description: str
    dose_mg: float
    formulation_type: FormulationType
    patient_type: str
    staggered_doses: List[Tuple[float, float, float]]
    nac_intervention: NACIntervention | None


def get_acute_overdose_scenario(dose_mg: float = 10000.0) -> Tuple[SimulationResult, pd.DataFrame]:
    """
    Scenario 1A: Acute Bolus Overdose (10,000 mg Napa at once).
    Standard presentation in emergency toxicology departments.
    """
    tablets = int(round(dose_mg / 500.0))
    formulation = Formulation.standard_napa(tablets=tablets)

    single_result = simulate_patient(
        formulation=formulation,
        initial_gsh=100.0,
    )
    cohort_df = simulate_cohort(
        n_patients=100,
        dose_mg=dose_mg,
        formulation_type=FormulationType.STANDARD,
        patient_type="normal",
        random_seed=101,
    )
    return single_result, cohort_df


def get_staggered_overdose_scenario(
    total_dose_mg: float = 10000.0,
    n_doses: int = 5,
    interval_hr: float = 2.5,
) -> Tuple[SimulationResult, pd.DataFrame]:
    """
    Scenario 1B: Staggered Ingestion Overdose.
    Total 10,000 mg ingested as 2,000 mg (4 tablets) every 2.5 hours x 5 doses.
    Frequently causes delayed clinical presentation and insidiously depleted GSH.
    """
    dose_per_intake = total_dose_mg / n_doses
    # Staggered list: (time_hr, imm_mg, sust_mg)
    # First dose at t=0 is absorbed via initial formulation loading
    events = [(i * interval_hr, dose_per_intake, 0.0) for i in range(1, n_doses)]

    # Initial loading formulation (first intake)
    initial_form = Formulation.standard_napa(tablets=int(round(dose_per_intake / 500.0)))

    single_result = simulate_patient(
        formulation=initial_form,
        initial_gsh=100.0,
        staggered_doses=events,
    )
    cohort_df = simulate_cohort(
        n_patients=100,
        dose_mg=total_dose_mg,
        formulation_type=FormulationType.STANDARD,
        patient_type="normal",
        staggered_doses=events,
        random_seed=102,
    )
    return single_result, cohort_df


def get_malnutrition_scenario(dose_mg: float = 5000.0) -> Dict[str, Any]:
    """
    Scenario 2: Borderline Supra-Therapeutic Dosing (5,000 mg / 24h).
    Compares well-nourished subjects (GSH baseline 100%) against malnourished / fasting
    individuals (GSH baseline ~54%).
    """
    tablets = int(round(dose_mg / 500.0))
    form = Formulation.standard_napa(tablets=tablets)

    single_normal = simulate_patient(formulation=form, initial_gsh=100.0)
    single_malnourished = simulate_patient(formulation=form, initial_gsh=54.0)

    cohort_normal = simulate_cohort(
        n_patients=100,
        dose_mg=dose_mg,
        formulation_type=FormulationType.STANDARD,
        patient_type="normal",
        random_seed=201,
    )
    cohort_malnourished = simulate_cohort(
        n_patients=100,
        dose_mg=dose_mg,
        formulation_type=FormulationType.STANDARD,
        patient_type="malnourished",
        random_seed=202,
    )

    return {
        "single_normal": single_normal,
        "single_malnourished": single_malnourished,
        "cohort_normal": cohort_normal,
        "cohort_malnourished": cohort_malnourished,
    }


def get_nac_timing_scenario(dose_mg: float = 12000.0) -> Dict[str, Any]:
    """
    Scenario 3: Antidote Therapeutic Window in Severe Overdose (12,000 mg).
    Compares:
    - Early NAC Rescue: Started at 8 hours post-ingestion (within Rumack-Matthew window).
    - Delayed NAC Rescue: Started at 16 hours post-ingestion.
    - Untreated Overdose: No antidote.
    """
    tablets = int(round(dose_mg / 500.0))
    form = Formulation.standard_napa(tablets=tablets)

    nac_early = NACIntervention(start_time_hr=8.0, duration_hr=21.0, boost_rate=18.5)
    nac_delayed = NACIntervention(start_time_hr=16.0, duration_hr=21.0, boost_rate=18.5)

    res_untreated = simulate_patient(formulation=form, initial_gsh=100.0, nac_intervention=None)
    res_early = simulate_patient(formulation=form, initial_gsh=100.0, nac_intervention=nac_early)
    res_delayed = simulate_patient(formulation=form, initial_gsh=100.0, nac_intervention=nac_delayed)

    cohort_untreated = simulate_cohort(
        n_patients=100,
        dose_mg=dose_mg,
        patient_type="normal",
        nac_intervention=None,
        random_seed=301,
    )
    cohort_early = simulate_cohort(
        n_patients=100,
        dose_mg=dose_mg,
        patient_type="normal",
        nac_intervention=nac_early,
        random_seed=302,
    )
    cohort_delayed = simulate_cohort(
        n_patients=100,
        dose_mg=dose_mg,
        patient_type="normal",
        nac_intervention=nac_delayed,
        random_seed=303,
    )

    return {
        "single_untreated": res_untreated,
        "single_early": res_early,
        "single_delayed": res_delayed,
        "cohort_untreated": cohort_untreated,
        "cohort_early": cohort_early,
        "cohort_delayed": cohort_delayed,
    }


def run_all_scenarios() -> Dict[str, Any]:
    """Executes and compiles all clinical scenarios."""
    single_acute, cohort_acute = get_acute_overdose_scenario(10000.0)
    single_stag, cohort_stag = get_staggered_overdose_scenario(10000.0)
    malnutrition_data = get_malnutrition_scenario(5000.0)
    nac_data = get_nac_timing_scenario(12000.0)

    return {
        "acute": {"single": single_acute, "cohort": cohort_acute},
        "staggered": {"single": single_stag, "cohort": cohort_stag},
        "malnutrition": malnutrition_data,
        "nac_timing": nac_data,
    }
