"""
Simulation Engine for Single-Patient Integration and Monte Carlo Cohorts.

Implements:
1. `simulate_patient()`: Deterministic ODE solver using scipy.integrate.solve_ivp (RK45)
   with support for multi-dose staggered ingestion and timed NAC rescue infusion.
2. `simulate_cohort()`: High-throughput Monte Carlo virtual clinical trial simulator
   sampling biological parameters across normal and log-normal physiological distributions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from napa_simulation.config import (
    SimulationParams,
    MC_VD_MEAN,
    MC_VD_STD,
    MC_KEL_MEAN,
    MC_KEL_STD,
    MC_CYP_MEAN,
    MC_CYP_STD,
    MC_GSH_NORMAL_MEAN,
    MC_GSH_NORMAL_STD,
    MC_GSH_MALNOURISHED_MEAN,
    MC_GSH_MALNOURISHED_STD,
    DEFAULT_VMAX_GLUC,
    DEFAULT_KM_GLUC,
    DEFAULT_VMAX_SULF,
    DEFAULT_KM_SULF,
    DEFAULT_K_A_IMMEDIATE,
    DEFAULT_K_A_SUSTAINED,
    DEFAULT_K_DETOX,
    DEFAULT_K_NECROSIS,
    CRITICAL_GSH_THRESHOLD,
    NAC_GSH_REPLENISHMENT_BOOST,
)
from napa_simulation.models.formulations import Formulation, FormulationType
from napa_simulation.models.pkpd import system_derivatives, NACIntervention


def classify_clinical_risk(final_necrosis: float, min_gsh: float, peak_c_plasma: float) -> str:
    """
    Classifies hepatotoxic risk based on biochemical indices and histopathological necrosis.

    Categories:
    - Safe / Therapeutic (No Liver Injury): Necrosis < 1%, GSH > 50%
    - Mild Risk / Subclinical Transaminitis: Necrosis < 5%, GSH >= 30%
    - Severe Acute Liver Injury (ALI): Necrosis 5-30%, GSH < 30%
    - Fulminant Hepatic Failure (FHF): Necrosis >= 30% or GSH collapsed < 15%
    """
    if final_necrosis >= 30.0 or min_gsh < 15.0 or peak_c_plasma > 300.0:
        return "Fulminant Hepatic Failure (FHF)"
    elif final_necrosis >= 5.0 or min_gsh < 30.0:
        return "Severe Acute Liver Injury (ALI)"
    elif final_necrosis >= 1.0 or min_gsh < 50.0:
        return "Mild Risk / Subclinical Transaminitis"
    else:
        return "Safe / Therapeutic (No Liver Injury)"


@dataclass
class SimulationResult:
    """
    Comprehensive simulation trajectory and summary diagnostics for an individual.
    """
    time_hr: np.ndarray
    a_gut_imm: np.ndarray
    a_gut_sust: np.ndarray
    c_plasma: np.ndarray
    gsh: np.ndarray
    free_napqi: np.ndarray
    necrosis_index: np.ndarray

    formulation_name: str
    total_dose_mg: float
    initial_gsh: float
    params: SimulationParams
    nac_intervention: Optional[NACIntervention]

    @property
    def peak_c_plasma(self) -> float:
        """Peak plasma concentration (mg/L)."""
        return float(np.max(self.c_plasma))

    @property
    def t_max(self) -> float:
        """Time of peak plasma concentration (hours)."""
        idx = int(np.argmax(self.c_plasma))
        return float(self.time_hr[idx])

    @property
    def min_gsh(self) -> float:
        """Nadir of hepatic Glutathione reserves (%)."""
        return float(np.min(self.gsh))

    @property
    def peak_napqi(self) -> float:
        """Peak reactive electrophile concentration (mg/L)."""
        return float(np.max(self.free_napqi))

    @property
    def final_necrosis(self) -> float:
        """Final cumulative hepatocyte necrosis index at endpoint (%)."""
        return float(self.necrosis_index[-1])

    @property
    def clinical_risk(self) -> str:
        """Qualitative clinical assessment string."""
        return classify_clinical_risk(self.final_necrosis, self.min_gsh, self.peak_c_plasma)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the full simulation timeline into a pandas DataFrame."""
        return pd.DataFrame({
            "Time_hr": self.time_hr,
            "A_gut_imm_mg": self.a_gut_imm,
            "A_gut_sust_mg": self.a_gut_sust,
            "C_plasma_mg_L": self.c_plasma,
            "GSH_percent": self.gsh,
            "Free_NAPQI_mg_L": self.free_napqi,
            "Necrosis_Index_percent": self.necrosis_index,
        })


def simulate_patient(
    formulation: Optional[Formulation] = None,
    dose_mg: Optional[float] = None,
    formulation_type: FormulationType = FormulationType.STANDARD,
    initial_gsh: float = 100.0,
    params: Optional[SimulationParams] = None,
    staggered_doses: Optional[List[Tuple[float, float, float]]] = None,
    nac_intervention: Optional[NACIntervention] = None,
    t_span: Tuple[float, float] = (0.0, 72.0),
    n_points: int = 500,
) -> SimulationResult:
    """
    Executes a deterministic ODE integration for an individual subject.

    Args:
        formulation: Optional pre-configured Formulation instance.
        dose_mg: Total APAP dose in mg if formulation is omitted.
        formulation_type: Type of Napa product if formulation is omitted.
        initial_gsh: Starting Glutathione baseline (100% normal, ~54% malnourished).
        params: Simulation parameters (defaults to literature values).
        staggered_doses: List of (time_hr, immediate_dose_mg, sustained_dose_mg) for staggered ingestion.
        nac_intervention: Optional NAC rescue infusion timing and parameters.
        t_span: Tuple of (t_start, t_end) in hours (default 0 to 72 hrs).
        n_points: Number of discrete evaluation points.

    Returns:
        SimulationResult instance containing trajectories and endpoint metrics.
    """
    if params is None:
        params = SimulationParams()

    # Determine formulation kinetics
    if formulation is None:
        if dose_mg is not None:
            # Reconstruct formulation from dose
            if formulation_type == FormulationType.STANDARD:
                tablets = max(1, int(round(dose_mg / 500.0)))
                formulation = Formulation.standard_napa(tablets)
            elif formulation_type == FormulationType.EXTEND:
                tablets = max(1, int(round(dose_mg / 665.0)))
                formulation = Formulation.napa_extend(tablets)
            else:
                tablets = max(1, int(round(dose_mg / 500.0)))
                formulation = Formulation.napa_extra(tablets)
        else:
            formulation = Formulation.standard_napa(tablets=2)

    # Adjust CYP rate if formulation has metabolic modulators (e.g., caffeine in Napa Extra)
    if formulation.cyp_multiplier != 1.0:
        params = SimulationParams(
            v_d=params.v_d,
            k_a_imm=params.k_a_imm,
            k_a_sust=params.k_a_sust,
            k_el=params.k_el,
            vmax_gluc=params.vmax_gluc,
            km_gluc=params.km_gluc,
            vmax_sulf=params.vmax_sulf,
            km_sulf=params.km_sulf,
            k_cyp2e1=params.k_cyp2e1 * formulation.cyp_multiplier,
            gsh_baseline=params.gsh_baseline,
            gsh_synthesis_rate=params.gsh_synthesis_rate,
            k_detox=params.k_detox,
            critical_gsh_threshold=params.critical_gsh_threshold,
            k_necrosis=params.k_necrosis,
            nac_boost=params.nac_boost,
        )

    # Initial condition setup: y = [A_imm, A_sust, C_plasma, GSH, Free_NAPQI, Necrosis]
    imm_0, sust_0 = formulation.get_initial_gut_amounts()
    current_state = np.array([imm_0, sust_0, 0.0, float(initial_gsh), 0.0, 0.0], dtype=float)

    t_start, t_end = t_span
    t_eval = np.linspace(t_start, t_end, n_points)

    if staggered_doses is None or len(staggered_doses) == 0:
        # Standard single bolus ingestion
        sol = solve_ivp(
            fun=lambda t, y: system_derivatives(t, y, params, nac_intervention),
            t_span=(t_start, t_end),
            y0=current_state,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        t_arr = sol.t
        y_arr = sol.y
    else:
        # Piecewise continuous integration across staggered ingestion events
        sorted_events = sorted(staggered_doses, key=lambda x: x[0])
        all_t = [t_start]
        all_y = [current_state.copy()]

        # Filter events within t_span
        events = [ev for ev in sorted_events if t_start < ev[0] < t_end]
        time_checkpoints = [t_start] + [ev[0] for ev in events] + [t_end]

        accumulated_t = []
        accumulated_y = []

        curr_y = current_state.copy()

        for i in range(len(time_checkpoints) - 1):
            sub_t_start = time_checkpoints[i]
            sub_t_end = time_checkpoints[i + 1]

            if sub_t_end <= sub_t_start:
                continue

            sub_t_eval = np.linspace(sub_t_start, sub_t_end, max(10, int(n_points * (sub_t_end - sub_t_start) / (t_end - t_start))))

            sub_sol = solve_ivp(
                fun=lambda t, y: system_derivatives(t, y, params, nac_intervention),
                t_span=(sub_t_start, sub_t_end),
                y0=curr_y,
                t_eval=sub_t_eval,
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
            )

            # Append trajectory (omit duplicate junction point)
            if len(accumulated_t) > 0:
                accumulated_t.extend(sub_sol.t[1:].tolist())
                accumulated_y.append(sub_sol.y[:, 1:])
            else:
                accumulated_t.extend(sub_sol.t.tolist())
                accumulated_y.append(sub_sol.y)

            # End state of current interval
            curr_y = sub_sol.y[:, -1].copy()

            # If an ingestion event occurs at sub_t_end, deposit into gut
            if i < len(events):
                event_t, ev_imm, ev_sust = events[i]
                curr_y[0] += ev_imm
                curr_y[1] += ev_sust

        t_arr = np.array(accumulated_t)
        y_arr = np.hstack(accumulated_y)

    return SimulationResult(
        time_hr=t_arr,
        a_gut_imm=y_arr[0],
        a_gut_sust=y_arr[1],
        c_plasma=y_arr[2],
        gsh=y_arr[3],
        free_napqi=y_arr[4],
        necrosis_index=y_arr[5],
        formulation_name=formulation.name,
        total_dose_mg=formulation.total_apap_mg,
        initial_gsh=initial_gsh,
        params=params,
        nac_intervention=nac_intervention,
    )


def simulate_cohort(
    n_patients: int = 100,
    dose_mg: float = 10000.0,
    formulation_type: FormulationType = FormulationType.STANDARD,
    patient_type: str = "normal",
    staggered_doses: Optional[List[Tuple[float, float, float]]] = None,
    nac_intervention: Optional[NACIntervention] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Simulates a virtual patient population via Monte Carlo parameter sampling.

    Args:
        n_patients: Number of virtual individuals in cohort.
        dose_mg: Total paracetamol dose in mg.
        formulation_type: Napa product variant.
        patient_type: 'normal' (well-nourished) or 'malnourished' / 'fasting'.
        staggered_doses: Optional staggered dosing schedule.
        nac_intervention: Optional therapeutic NAC protocol.
        random_seed: Reproducibility seed for numpy RNG.

    Returns:
        Structured pandas DataFrame containing patient parameters and toxicological outcomes.
    """
    rng = np.random.default_rng(random_seed)

    records = []

    # Map dose into formulation tablets
    if formulation_type == FormulationType.EXTEND:
        tablets = max(1, int(round(dose_mg / 665.0)))
        base_form = Formulation.napa_extend(tablets)
    elif formulation_type == FormulationType.EXTRA:
        tablets = max(1, int(round(dose_mg / 500.0)))
        base_form = Formulation.napa_extra(tablets)
    else:
        tablets = max(1, int(round(dose_mg / 500.0)))
        base_form = Formulation.standard_napa(tablets)

    for i in range(n_patients):
        # Sample physiological parameters
        v_d = float(np.clip(rng.normal(MC_VD_MEAN, MC_VD_STD), 35.0, 75.0))
        k_el = float(np.clip(rng.normal(MC_KEL_MEAN, MC_KEL_STD), 0.025, 0.08))
        k_cyp = float(np.clip(rng.normal(MC_CYP_MEAN, MC_CYP_STD), 0.028, 0.075))
        vmax_gluc = float(np.clip(rng.normal(DEFAULT_VMAX_GLUC, 120.0), 900.0, 2000.0))
        vmax_sulf = float(np.clip(rng.normal(DEFAULT_VMAX_SULF, 35.0), 220.0, 500.0))

        # Sample baseline Glutathione reserve according to nutritional status
        if patient_type.lower() in ["malnourished", "fasting"]:
            gsh_base = float(np.clip(rng.normal(MC_GSH_MALNOURISHED_MEAN, MC_GSH_MALNOURISHED_STD), 25.0, 80.0))
        else:
            gsh_base = float(np.clip(rng.normal(MC_GSH_NORMAL_MEAN, MC_GSH_NORMAL_STD), 75.0, 125.0))

        patient_params = SimulationParams(
            v_d=v_d,
            k_a_imm=DEFAULT_K_A_IMMEDIATE,
            k_a_sust=DEFAULT_K_A_SUSTAINED,
            k_el=k_el,
            vmax_gluc=vmax_gluc,
            km_gluc=DEFAULT_KM_GLUC,
            vmax_sulf=vmax_sulf,
            km_sulf=DEFAULT_KM_SULF,
            k_cyp2e1=k_cyp,
            gsh_baseline=100.0,
            gsh_synthesis_rate=4.0,
            k_detox=DEFAULT_K_DETOX,
            critical_gsh_threshold=CRITICAL_GSH_THRESHOLD,
            k_necrosis=DEFAULT_K_NECROSIS,
            nac_boost=NAC_GSH_REPLENISHMENT_BOOST,
        )

        res = simulate_patient(
            formulation=base_form,
            initial_gsh=gsh_base,
            params=patient_params,
            staggered_doses=staggered_doses,
            nac_intervention=nac_intervention,
            t_span=(0.0, 72.0),
            n_points=250,
        )

        records.append({
            "patient_id": i + 1,
            "formulation": base_form.name,
            "dose_mg": base_form.total_apap_mg,
            "patient_type": "Malnourished" if patient_type.lower() in ["malnourished", "fasting"] else "Normal",
            "v_d_L": v_d,
            "k_el": k_el,
            "k_cyp2e1": k_cyp,
            "baseline_gsh_pct": gsh_base,
            "peak_c_plasma_mg_L": res.peak_c_plasma,
            "t_max_hr": res.t_max,
            "min_gsh_pct": res.min_gsh,
            "peak_napqi_mg_L": res.peak_napqi,
            "final_necrosis_pct": res.final_necrosis,
            "clinical_risk": res.clinical_risk,
        })

    return pd.DataFrame(records)
