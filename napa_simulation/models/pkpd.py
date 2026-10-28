"""
Pharmacokinetic-Pharmacodynamic (PK-PD) Coupled Differential Equation System.

This module formalizes the mathematical system of Ordinary Differential Equations (ODEs)
governing paracetamol gastrointestinal absorption, central distribution, non-linear
saturable hepatic conjugation, CYP2E1 toxic bioactivation into NAPQI, glutathione (GSH)
depletion, and macromolecular arylation leading to irreversible hepatocyte necrosis.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import numpy as np
from napa_simulation.config import SimulationParams, CRITICAL_GSH_THRESHOLD


@dataclass
class PKPDState:
    """Named representation of the ODE state vector."""
    a_gut_imm: float     # mg (Immediate-release APAP in gut)
    a_gut_sust: float    # mg (Sustained-release APAP in gut matrix)
    c_plasma: float      # mg/L (Plasma paracetamol concentration)
    gsh: float           # % (Hepatic Glutathione level, 100% normal)
    free_napqi: float    # mg/L equivalent (Unbound reactive electrophile)
    necrosis_index: float # % (Hepatocyte death index, 0-100%)

    @classmethod
    def from_array(cls, y: np.ndarray) -> "PKPDState":
        return cls(
            a_gut_imm=float(y[0]),
            a_gut_sust=float(y[1]),
            c_plasma=float(y[2]),
            gsh=float(y[3]),
            free_napqi=float(y[4]),
            necrosis_index=float(y[5]),
        )

    def to_array(self) -> np.ndarray:
        return np.array([
            self.a_gut_imm,
            self.a_gut_sust,
            self.c_plasma,
            self.gsh,
            self.free_napqi,
            self.necrosis_index,
        ], dtype=float)


@dataclass
class NACIntervention:
    """
    N-Acetylcysteine (NAC) therapeutic infusion protocol.

    Attributes:
        start_time_hr: Time of administration start in hours post-ingestion.
        duration_hr: Duration of continuous antidote infusion in hours.
        boost_rate: Supplemental GSH synthesis replenishment rate (%/h).
    """
    start_time_hr: float
    duration_hr: float = 21.0
    boost_rate: float = 18.5

    def is_active(self, t: float) -> bool:
        """Determines if NAC infusion is actively running at time t."""
        return self.start_time_hr <= t <= (self.start_time_hr + self.duration_hr)


def system_derivatives(
    t: float,
    y: List[float],
    params: SimulationParams,
    nac_protocol: Optional[NACIntervention] = None,
) -> List[float]:
    """
    Coupled ODE derivative function for scipy.integrate.solve_ivp.

    State Vector y:
    - y[0]: A_gut_imm (mg) - Immediate absorption compartment
    - y[1]: A_gut_sust (mg) - Sustained release depot compartment
    - y[2]: C_plasma (mg/L) - Central paracetamol concentration
    - y[3]: GSH (%) - Hepatic glutathione pool
    - y[4]: Free_NAPQI (mg/L equivalent) - Reactive electrophilic metabolite
    - y[5]: Necrosis_Index (%) - Cumulative hepatic necrosis percentage

    Mathematical Formulation:
    1. Absorption:
       dA_imm/dt = - k_a_imm * A_imm
       dA_sust/dt = - k_a_sust * A_sust
       R_abs = k_a_imm * A_imm + k_a_sust * A_sust

    2. Hepatic Elimination & Saturable Metabolism:
       Glucuronidation (Michaelis-Menten): R_gluc = (Vmax_gluc * C_plasma) / (Km_gluc + C_plasma)
       Sulfation (Michaelis-Menten):       R_sulf = (Vmax_sulf * C_plasma) / (Km_sulf + C_plasma)
       Renal Elimination (Linear):         R_el   = k_el * V_d * C_plasma
       Toxic Bioactivation (CYP2E1):       R_cyp  = k_cyp2e1 * V_d * C_plasma

       dC_plasma/dt = [ R_abs - (R_gluc + R_sulf + R_el + R_cyp) ] / V_d

    3. Glutathione & Toxic Metabolite (NAPQI) Dynamics:
       r_gen_napqi = k_cyp2e1 * C_plasma
       r_detox = k_detox * Free_NAPQI * (GSH / 100.0)

       r_nac = nac_protocol.boost_rate if (active) else 0.0
       r_synth = params.gsh_synthesis_rate * max(0.0, 1.0 - (GSH / params.gsh_baseline)) + r_nac

       dGSH/dt = r_synth - (r_detox * 4.5)
       dFree_NAPQI/dt = r_gen_napqi - r_detox - (0.8 * Free_NAPQI)

    4. Irreversible Hepatocyte Necrosis:
       Activated when GSH falls below critical threshold (e.g., 30% baseline):
       If GSH < critical_gsh_threshold:
           vulnerability = (critical_gsh_threshold - GSH) / critical_gsh_threshold
           dNecrosis/dt = k_necrosis * Free_NAPQI * vulnerability * (1 - Necrosis / 100)
       Else:
           dNecrosis/dt = 0
    """
    # Enforce non-negativity boundary conditions
    a_imm = max(0.0, float(y[0]))
    a_sust = max(0.0, float(y[1]))
    c_plasma = max(0.0, float(y[2]))
    gsh = max(0.0, float(y[3]))
    napqi = max(0.0, float(y[4]))
    necrosis = np.clip(float(y[5]), 0.0, 100.0)

    # 1. Gut absorption rates
    da_imm_dt = -params.k_a_imm * a_imm
    da_sust_dt = -params.k_a_sust * a_sust
    r_absorption = (params.k_a_imm * a_imm) + (params.k_a_sust * a_sust) # mg/h

    # 2. Hepatic non-linear conjugation and clearance
    # Glucuronidation: high capacity, saturates at elevated overdoses
    r_gluc = (params.vmax_gluc * c_plasma) / (params.km_gluc + c_plasma) if c_plasma > 0 else 0.0
    # Sulfation: low capacity, saturates early in therapeutic-to-mild range
    r_sulf = (params.vmax_sulf * c_plasma) / (params.km_sulf + c_plasma) if c_plasma > 0 else 0.0
    # Linear renal filtration of unchanged drug
    r_renal = params.k_el * params.v_d * c_plasma
    # Toxic oxidation shunt by Cytochrome P450 2E1
    r_cyp = params.k_cyp2e1 * params.v_d * c_plasma

    total_elimination_rate = r_gluc + r_sulf + r_renal + r_cyp # mg/h
    dc_plasma_dt = (r_absorption - total_elimination_rate) / params.v_d

    # 3. NAPQI bioactivation and GSH conjugation
    # Generation rate of NAPQI in concentration terms
    r_gen_napqi = params.k_cyp2e1 * c_plasma * 2.1
    # Detoxification velocity via GSH S-transferase conjugation
    gsh_fraction = max(0.0, gsh / 100.0)
    r_detox = params.k_detox * napqi * gsh_fraction

    # GSH synthesis and replenishment
    r_nac = 0.0
    if nac_protocol is not None and nac_protocol.is_active(t):
        r_nac = nac_protocol.boost_rate

    # Endogenous turnover replenishes toward baseline; NAC supplies exogenous cysteine substrate
    gsh_deficit_ratio = max(0.0, 1.0 - (gsh / params.gsh_baseline))
    r_synth = (params.gsh_synthesis_rate * gsh_deficit_ratio) + r_nac

    # Rate of GSH consumption (saturable affinity for glutathione pool)
    gsh_avail_factor = gsh / (10.0 + gsh) if gsh > 0 else 0.0
    r_gsh_consumption = r_gen_napqi * 2.3 * gsh_avail_factor
    dgsh_dt = r_synth - r_gsh_consumption

    # Free NAPQI net accumulation
    # Unconjugated NAPQI undergoes detoxification or decay/binding
    dnapqi_dt = r_gen_napqi - r_detox - (0.25 * napqi)

    # 4. Hepatocyte Necrosis Index
    # When GSH drops below critical threshold (30%), NAPQI covalently binds macromolecular proteins
    if gsh < params.critical_gsh_threshold:
        vulnerability = ((params.critical_gsh_threshold - gsh) / params.critical_gsh_threshold) ** 2.0
        necrosis_capacity = max(0.0, 1.0 - (necrosis / 100.0))
        dnecrosis_dt = params.k_necrosis * napqi * vulnerability * necrosis_capacity
    else:
        dnecrosis_dt = 0.0

    return [da_imm_dt, da_sust_dt, dc_plasma_dt, dgsh_dt, dnapqi_dt, dnecrosis_dt]
