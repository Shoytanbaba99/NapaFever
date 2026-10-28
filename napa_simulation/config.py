"""
Pharmacokinetic and Pharmacodynamic (PK-PD) Configuration and Constants.

This module stores literature-validated biological, physiological, and kinetic
parameters for Paracetamol (N-acetyl-p-aminophenol / APAP) metabolism,
glutathione depletion, and reactive metabolite hepatotoxicity.

Literature References:
- Rumack BH, Matthew H. (1975). Acetaminophen poisoning and toxicity. Pediatrics, 55(6), 871-876.
- Prescott LF. (1980). Kinetics and metabolism of paracetamol and phenacetin. Br J Clin Pharmacol, 10(S2), 291S-298S.
- Mitchell JR, Jollow DJ, Potter WZ, Gillette JR, Brodie BB. (1973). Acetaminophen-induced hepatic necrosis.
  IV. Protective role of glutathione. J Pharmacol Exp Ther, 187(1), 211-217.
- Critchley JA, Nimmo GR, Gregson CA, Woolhouse NM, Prescott LF. (2005). Inter-subject and intra-subject
  variation in paracetamol metabolism. Br J Clin Pharmacol, 22(6), 649-655.
- Graudins A, Chiew A, Chan B. (2010). Overdose with modified-release paracetamol: clinical review.
  Emerg Med Australas, 22(4), 286-296.
- Manyike PT, Kharasch ED, Kalhorn TF, Slattery JT. (2000). Contribution of CYP2E1 and CYP3A to acetaminophen
  reactive metabolite formation. Clin Pharmacol Ther, 67(3), 275-282.
"""

from dataclasses import dataclass
from typing import Dict, Any


# Molecular Constants
APAP_MOLECULAR_WEIGHT = 151.163   # g/mol (Paracetamol / Acetaminophen)
CAFFEINE_MOLECULAR_WEIGHT = 194.19 # g/mol (1,3,7-Trimethylxanthine)

# Compartment & Absorption Defaults (Adult standard ~ 70 kg)
DEFAULT_V_D = 50.0                # L (Central volume of distribution, ~0.7-0.9 L/kg; Prescott 1980)
DEFAULT_K_A_IMMEDIATE = 1.45      # 1/h (First-order absorption rate constant for immediate release)
DEFAULT_K_A_SUSTAINED = 0.22      # 1/h (Absorption rate constant for sustained-release matrix depot)
DEFAULT_K_EL = 0.05               # 1/h (Basal renal/unchanged elimination rate constant)

# Hepatic Safe Conjugation Kinetics (Michaelis-Menten Non-linear Pathways)
# Prescott et al. (1980), Critchley et al. (2005)
# Glucuronidation: High capacity, moderate affinity
DEFAULT_VMAX_GLUC = 700.0         # mg/h (Maximum velocity of glucuronidation)
DEFAULT_KM_GLUC = 65.0            # mg/L (Michaelis constant for glucuronidation)

# Sulfation: Low capacity, high affinity (saturates early at mild-to-moderate doses)
DEFAULT_VMAX_SULF = 180.0         # mg/h (Maximum velocity of sulfation)
DEFAULT_KM_SULF = 16.0            # mg/L (Michaelis constant for sulfation)

# Toxic Shunt (Cytochrome P450 / CYP2E1 Oxidation to NAPQI)
DEFAULT_K_CYP2E1 = 0.045          # 1/h (Rate constant of oxidative bioactivation into NAPQI)
CAFFEINE_CYP_MULTIPLIER = 1.22    # +22% metabolic modulation / CYP induction in Napa Extra

# Glutathione (GSH) & Detoxification Kinetics
# Mitchell et al. (1973)
DEFAULT_GSH_BASELINE = 100.0      # % (Normalized baseline hepatic GSH pool; ~5.5-7.0 mM)
DEFAULT_GSH_SYNTHESIS_RATE = 4.0  # %/h (Endogenous zero/fractional replenishment rate)
DEFAULT_K_DETOX = 0.75            # (mg/L)^(-1) * h^(-1) (Conjugation rate between GSH and NAPQI)
CRITICAL_GSH_THRESHOLD = 30.0     # % (Critical threshold below which macromolecular arylation occurs)

# Hepatocyte Necrosis Dynamics
DEFAULT_K_NECROSIS = 0.32         # (mg/L)^(-1) * h^(-1) (Rate of irreversible hepatocyte necrosis)
MAX_NECROSIS_INDEX = 100.0        # % (100% represents massive fulminant hepatic necrosis)

# Therapeutic Antidote (N-Acetylcysteine / NAC)
NAC_GSH_REPLENISHMENT_BOOST = 35.0 # %/h (Exogenous replenishment rate during active NAC infusion)
DEFAULT_NAC_INFUSION_DURATION = 21.0 # hours (Standard intravenous 3-bag Prescott protocol duration)

# Monte Carlo Population Variance Parameters
MC_VD_MEAN = 50.0
MC_VD_STD = 5.0

MC_KEL_MEAN = 0.035
MC_KEL_STD = 0.005

MC_CYP_MEAN = 0.045
MC_CYP_STD = 0.006

MC_GSH_NORMAL_MEAN = 100.0
MC_GSH_NORMAL_STD = 7.5

MC_GSH_MALNOURISHED_MEAN = 50.0
MC_GSH_MALNOURISHED_STD = 7.0


@dataclass(frozen=True)
class SimulationParams:
    """Immutable parameter container for ODE integration."""
    v_d: float = DEFAULT_V_D
    k_a_imm: float = DEFAULT_K_A_IMMEDIATE
    k_a_sust: float = DEFAULT_K_A_SUSTAINED
    k_el: float = DEFAULT_K_EL
    vmax_gluc: float = DEFAULT_VMAX_GLUC
    km_gluc: float = DEFAULT_KM_GLUC
    vmax_sulf: float = DEFAULT_VMAX_SULF
    km_sulf: float = DEFAULT_KM_SULF
    k_cyp2e1: float = DEFAULT_K_CYP2E1
    gsh_baseline: float = DEFAULT_GSH_BASELINE
    gsh_synthesis_rate: float = DEFAULT_GSH_SYNTHESIS_RATE
    k_detox: float = DEFAULT_K_DETOX
    critical_gsh_threshold: float = CRITICAL_GSH_THRESHOLD
    k_necrosis: float = DEFAULT_K_NECROSIS
    nac_boost: float = NAC_GSH_REPLENISHMENT_BOOST

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v_d": self.v_d,
            "k_a_imm": self.k_a_imm,
            "k_a_sust": self.k_a_sust,
            "k_el": self.k_el,
            "vmax_gluc": self.vmax_gluc,
            "km_gluc": self.km_gluc,
            "vmax_sulf": self.vmax_sulf,
            "km_sulf": self.km_sulf,
            "k_cyp2e1": self.k_cyp2e1,
            "gsh_baseline": self.gsh_baseline,
            "gsh_synthesis_rate": self.gsh_synthesis_rate,
            "k_detox": self.k_detox,
            "critical_gsh_threshold": self.critical_gsh_threshold,
            "k_necrosis": self.k_necrosis,
            "nac_boost": self.nac_boost,
        }
