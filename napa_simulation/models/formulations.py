"""
Drug Formulations and Absorption Profiles for Napa Variations.

This module models the pharmaceutical release profiles of three prevalent
Bangladeshi Paracetamol formulations:
1. Standard Napa (500 mg immediate-release APAP)
2. Napa Extend (665 mg bi-layer modified-release: 36% immediate, 64% sustained)
3. Napa Extra (500 mg APAP + 65 mg Caffeine; caffeine acts as an inducer/modulator of CYP2E1)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple
from napa_simulation.config import CAFFEINE_CYP_MULTIPLIER


class FormulationType(str, Enum):
    """Enumeration of Napa commercial product lines."""
    STANDARD = "Standard Napa"
    EXTEND = "Napa Extend"
    EXTRA = "Napa Extra"


@dataclass(frozen=True)
class Formulation:
    """
    Encapsulates pharmacokinetic formulation attributes.

    Attributes:
        name: Clinical brand/variation name.
        formulation_type: FormulationType enum.
        tablets: Number of tablets ingested.
        apap_per_tablet_mg: Paracetamol strength per tablet in mg.
        total_apap_mg: Total paracetamol dose in mg.
        immediate_dose_mg: Dose partitioned into immediate-release gut depot.
        sustained_dose_mg: Dose partitioned into sustained-release matrix depot.
        caffeine_per_tablet_mg: Caffeine content per tablet in mg.
        total_caffeine_mg: Total caffeine ingested in mg.
        cyp_multiplier: Kinetic multiplier applied to k_cyp2e1.
    """
    name: str
    formulation_type: FormulationType
    tablets: int
    apap_per_tablet_mg: float
    total_apap_mg: float
    immediate_dose_mg: float
    sustained_dose_mg: float
    caffeine_per_tablet_mg: float
    total_caffeine_mg: float
    cyp_multiplier: float

    @classmethod
    def standard_napa(cls, tablets: int = 1) -> "Formulation":
        """
        Standard Napa: 500 mg immediate-release Paracetamol per tablet.
        100% immediate dissolution.
        """
        tablets = max(1, int(tablets))
        apap_per_tab = 500.0
        total_dose = apap_per_tab * tablets
        return cls(
            name="Standard Napa (500mg)",
            formulation_type=FormulationType.STANDARD,
            tablets=tablets,
            apap_per_tablet_mg=apap_per_tab,
            total_apap_mg=total_dose,
            immediate_dose_mg=total_dose,
            sustained_dose_mg=0.0,
            caffeine_per_tablet_mg=0.0,
            total_caffeine_mg=0.0,
            cyp_multiplier=1.0,
        )

    @classmethod
    def napa_extend(cls, tablets: int = 1) -> "Formulation":
        """
        Napa Extend: 665 mg bi-layer modified release Paracetamol per tablet.
        Partitioned into ~36% immediate bolus and ~64% sustained-release erosion matrix.
        Reference: Graudins et al. (2010), modified release paracetamol formulations.
        """
        tablets = max(1, int(tablets))
        apap_per_tab = 665.0
        total_dose = apap_per_tab * tablets
        immediate_fraction = 0.36
        sustained_fraction = 0.64

        return cls(
            name="Napa Extend (665mg Modified Release)",
            formulation_type=FormulationType.EXTEND,
            tablets=tablets,
            apap_per_tablet_mg=apap_per_tab,
            total_apap_mg=total_dose,
            immediate_dose_mg=total_dose * immediate_fraction,
            sustained_dose_mg=total_dose * sustained_fraction,
            caffeine_per_tablet_mg=0.0,
            total_caffeine_mg=0.0,
            cyp_multiplier=1.0,
        )

    @classmethod
    def napa_extra(cls, tablets: int = 1) -> "Formulation":
        """
        Napa Extra: 500 mg Paracetamol + 65 mg Caffeine per tablet.
        Immediate release with caffeine-mediated CYP enzyme kinetic induction (+15% - 25%).
        """
        tablets = max(1, int(tablets))
        apap_per_tab = 500.0
        caff_per_tab = 65.0
        total_dose = apap_per_tab * tablets
        total_caffeine = caff_per_tab * tablets

        return cls(
            name="Napa Extra (500mg APAP + 65mg Caffeine)",
            formulation_type=FormulationType.EXTRA,
            tablets=tablets,
            apap_per_tablet_mg=apap_per_tab,
            total_apap_mg=total_dose,
            immediate_dose_mg=total_dose,
            sustained_dose_mg=0.0,
            caffeine_per_tablet_mg=caff_per_tab,
            total_caffeine_mg=total_caffeine,
            cyp_multiplier=CAFFEINE_CYP_MULTIPLIER,
        )

    @classmethod
    def from_type(cls, form_type: FormulationType, tablets: int = 1) -> "Formulation":
        """Factory dispatcher by FormulationType."""
        if form_type == FormulationType.STANDARD:
            return cls.standard_napa(tablets)
        elif form_type == FormulationType.EXTEND:
            return cls.napa_extend(tablets)
        elif form_type == FormulationType.EXTRA:
            return cls.napa_extra(tablets)
        else:
            raise ValueError(f"Unknown formulation type: {form_type}")

    def get_initial_gut_amounts(self) -> Tuple[float, float]:
        """Returns tuple of (A_gut_immediate_mg, A_gut_sustained_mg)."""
        return self.immediate_dose_mg, self.sustained_dose_mg
