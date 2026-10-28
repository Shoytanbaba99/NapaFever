#!/usr/bin/env python3
"""
Liver Damage Simulation of Napa (Paracetamol) Overdose:
A Pharmacokinetic-Pharmacodynamic (PK-PD) Compartmental Model with
Monte Carlo Variations and Statistical Hypothesis Testing.

Course: CSE 413: Simulation & Modeling Lab - Section 7B
Team Name: K.G.B.P
Authors:
  - Mehedi Hassan Bhuiyan (ID: 0432320005101080)
  - Md. Masud Rahman (ID: 0432320005101064)
  - Hasib Al Mahmud Siddique (ID: 0432320005101095)
  - Rudro Antony Mrong (ID: 0432320005101059)
"""

import sys
import os
import time

# Ensure repository root is in python module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from napa_simulation.models.formulations import Formulation, FormulationType
from napa_simulation.simulation.engine import simulate_patient
from napa_simulation.models.pkpd import NACIntervention
from napa_simulation.analysis.hypothesis_testing import run_statistical_test_suite
from napa_simulation.visualization.plots import generate_all_publication_figures


BANNER = r"""
========================================================================================
   LIVER DAMAGE SIMULATION OF NAPA (PARACETAMOL) OVERDOSE: A PK-PD COMPARTMENTAL MODEL
       WITH MONTE CARLO VARIATIONS AND STATISTICAL HYPOTHESIS TESTING
========================================================================================
  Academic Course : CSE 413: Simulation & Modeling Lab - Section 7B
  Project Team    : K.G.B.P
  Authors         :
    1. Mehedi Hassan Bhuiyan       (ID: 0432320005101080)
    2. Md. Masud Rahman             (ID: 0432320005101064)
    3. Hasib Al Mahmud Siddique     (ID: 0432320005101095)
    4. Rudro Antony Mrong           (ID: 0432320005101059)
========================================================================================
"""


def print_section_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(f" >>> {title.upper()}")
    print("=" * 88)


def run_deterministic_baselines() -> None:
    print_section_header("Phase 1: Deterministic Formulation & Dose-Response Analysis")

    scenarios = [
        ("Standard Napa (Therapeutic 1g)", Formulation.standard_napa(tablets=2), 100.0, None),
        ("Napa Extend (Therapeutic 1.33g)", Formulation.napa_extend(tablets=2), 100.0, None),
        ("Napa Extra (Therapeutic + Caff)", Formulation.napa_extra(tablets=2), 100.0, None),
        ("Acute Napa Overdose (10,000 mg)", Formulation.standard_napa(tablets=20), 100.0, None),
        ("Napa Extra Overdose (10,000 mg)", Formulation.napa_extra(tablets=20), 100.0, None),
        ("Borderline Napa in Malnutrition (5g)", Formulation.standard_napa(tablets=10), 50.0, None),
    ]

    header_fmt = "{:<36} {:<10} {:<14} {:<10} {:<12} {:<12} {:<24}"
    row_fmt    = "{:<36} {:<10.0f} {:<14.2f} {:<10.2f} {:<12.1f} {:<12.2f} {:<24}"

    print(header_fmt.format("Simulation Scenario", "Dose(mg)", "Peak APAP(mg/L)", "Tmax(hr)", "Min GSH(%)", "Necrosis(%)", "Clinical Risk"))
    print("-" * 125)

    for name, form, gsh, nac in scenarios:
        res = simulate_patient(formulation=form, initial_gsh=gsh, nac_intervention=nac)
        print(row_fmt.format(
            name,
            res.total_dose_mg,
            res.peak_c_plasma,
            res.t_max,
            res.min_gsh,
            res.final_necrosis,
            res.clinical_risk,
        ))
    print("-" * 125)


def run_hypothesis_trials() -> None:
    print_section_header("Phase 2: Monte Carlo Virtual Clinical Trials & Statistical Suite")
    print("Running N=100 stochastic virtual patients per cohort across 2 overdose damage clinical trials...")
    t0 = time.time()
    results_list, summary_df = run_statistical_test_suite(n_patients=100, alpha=0.05)
    t_elapsed = time.time() - t0
    print(f"Monte Carlo simulations and statistical hypothesis testing completed in {t_elapsed:.2f}s.\n")

    for res in results_list:
        print(res.format_defense_report())
        print()

    print_section_header("Consolidated Clinical Trial Hypothesis Testing Summary")
    # Clean display of summary dataframe
    cols = ["Experiment", "H0", "t-Statistic", "p-Value (t-test)", "Cohen's d", "Verdict"]
    print(summary_df[cols].to_string(index=False))


def generate_figures() -> None:
    print_section_header("Phase 3: Publication-Grade Scientific Visualizations")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_figures")
    print(f"Generating 300 DPI high-resolution figures into '{output_dir}/'...")
    paths = generate_all_publication_figures(output_dir=output_dir)
    for key, path in paths.items():
        print(f"  [+] Saved {key.upper()}: {path}")


def print_system_summary() -> None:
    print_section_header("Phase 4: Mechanistic System Architecture and Verification")
    print("""
1. MATHEMATICAL ENGINE:
   - System of 6 coupled non-linear Ordinary Differential Equations (ODEs).
   - Solved numerically using Runge-Kutta 4th/5th order adaptive step-size solver (scipy.integrate.solve_ivp, RK45).
   - Mechanistically models gut dissolution, first-order absorption, saturable Michaelis-Menten glucuronidation/sulfation,
     toxic CYP2E1 oxidation to NAPQI, glutathione (GSH) replenishment, and hepatocyte macromolecular arylation.

2. FORMULATION COMPARATIVE PHYSIOLOGY:
   - Standard Napa: Rapid immediate release, high initial C_plasma peak.
   - Napa Extend: Bi-layer release (36% immediate, 64% sustained over 6-8h) shifts peak time and flattens C_plasma curve.
   - Napa Extra: Includes 65mg caffeine per tablet, causing enzymatic CYP induction (+22%), accelerating toxic NAPQI formation.

3. MONTE CARLO TO HYPOTHESIS TESTING BRIDGE:
   - Deterministic ODEs provide one trajectory; Monte Carlo samples 100 virtual patients per group with natural biological
     variance in V_d (~N(50, 5) L), baseline GSH (Normal: ~N(100, 8)%, Malnourished: ~N(54, 9.5)%), and clearance rates.
   - Empirical distributions are analyzed via Welch's Two-Sample t-tests, Mann-Whitney U tests, and Cohen's d effect sizes.
   - Rejection of H0 (p < 0.05) proves statistically significant toxicological divergence across clinical cohorts.
""")


def main() -> None:
    print(BANNER)
    run_deterministic_baselines()
    run_hypothesis_trials()
    generate_figures()
    print_system_summary()
    print("\n[SUCCESS] Napa PK-PD Simulation Suite execution completed successfully.\n")


if __name__ == "__main__":
    main()
