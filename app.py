"""
Interactive Streamlit Dashboard:
Liver Damage Simulation of Napa (Paracetamol) Overdose.

A Pharmacokinetic-Pharmacodynamic (PK-PD) Compartmental Simulation Suite
with Monte Carlo Virtual Cohorts and Inferential Hypothesis Testing.

Course: CSE 413: Simulation & Modeling Lab - Section 7B
Team: K.G.B.P
"""

import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local packages are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from napa_simulation.models.formulations import Formulation, FormulationType
from napa_simulation.models.pkpd import NACIntervention, CRITICAL_GSH_THRESHOLD
from napa_simulation.config import SimulationParams
from napa_simulation.simulation.engine import (
    simulate_patient,
    simulate_cohort,
    classify_clinical_risk,
)
from napa_simulation.analysis.hypothesis_testing import (
    run_experiment_1_acute_vs_staggered,
    run_experiment_2_malnutrition,
    run_experiment_3_nac_early_vs_delayed,
    run_statistical_test_suite,
)
from napa_simulation.visualization.plots import (
    plot_timeline_multipanel,
    plot_cohort_distributions,
    plot_antidote_efficacy_curve,
    set_academic_style,
)

# Page configuration
st.set_page_config(
    page_title="Napa PK-PD Liver Toxicity Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .team-badge {
        background-color: #f3f4f6;
        border-left: 4px solid #2563eb;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title & Team Header
st.markdown('<div class="main-title">Liver Damage Simulation of Napa Overdose</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A Mechanistic PK-PD Compartmental Model with Monte Carlo Variations & Statistical Hypothesis Testing</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="team-badge">
        <b>Academic Course:</b> CSE 413: Simulation & Modeling Lab &nbsp;|&nbsp; <b>Section:</b> 7B<br>
        <b>Team Name:</b> <b>K.G.B.P</b> &nbsp;|&nbsp; 
        <b>Authors:</b> Mehedi Hassan Bhuiyan (0432320005101080), Md. Masud Rahman (0432320005101064), 
        Hasib Al Mahmud Siddique (0432320005101095), Rudro Antony Mrong (0432320005101059)
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("🕹️ Simulation Parameters")

# Formulation Selection
form_choice = st.sidebar.selectbox(
    "Napa Product Variation",
    [
        "Standard Napa (500mg Immediate)",
        "Napa Extend (665mg Bi-layer MR)",
        "Napa Extra (500mg APAP + 65mg Caffeine)",
    ],
    index=0,
)

if "Extend" in form_choice:
    form_type = FormulationType.EXTEND
    tablet_strength = 665.0
elif "Extra" in form_choice:
    form_type = FormulationType.EXTRA
    tablet_strength = 500.0
else:
    form_type = FormulationType.STANDARD
    tablet_strength = 500.0

tablets = st.sidebar.slider("Number of Tablets Ingested", min_value=1, max_value=40, value=10, step=1)
total_dose_mg = tablets * tablet_strength

st.sidebar.markdown(f"**Total Dose:** `{total_dose_mg:,.0f} mg` Paracetamol")
if form_type == FormulationType.EXTRA:
    st.sidebar.markdown(f"**Total Caffeine:** `{tablets * 65.0:,.0f} mg` (+22% CYP2E1 induction)")

# Ingestion Regimen
st.sidebar.subheader("🕒 Ingestion Pattern")
intake_mode = st.sidebar.radio(
    "Intake Schedule",
    ["Single Acute Bolus", "Staggered Repeated Ingestion"],
    index=0,
)

staggered_events = None
if intake_mode == "Staggered Repeated Ingestion":
    stagger_doses = st.sidebar.slider("Number of Divided Ingestions", min_value=2, max_value=8, value=4, step=1)
    stagger_interval = st.sidebar.slider("Interval Between Ingestions (Hours)", min_value=1.0, max_value=8.0, value=3.0, step=0.5)
    dose_per_event = total_dose_mg / stagger_doses
    # Staggered events: list of (t, imm_mg, sust_mg)
    imm_ratio = 0.36 if form_type == FormulationType.EXTEND else 1.0
    sust_ratio = 0.64 if form_type == FormulationType.EXTEND else 0.0
    staggered_events = [
        (i * stagger_interval, dose_per_event * imm_ratio, dose_per_event * sust_ratio)
        for i in range(1, stagger_doses)
    ]
    # First dose at t=0 is handled by initial formulation loading
    form_instance = Formulation(
        name=form_choice,
        formulation_type=form_type,
        tablets=int(round(tablets / stagger_doses)),
        apap_per_tablet_mg=tablet_strength,
        total_apap_mg=dose_per_event,
        immediate_dose_mg=dose_per_event * imm_ratio,
        sustained_dose_mg=dose_per_event * sust_ratio,
        caffeine_per_tablet_mg=65.0 if form_type == FormulationType.EXTRA else 0.0,
        total_caffeine_mg=(tablets / stagger_doses) * 65.0 if form_type == FormulationType.EXTRA else 0.0,
        cyp_multiplier=1.22 if form_type == FormulationType.EXTRA else 1.0,
    )
else:
    form_instance = Formulation.from_type(form_type, tablets=tablets)

# Nutritional Status
st.sidebar.subheader("🥗 Patient Physiological State")
nutrition_status = st.sidebar.selectbox(
    "Nutritional / Fasting Status",
    ["Well-Nourished (100% GSH Baseline)", "Malnourished / Fasted (54% GSH Baseline)", "Custom Baseline"],
    index=0,
)

if nutrition_status == "Well-Nourished (100% GSH Baseline)":
    baseline_gsh = 100.0
elif nutrition_status == "Malnourished / Fasted (54% GSH Baseline)":
    baseline_gsh = 54.0
else:
    baseline_gsh = st.sidebar.slider("Custom Glutathione Baseline (%)", 20.0, 120.0, 80.0, 5.0)

v_d_input = st.sidebar.slider("Volume of Distribution V_d (L)", min_value=35.0, max_value=75.0, value=50.0, step=1.0)

# Execution of Single Deterministic Simulation
custom_params = SimulationParams(v_d=v_d_input)
sim_result = simulate_patient(
    formulation=form_instance,
    initial_gsh=baseline_gsh,
    params=custom_params,
    staggered_doses=staggered_events,
    t_span=(0.0, 72.0),
    n_points=500,
)

# ----------------- TOP METRIC TILES -----------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Peak Plasma APAP (Cmax)", f"{sim_result.peak_c_plasma:.1f} mg/L", f"Tmax: {sim_result.t_max:.1f} hr")

with col2:
    gsh_delta = sim_result.min_gsh - baseline_gsh
    st.metric("Nadir Glutathione (Min GSH)", f"{sim_result.min_gsh:.1f}%", f"{gsh_delta:.1f}% from baseline", delta_color="inverse")

with col3:
    st.metric("Hepatocyte Necrosis", f"{sim_result.final_necrosis:.2f}%", "Irreversible cell death")

with col4:
    risk = sim_result.clinical_risk
    if "Safe" in risk:
        st.success(f"**Clinical Status:**\n{risk}")
    elif "Mild" in risk:
        st.warning(f"**Clinical Status:**\n{risk}")
    elif "Severe" in risk:
        st.error(f"**Clinical Status:**\n{risk}")
    else:
        st.error(f"**CRITICAL ALERT:**\n{risk}")

# ----------------- MAIN TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Single Patient Dynamics",
    "🎲 Monte Carlo Population Trial",
    "🔬 Statistical Hypothesis Testing",
    "📐 Mathematical Model & Specifications",
])

# TAB 1: Single Patient Trajectories
with tab1:
    st.subheader("Coupled Pharmacokinetic-Pharmacodynamic Timecourse (0 - 72 Hours)")

    set_academic_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    t = sim_result.time_hr

    # Subplot 1: Plasma APAP
    ax1 = axes[0, 0]
    ax1.plot(t, sim_result.c_plasma, color="#1e40af", lw=2.5, label="Plasma APAP (mg/L)")
    ax1.axhline(150.0, color="#b91c1c", linestyle=":", alpha=0.8, label="Rumack-Matthew Threshold (150 mg/L @ 4h)")
    ax1.set_ylabel("Concentration (mg/L)", fontweight="bold")
    ax1.set_title("A. Central Compartment Concentration", loc="left", fontweight="bold")
    ax1.legend(loc="upper right")

    # Subplot 2: Glutathione Pool
    ax2 = axes[0, 1]
    ax2.plot(t, sim_result.gsh, color="#047857", lw=2.5, label="Hepatic GSH (%)")
    ax2.axhline(CRITICAL_GSH_THRESHOLD, color="#dc2626", linestyle="--", lw=1.8, label="Critical Necrosis Threshold (30% GSH)")
    ax2.set_ylabel("GSH (% Baseline)", fontweight="bold")
    ax2.set_title("B. Hepatic Glutathione Depletion & Stress", loc="left", fontweight="bold")
    ax2.set_ylim(-2, max(115.0, baseline_gsh + 10))
    ax2.legend(loc="upper right")

    # Subplot 3: Free NAPQI
    ax3 = axes[1, 0]
    ax3.plot(t, sim_result.free_napqi, color="#b45309", lw=2.5, label="Free NAPQI (mg/L equiv)")
    ax3.set_xlabel("Time Post-Ingestion (hours)", fontweight="bold")
    ax3.set_ylabel("NAPQI (mg/L equiv)", fontweight="bold")
    ax3.set_title("C. Reactive Electrophilic Metabolite Flux", loc="left", fontweight="bold")
    ax3.legend(loc="upper right")

    # Subplot 4: Hepatocyte Necrosis
    ax4 = axes[1, 1]
    ax4.plot(t, sim_result.necrosis_index, color="#991b1b", lw=2.8, label="Necrosis Index (%)")
    ax4.set_xlabel("Time Post-Ingestion (hours)", fontweight="bold")
    ax4.set_ylabel("Necrosis (% Death)", fontweight="bold")
    ax4.set_title("D. Cumulative Irreversible Liver Cell Necrosis", loc="left", fontweight="bold")
    ax4.set_ylim(-2, 102)
    ax4.legend(loc="upper left")

    fig.tight_layout()
    st.pyplot(fig)

    # Download button for CSV data
    df_single = sim_result.to_dataframe()
    csv_bytes = df_single.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Patient Timecourse Data (CSV)",
        data=csv_bytes,
        file_name="napa_patient_timecourse.csv",
        mime="text/csv",
    )

# TAB 2: Monte Carlo Population Trial
with tab2:
    st.subheader("🎲 High-Throughput Monte Carlo Virtual Clinical Trial")
    st.markdown(
        """
        Simulate an entire cohort of virtual patients with natural human biological variation:
        - Volume of Distribution $V_d \\sim \\mathcal{N}(50, 5)\\text{ L}$
        - Elimination clearance $k_{el} \\sim \\mathcal{N}(0.05, 0.007)\\text{ h}^{-1}$
        - CYP2E1 oxidative capacity $k_{\\text{cyp}} \\sim \\mathcal{N}(0.048, 0.006)$
        - Baseline Glutathione reserve (Normal vs Malnourished populations)
        """
    )

    col_mc1, col_mc2, col_mc3 = st.columns(3)
    with col_mc1:
        n_patients = st.slider("Virtual Cohort Size (Patients per group)", min_value=25, max_value=300, value=100, step=25)
    with col_mc2:
        mc_dose = st.slider("Cohort APAP Dose (mg)", min_value=1000, max_value=15000, value=int(total_dose_mg), step=500)
    with col_mc3:
        run_mc = st.button("🚀 Run Monte Carlo Population Trial", type="primary")

    if run_mc or "mc_cohorts" in st.session_state:
        if run_mc:
            with st.spinner(f"Simulating {n_patients * 3} virtual patients across cohorts..."):
                cohort_std = simulate_cohort(
                    n_patients=n_patients,
                    dose_mg=mc_dose,
                    formulation_type=FormulationType.STANDARD,
                    patient_type="normal",
                    random_seed=42,
                )
                cohort_extend = simulate_cohort(
                    n_patients=n_patients,
                    dose_mg=mc_dose,
                    formulation_type=FormulationType.EXTEND,
                    patient_type="normal",
                    random_seed=43,
                )
                cohort_malnourished = simulate_cohort(
                    n_patients=n_patients,
                    dose_mg=mc_dose,
                    formulation_type=FormulationType.STANDARD,
                    patient_type="malnourished",
                    random_seed=44,
                )
                st.session_state["mc_cohorts"] = {
                    "Standard Napa (Normal)": cohort_std,
                    "Napa Extend (Normal)": cohort_extend,
                    "Standard Napa (Malnourished)": cohort_malnourished,
                }

        cohorts = st.session_state["mc_cohorts"]

        fig_dist = plot_cohort_distributions(
            cohorts,
            metric="final_necrosis_pct",
            title=f"Population Necrosis Distributions at {mc_dose:,.0f} mg Ingestion (N={n_patients} per group)",
            ylabel="Hepatocyte Necrosis (%)",
        )
        st.pyplot(fig_dist)

        # Cohort summary statistics table
        summary_rows = []
        for name, df in cohorts.items():
            summary_rows.append({
                "Cohort": name,
                "Mean Necrosis (%)": f"{df['final_necrosis_pct'].mean():.2f}%",
                "Std Necrosis (%)": f"{df['final_necrosis_pct'].std():.2f}%",
                "Median Necrosis (%)": f"{df['final_necrosis_pct'].median():.2f}%",
                "Severe Injury Rate (Necrosis > 5%)": f"{(df['final_necrosis_pct'] > 5.0).mean() * 100:.1f}%",
                "Fulminant Failure Rate (Necrosis >= 30%)": f"{(df['final_necrosis_pct'] >= 30.0).mean() * 100:.1f}%",
            })
        st.table(pd.DataFrame(summary_rows))

# TAB 3: Statistical Hypothesis Testing
with tab3:
    st.subheader("🔬 Automated Statistical Hypothesis Testing Suite")
    st.markdown(
        """
        Translating computational simulations into formal medical science:
        We test explicit null hypotheses ($H_0$) against alternative hypotheses ($H_1$)
        using **Welch's Two-Sample t-test**, **Mann-Whitney U rank-sum test**, and **Cohen's d** effect size.
        """
    )

    if st.button("🧪 Execute Complete Clinical Hypothesis Suite (N=100 per trial)", type="primary"):
        with st.spinner("Executing virtual clinical trials and hypothesis calculations..."):
            test_results, summary_table = run_statistical_test_suite(n_patients=100, alpha=0.05)
            st.session_state["hyp_results"] = test_results
            st.session_state["hyp_summary"] = summary_table

    if "hyp_results" in st.session_state:
        results = st.session_state["hyp_results"]
        summary = st.session_state["hyp_summary"]

        st.dataframe(summary, use_container_width=True)

        for res in results:
            with st.expander(f"📋 Experiment {res.experiment_id}: {res.title} (Click to inspect defense report)", expanded=True):
                st.code(res.format_defense_report(), language="text")

# TAB 4: Mathematical Model & Specifications
with tab4:
    st.subheader("📐 Mathematical Formulation & Physiological Invariants")
    st.markdown(
        """
        ### 1. Coupled Ordinary Differential Equation System
        The multi-compartmental model tracks six state variables solved over t in [0, 72] hours:
        - Absorption:
          $$\\frac{dA_{\\text{imm}}}{dt} = - k_{a,\\text{imm}} A_{\\text{imm}}$$
          $$\\frac{dA_{\\text{sust}}}{dt} = - k_{a,\\text{sust}} A_{\\text{sust}}$$
          $$\\dot{R}_{\\text{abs}}(t) = k_{a,\\text{imm}} A_{\\text{imm}} + k_{a,\\text{sust}} A_{\\text{sust}}$$

        - Central Plasma Compartment:
          $$\\frac{dC_{\\text{plasma}}}{dt} = \\frac{1}{V_d} \\left[ \\dot{R}_{\\text{abs}}(t) - \\left( \\frac{V_{\\max,\\text{gluc}} C}{K_{m,\\text{gluc}} + C} + \\frac{V_{\\max,\\text{sulf}} C}{K_{m,\\text{sulf}} + C} + k_{el} V_d C + k_{\\text{cyp}} V_d C \\right) \\right]$$

        - Glutathione Depletion:
          $$\\frac{d[\\text{GSH}]}{dt} = \\alpha \\max\\left(0, 1 - \\frac{[\\text{GSH}]}{[\\text{GSH}]_{\\text{base}}}\\right) - \\left( r_{\\text{gen,napqi}} \\cdot 2.3 \\cdot \\frac{[\\text{GSH}]}{10.0 + [\\text{GSH}]} \\right)$$

        - Reactive Electrophilic Metabolite (NAPQI):
          $$\\frac{d[\\text{Free NAPQI}]}{dt} = 2.1 \\cdot k_{\\text{cyp}} C - k_{\\text{detox}} [\\text{NAPQI}] \\left(\\frac{[\\text{GSH}]}{100}\\right) - 0.25 [\\text{NAPQI}]$$

        - Cumulative Hepatocellular Necrosis Index:
          $$\\frac{d(\\text{Necrosis})}{dt} = k_{\\text{necrosis}} [\\text{NAPQI}] \\left(\\frac{30 - [\\text{GSH}]}{30}\\right)^2 \\left(1 - \\frac{\\text{Necrosis}}{100}\\right) \\quad \\text{when } [\\text{GSH}] < 30\\%$$

        ---

        ### 2. Parameter Standards
        - Glucuronidation: $V_{\\max} = 700.0\\text{ mg/h}, K_m = 65.0\\text{ mg/L}$
        - Sulfation: $V_{\\max} = 180.0\\text{ mg/h}, K_m = 16.0\\text{ mg/L}$
        - Critical GSH Necrosis Threshold: $30.0\\%$ of physiological baseline
        - Volume of Distribution: $50.0\\text{ L}$ (~0.7 to 0.9 L/kg adult standard)
        """
    )
