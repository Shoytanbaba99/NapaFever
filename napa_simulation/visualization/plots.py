"""
Publication-Grade Biomedical Visualization Suite.

Generates high-resolution multi-panel pharmacokinetic timecourses,
Monte Carlo population violin/box distributions, and clinical antidote
rescue efficacy curves.
"""

from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from napa_simulation.simulation.engine import SimulationResult, simulate_patient
from napa_simulation.models.formulations import Formulation
from napa_simulation.models.pkpd import NACIntervention
from napa_simulation.config import CRITICAL_GSH_THRESHOLD


def set_academic_style() -> None:
    """Applies clean publication styling to matplotlib/seaborn."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "figure.dpi": 300,
        "lines.linewidth": 2.0,
    })


def plot_timeline_multipanel(
    results_map: Dict[str, SimulationResult],
    title: str = "Hepatic PK-PD Trajectories Across Paracetamol Exposure Scenarios",
    figsize: Tuple[int, int] = (12, 14),
) -> plt.Figure:
    """
    Figure 1: 4-row multi-panel temporal dynamics:
    Panel A: Plasma Paracetamol Concentration (mg/L)
    Panel B: Hepatic Glutathione Reserve (% baseline) with critical 30% threshold
    Panel C: Reactive Intermediate (Free NAPQI, mg/L equivalent)
    Panel D: Cumulative Hepatocyte Necrosis Index (%)
    """
    set_academic_style()
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77"]

    # Panel A: Plasma APAP
    ax1 = axes[0]
    for idx, (label, res) in enumerate(results_map.items()):
        c = colors[idx % len(colors)]
        ax1.plot(res.time_hr, res.c_plasma, label=label, color=c, lw=2.2)
    ax1.set_ylabel("Plasma APAP\n(mg/L)", fontweight="bold")
    ax1.set_title("A. Central Compartment Pharmacokinetics", loc="left", fontweight="bold")
    ax1.axhline(150.0, color="gray", linestyle=":", alpha=0.7, label="Rumack-Matthew Toxic Baseline (150 mg/L @ 4h)")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)

    # Panel B: Glutathione Pool
    ax2 = axes[1]
    for idx, (label, res) in enumerate(results_map.items()):
        c = colors[idx % len(colors)]
        ax2.plot(res.time_hr, res.gsh, label=label, color=c, lw=2.2)
    ax2.axhline(CRITICAL_GSH_THRESHOLD, color="#d90429", linestyle="--", lw=1.8,
                label=f"Critical Necrosis Threshold ({CRITICAL_GSH_THRESHOLD:.0f}% GSH)")
    ax2.set_ylabel("Hepatic GSH\n(% Baseline)", fontweight="bold")
    ax2.set_title("B. Intracellular Antioxidant Depletion & Replenishment", loc="left", fontweight="bold")
    ax2.set_ylim(-2, 115)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    # Panel C: Free NAPQI
    ax3 = axes[2]
    for idx, (label, res) in enumerate(results_map.items()):
        c = colors[idx % len(colors)]
        ax3.plot(res.time_hr, res.free_napqi, label=label, color=c, lw=2.2)
    ax3.set_ylabel("Free NAPQI\n(mg/L equiv)", fontweight="bold")
    ax3.set_title("C. Unconjugated Reactive Metabolite Flux", loc="left", fontweight="bold")
    ax3.legend(loc="upper right", frameon=True, framealpha=0.9)

    # Panel D: Necrosis Index
    ax4 = axes[3]
    for idx, (label, res) in enumerate(results_map.items()):
        c = colors[idx % len(colors)]
        ax4.plot(res.time_hr, res.necrosis_index, label=label, color=c, lw=2.4)
    ax4.set_ylabel("Hepatocyte Necrosis\n(% Death)", fontweight="bold")
    ax4.set_xlabel("Time Post-Ingestion (hours)", fontweight="bold")
    ax4.set_title("D. Irreversible Parenchymal Histopathology", loc="left", fontweight="bold")
    ax4.set_ylim(-2, 102)
    ax4.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout()
    return fig


def plot_cohort_distributions(
    cohorts_dict: Dict[str, pd.DataFrame],
    metric: str = "final_necrosis_pct",
    title: str = "Monte Carlo Virtual Clinical Trials: Population Toxicity Distributions",
    ylabel: str = "Hepatocyte Necrosis Index (%)",
    figsize: Tuple[int, int] = (11, 6),
) -> plt.Figure:
    """
    Figure 2: Violin and overlaid boxplot distribution comparing virtual patient cohorts.
    """
    set_academic_style()
    fig, ax = plt.subplots(figsize=figsize)

    combined_rows = []
    for group_name, df in cohorts_dict.items():
        temp = df[[metric]].copy()
        temp["Cohort"] = group_name
        combined_rows.append(temp)

    plot_df = pd.concat(combined_rows, ignore_index=True)

    # Violin plot for KDE shape
    palette = sns.color_palette("muted", n_colors=len(cohorts_dict))
    sns.violinplot(
        data=plot_df,
        x="Cohort",
        y=metric,
        hue="Cohort",
        legend=False,
        ax=ax,
        palette=palette,
        inner=None,
        cut=0,
        alpha=0.6,
    )

    # Boxplot for quartiles and median
    sns.boxplot(
        data=plot_df,
        x="Cohort",
        y=metric,
        ax=ax,
        width=0.22,
        color="white",
        boxprops=dict(alpha=0.9),
        showcaps=True,
    )

    # Strip plot with jitter for individual patient outcomes
    sns.stripplot(
        data=plot_df,
        x="Cohort",
        y=metric,
        ax=ax,
        size=4.5,
        color="#222222",
        alpha=0.45,
        jitter=0.18,
    )

    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlabel("Simulated Clinical Cohort", fontweight="bold")
    ax.set_ylim(bottom=max(-2, plot_df[metric].min() - 5))

    fig.tight_layout()
    return fig


def plot_antidote_efficacy_curve(
    delay_hours_list: Optional[List[float]] = None,
    dose_mg: float = 12000.0,
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Figure 3: Antidote rescue window curve illustrating hepatocyte salvage vs. NAC delay.
    """
    set_academic_style()
    if delay_hours_list is None:
        delay_hours_list = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0]

    records = []
    form = Formulation.standard_napa(tablets=int(round(dose_mg / 500.0)))

    for delay in delay_hours_list:
        nac = NACIntervention(start_time_hr=delay, duration_hr=21.0, boost_rate=18.5)
        res = simulate_patient(formulation=form, initial_gsh=100.0, nac_intervention=nac)
        surviving_liver = max(0.0, 100.0 - res.final_necrosis)
        records.append({
            "Delay_hr": delay,
            "Final_Necrosis_pct": res.final_necrosis,
            "Surviving_Liver_pct": surviving_liver,
            "Min_GSH_pct": res.min_gsh,
            "Peak_APAP_mg_L": res.peak_c_plasma,
        })

    # Also compute untreated baseline
    res_no_nac = simulate_patient(formulation=form, initial_gsh=100.0, nac_intervention=None)
    records.append({
        "Delay_hr": 99.0, # Untreated sentinel
        "Final_Necrosis_pct": res_no_nac.final_necrosis,
        "Surviving_Liver_pct": max(0.0, 100.0 - res_no_nac.final_necrosis),
        "Min_GSH_pct": res_no_nac.min_gsh,
        "Peak_APAP_mg_L": res_no_nac.peak_c_plasma,
    })

    df_curve = pd.DataFrame(records)
    plot_points = df_curve[df_curve["Delay_hr"] <= 24.0]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot salvage tissue curve
    ax.plot(plot_points["Delay_hr"], plot_points["Surviving_Liver_pct"], "o-",
            color="#2a9d8f", lw=2.8, markersize=8, label="Surviving Viable Parenchyma (%)")

    # Plot necrosis curve
    ax.plot(plot_points["Delay_hr"], plot_points["Final_Necrosis_pct"], "s--",
            color="#e76f51", lw=2.4, markersize=7, label="Cumulative Necrosis (%)")

    # Highlight Rumack-Matthew therapeutic golden window (<= 8 hours)
    ax.axvspan(0, 8, color="#2a9d8f", alpha=0.15, label="Optimal Clinical Window (<= 8h Post-Ingestion)")
    ax.axvline(8.0, color="#264653", linestyle="--", lw=1.5)
    ax.text(8.2, 50, "8-Hour Rumack-Matthew\nCritical Inflection", color="#264653", fontsize=9, fontweight="bold")

    # Mark untreated reference
    untreated_surv = df_curve[df_curve["Delay_hr"] == 99.0]["Surviving_Liver_pct"].values[0]
    ax.axhline(untreated_surv, color="#777777", linestyle=":", lw=1.5,
               label=f"Untreated Overdose Survival ({untreated_surv:.1f}%)")

    ax.set_title(f"Therapeutic Window: Hepatocyte Salvage vs. NAC Administration Delay ({dose_mg:,.0f} mg Overdose)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("NAC Administration Delay (Hours Post-Ingestion)", fontweight="bold")
    ax.set_ylabel("Liver Tissue Percentage (%)", fontweight="bold")
    ax.set_ylim(-2, 105)
    ax.set_xlim(-0.5, 25.0)
    ax.legend(loc="center left", frameon=True, framealpha=0.92)

    fig.tight_layout()
    return fig, df_curve


def generate_all_publication_figures(output_dir: str = "output_figures") -> Dict[str, str]:
    """
    Executes and saves all 3 core figures to disk at 300 DPI.

    Returns dict mapping figure key to output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    # Figure 1: Timeline comparing Standard vs Extend vs Overdose
    res_std = simulate_patient(formulation=Formulation.standard_napa(tablets=2)) # 1,000 mg
    res_ext = simulate_patient(formulation=Formulation.napa_extend(tablets=2))   # 1,330 mg
    res_extra = simulate_patient(formulation=Formulation.napa_extra(tablets=20)) # 10,000 mg + caffeine
    res_acute = simulate_patient(formulation=Formulation.standard_napa(tablets=20)) # 10,000 mg

    fig1 = plot_timeline_multipanel(
        {
            "Therapeutic Napa (1,000 mg)": res_std,
            "Therapeutic Extend (1,330 mg)": res_ext,
            "Overdose Standard (10,000 mg)": res_acute,
            "Overdose Extra (10,000 mg + Caff)": res_extra,
        },
        title="Figure 1: Pharmacokinetics and Toxicodynamics Across Napa Formulations",
    )
    p1 = os.path.join(output_dir, "figure1_timeline_pkpd.png")
    fig1.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    saved_paths["figure1"] = p1

    # Figure 2: Monte Carlo population distributions
    from napa_simulation.simulation.scenarios import (
        get_acute_overdose_scenario,
        get_staggered_overdose_scenario,
        get_malnutrition_scenario,
    )
    _, df_acute = get_acute_overdose_scenario(10000.0)
    _, df_stag = get_staggered_overdose_scenario(10000.0)
    mal_scen = get_malnutrition_scenario(5000.0)

    fig2 = plot_cohort_distributions(
        {
            "Acute (10g)": df_acute,
            "Staggered (10g)": df_stag,
            "Normal (5g)": mal_scen["cohort_normal"],
            "Malnourished (5g)": mal_scen["cohort_malnourished"],
        },
        title="Figure 2: Monte Carlo Cohort Necrosis Distributions Across Clinical Populations",
    )
    p2 = os.path.join(output_dir, "figure2_cohort_distributions.png")
    fig2.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    saved_paths["figure2"] = p2

    return saved_paths
