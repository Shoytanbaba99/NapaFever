"""
Automated Statistical Hypothesis Testing Suite for Toxicological Investigations.

Implements rigorous inferential statistics for two core overdose damage experiments:
1. Acute Bolus Overdose vs. Staggered Ingestion Overdose (Two-tailed)
2. Well-Nourished vs. Malnourished Cohort under Borderline Dosing (Right-tailed / One-tailed)

Statistical Methods:
- Welch's Two-Sample Independent t-Test (robust against unequal variances)
- Non-parametric Mann-Whitney U Test (distribution-free rank-sum verification)
- Cohen's d Effect Size calculation and categorical magnitude interpretation
- Formal Null / Alternative Hypothesis decision rule (alpha = 0.05)
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from napa_simulation.simulation.scenarios import (
    get_acute_overdose_scenario,
    get_staggered_overdose_scenario,
    get_malnutrition_scenario,
)
from napa_simulation.simulation.engine import simulate_cohort
from napa_simulation.models.formulations import FormulationType
from napa_simulation.models.pkpd import NACIntervention


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Computes Cohen's d standardized effect size between two independent groups.

    Formula:
        d = (x_bar_1 - x_bar_2) / s_pooled
        where s_pooled = sqrt( [ (n1 - 1)*s1^2 + (n2 - 1)*s2^2 ] / (n1 + n2 - 2) )
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    pooled_variance = (((n1 - 1) * var1) + ((n2 - 1) * var2)) / (n1 + n2 - 2)
    s_pooled = np.sqrt(max(pooled_variance, 1e-12))

    mean_diff = np.mean(group1) - np.mean(group2)
    return float(mean_diff / s_pooled)


def interpret_cohens_d(d: float) -> str:
    """Categorizes effect size magnitude according to Cohen's thresholds."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible effect"
    elif abs_d < 0.5:
        return "Small effect"
    elif abs_d < 0.8:
        return "Medium effect"
    elif abs_d < 1.2:
        return "Large effect"
    else:
        return "Very Large / Extreme effect"


@dataclass
class HypothesisTestResult:
    """Encapsulates the inferential output and academic verdict of a trial."""
    experiment_id: int
    title: str
    null_hypothesis: str
    alt_hypothesis: str
    test_type: str
    group1_name: str
    group1_n: int
    group1_mean: float
    group1_std: float
    group2_name: str
    group2_n: int
    group2_mean: float
    group2_std: float
    t_statistic: float
    p_value_ttest: float
    mann_whitney_u: float
    p_value_mw: float
    cohens_d: float
    effect_size_label: str
    alpha: float
    reject_h0: bool
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Experiment": f"Exp {self.experiment_id}: {self.title}",
            "H0": self.null_hypothesis,
            "H1": self.alt_hypothesis,
            f"{self.group1_name} Mean (SD)": f"{self.group1_mean:.2f}% (±{self.group1_std:.2f})",
            f"{self.group2_name} Mean (SD)": f"{self.group2_mean:.2f}% (±{self.group2_std:.2f})",
            "t-Statistic": f"{self.t_statistic:.4f}",
            "p-Value (t-test)": f"{self.p_value_ttest:.4e}" if self.p_value_ttest < 0.0001 else f"{self.p_value_ttest:.4f}",
            "Mann-Whitney U": f"{self.mann_whitney_u:.1f}",
            "p-Value (MW-U)": f"{self.p_value_mw:.4e}" if self.p_value_mw < 0.0001 else f"{self.p_value_mw:.4f}",
            "Cohen's d": f"{self.cohens_d:.3f} ({self.effect_size_label})",
            "Verdict": "REJECT H0" if self.reject_h0 else "FAIL TO REJECT H0",
        }

    def format_defense_report(self) -> str:
        """Generates a structured terminal and publication report block."""
        border = "=" * 80
        sub_border = "-" * 80
        verdict_str = "[ REJECT H0 - STATISTICALLY SIGNIFICANT ]" if self.reject_h0 else "[ FAIL TO REJECT H0 - NOT SIGNIFICANT ]"

        report = [
            border,
            f"EXPERIMENT {self.experiment_id}: {self.title.upper()}",
            border,
            f"Null Hypothesis (H0)       : {self.null_hypothesis}",
            f"Alternative Hypothesis (H1): {self.alt_hypothesis}",
            f"Statistical Framework      : {self.test_type} (alpha = {self.alpha})",
            sub_border,
            f"Cohort 1 [{self.group1_name}]: N = {self.group1_n}, Mean Necrosis = {self.group1_mean:.2f}%, SD = {self.group1_std:.2f}%",
            f"Cohort 2 [{self.group2_name}]: N = {self.group2_n}, Mean Necrosis = {self.group2_mean:.2f}%, SD = {self.group2_std:.2f}%",
            sub_border,
            f"Welch's t-Statistic        : t = {self.t_statistic:.4f}, p-value = {self.p_value_ttest:.4e}",
            f"Mann-Whitney U Rank Sum    : U = {self.mann_whitney_u:.1f}, p-value = {self.p_value_mw:.4e}",
            f"Standardized Effect Size   : Cohen's d = {self.cohens_d:.3f} ({self.effect_size_label})",
            sub_border,
            f"DECISION                   : {verdict_str}",
            f"ACADEMIC INTERPRETATION    : {self.verdict}",
            border,
        ]
        return "\n".join(report)


def run_experiment_1_acute_vs_staggered(n_patients: int = 100, alpha: float = 0.05) -> HypothesisTestResult:
    """
    Experiment 1: Acute Bolus (10,000 mg) vs. Staggered Ingestion (2,000 mg every 2.5 h x 5).

    H0: mu_acute == mu_staggered (Mean peak necrosis is equivalent).
    H1: mu_acute != mu_staggered (Two-tailed difference in hepatocyte necrosis).
    """
    _, df_acute = get_acute_overdose_scenario(10000.0)
    _, df_staggered = get_staggered_overdose_scenario(10000.0, n_doses=5, interval_hr=2.5)

    g1 = df_acute["final_necrosis_pct"].to_numpy()
    g2 = df_staggered["final_necrosis_pct"].to_numpy()

    t_stat, p_val_ttest = stats.ttest_ind(g1, g2, equal_var=False, alternative="two-sided")
    mw_res = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    d = calculate_cohens_d(g1, g2)
    reject = bool(p_val_ttest < alpha)

    if reject:
        if np.mean(g1) > np.mean(g2):
            verdict = (
                f"Statistically significant difference detected (p = {p_val_ttest:.4e} < {alpha}). "
                f"Acute bolus produces significantly higher peak necrosis ({np.mean(g1):.2f}% vs {np.mean(g2):.2f}%) "
                f"due to rapid saturation of safe glucuronide/sulfate conjugating pathways."
            )
        else:
            verdict = (
                f"Statistically significant difference detected (p = {p_val_ttest:.4e} < {alpha}). "
                f"Staggered dosing produces prolonged toxic intermediate exposure."
            )
    else:
        verdict = f"Insufficient statistical evidence to reject H0 at alpha = {alpha}."

    return HypothesisTestResult(
        experiment_id=1,
        title="Acute Bolus vs. Staggered Overdose (10,000 mg APAP)",
        null_hypothesis="Mean peak hepatocyte necrosis in acute overdose == Mean peak necrosis in staggered overdose.",
        alt_hypothesis="Mean peak hepatocyte necrosis in acute overdose != Mean peak necrosis in staggered overdose.",
        test_type="Two-Sample Welch's t-test & Mann-Whitney U (Two-Tailed)",
        group1_name="Acute Bolus (10g)",
        group1_n=len(g1),
        group1_mean=float(np.mean(g1)),
        group1_std=float(np.std(g1, ddof=1)),
        group2_name="Staggered Ingestion (2g x 5)",
        group2_n=len(g2),
        group2_mean=float(np.mean(g2)),
        group2_std=float(np.std(g2, ddof=1)),
        t_statistic=float(t_stat),
        p_value_ttest=float(p_val_ttest),
        mann_whitney_u=float(mw_res.statistic),
        p_value_mw=float(mw_res.pvalue),
        cohens_d=d,
        effect_size_label=interpret_cohens_d(d),
        alpha=alpha,
        reject_h0=reject,
        verdict=verdict,
    )


def run_experiment_2_malnutrition(n_patients: int = 100, alpha: float = 0.05) -> HypothesisTestResult:
    """
    Experiment 2: Malnourished vs. Well-Nourished Cohort at Borderline Dosing (5,000 mg/24h).

    H0: mu_malnourished <= mu_normal (Malnutrition does not increase mean necrosis).
    H1: mu_malnourished > mu_normal (Right-tailed / one-tailed hypothesis).
    """
    scen = get_malnutrition_scenario(5000.0)
    df_norm = scen["cohort_normal"]
    df_mal = scen["cohort_malnourished"]

    g_mal = df_mal["final_necrosis_pct"].to_numpy()
    g_norm = df_norm["final_necrosis_pct"].to_numpy()

    t_stat, p_val_ttest = stats.ttest_ind(g_mal, g_norm, equal_var=False, alternative="greater")
    mw_res = stats.mannwhitneyu(g_mal, g_norm, alternative="greater")
    d = calculate_cohens_d(g_mal, g_norm)
    reject = bool(p_val_ttest < alpha)

    if reject:
        verdict = (
            f"Statistically significant evidence (p = {p_val_ttest:.4e} < {alpha}) demonstrating that "
            f"nutritional deficiency (GSH baseline ~54%) shifts therapeutic tolerance downward. "
            f"Under borderline 5,000 mg dosing, malnourished patients suffer significant necrosis ({np.mean(g_mal):.2f}%) "
            f"compared to safe baseline tolerance in well-nourished subjects ({np.mean(g_norm):.2f}%)."
        )
    else:
        verdict = f"Insufficient statistical evidence to reject H0 at alpha = {alpha}."

    return HypothesisTestResult(
        experiment_id=2,
        title="Nutritional Vulnerability at Borderline Dosing (5,000 mg APAP)",
        null_hypothesis="Malnutrition does not increase mean hepatocyte necrosis under borderline Napa dosing.",
        alt_hypothesis="Malnutrition significantly increases mean hepatocyte necrosis under borderline Napa dosing.",
        test_type="Right-Tailed Welch's t-test & Mann-Whitney U",
        group1_name="Malnourished Cohort",
        group1_n=len(g_mal),
        group1_mean=float(np.mean(g_mal)),
        group1_std=float(np.std(g_mal, ddof=1)),
        group2_name="Well-Nourished Cohort",
        group2_n=len(g_norm),
        group2_mean=float(np.mean(g_norm)),
        group2_std=float(np.std(g_norm, ddof=1)),
        t_statistic=float(t_stat),
        p_value_ttest=float(p_val_ttest),
        mann_whitney_u=float(mw_res.statistic),
        p_value_mw=float(mw_res.pvalue),
        cohens_d=d,
        effect_size_label=interpret_cohens_d(d),
        alpha=alpha,
        reject_h0=reject,
        verdict=verdict,
    )


def run_statistical_test_suite(n_patients: int = 100, alpha: float = 0.05) -> Tuple[List[HypothesisTestResult], pd.DataFrame]:
    """
    Executes both overdose damage experiments and returns structured results and summary DataFrame.
    """
    res1 = run_experiment_1_acute_vs_staggered(n_patients=n_patients, alpha=alpha)
    res2 = run_experiment_2_malnutrition(n_patients=n_patients, alpha=alpha)

    results_list = [res1, res2]
    summary_df = pd.DataFrame([r.to_dict() for r in results_list])
    return results_list, summary_df
