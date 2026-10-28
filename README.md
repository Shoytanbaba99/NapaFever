# Liver Damage Simulation of Napa (Paracetamol) Overdose
### A Pharmacokinetic-Pharmacodynamic (PK-PD) Compartmental Model with Monte Carlo Variations and Statistical Hypothesis Testing

***

### Academic Metadata
- **Course:** CSE 413: Simulation & Modeling Lab
- **Section:** 7B
- **Team Name:** K.G.B.P
- **Authors:**
  - Mehedi Hassan Bhuiyan (ID: 0432320005101080)
  - Md. Masud Rahman (ID: 0432320005101064)
  - Hasib Al Mahmud Siddique (ID: 0432320005101095)
  - Rudro Antony Mrong (ID: 0432320005101059)

***

## 1. Introduction and Toxicological Background

Paracetamol (N-acetyl-p-aminophenol / APAP) is marketed extensively in Bangladesh under the trade name Napa (Beximco Pharmaceuticals Ltd.). Clinical use spans three dominant oral formulations:
1. Standard Napa: 500 mg immediate-release tablet.
2. Napa Extend: 665 mg bi-layer modified-release tablet (36% immediate-release fraction, 64% sustained-release erosion matrix active over 6 to 8 hours).
3. Napa Extra: 500 mg Paracetamol compounded with 65 mg Caffeine; caffeine acts as an enzymatic modulator and inducer of Cytochrome P450 enzymes.

Under therapeutic dosing (1000 mg every 4 to 6 hours, max 4000 mg/day), 85% to 90% of paracetamol undergoes non-toxic hepatic phase II conjugation into paracetamol-glucuronide and paracetamol-sulfate. A minor fraction (5% to 10%) is oxidized via Cytochrome P450 2E1 (CYP2E1) into the reactive electrophile N-acetyl-p-benzoquinone imine (NAPQI). Under physiological conditions, endogenous Glutathione (GSH) detoxifies NAPQI via GST conjugation into inert mercapturic acid conjugates.

In acute or staggered overdose (> 4000 to 10000 mg), hepatic glucuronosyltransferases and sulfotransferases undergo capacity-limited saturation. The metabolic burden shunts toward CYP2E1 oxidation, generating toxic concentrations of NAPQI. Once intracellular GSH reserves deplete below 30% of normal baseline, free NAPQI binds covalently to nucleophilic cysteine sulfhydryl groups on hepatic mitochondrial proteins. This triggers mitochondrial permeability transition pore opening, collapse of ATP synthesis, and centrilobular hepatocyte necrosis.

This project delivers a multi-compartmental systems pharmacology model integrating ordinary differential equations (ODEs), Monte Carlo population sampling, and formal inferential statistical hypothesis testing.

***

## 2. Mathematical Formulation and System Dynamics

The system is formulated as a 6-compartment coupled non-linear ODE system solved over t in [0, 72] hours using adaptive Runge-Kutta 4th/5th order numerical integration (scipy.integrate.solve_ivp, RK45).

### 2.1 State Vector Definition
The dynamic state vector is defined as:
```text
y(t) = [ A_gut_imm(t), A_gut_sust(t), C_plasma(t), GSH(t), Free_NAPQI(t), Necrosis_Index(t) ]
```

- A_gut_imm: Paracetamol mass in immediate gastrointestinal solution (mg)
- A_gut_sust: Paracetamol mass remaining in sustained-release matrix depot (mg)
- C_plasma: Paracetamol concentration in central vascular compartment (mg/L)
- GSH: Hepatic reduced Glutathione pool relative to physiological baseline (percent, baseline = 100%)
- Free_NAPQI: Unconjugated reactive electrophilic metabolite concentration (mg/L equivalent)
- Necrosis_Index: Cumulative irreversible hepatocyte loss index (0% to 100%)

### 2.2 Differential Equations

#### Gastrointestinal Dissolution and Absorption
Immediate and sustained depot absorption rates follow first-order release:

dA_gut_imm / dt = - k_a_imm * A_gut_imm

dA_gut_sust / dt = - k_a_sust * A_gut_sust

R_abs(t) = (k_a_imm * A_gut_imm) + (k_a_sust * A_gut_sust)

Where k_a_imm = 1.45 h^-1 and k_a_sust = 0.22 h^-1.

#### Hepatic Metabolism and Central Clearance
Safe elimination routes are saturable enzymes governed by Michaelis-Menten kinetics:

- Glucuronidation: R_gluc = (Vmax_gluc * C_plasma) / (Km_gluc + C_plasma)
  [Vmax_gluc = 700.0 mg/h, Km_gluc = 65.0 mg/L]
- Sulfation: R_sulf = (Vmax_sulf * C_plasma) / (Km_sulf + C_plasma)
  [Vmax_sulf = 180.0 mg/h, Km_sulf = 16.0 mg/L]
- Renal Filtration: R_renal = k_el * V_d * C_plasma
  [k_el = 0.035 h^-1, V_d = 50.0 L]
- CYP2E1 Bioactivation: R_cyp = k_cyp2e1 * V_d * C_plasma
  [k_cyp2e1 = 0.045 h^-1]

The central compartment concentration changes according to:

dC_plasma / dt = [ R_abs(t) - (R_gluc + R_sulf + R_renal + R_cyp) ] / V_d

#### Reactive Intermediate and Glutathione Dynamics
NAPQI generation rate:
r_gen_napqi = 2.1 * k_cyp2e1 * C_plasma

Detoxification rate via GST conjugation:
r_detox = k_detox * Free_NAPQI * (GSH / 100.0)
[k_detox = 0.75 (mg/L)^-1 * h^-1]

Glutathione replenishment incorporates endogenous zero/fractional turnover and exogenous N-acetylcysteine (NAC) infusion:
r_synth = alpha_gsh * max(0, 1 - GSH / GSH_baseline) + r_NAC(t)
[alpha_gsh = 4.0 %/h, r_NAC = 35.0 %/h during active antidote infusion]

The rate of change of GSH and Free NAPQI:
dGSH / dt = r_synth - [ r_gen_napqi * 2.3 * (GSH / (10.0 + GSH)) ]

dFree_NAPQI / dt = r_gen_napqi - r_detox - 0.25 * Free_NAPQI

#### Irreversible Hepatocyte Necrosis
When intracellular GSH falls below the critical threshold (GSH < 30%), macromolecular arylation proceeds:

If GSH < 30.0:
  vulnerability = ((30.0 - GSH) / 30.0)^2
  dNecrosis / dt = k_necrosis * Free_NAPQI * vulnerability * (1 - Necrosis / 100.0)
Else:
  dNecrosis / dt = 0

[k_necrosis = 0.32 (mg/L)^-1 * h^-1]

***

## 3. Product Formulations and Pharmacokinetics

| Product Variant | Paracetamol / Tab | Caffeine / Tab | Dissolution Architecture | Kinetic Adjustment |
| :--- | :---: | :---: | :--- | :--- |
| Standard Napa | 500 mg | 0 mg | 100% Immediate dissolution | k_a_imm = 1.45 h^-1, k_a_sust = 0.0 |
| Napa Extend | 665 mg | 0 mg | Bi-layer (36% bolus, 64% matrix erosion) | 36% immediate, 64% sustained (k_a_sust = 0.22 h^-1) |
| Napa Extra | 500 mg | 65 mg | Immediate release + Caffeine | k_cyp2e1 scaled by 1.22 (+22% reactive flux) |

***

## 4. Monte Carlo Stochastic Population Modeling

Deterministic differential equations describe an idealized single patient. To simulate clinical variability across human populations, Monte Carlo sampling instantiates virtual cohorts (N = 100 individuals per cohort) by sampling parameters from physiological distributions:
- Central volume of distribution: V_d ~ N(50.0, 5.0) L
- Renal clearance constant: k_el ~ N(0.035, 0.005) h^-1
- CYP2E1 activity: k_cyp2e1 ~ N(0.045, 0.006) h^-1
- Baseline GSH (Well-Nourished): GSH_base ~ N(100.0, 7.5)%
- Baseline GSH (Malnourished / Fasted): GSH_base ~ N(50.0, 7.0)%

***

## 5. Statistical Hypothesis Testing Suite

Two distinct virtual clinical trials were conducted to evaluate toxicological damage hypotheses. In each experiment, continuous patient outcomes (final hepatocyte necrosis index at 72 hours) were analyzed using Welch's two-sample t-test (accounting for unequal variances), the non-parametric Mann-Whitney U test, and Cohen's d standardized effect size (significance threshold alpha = 0.05).

### Summary of Experimental Outcomes

| Trial | Investigated Regimens | Null Hypothesis (H0) | Welch's t | p-value (t-test) | Cohen's d | Decision |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Exp 1 | Acute Bolus (10g) vs. Staggered Ingestion (2g x 5) | Mean necrosis acute == Mean necrosis staggered | -28.52 | 5.11e-72 | -4.033 | Reject H0 |
| Exp 2 | Malnourished vs. Well-Nourished at Borderline Dosing (5g) | Malnutrition does not increase mean necrosis | +6.54 | 1.33e-09 | +0.926 | Reject H0 |

### Detailed Experimental Analysis

#### Experiment 1: Acute Bolus vs. Staggered Ingestion (10,000 mg APAP)
- Cohort 1: Acute bolus ingestion (10000 mg at t=0). Mean Necrosis = 17.94% (SD = 10.35%).
- Cohort 2: Staggered ingestion (2000 mg every 2.5 hours x 5 doses). Mean Necrosis = 59.16% (SD = 10.10%).
- Inferential Statistics: t = -28.52, p = 5.11e-72; Mann-Whitney U = 68.0, p = 1.95e-33; Cohen's d = -4.033.
- Scientific Conclusion: H0 is rejected. Staggered ingestion sustains plasma paracetamol concentrations above clearance saturation thresholds over a 15-hour period. This prevents the physiological replenishment of glutathione and leads to a significantly greater cumulative necrotic index than a single acute bolus of equivalent total mass.

#### Experiment 2: Nutritional Vulnerability under Borderline Dosing (5,000 mg APAP)
- Cohort 1: Malnourished cohort (baseline GSH ~ 50%). Mean Necrosis = 0.98% (SD = 1.50%).
- Cohort 2: Well-nourished cohort (baseline GSH ~ 100%). Mean Necrosis = 0.00% (SD = 0.00%).
- Inferential Statistics: t = +6.54, p = 1.33e-09; Mann-Whitney U = 6750.0, p = 2.06e-08; Cohen's d = +0.926.
- Scientific Conclusion: H0 is rejected. Fasting or chronic malnutrition suppresses basal glutathione reserves, reducing the metabolic buffer capacity. Borderline doses that produce zero hepatotoxicity in well-nourished individuals cause significant hepatocellular injury in depleted patients.

***

## 6. Execution Instructions

### Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Standalone CLI Verification
```bash
python main.py
```
Executes single-patient deterministic trajectories, runs the Monte Carlo virtual clinical trials, prints statistical hypothesis testing tables, and exports high-resolution 300 DPI plots into `output_figures/`.

### Run Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Launches the interactive dashboard allowing parameter manipulation across formulation selection, tablet counts, and nutritional baselines.

### Run in Jupyter Notebook or Google Colab
```bash
jupyter notebook napa_simulation_lab.ipynb
```
An interactive Jupyter Notebook (`napa_simulation_lab.ipynb`) is provided for step-by-step laboratory demonstration, interactive in-cell sliders (`ipywidgets`), and Google Colab execution.
