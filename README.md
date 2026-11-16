# Liver Damage Simulation of Napa (Paracetamol) Overdose
### Hypothesis Testing using Two-Sample t-Test

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

## 1. Project Overview

In Bangladesh, Paracetamol (Napa) is a common over-the-counter medicine. Under normal therapeutic dosing (1000 mg every 6 hours), the liver safely metabolizes the drug. However, in overdose situations (> 4000 to 10000 mg), the safe clearance pathways saturate, and the drug is shunted into Cytochrome P450 enzymes to form a toxic intermediate called NAPQI.

When the liver's natural antioxidant pool (Glutathione / GSH) is depleted below 30% of its normal baseline, NAPQI binds to liver cell proteins, causing irreversible cell death (necrosis).

This project implements a Monte Carlo simulation in Python to generate virtual patient cohorts with biological variation, followed by Two-Sample t-Tests to determine whether dosing schedule and nutritional status significantly impact liver damage.

***

## 2. Experimental Hypotheses and Results

We evaluated two distinct clinical experiments using Two-Sample t-Tests at significance level alpha = 0.05.

### Experiment 1: Acute Bolus vs. Staggered Repeated Overdose (10,000 mg Napa)
- **Problem Statement:** Does taking 10,000 mg Napa all at once (acute) cause the same average liver damage as taking 2,000 mg every 2.5 hours across 5 doses (staggered)?
- **Null Hypothesis (H0):** Mean liver necrosis in acute overdose == Mean liver necrosis in staggered overdose.
- **Alternative Hypothesis (H1):** Mean liver necrosis in acute overdose != Mean liver necrosis in staggered overdose.
- **Sample Size:** N = 30 virtual patients per group.
- **Results:**
  - Sample 1 Mean (Staggered Overdose): 69.89% (SD = 9.83%)
  - Sample 2 Mean (Acute Overdose): 19.39% (SD = 7.75%)
  - t-Statistic: 22.10
  - Degrees of Freedom: 58
  - p-Value: 6.2043e-30
  - 95% Confidence Interval: [45.92, 55.06]
- **Decision:** Reject the null hypothesis (p < 0.05).
- **Plain-English Conclusion:** Staggered repeated overdosing causes significantly higher cumulative liver damage than a single acute bolus because the liver's safe pathways remain continuously saturated over an extended time, preventing glutathione recovery.

***

### Experiment 2: Malnourished vs. Well-Nourished Cohort at Borderline Dose (5,500 mg Napa)
- **Problem Statement:** Does a borderline dose (5,500 mg) cause significantly higher liver necrosis in malnourished patients with depleted glutathione reserves compared to healthy individuals?
- **Null Hypothesis (H0):** Mean liver necrosis in malnourished patients == Mean liver necrosis in well-nourished patients.
- **Alternative Hypothesis (H1):** Mean liver necrosis in malnourished patients != Mean liver necrosis in well-nourished patients.
- **Sample Size:** N = 30 virtual patients per group.
- **Results:**
  - Sample 1 Mean (Malnourished): 4.81% (SD = 5.01%)
  - Sample 2 Mean (Well-Nourished): 0.00% (SD = 0.00%)
  - t-Statistic: 5.26
  - Degrees of Freedom: 58
  - p-Value: 2.1646e-06
  - 95% Confidence Interval: [2.98, 6.64]
- **Decision:** Reject the null hypothesis (p < 0.05).
- **Plain-English Conclusion:** Fasting or malnutrition reduces baseline glutathione reserves to around 50%, removing the protective antioxidant safety buffer. As a result, doses that cause zero necrosis in healthy adults produce significant liver cell damage in malnourished individuals.

***

## 3. How to Run

### Requirements
Install the required packages:
```bash
pip install -r requirements.txt
```

### Run the Notebook
Open `napa_simulation_lab.ipynb` in Jupyter Notebook or Google Colab and run all cells:
```bash
jupyter notebook napa_simulation_lab.ipynb
```
All outputs, t-statistics, p-values, decision statements, and error bar plots are pre-rendered and will execute directly.
