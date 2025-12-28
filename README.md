# EFS-SFR for Early Pre-eclampsia (PE) Prediction

Research code for **early prediction of pre-eclampsia (PE)** from **routine first‑trimester clinical features** using:

- **EFS‑SFR** (Ensemble Feature Selection for Sorted Feature Ranking): an ensemble feature selection framework that combines  
  **SelectKBest**, **RFE**, and **RFECV**, then aggregates their rankings using a sorted/mean-based strategy to produce a stable, interpretable feature ranking.
- A suite of tabular ML classifiers (baseline + optional strong rivals) to compare performance **before vs after** feature selection.

> ⚠️ **Clinical disclaimer**: This repository is for research/education only and is **not** a medical device. Do not use it for clinical decisions without appropriate regulatory approvals and external validation.

---

## Method overview (EFS‑SFR)

EFS‑SFR runs three feature selection methods on the training set:

1. **SelectKBest** (filter method; configurable scoring function)
2. **RFE** (wrapper method)
3. **RFECV** (wrapper method with cross‑validation)

Each method produces a feature ranking. EFS‑SFR then aggregates these rankings using:

- **Arithmetic mean rank**
- **Geometric mean rank**

The aggregated rank is converted into a **normalized score** (higher = more important).  
Features are selected using a **threshold α** (default **α = 0.32**) and an optional **top_k** safeguard (default **top_k = 10**) to ensure a compact feature set.

---

## Reported study setting and results (from the manuscript)

- Cohort: **N = 1077** pregnancies (**PE = 84**, non‑PE = 993)
- Candidate features: **31** routine clinical/first‑trimester features
- Selected features (EFS‑SFR): **10** key predictors
- Optimal selection threshold: **α = 0.32**
- Best baseline classifier: **Random Forest** (reported accuracy **93%**, +1.97% after feature selection)
- Additional reported improvements: SVM (+3.13%), XGBoost (+1.77%), Naive Bayes (+21.15%)

These values are **reported** results; your numbers may differ depending on cohort, preprocessing, and split strategy.

---

## Repository contents

- `EnsembleFeatureSelection&Prediction.ipynb` — EFS‑SFR feature ranking + selection implementation,  train/evaluate models with **full** vs **selected** features; saves CSV outputs
- `test_main.py` — minimal synthetic (toy) run; **no patient data required**
- `data/README.md` — dataset placement and privacy notes
- `requirements.txt` — core dependencies
- `requirements-optional.txt` — optional strong tabular baselines (XGBoost/LightGBM/CatBoost)

---

## Installation

### Option A: pip + venv (recommended)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

