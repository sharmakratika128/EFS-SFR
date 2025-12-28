# EFS-SFR: Ensemble Feature Selection for Sorted Feature Ranking (PE Prediction)

This repository contains the code for the paper:
**"EFS-SFR: Ensemble Feature Selection for Sorted Feature Ranking for Early Prediction of Preeclampsia"**.

## Overview
- Task: Predict **preeclampsia (PE)** vs **non-PE** using first-trimester routine clinical variables.
- Proposed method: **EFS-SFR**, an ensemble feature selection framework combining:
  - SelectKBest
  - RFE
  - RFECV
  aggregated via a sorted ranking strategy.
- Models: RF, SVM, LR, k-NN, Naive Bayes, XGBoost.
- Additional comparisons: **recent competitive baselines** for tabular prediction (e.g., CatBoost, LightGBM, Extra Trees, stacking, MLP/ANN, Decision Tree).
---
## Repository contents

- `EnsembleFeatureSelection&Prediction.py` — EFS‑SFR feature ranking + selection implementation,  train/evaluate models with **full** vs **selected** features; saves CSV outputs;  minimal synthetic (toy) run; **no patient data required**
- `evaluationOnCompetitivebaseline.py` —strong tabular baselines (XGBoost/LightGBM/CatBoost)
- `README.md` — dataset placement and privacy notes
- `requirements.txt` — core dependencies
- 'Sample_Enabled_Patients_Data_Updated.csv' - samaple csv file


---
## **Data access**

Due to patient privacy, the raw clinical dataset is not publicly included.

A schema-only sample is provided in Sample_Enabled_Patients_Data_Updated .

To request access, contact the corresponding author -  sharma.kratika128@gmail.com 
## Installation

### Option A: pip + venv (recommended)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

---
---


