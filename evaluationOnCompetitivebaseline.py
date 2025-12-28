This code is designed to:

* Load your CSV
* Use  **10 selected features from EFS - SFR ** 
* Run **recent competitive baselines**: CatBoost, LightGBM, ExtraTrees, DecisionTree, MLP, Stacking
* Evaluate with **Stratified 5-fold CV**
* Report **Acc / Recall / F1 / AUROC**
* Also report **Train time (s)** and **Inference time (ms/sample)**
* Export a **LaTeX table** that you can paste into your paper (`results/tables/table_rival_methods.tex`)




## Cell 1 — Install dependencies (Colab)

```python
!pip -q install numpy pandas scikit-learn matplotlib pyyaml joblib xgboost lightgbm catboost
```

---

## Cell 2 — Imports + settings

import os, time, json, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from catboost import CatBoostClassifier
```

---

## Cell 3 — Mount Drive (optional) OR set local path

If your CSV is in GitHub repo, download or upload. Easiest: upload file directly in Colab.

```python
from google.colab import files
uploaded = files.upload()  # choose your CSV
```

Then set CSV name (after upload):

```python
CSV_PATH = next(iter(uploaded.keys()))
print("Using:", CSV_PATH)
```

---

## Cell 4 — Load data + select features


```python
TARGET_COL = "Preeclampsia(1)/NA(0)"

FEATURES_10 = [
    "Age",
    "Weight before pregnancy",
    "Weight in late first trimester",
    "BMI before pregnancy",
    "F B-hCG (ng/ml)",
    "BMI in late first trimester",
    "Hemoglobin level measured in the first trimester",
    "DBP in the late first trimester",
    "CRL",
    "BPD",
]

df = pd.read_csv(CSV_PATH)

# basic cleanup
df = df.replace(["NA", "na", "Na", "", " "], np.nan)

# keep only needed cols
keep_cols = FEATURES_10 + [TARGET_COL]
df = df[keep_cols].copy()

# drop rows with missing values (you can replace with imputation later)
df = df.dropna().copy()

# types
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(int)
for c in FEATURES_10:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna().copy()

X = df[FEATURES_10].values
y = df[TARGET_COL].values

print("Final shape:", X.shape, "PE cases:", int(y.sum()), "Non-PE:", int((y==0).sum()))
```

---

## Cell 5 — Define models (recent competitive baselines)

```python
def get_models(random_state=42):
    models = {}

    # CatBoost (good for tabular)
    models["CatBoost"] = CatBoostClassifier(
        depth=6, learning_rate=0.05, iterations=500,
        loss_function="Logloss", verbose=False,
        random_seed=random_state
    )

    # LightGBM
    models["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=random_state
    )

    # Extra Trees
    models["Extra Trees"] = ExtraTreesClassifier(
        n_estimators=500, random_state=random_state, n_jobs=-1
    )

    # Decision Tree (interpretable baseline)
    models["Decision Tree"] = DecisionTreeClassifier(
        random_state=random_state, max_depth=5
    )

    # MLP / ANN (needs scaling)
    models["MLP/ANN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            max_iter=500, random_state=random_state
        ))
    ])

    # Stacking (meta-ensemble)
    base_estimators = [
        ("lr", Pipeline([("scaler", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])),
        ("et", ExtraTreesClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)),
        ("lgbm", lgb.LGBMClassifier(n_estimators=300, random_state=random_state)),
    ]
    models["Stacking"] = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        passthrough=False,
        n_jobs=-1
    )

    return models

models = get_models()
list(models.keys())
```

---

## Cell 6 — CV evaluation with timing

```python
def evaluate_cv(models, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results = []

    for name, model in models.items():
        accs, recs, f1s, aucs = [], [], [], []
        train_times, infer_times = [], []

        for train_idx, test_idx in skf.split(X, y):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            # train time
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            t1 = time.perf_counter()
            train_times.append(t1 - t0)

            # inference time per sample
            t2 = time.perf_counter()
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(Xte)[:, 1]
            else:
                # fallback
                prob = model.predict(Xte)
            t3 = time.perf_counter()
            infer_times.append((t3 - t2) * 1000 / len(Xte))  # ms/sample

            pred = (prob >= 0.5).astype(int)

            accs.append(accuracy_score(yte, pred))
            recs.append(recall_score(yte, pred, zero_division=0))
            f1s.append(f1_score(yte, pred, zero_division=0))
            try:
                aucs.append(roc_auc_score(yte, prob))
            except:
                aucs.append(np.nan)

        results.append({
            "Model": name,
            "#Feat": X.shape[1],
            "Acc": np.mean(accs),
            "Recall": np.mean(recs),
            "F1": np.mean(f1s),
            "AUROC": np.nanmean(aucs),
            "Train (s)": np.mean(train_times),
            "Infer (ms)": np.mean(infer_times),
        })

    return pd.DataFrame(results)

res_df = evaluate_cv(models, X, y)
res_df
```

---

## Cell 7 — Format results nicely

```python
def fmt(df):
    out = df.copy()
    for c in ["Acc","Recall","F1","AUROC"]:
        out[c] = out[c].map(lambda x: f"{x:.3f}")
    out["Train (s)"] = out["Train (s)"].map(lambda x: f"{x:.3f}")
    out["Infer (ms)"] = out["Infer (ms)"].map(lambda x: f"{x:.3f}")
    return out.sort_values("AUROC", ascending=False)

fmt_df = fmt(res_df)
fmt_df
```

---





