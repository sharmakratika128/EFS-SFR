#!/usr/bin/env python3
"""
scripts/run_rival_methods.py

This script:
- Loads your CSV
- Uses 10 selected features from EFS-SFR
- Runs recent competitive baselines: CatBoost, LightGBM, ExtraTrees, DecisionTree, MLP, Stacking
- Evaluates with Stratified 5-fold CV
- Reports Acc / Recall / F1 / AUROC
- Measures Train time (s) and Inference time (ms/sample)
- Exports a LaTeX table to: results/tables/table_rival_methods.tex

Example:
  python scripts/run_rival_methods.py --data data/Enabled_Patients_Data.csv
"""

import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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


# -------------------------
# Config (edit if needed)
# -------------------------
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


# -------------------------
# Helpers
# -------------------------
@dataclass
class ModelResult:
    model: str
    model_type: str
    n_feat: int
    acc: float
    recall: float
    f1: float
    auroc: float
    train_s: float
    infer_ms: float


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(["NA", "na", "Na", "", " "], np.nan)
    keep_cols = FEATURES_10 + [TARGET_COL]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df[keep_cols].copy()

    # types
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    for c in FEATURES_10:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def get_models(random_state: int = 42) -> Dict[str, Tuple[object, str]]:
    """
    Returns dict: name -> (estimator, type_label)
    """
    models: Dict[str, Tuple[object, str]] = {}

    # CatBoost
    models["CatBoost"] = (
        CatBoostClassifier(
            depth=6,
            learning_rate=0.05,
            iterations=500,
            loss_function="Logloss",
            verbose=False,
            random_seed=random_state,
        ),
        "Boosting (trees)",
    )

    # LightGBM
    models["LightGBM"] = (
        lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        ),
        "Boosting (trees)",
    )

    # Extra Trees
    models["Extra Trees"] = (
        ExtraTreesClassifier(n_estimators=500, random_state=random_state, n_jobs=-1),
        "Bagging (trees)",
    )

    # Decision Tree
    models["Decision Tree"] = (
        DecisionTreeClassifier(random_state=random_state, max_depth=5),
        "Interpretable tree",
    )

    # MLP/ANN (needs scaling)
    models["MLP/ANN"] = (
        Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        max_iter=500,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Neural network",
    )

    # Stacking
    base_estimators = [
        (
            "lr",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ]
            ),
        ),
        ("et", ExtraTreesClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)),
        ("lgbm", lgb.LGBMClassifier(n_estimators=300, random_state=random_state)),
    ]

    models["Stacking"] = (
        StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=2000),
            passthrough=False,
            n_jobs=-1,
        ),
        "Meta-ensemble",
    )

    return models


def evaluate_cv(
    models: Dict[str, Tuple[object, str]],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> List[ModelResult]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results: List[ModelResult] = []

    for name, (model, type_label) in models.items():
        accs, recs, f1s, aucs = [], [], [], []
        train_times, infer_times = [], []

        for train_idx, test_idx in skf.split(X, y):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            # training time
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            t1 = time.perf_counter()
            train_times.append(t1 - t0)

            # inference time per sample
            t2 = time.perf_counter()
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(Xte)[:, 1]
            else:
                # fallback: no probas available
                pred = model.predict(Xte)
                prob = pred.astype(float)
            t3 = time.perf_counter()
            infer_times.append((t3 - t2) * 1000.0 / len(Xte))  # ms/sample

            pred = (prob >= 0.5).astype(int)

            accs.append(accuracy_score(yte, pred))
            recs.append(recall_score(yte, pred, zero_division=0))
            f1s.append(f1_score(yte, pred, zero_division=0))
            try:
                aucs.append(roc_auc_score(yte, prob))
            except Exception:
                aucs.append(np.nan)

        results.append(
            ModelResult(
                model=name,
                model_type=type_label,
                n_feat=X.shape[1],
                acc=float(np.mean(accs)),
                recall=float(np.mean(recs)),
                f1=float(np.mean(f1s)),
                auroc=float(np.nanmean(aucs)),
                train_s=float(np.mean(train_times)),
                infer_ms=float(np.mean(infer_times)),
            )
        )

    return results


def save_latex_table(
    results: List[ModelResult],
    out_path: str = "results/tables/table_rival_methods.tex",
    caption: str = (
        "Comparative evaluation with recent competitive baselines on the proposed cohort using "
        "the EFS-SFR selected subset ($\\alpha=0.32$). \\#Feat denotes the number of input features "
        "used by each model (10 for all methods in this experiment). Results are mean values under "
        "stratified 5-fold CV. Systolic/diastolic BP may be in normalized units depending on the dataset."
    ),
    label: str = "tab:rival_methods",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # sort by AUROC desc
    results_sorted = sorted(results, key=lambda r: (r.auroc if not np.isnan(r.auroc) else -1.0), reverse=True)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4.2pt}")
    lines.append(r"\begin{tabular}{lcccccccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Model} & \textbf{Type} & \textbf{\#Feat} & \textbf{Acc.} & \textbf{Recall} & \textbf{F1} & \textbf{AUROC} & \textbf{Train (s)} & \textbf{Infer (ms)} \\")
    lines.append(r"\hline")

    for r in results_sorted:
        lines.append(
            f"{r.model} & {r.model_type} & {r.n_feat:d} & "
            f"{r.acc:.3f} & {r.recall:.3f} & {r.f1:.3f} & {r.auroc:.3f} & "
            f"{r.train_s:.3f} & {r.infer_ms:.3f} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run recent competitive baselines with stratified 5-fold CV.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--out", type=str, default="results/tables/table_rival_methods.tex", help="Output LaTeX file path.")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = _clean_df(df)

    X = df[FEATURES_10].values
    y = df[TARGET_COL].values

    print(f"Loaded: {args.data}")
    print(f"Samples: {len(df)} | PE: {int(y.sum())} | Non-PE: {int((y==0).sum())} | Features: {X.shape[1]}")

    models = get_models(random_state=args.seed)
    results = evaluate_cv(models, X, y, n_splits=args.folds, seed=args.seed)

    # Print summary
    print("\nResults (mean over folds):")
    for r in sorted(results, key=lambda rr: rr.auroc, reverse=True):
        print(
            f"{r.model:12s}  Acc={r.acc:.3f}  Recall={r.recall:.3f}  F1={r.f1:.3f}  "
            f"AUROC={r.auroc:.3f}  Train(s)={r.train_s:.3f}  Infer(ms)={r.infer_ms:.3f}"
        )

    save_latex_table(results, out_path=args.out)
    print(f"\nSaved LaTeX table to: {args.out}")


if __name__ == "__main__":
    main()
