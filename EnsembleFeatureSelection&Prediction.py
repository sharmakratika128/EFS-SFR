#!/usr/bin/env python3
"""
EnsembleFeatureSelectionAndPrediction.py

Paper-oriented, reproducible script for:
1) EFS-SFR feature ranking + selection (SelectKBest + RFE + RFECV with sorted aggregation)
2) Model training/evaluation with FULL vs SELECTED features (Stratified 5-fold CV)
3) Saves CSV outputs + LaTeX-ready results folder
4) Includes a minimal synthetic/toy run by default (no patient data required)

USAGE (toy run; no data needed):
  python EnsembleFeatureSelectionAndPrediction.py

USAGE (with your CSV):
  python EnsembleFeatureSelectionAndPrediction.py --data Sample_Enabled_Patients_Data_Updated.csv

NOTES:
- Do NOT commit patient data to GitHub. Add data/*.csv to .gitignore.
- Update TARGET_COL / FEATURES if your CSV headers differ.
"""

import os
import time
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# -------------------------
# Default config (edit if needed)
# -------------------------
TARGET_COL = "Preeclampsia(1)/NA(0)"

# Use these only when --data is provided and you want a fixed 10-feature setting.
# Otherwise the script will infer features as all numeric columns except target.
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
# Utilities
# -------------------------
EPS = 1e-12


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmin(x))
    mn, mx = float(np.min(x)), float(np.max(x))
    if abs(mx - mn) < EPS:
        return np.ones_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def ranking_to_score(ranking: np.ndarray) -> np.ndarray:
    """
    Convert sklearn-style rankings (1=best) to [0,1] score where 1 is best.
    """
    r = np.asarray(ranking, dtype=float)
    n = len(r)
    if n <= 1:
        return np.ones_like(r)
    return 1.0 - (r - 1.0) / (n - 1.0)


def geometric_mean_scores(scores: np.ndarray) -> float:
    s = np.clip(scores, EPS, 1.0)
    return float(np.exp(np.mean(np.log(s))))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class FeatureRankingResult:
    ranking_df: pd.DataFrame
    selected_features: List[str]


@dataclass
class FoldMetrics:
    acc: float
    recall: float
    f1: float
    auroc: float
    train_s: float
    infer_ms: float


# -------------------------
# EFS-SFR implementation
# -------------------------
def efs_sfr_rank(
    X: pd.DataFrame,
    y: np.ndarray,
    alpha: float = 0.32,
    top_k: Optional[int] = 10,
    random_state: int = 42,
) -> FeatureRankingResult:
    """
    EFS-SFR:
      - SelectKBest (mutual information, k='all') -> normalized importance
      - RFE (LogReg) -> ranking -> score
      - RFECV (LogReg) -> ranking -> score
      - Aggregate via arithmetic + geometric mean; final_score=(arith+geom)/2
      - Select features by:
          * if top_k is not None: top_k by final_score
          * else: final_score >= alpha

    Returns:
      ranking_df: full table of feature scores/ranks
      selected_features: selected feature list
    """
    feature_names = list(X.columns)
    Xv = X.values

    # Base estimator for RFE/RFECV
    base_est = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    )

    # 1) SelectKBest scores (k='all' to score every feature)
    skb = SelectKBest(score_func=mutual_info_classif, k="all")
    skb.fit(Xv, y)
    skb_scores = minmax01(skb.scores_)

    # 2) RFE rankings -> scores
    rfe = RFE(estimator=base_est, n_features_to_select=max(1, int(len(feature_names) / 2)))
    rfe.fit(Xv, y)
    rfe_scores = ranking_to_score(rfe.ranking_)

    # 3) RFECV rankings -> scores (uses internal CV on training data)
    rfecv = RFECV(
        estimator=base_est,
        step=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring="roc_auc",
        min_features_to_select=1,
    )
    rfecv.fit(Xv, y)
    rfecv_scores = ranking_to_score(rfecv.ranking_)

    # Aggregation (arith + geom)
    stacked = np.vstack([skb_scores, rfe_scores, rfecv_scores]).T
    arith = stacked.mean(axis=1)

    geom = np.array([geometric_mean_scores(stacked[i, :]) for i in range(stacked.shape[0])], dtype=float)
    final = 0.5 * (arith + geom)

    # Build ranking df
    df_rank = pd.DataFrame(
        {
            "Feature": feature_names,
            "Score_SelectKBest": skb_scores,
            "Score_RFE": rfe_scores,
            "Score_RFECV": rfecv_scores,
            "ArithMean": arith,
            "GeomMean": geom,
            "FinalScore": final,
        }
    ).sort_values("FinalScore", ascending=False)

    df_rank["Rank"] = np.arange(1, len(df_rank) + 1)

    # Selection rule
    if top_k is not None:
        selected = df_rank.head(int(top_k))["Feature"].tolist()
    else:
        selected = df_rank[df_rank["FinalScore"] >= float(alpha)]["Feature"].tolist()

    return FeatureRankingResult(ranking_df=df_rank, selected_features=selected)


# -------------------------
# Models + evaluation
# -------------------------
def get_models(seed: int = 42) -> Dict[str, object]:
    """
    Baseline models used in your paper.
    (You already have competitive baselines in evaluationOnCompetitivebaseline.py)
    """
    models: Dict[str, object] = {}

    models["RF"] = RandomForestClassifier(
        n_estimators=500, random_state=seed, n_jobs=-1, class_weight="balanced"
    )

    models["SVM"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        ]
    )

    models["LR"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=seed)),
        ]
    )

    models["k-NN"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ]
    )

    models["Naive Bayes"] = GaussianNB()

    if _HAS_XGB:
        # scale_pos_weight improves training on imbalanced classes
        models["XGBoost"] = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        )
    return models


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # fallback
    pred = model.predict(X).astype(float)
    return pred


def eval_one_fold(model, Xtr, ytr, Xte, yte) -> FoldMetrics:
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    t1 = time.perf_counter()
    train_s = t1 - t0

    t2 = time.perf_counter()
    prob = _predict_proba(model, Xte)
    t3 = time.perf_counter()
    infer_ms = (t3 - t2) * 1000.0 / max(1, len(Xte))

    pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(yte, pred)
    rec = recall_score(yte, pred, zero_division=0)
    f1 = f1_score(yte, pred, zero_division=0)
    try:
        auc = roc_auc_score(yte, prob)
    except Exception:
        auc = float("nan")

    return FoldMetrics(acc=acc, recall=rec, f1=f1, auroc=auc, train_s=train_s, infer_ms=infer_ms)


def evaluate_full_vs_selected(
    X: pd.DataFrame,
    y: np.ndarray,
    alpha: float = 0.32,
    top_k: int = 10,
    folds: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 5-fold CV:
      - FULL: train/eval using all features
      - SELECTED: in each fold, compute EFS-SFR on training fold only, select top_k,
                  then train/eval using selected features on that fold.
    Returns:
      metrics_full_df, metrics_sel_df
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    models = get_models(seed=seed)

    full_rows = []
    sel_rows = []

    for model_name, model in models.items():
        fold_metrics_full = []
        fold_metrics_sel = []

        for tr_idx, te_idx in skf.split(X.values, y):
            Xtr_df, Xte_df = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]

            # FULL
            m_full = eval_one_fold(model, Xtr_df.values, ytr, Xte_df.values, yte)
            fold_metrics_full.append(m_full)

            # SELECTED (fit selector on training fold only)
            fs = efs_sfr_rank(Xtr_df, ytr, alpha=alpha, top_k=top_k, random_state=seed)
            sel_feats = fs.selected_features

            Xtr_sel = Xtr_df[sel_feats].values
            Xte_sel = Xte_df[sel_feats].values

            # Create a fresh model instance per fold to avoid leakage/state issues
            # (pipelines/classifiers keep fitted state)
            models2 = get_models(seed=seed)
            model2 = models2[model_name]

            m_sel = eval_one_fold(model2, Xtr_sel, ytr, Xte_sel, yte)
            fold_metrics_sel.append(m_sel)

        # Aggregate over folds
        def agg(ms: List[FoldMetrics]) -> Dict[str, float]:
            return {
                "Acc": float(np.mean([m.acc for m in ms])),
                "Recall": float(np.mean([m.recall for m in ms])),
                "F1": float(np.mean([m.f1 for m in ms])),
                "AUROC": float(np.nanmean([m.auroc for m in ms])),
                "Train(s)": float(np.mean([m.train_s for m in ms])),
                "Infer(ms)": float(np.mean([m.infer_ms for m in ms])),
            }

        a_full = agg(fold_metrics_full)
        a_sel = agg(fold_metrics_sel)

        full_rows.append({"Model": model_name, "#Feat": X.shape[1], **a_full})
        sel_rows.append({"Model": model_name, "#Feat": top_k, **a_sel})

    return pd.DataFrame(full_rows), pd.DataFrame(sel_rows)


# -------------------------
# Data loading (CSV or toy)
# -------------------------
def load_data(
    csv_path: Optional[str],
    target_col: str,
    features: Optional[List[str]],
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    If csv_path is None -> toy synthetic dataset.
    If csv_path provided -> read CSV and keep numeric features.
    """
    set_seed(seed)

    if csv_path is None:
        X_arr, y_arr = make_classification(
            n_samples=400,
            n_features=31,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=2,
            weights=[0.92, 0.08],
            class_sep=1.0,
            random_state=seed,
        )
        cols = [f"f{i:02d}" for i in range(X_arr.shape[1])]
        X_df = pd.DataFrame(X_arr, columns=cols)
        y = y_arr.astype(int)
        return X_df, y

    df = pd.read_csv(csv_path)
    df = df.replace(["NA", "na", "Na", "", " "], np.nan)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)

    # Choose feature columns
    if features is not None:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing requested features in CSV: {missing}")
        X_df = df[features].copy()
    else:
        # all numeric columns except target
        num_cols = []
        for c in df.columns:
            if c == target_col:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                num_cols.append(c)
        X_df = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows with NA in any feature
    X_df = X_df.dropna().copy()
    y = df.loc[X_df.index, target_col].values.astype(int)

    return X_df, y


# -------------------------
# Main
# -------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="EFS-SFR feature selection + full vs selected model evaluation."
    )
    parser.add_argument("--data", type=str, default=None, help="Path to CSV (optional). If omitted, runs a toy synthetic demo.")
    parser.add_argument("--target", type=str, default=TARGET_COL, help="Target column name in CSV.")
    parser.add_argument("--use_features10", action="store_true", help="If set, use the fixed 10-feature list FEATURES_10 (for your PE CSV).")
    parser.add_argument("--alpha", type=float, default=0.32, help="Selection threshold (used if --top_k is None).")
    parser.add_argument("--top_k", type=int, default=10, help="Select top_k features by EFS-SFR FinalScore.")
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory for CSV files.")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "tables"))

    features = FEATURES_10 if args.use_features10 else None
    X_df, y = load_data(args.data, args.target, features, seed=args.seed)

    print("\n=== Dataset summary ===")
    print(f"Samples: {len(X_df)} | Features: {X_df.shape[1]} | Positives: {int(y.sum())} | Negatives: {int((y==0).sum())}")
    print(f"Using fixed FEATURES_10: {bool(args.use_features10)}")
    if args.data is None:
        print("Running TOY synthetic demo (no patient data).")
    else:
        print(f"Loaded data from: {args.data}")

    # Global EFS-SFR ranking on full data (for reporting)
    fs_global = efs_sfr_rank(X_df, y, alpha=args.alpha, top_k=args.top_k, random_state=args.seed)
    ranking_csv = os.path.join(args.outdir, "efs_sfr_feature_ranking.csv")
    fs_global.ranking_df.to_csv(ranking_csv, index=False)

    selected_csv = os.path.join(args.outdir, "efs_sfr_selected_features.csv")
    pd.DataFrame({"SelectedFeature": fs_global.selected_features}).to_csv(selected_csv, index=False)

    print("\n=== EFS-SFR selected features (global) ===")
    print(fs_global.selected_features)
    print(f"Saved ranking: {ranking_csv}")
    print(f"Saved selected list: {selected_csv}")

    # Evaluate models full vs selected (fold-wise fair selection)
    metrics_full, metrics_sel = evaluate_full_vs_selected(
        X_df, y,
        alpha=args.alpha,
        top_k=args.top_k,
        folds=args.folds,
        seed=args.seed,
    )

    full_csv = os.path.join(args.outdir, "model_metrics_full_features.csv")
    sel_csv = os.path.join(args.outdir, "model_metrics_selected_features.csv")
    metrics_full.to_csv(full_csv, index=False)
    metrics_sel.to_csv(sel_csv, index=False)

    print("\n=== Metrics (FULL features) ===")
    print(metrics_full.sort_values("AUROC", ascending=False).to_string(index=False))

    print("\n=== Metrics (SELECTED features) ===")
    print(metrics_sel.sort_values("AUROC", ascending=False).to_string(index=False))

    print(f"\nSaved metrics: {full_csv}")
    print(f"Saved metrics: {sel_csv}")

    # Optional: export a LaTeX table for selected-features results
    latex_path = os.path.join(args.outdir, "tables", "table_full_vs_selected.tex")
    def _to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
        # format numeric cols
        df2 = df.copy()
        for c in ["Acc", "Recall", "F1", "AUROC", "Train(s)", "Infer(ms)"]:
            if c in df2.columns:
                df2[c] = df2[c].map(lambda v: f"{float(v):.3f}")
        return (
            "\\begin{table*}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\scriptsize\n"
            "\\setlength{\\tabcolsep}{4.5pt}\n"
            + df2.to_latex(index=False, escape=False)
            + "\\end{table*}\n"
        )

    latex_text = _to_latex(
        metrics_sel.sort_values("AUROC", ascending=False),
        caption=(
            "Model performance using EFS-SFR selected features (mean values under stratified 5-fold CV). "
            "Training and inference times are measured on the same machine for all methods."
        ),
        label="tab:efs_sfr_selected_results",
    )
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_text)

    print(f"Saved LaTeX table: {latex_path}")


if __name__ == "__main__":
    main()
