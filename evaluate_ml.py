"""
CKD Random Forest Classifier - Evaluation Script
=================================================
Evaluates the trained Random Forest model on the held-out test set
and via 5-fold stratified cross-validation.

Metrics: Accuracy, Precision, Recall (Sensitivity), F1, ROC-AUC
Plots  : Confusion Matrix, ROC Curve, SHAP Feature Importance, SHAP Beeswarm

Usage
-----
  python evaluate_ml.py

Output
------
  Console : per-metric scores for test set and cross-validation
  reports/ : confusion_matrix.png, roc_curve.png, shap_importance.png, shap_beeswarm.png
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve,
)

from src.ml_model import MLModelPipeline, TOP_FEATURES

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(pipeline: MLModelPipeline):
    """
    Run the full preprocessing pipeline identical to training,
    returning feature matrix X and target vector y.
    """
    df = pipeline.load_and_prepare_dataset()
    df = pipeline.handling_missing_values(df)

    # Normalise 'notpresent' -> 'absent' (same as train())
    categorical_cols = [c for c in df.columns if df[c].dtype == "object"]
    df[categorical_cols] = df[categorical_cols].apply(
        lambda s: np.where(s == "notpresent", "absent", s)
    )

    # Encode with the saved (fitted) encoders - do NOT re-fit
    encoders = joblib.load("./data/encoders.pkl")
    for col in categorical_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    X = df[TOP_FEATURES]
    y = df["class"].dropna().astype(int)
    X = X.loc[y.index]

    return X, y


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_metrics(y_true, y_pred, y_prob, label: str) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)

    print(f"\n{'-'*50}")
    print(f"  {label}")
    print(f"{'-'*50}")
    print(f"  {'Accuracy':<30} {acc:.4f}")
    print(f"  {'Precision':<30} {prec:.4f}")
    print(f"  {'Recall  (Sensitivity)':<30} {rec:.4f}")
    print(f"  {'F1-Score':<30} {f1:.4f}")
    print(f"  {'ROC-AUC':<30} {auc:.4f}")
    print(f"{'-'*50}")

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)


def plot_confusion_matrix(y_true, y_pred, path: str):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No CKD", "CKD"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix - Test Set")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_roc_curve(y_true, y_prob, auc: float, path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve - CKD Classifier")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_shap(model, X_test: pd.DataFrame, save_dir: str):
    print("\n  Computing SHAP values (TreeExplainer)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values may be a list [cls0, cls1] or a 3D array (samples, features, classes)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Bar plot (mean |SHAP|)
    mean_abs  = np.abs(sv).mean(axis=0)
    order     = np.argsort(mean_abs)
    features  = np.array(TOP_FEATURES)[order].tolist()
    values    = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(features, values, color="steelblue")
    ax.set_xlabel("Mean |SHAP Value| (impact on CKD prediction)")
    ax.set_title("Feature Importance - SHAP")
    fig.tight_layout()
    bar_path = os.path.join(save_dir, "shap_importance.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {bar_path}")

    # Beeswarm / summary plot
    shap.summary_plot(sv, X_test, feature_names=TOP_FEATURES, show=False)
    beeswarm_path = os.path.join(save_dir, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {beeswarm_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 50)
    print("  CKD Classifier - SHAP + Metrics Evaluation")
    print("=" * 50)

    pipeline = MLModelPipeline(verbose=False)

    print("\nPreparing dataset...")
    X, y = prepare_dataset(pipeline)
    print(f"  Samples : {len(X)}  |  CKD : {int(y.sum())}  |  No CKD : {int((y == 0).sum())}")

    model = joblib.load("./data/kidney_disease_rf_model.pkl")
    print(f"  Model   : RandomForestClassifier  "
          f"(n_estimators={model.n_estimators})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = print_metrics(y_test, y_pred, y_prob, f"Test Set (n={len(X_test)})")

    print("\nGenerating plots...")
    plot_confusion_matrix(
        y_test, y_pred,
        os.path.join(REPORTS_DIR, "confusion_matrix.png")
    )
    plot_roc_curve(
        y_test, y_prob, metrics["roc_auc"],
        os.path.join(REPORTS_DIR, "roc_curve.png")
    )

    print("\nRunning 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_result = cross_validate(
        model, X, y,
        cv=skf,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=False,
        n_jobs=-1,
    )

    print(f"\n{'-'*50}")
    print("  5-Fold Cross-Validation")
    print(f"{'-'*50}")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        scores = cv_result[f"test_{metric}"]
        label  = metric.replace("_", " ").capitalize()
        print(f"  {label:<30} {scores.mean():.4f} +/- {scores.std():.4f}")
    print(f"{'-'*50}")

    plot_shap(model, X_test, REPORTS_DIR)

    print(f"\nAll reports saved to: {REPORTS_DIR}/")
    print("Evaluation complete.\n")


if __name__ == "__main__":
    main()
