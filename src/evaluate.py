"""
Evaluation and Diagnostic Module for Scholarship Eligibility Prediction.

This module provides:
1. Performance metric computation (Accuracy, Precision, Recall, F1-Score)
2. Confusion matrix generation and visualization for all 4 models
3. Model comparison bar chart visualization
4. Feature importance extraction and visualization for Random Forest
5. Exporting metrics to CSV and JSON formats
"""

import json
import os
import sys
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.feature_engineering import get_feature_names_after_preprocessing


PLOTS_DIR = os.path.join("outputs", "plots")
METRICS_DIR = os.path.join("outputs", "metrics")


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    pos_label: str = "Eligible"
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for a model.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["Not Eligible", "Eligible"])

    return {
        "Model": model_name,
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1-Score": round(float(f1), 4),
        "Confusion_Matrix": cm.tolist()
    }


def plot_confusion_matrices(
    evaluation_results: List[Dict[str, Any]],
    output_dir: str = PLOTS_DIR
) -> str:
    """
    Plot 2x2 grid of confusion matrices for all evaluated models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    labels = ["Not Eligible", "Eligible"]

    for idx, res in enumerate(evaluation_results):
        cm = np.array(res["Confusion_Matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=labels,
            yticklabels=labels,
            cbar=False
        )
        axes[idx].set_title(f"{res['Model']}\n(Accuracy: {res['Accuracy']*100:.1f}%, F1: {res['F1-Score']:.3f})")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_rf_confusion_matrix(
    rf_metrics: Dict[str, Any],
    output_dir: str = PLOTS_DIR
) -> str:
    """Plot dedicated high-resolution confusion matrix for Random Forest."""
    cm = np.array(rf_metrics["Confusion_Matrix"])
    plt.figure(figsize=(7, 6))
    labels = ["Not Eligible", "Eligible"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 16, "weight": "bold"},
        cbar=True
    )
    plt.title(f"Random Forest Confusion Matrix\n(Accuracy: {rf_metrics['Accuracy']*100:.1f}%, F1-Score: {rf_metrics['F1-Score']:.4f})", pad=15)
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix_random_forest.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[SAVED] Random Forest confusion matrix plot -> {path}")
    return path


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_dir: str = PLOTS_DIR
) -> str:
    """
    Plot grouped bar chart comparing Accuracy, Precision, Recall, and F1-Score across models.
    """
    melted_df = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(11, 6))
    palette = ["#1971c2", "#2f9e44", "#f59f00", "#e03131"]

    ax = sns.barplot(
        data=melted_df,
        x="Model",
        y="Score",
        hue="Metric",
        palette=palette
    )
    plt.title("Model Performance Comparison (Course Review Presentation)", pad=15)
    plt.xlabel("Classification Algorithm")
    plt.ylabel("Score (0.0 to 1.0)")
    plt.ylim(0.65, 1.02)
    plt.legend(title="Evaluation Metric", loc="lower right", frameon=True)

    # Add numeric score annotations above bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=8, rotation=0, xytext=(0, 2),
                textcoords='offset points'
            )

    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_random_forest_feature_importance(
    rf_pipeline: Any,
    output_dir: str = PLOTS_DIR,
    top_n: int = 12
) -> Tuple[str, pd.DataFrame]:
    """
    Extract and visualize Gini feature importances from the Random Forest pipeline.
    """
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    classifier = rf_pipeline.named_steps["classifier"]

    feature_names = get_feature_names_after_preprocessing(preprocessor)
    importances = classifier.feature_importances_

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Top features for plotting
    plot_df = fi_df.head(top_n).sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(10, 6))
    colors = ["#1971c2" if f in ["FamilyIncome", "12thMarks"] else "#748ffc" for f in plot_df["Feature"]]
    plt.barh(plot_df["Feature"], plot_df["Importance"], color=colors)

    plt.title("Random Forest: Feature Importance (Gini Impurity Reduction)", pad=15)
    plt.xlabel("Mean Relative Importance")
    plt.ylabel("Feature")

    for index, value in enumerate(plot_df["Importance"]):
        plt.text(value + 0.005, index, f"{value:.3f} ({value*100:.1f}%)", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "random_forest_feature_importance.png")
    plt.savefig(path, dpi=300)
    plt.close()

    return path, fi_df


def save_metrics(
    metrics_list: List[Dict[str, Any]],
    output_dir: str = METRICS_DIR
) -> pd.DataFrame:
    """Save evaluation metrics to CSV and JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    df_metrics = pd.DataFrame(metrics_list)[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]]

    csv_path = os.path.join(output_dir, "model_comparison.csv")
    df_metrics.to_csv(csv_path, index=False)

    json_path = os.path.join(output_dir, "model_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=4)

    print(f"[SAVED] Metrics table -> {csv_path}")
    print(f"[SAVED] Metrics JSON  -> {json_path}")

    return df_metrics
