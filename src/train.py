"""
Model Training and Comparison Module for Scholarship Eligibility Prediction.

This module trains and compares four classification algorithms:
1. Logistic Regression (Linear baseline)
2. Decision Tree (Non-linear rule-based classifier)
3. Random Forest (Ensemble bagging classifier)
4. Naïve Bayes (Probabilistic Bayesian classifier)

It exports:
- outputs/metrics/model_comparison.csv
- outputs/metrics/model_comparison.json
- outputs/plots/model_comparison.png
- outputs/plots/confusion_matrices.png
- outputs/plots/random_forest_feature_importance.png
- models/best_model.joblib (Best performing scikit-learn pipeline)
"""

import os
import sys
from typing import Dict, Any, List, Tuple
import joblib
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from src.data_preprocessing import load_raw_data, clean_data, split_data, save_processed_splits
from src.feature_engineering import build_preprocessor, prepare_features
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_random_forest_feature_importance,
    save_metrics
)


MODELS_DIR = "models"
PROCESSED_DATA_DIR = os.path.join("data", "processed")


def get_models() -> Dict[str, Any]:
    """Define the 4 required classification models with fixed random seeds."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        "Naïve Bayes": GaussianNB()
    }


def train_and_evaluate_models() -> Tuple[pd.DataFrame, str]:
    """
    Train all 4 models, evaluate performance, generate diagnostic plots,
    and save the optimal model pipeline.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load or prepare train/test splits
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("[INFO] Processed splits not found. Running preprocessing...")
        raw_df = load_raw_data()
        clean_df = clean_data(raw_df)
        X_train, X_test, y_train, y_test = split_data(clean_df)
        save_processed_splits(X_train, X_test, y_train, y_test)
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        X_train, y_train = prepare_features(train_df)
        X_test, y_test = prepare_features(test_df)

    print("=" * 65)
    print("TRAINING & COMPARING MACHINE LEARNING CLASSIFIERS")
    print("=" * 65)
    print(f"Training observations: {len(X_train)} | Testing observations: {len(X_test)}")

    models = get_models()
    evaluation_results: List[Dict[str, Any]] = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, clf in models.items():
        print(f"\n[TRAINING] {name}...")
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, model_name=name)
        evaluation_results.append(metrics)
        fitted_pipelines[name] = pipeline

        print(f"  -> Accuracy:  {metrics['Accuracy'] * 100:.2f}%")
        print(f"  -> Precision: {metrics['Precision']:.4f}")
        print(f"  -> Recall:    {metrics['Recall']:.4f}")
        print(f"  -> F1-Score:  {metrics['F1-Score']:.4f}")

    # 2. Save metrics & generate comparative visuals
    metrics_df = save_metrics(evaluation_results)
    plot_confusion_matrices(evaluation_results)
    plot_model_comparison(metrics_df)

    # 3. Random Forest feature importance
    rf_pipeline = fitted_pipelines["Random Forest"]
    fi_plot_path, fi_df = plot_random_forest_feature_importance(rf_pipeline)
    print(f"[SAVED] Random Forest feature importance plot -> {fi_plot_path}")

    # 4. Display comparison table
    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 65)
    print(metrics_df.to_string(index=False))
    print("=" * 65)

    # 5. Select and save best model
    # Priority: F1-Score (essential for moderately imbalanced targets)
    best_row = metrics_df.sort_values(by="F1-Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline = fitted_pipelines[best_model_name]
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"[BEST MODEL] Selected: {best_model_name} (F1-Score: {best_row['F1-Score']:.4f})")
    print(f"[SAVED] Exported complete pipeline to: {best_model_path}")

    # 6. Reference PPT Comparison & Diagnostic Report
    ref_rf_acc = 0.955
    ref_rf_f1 = 0.966
    rf_metrics = [m for m in evaluation_results if m["Model"] == "Random Forest"][0]
    actual_rf_acc = rf_metrics["Accuracy"]
    actual_rf_f1 = rf_metrics["F1-Score"]

    print("\n" + "=" * 65)
    print("PRESENTATION CONSISTENCY & DISCREPANCY REPORT")
    print("=" * 65)
    print("Reference Presentation (Slide 10):")
    print(f"  Random Forest Accuracy: {ref_rf_acc * 100:.1f}%")
    print(f"  Random Forest F1-Score: {ref_rf_f1:.3f}")
    print(f"  Strongest Predictors:   1. FamilyIncome, 2. 12thMarks")
    print("\nActual Experimental Measurement:")
    print(f"  Random Forest Accuracy: {actual_rf_acc * 100:.2f}% (Diff: {(actual_rf_acc - ref_rf_acc) * 100:+.2f}%)")
    print(f"  Random Forest F1-Score: {actual_rf_f1:.4f} (Diff: {actual_rf_f1 - ref_rf_f1:+.4f})")

    top_features = fi_df.head(2)["Feature"].tolist()
    print(f"  Top Measured Predictors: 1. {top_features[0]}, 2. {top_features[1]}")

    if top_features == ["FamilyIncome", "12thMarks"] or (
        "FamilyIncome" in top_features and "12thMarks" in top_features
    ):
        print("  -> Predictor ranking aligns with presentation findings!")
    else:
        print(f"  -> Discrepancy noted in top predictor order: {top_features}")

    print("=" * 65)
    return metrics_df, best_model_name


if __name__ == "__main__":
    train_and_evaluate_models()
