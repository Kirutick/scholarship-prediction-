"""Verification script - runs all 4 models and prints exact metrics + confusion matrix."""
import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.feature_engineering import build_preprocessor, prepare_features

train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")
X_train, y_train = prepare_features(train_df)
X_test,  y_test  = prepare_features(test_df)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Test class distribution: {dict(y_test.value_counts())}\n")

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Naive Bayes":         GaussianNB(),
}

print("=" * 70)
print("ALL 4 MODELS - FRESH TRAINING RESULTS (random_state=42)")
print("=" * 70)

for name, clf in classifiers.items():
    pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Eligible", zero_division=0)
    rec  = recall_score(y_test, y_pred, pos_label="Eligible", zero_division=0)
    f1   = f1_score(y_test, y_pred, pos_label="Eligible", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=["Not Eligible", "Eligible"])

    print(f"\n[{name}]")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Confusion Matrix (rows=True, cols=Predicted):")
    print(f"                  Pred:Not-Elig  Pred:Eligible")
    print(f"  True:Not-Elig      TN={cm[0,0]:3d}           FP={cm[0,1]:3d}")
    print(f"  True:Eligible      FN={cm[1,0]:3d}           TP={cm[1,1]:3d}")

# Now verify saved model
print("\n" + "=" * 70)
print("SAVED MODEL VERIFICATION (random_forest_pipeline.joblib)")
print("=" * 70)
saved_model = joblib.load("models/random_forest_pipeline.joblib")
y_pred_saved = saved_model.predict(X_test)
acc  = accuracy_score(y_test, y_pred_saved)
prec = precision_score(y_test, y_pred_saved, pos_label="Eligible", zero_division=0)
rec  = recall_score(y_test, y_pred_saved, pos_label="Eligible", zero_division=0)
f1   = f1_score(y_test, y_pred_saved, pos_label="Eligible", zero_division=0)
cm   = confusion_matrix(y_test, y_pred_saved, labels=["Not Eligible", "Eligible"])

print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
print(f"  Model loaded from: models/random_forest_pipeline.joblib")
print(f"  Does NOT retrain. Loads pre-fitted pipeline.")
print(f"  Classes: {saved_model.classes_}")
