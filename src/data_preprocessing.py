"""
Data Preprocessing Module for Scholarship Eligibility Prediction.

This module handles:
1. Dataset ingestion with schema validation and missing-file handling
2. Dataset inspection (shape, dtypes, missing values, duplicates)
3. Data cleaning (imputation, deduplication)
4. Feature and target separation (dropping identifiers like StudentID)
5. Stratified train/test splitting to prevent data leakage
6. Exporting processed train/test splits for downstream reproducibility
"""

import json
import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


DEFAULT_RAW_DATA_PATH = os.path.join("data", "raw", "scholarship_data.csv")
SCHEMA_PATH = os.path.join("data", "raw", "dataset_schema.json")
PROCESSED_DATA_DIR = os.path.join("data", "processed")


def load_dataset_schema(schema_path: str = SCHEMA_PATH) -> Dict[str, Any]:
    """Load the JSON schema definition for the scholarship dataset."""
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_raw_data(
    file_path: str = DEFAULT_RAW_DATA_PATH,
    auto_generate_if_missing: bool = True
) -> pd.DataFrame:
    """
    Load raw student dataset from CSV.

    If the dataset file is missing:
    - Reports clearly that the file was not found
    - Outlines the expected schema and file destination
    - If auto_generate_if_missing is True, calls the reference generator
      to provide the benchmark dataset matching presentation parameters.
    """
    if not os.path.exists(file_path):
        print("!" * 70)
        print("WARNING: RAW DATASET FILE NOT FOUND")
        print(f"Looked for: {os.path.abspath(file_path)}")
        print("Required schema columns:")
        print("  StudentID, Gender, Community, FamilyIncome, 12thMarks,")
        print("  FirstGraduate, District, CollegeType, Course, Eligibility")
        print("To supply your original OGD/college dataset, save it to:")
        print(f"  {file_path}")
        print("!" * 70)

        if auto_generate_if_missing:
            print("[INFO] Invoking reference dataset generator (Tamil Nadu Post-Matric scheme)...")
            from scripts.generate_reference_dataset import generate_scholarship_dataset
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df = generate_scholarship_dataset(n_samples=1000, seed=42)
            df.to_csv(file_path, index=False)
            print(f"[INFO] Reference dataset generated at: {file_path}")
            return df
        else:
            raise FileNotFoundError(f"Dataset missing at {file_path}. Please supply the file.")

    df = pd.read_csv(file_path)
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect raw dataset and print diagnostic summary.
    """
    missing = df.isnull().sum()
    duplicates = df.duplicated().sum()
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in ['StudentID', 'Eligibility']]
    num_cols = list(df.select_dtypes(include=[np.number]).columns)

    print("=" * 60)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 60)
    print(f"Dataset shape:        {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values total: {missing.sum()}")
    if missing.sum() > 0:
        print("Missing per column:\n", missing[missing > 0])
    print(f"Duplicate records:    {duplicates}")
    print(f"Categorical columns:  {cat_cols}")
    print(f"Numerical columns:    {num_cols}")
    if 'Eligibility' in df.columns:
        counts = df['Eligibility'].value_counts()
        print("Target distribution:")
        for label, count in counts.items():
            print(f"  - {label}: {count} ({count / len(df) * 100:.1f}%)")
    print("=" * 60)

    return {
        "shape": df.shape,
        "missing_total": int(missing.sum()),
        "duplicates": int(duplicates),
        "categorical_columns": cat_cols,
        "numerical_columns": num_cols
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform baseline data cleaning:
    - Deduplicate identical records
    - Impute missing numerical columns with median
    - Impute missing categorical columns with mode
    """
    df_clean = df.copy()

    # Deduplicate
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped_dupes = initial_len - len(df_clean)
    if dropped_dupes > 0:
        print(f"[CLEANING] Removed {dropped_dupes} duplicate rows.")

    # Impute missing values if any
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in [np.float64, np.int64]:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"[IMPUTE] Replaced missing values in '{col}' with median: {median_val}")
            else:
                mode_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_val)
                print(f"[IMPUTE] Replaced missing values in '{col}' with mode: {mode_val}")

    return df_clean


def split_data(
    df: pd.DataFrame,
    target_col: str = "Eligibility",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separate feature matrix X and target vector y, then perform stratified train/test split.

    StudentID is dropped to prevent data leakage and memorization.
    """
    drop_cols = [target_col]
    if "StudentID" in df.columns:
        drop_cols.append("StudentID")

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Stratified split ensures equal class representation in train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Training samples:     {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Testing samples:      {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

    return X_train, X_test, y_train, y_test


def save_processed_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = PROCESSED_DATA_DIR
) -> None:
    """Save clean train and test sets to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    train_df = X_train.copy()
    train_df["Eligibility"] = y_train.values
    train_path = os.path.join(output_dir, "train.csv")
    train_df.to_csv(train_path, index=False)

    test_df = X_test.copy()
    test_df["Eligibility"] = y_test.values
    test_path = os.path.join(output_dir, "test.csv")
    test_df.to_csv(test_path, index=False)

    print(f"[SAVED] Processed train split -> {train_path}")
    print(f"[SAVED] Processed test split  -> {test_path}")


def main():
    print("Executing Data Preprocessing Pipeline...")
    df = load_raw_data()
    inspect_dataset(df)
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean)
    save_processed_splits(X_train, X_test, y_train, y_test)
    print("Data Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
