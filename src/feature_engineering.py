"""
Feature Engineering Module for Scholarship Eligibility Prediction.

This module provides:
1. Feature selection (dropping non-informative IDs like StudentID)
2. Specification of numerical and categorical feature pipelines
3. Construction of a leakage-free ColumnTransformer
4. Feature name extraction for interpretability and feature importance analysis
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


NUMERICAL_FEATURES: List[str] = ['FamilyIncome', '12thMarks']
CATEGORICAL_FEATURES: List[str] = ['Community', 'FirstGraduate', 'District', 'CollegeType', 'Course', 'Gender']
IDENTIFIER_COLUMNS: List[str] = ['StudentID']
TARGET_COLUMN: str = 'Eligibility'


def get_feature_lists() -> Tuple[List[str], List[str], List[str], str]:
    """Return lists of feature column names by type."""
    return NUMERICAL_FEATURES, CATEGORICAL_FEATURES, IDENTIFIER_COLUMNS, TARGET_COLUMN


from sklearn.impute import SimpleImputer


def build_preprocessor() -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for unified feature preprocessing.

    Transformations:
    - Numerical features (FamilyIncome, 12thMarks):
      1. SimpleImputer(strategy='median')
      2. StandardScaler()
      Centers the distribution around 0 and standardizes unit variance.
      Crucial for Logistic Regression convergence and fair regularization.
    - Categorical features (Community, FirstGraduate, District, CollegeType, Course, Gender):
      1. SimpleImputer(strategy='most_frequent')
      2. OneHotEncoder(handle_unknown='ignore', sparse_output=False)
      Converts discrete nominal classes into binary indicators without imposing
      artificial ordinal hierarchy.

    Zero-Leakage Assurance:
    The preprocessor is fitted strictly on X_train during pipeline training,
    ensuring test observations never contaminate training statistics.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'  # Drop identifiers or unhandled columns
    )

    return preprocessor


def get_feature_names_after_preprocessing(fitted_preprocessor: ColumnTransformer) -> List[str]:
    """
    Retrieve feature names generated after ColumnTransformer has been fitted.

    Useful for Random Forest feature importance visualization and model interpretability.
    """
    feature_names = []

    # Numerical feature names
    feature_names.extend(NUMERICAL_FEATURES)

    # OneHotEncoded categorical feature names from the nested pipeline
    cat_step = fitted_preprocessor.named_transformers_['cat']
    if isinstance(cat_step, Pipeline):
        cat_encoder = cat_step.named_steps['encoder']
    else:
        cat_encoder = cat_step

    encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names.extend(encoded_cat_names)

    return feature_names


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a raw DataFrame into feature matrix X and target Series y (if target exists).
    """
    drop_cols = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
    if TARGET_COLUMN in df.columns:
        drop_cols.append(TARGET_COLUMN)
        y = df[TARGET_COLUMN]
    else:
        y = None

    X = df.drop(columns=drop_cols)
    return X, y
