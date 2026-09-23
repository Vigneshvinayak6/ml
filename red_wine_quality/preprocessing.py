"""Feature/target splitting and sklearn preprocessing."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import TARGET


def split_features_target(frame: pd.DataFrame):
    """Return X and y, preserving numeric feature names."""
    if TARGET not in frame:
        raise ValueError(f"Missing target column: {TARGET}")
    return frame.drop(columns=[TARGET]), frame[TARGET]


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build a leakage-safe numeric imputation/scaling transformer."""
    numeric = list(features.select_dtypes(include="number").columns)
    if not numeric:
        raise ValueError("No numeric features found")
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    return ColumnTransformer([("numeric", numeric_pipe, numeric)], remainder="drop")
