"""Predict quality from a CSV of feature rows or the canonical example row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from red_wine_quality.model import load_model, predict


DEFAULT_ROW = {
    "fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00,
    "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4,
}
FEATURES = list(DEFAULT_ROW)


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize prediction input with actionable error messages."""
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    values = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    invalid = values.isna().any()
    if invalid.any():
        raise ValueError("Features must be numeric; invalid columns: " + ", ".join(invalid[invalid].index))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="CSV containing feature columns; omit for one example")
    parser.add_argument("--model", default="models/red_wine_quality_model.joblib")
    args = parser.parse_args()
    frame = pd.read_csv(args.input) if args.input else pd.DataFrame([DEFAULT_ROW])
    frame = validate_features(frame)
    print(json.dumps({"predictions": predict(load_model(args.model), frame)}))


if __name__ == "__main__":
    main()
