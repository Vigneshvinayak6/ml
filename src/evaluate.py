"""Evaluate a persisted regression pipeline on the canonical holdout split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from red_wine_quality.data import load_data
from red_wine_quality.model import load_model
from red_wine_quality.preprocessing import split_features_target


def evaluate(data_path: str | Path, model_path: str | Path) -> dict[str, float]:
    frame = load_data(data_path)
    X, y = split_features_target(frame)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    predictions = load_model(model_path).predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/winequality-red.csv")
    parser.add_argument("--model", default="models/red_wine_quality_model.joblib")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data, args.model), indent=2))


if __name__ == "__main__":
    main()
