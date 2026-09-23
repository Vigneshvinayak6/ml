"""Train the primary regression model and print holdout/CV metrics.

Run with ``python -m src.train`` from the repository root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from red_wine_quality.model import compare_regressors, cross_validate_model, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true", help="Use the fast Ridge baseline")
    parser.add_argument("--data", default="data/winequality-red.csv")
    parser.add_argument("--model", default="models/red_wine_quality_model.joblib")
    args = parser.parse_args()
    fitted, holdout = train(args.data, args.model, tune=not args.no_tune)
    tuning = fitted.best_params_ if hasattr(fitted, "best_params_") else {}
    print(json.dumps({
        "holdout": holdout,
        "tuning": tuning,
        "cv": cross_validate_model(args.data),
        "comparison_cv": compare_regressors(args.data),
    }, indent=2))


if __name__ == "__main__":
    main()
