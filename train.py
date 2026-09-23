"""Command-line training entry point."""
import argparse
import json

from red_wine_quality.model import cross_validate_model, train

parser = argparse.ArgumentParser()
parser.add_argument("--no-tune", action="store_true", help="Use fast Ridge baseline")
parser.add_argument("--data", default="data/winequality-red.csv")
parser.add_argument("--model", default="models/wine_quality_regressor.joblib")
args = parser.parse_args()
_, holdout = train(args.data, args.model, tune=not args.no_tune)
print(json.dumps({"holdout": holdout, "cv": cross_validate_model(args.data)}, indent=2))
