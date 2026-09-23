import pandas as pd

from red_wine_quality.model import build_regressor, predict
from red_wine_quality.preprocessing import split_features_target
from src.predict import DEFAULT_ROW, validate_features


def test_pipeline_predicts_without_target_leakage():
    frame = pd.DataFrame({
        "fixed acidity": [7.4, 7.8, 7.8, 11.2],
        "volatile acidity": [0.7, 0.88, 0.76, 0.28],
        "alcohol": [9.4, 9.8, 9.8, 9.8],
        "quality": [5, 5, 5, 6],
    })
    X, y = split_features_target(frame)
    model = build_regressor(X, tune=False).fit(X, y)
    assert "quality" not in X
    assert len(predict(model, X)) == len(frame)


def test_prediction_validation_reports_missing_and_non_numeric_columns():
    with __import__("pytest").raises(ValueError, match="Missing required"):
        validate_features(pd.DataFrame({"alcohol": [9.4]}))
    invalid = pd.DataFrame([DEFAULT_ROW])
    invalid["alcohol"] = "bad"
    with __import__("pytest").raises(ValueError, match="must be numeric"):
        validate_features(invalid)
