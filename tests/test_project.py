import pandas as pd
from sklearn.pipeline import Pipeline

from red_wine_quality.model import build_regressor, predict
from red_wine_quality.preprocessing import split_features_target


def sample_frame():
    return pd.DataFrame({
        "fixed acidity": [7.4, 7.8, 7.8, 11.2],
        "volatile acidity": [0.7, 0.88, 0.76, 0.28],
        "alcohol": [9.4, 9.8, 9.8, 9.8],
        "quality": [5, 5, 5, 6],
    })


def test_split_and_pipeline_predict():
    X, y = split_features_target(sample_frame())
    model = build_regressor(X, tune=False).fit(X, y)
    assert isinstance(model, Pipeline)
    assert len(predict(model, X)) == len(X)


def test_target_is_not_feature():
    X, _ = split_features_target(sample_frame())
    assert "quality" not in X
