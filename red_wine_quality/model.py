"""Training, evaluation, persistence, and prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import load_data
from .preprocessing import make_preprocessor, split_features_target


def build_regressor(features: pd.DataFrame, tune: bool = True) -> Any:
    """Return a pipeline, optionally with a modest random-forest grid search."""
    preprocessor = make_preprocessor(features)
    if tune:
        estimator = GridSearchCV(
            Pipeline([("preprocessor", preprocessor), ("model", RandomForestRegressor(random_state=42, n_jobs=-1))]),
            {"model__n_estimators": [150, 250], "model__max_depth": [None, 12], "model__min_samples_leaf": [1, 2]},
            cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
    else:
        estimator = Pipeline([("preprocessor", preprocessor), ("model", Ridge(alpha=1.0))])
    return estimator


def compare_regressors(
    data_path: str | Path = "data/winequality-red.csv",
    folds: int = 5,
) -> dict[str, dict[str, float]]:
    """Compare baseline regressors with shuffled, reproducible K-fold CV."""
    frame = load_data(data_path)
    X, y = split_features_target(frame)
    models = {
        "LinearRegression": Pipeline([("preprocessor", make_preprocessor(X)), ("model", LinearRegression())]),
        "Ridge": Pipeline([("preprocessor", make_preprocessor(X)), ("model", Ridge(alpha=1.0))]),
        "RandomForestRegressor": Pipeline([
            ("preprocessor", make_preprocessor(X)),
            ("model", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)),
        ]),
        "GradientBoostingRegressor": Pipeline([
            ("preprocessor", make_preprocessor(X)),
            ("model", GradientBoostingRegressor(random_state=42, n_estimators=150)),
        ]),
    }
    cv = KFold(folds, shuffle=True, random_state=42)
    scores: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        result = cross_validate(
            model, X, y, cv=cv,
            scoring=("neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error", "r2"),
        )
        scores[name] = {
            "mae": float(-result["test_neg_mean_absolute_error"].mean()),
            "mse": float(-result["test_neg_mean_squared_error"].mean()),
            "rmse": float(-result["test_neg_root_mean_squared_error"].mean()),
            "r2": float(result["test_r2"].mean()),
        }
    return scores


def build_classifier(features: pd.DataFrame) -> Pipeline:
    """Build an optional classifier for the dataset's discrete quality labels."""
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(features)),
            ("model", RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)),
        ]
    )


def train_classifier(data_path: str | Path = "data/winequality-red.csv") -> tuple[Any, dict[str, float]]:
    """Train an optional classifier and report accuracy and macro-F1."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

    frame = load_data(data_path)
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    estimator = build_classifier(X_train).fit(X_train, y_train)
    predicted = estimator.predict(X_test)
    return estimator, {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision_macro": float(precision_score(y_test, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predicted, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
    }


def train(
    data_path: str | Path = "data/winequality-red.csv",
    model_path: str | Path = "models/wine_quality_regressor.joblib",
    tune: bool = True,
    test_size: float = 0.2,
) -> tuple[Any, dict[str, float]]:
    """Fit a regressor and return the fitted pipeline plus holdout metrics."""
    frame = load_data(data_path)
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    estimator = build_regressor(X_train, tune=tune)
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "mse": float(mean_squared_error(y_test, pred)),
        "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "r2": float(r2_score(y_test, pred)),
    }
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)
    return estimator, metrics


def cross_validate_model(data_path: str | Path = "data/winequality-red.csv", folds: int = 5) -> dict[str, float]:
    """Compute reproducible K-fold regression scores."""
    frame = load_data(data_path)
    X, y = split_features_target(frame)
    result = cross_validate(
        build_regressor(X, tune=False), X, y,
        cv=KFold(folds, shuffle=True, random_state=42),
        scoring=("neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error", "r2"),
    )
    return {
        "cv_mae": float(-result["test_neg_mean_absolute_error"].mean()),
        "cv_mse": float(-result["test_neg_mean_squared_error"].mean()),
        "cv_rmse": float(-result["test_neg_root_mean_squared_error"].mean()),
        "cv_r2": float(result["test_r2"].mean()),
    }


def load_model(model_path: str | Path = "models/red_wine_quality_model.joblib") -> Any:
    return joblib.load(model_path)


def predict(model: Any, features: pd.DataFrame) -> list[float]:
    """Predict quality for one or more rows."""
    return [float(value) for value in model.predict(features)]
