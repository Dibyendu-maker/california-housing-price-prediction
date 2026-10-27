"""
Model definitions and training routines for Property Price Prediction.
Implements:
1. Simple Linear Regression (Median Income -> Median House Value)
2. Multiple Linear Regression (All features, One-Hot encoded categories)
3. Multiple Linear Regression with Engineered District Ratios
4. Regularized Ridge Regression with Cross-Validated Alpha Selection
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline

from src.config import (
    SIMPLE_PREDICTOR,
    MODELS_DIR,
    RANDOM_SEED,
    CV_FOLDS,
)
from src.data import (
    PropertyFeatureEngineer,
    build_preprocessor,
)


def train_simple_linear_regression(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train Simple Linear Regression using median_income as the sole predictor.
    Returns:
        model: Fitted scikit-learn LinearRegression model
        summary: Dictionary with learned coefficient, intercept, and formula
    """
    X_simple = X_train[[SIMPLE_PREDICTOR]].copy()
    model = LinearRegression()
    model.fit(X_simple, y_train)

    summary = {
        "model_name": "Simple Linear Regression",
        "predictor": SIMPLE_PREDICTOR,
        "intercept": float(model.intercept_),
        "coefficient": float(model.coef_[0]),
        "formula": f"median_house_value = {model.intercept_:,.2f} + {model.coef_[0]:,.2f} * {SIMPLE_PREDICTOR}",
    }
    return model, summary


def train_multiple_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    include_feature_engineering: bool = False,
):
    """
    Train Multiple Linear Regression with full scikit-learn Pipeline.
    Includes imputation, standard scaling, and one-hot encoding.
    """
    preprocessor = build_preprocessor(include_feature_engineering=include_feature_engineering)

    steps = []
    if include_feature_engineering:
        steps.append(("feat_eng", PropertyFeatureEngineer(add_features=True)))
    steps.append(("preprocessor", preprocessor))
    steps.append(("regressor", LinearRegression()))

    pipeline = Pipeline(steps=steps)
    pipeline.fit(X_train, y_train)

    return pipeline


def train_ridge_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    include_feature_engineering: bool = True,
    alphas=np.logspace(-2, 4, 60),
    cv: int = CV_FOLDS,
):
    """
    Train Ridge Regression with cross-validated alpha selection on training set.
    Shrinks collinear coefficients and controls variance.
    """
    preprocessor = build_preprocessor(include_feature_engineering=include_feature_engineering)

    steps = []
    if include_feature_engineering:
        steps.append(("feat_eng", PropertyFeatureEngineer(add_features=True)))
    steps.append(("preprocessor", preprocessor))
    steps.append((
        "regressor",
        RidgeCV(
            alphas=alphas,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        ),
    ))

    pipeline = Pipeline(steps=steps)
    pipeline.fit(X_train, y_train)

    best_alpha = float(pipeline.named_steps["regressor"].alpha_)
    return pipeline, best_alpha


def save_model_artifact(model, filepath: Path):
    """Serialize model artifact to disk using joblib."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model_artifact(filepath: Path):
    """Load serialized model artifact from disk."""
    return joblib.load(filepath)
