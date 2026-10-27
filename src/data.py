"""
Data loading and preprocessing utilities for Property Price Prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    RANDOM_SEED,
    TEST_SIZE,
)


def load_dataset(file_path=DATA_PATH) -> pd.DataFrame:
    """Load the dataset from the specified Excel file."""
    df = pd.read_excel(file_path)
    return df


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_SEED):
    """
    Split the dataset into training and testing feature/target sets.
    Preserves clean separation to prevent data leakage.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


class PropertyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to engineer district-level ratio features:
    - rooms_per_household: total_rooms / households
    - bedrooms_per_room: total_bedrooms / total_rooms
    - population_per_household: population / households
    """
    def __init__(self, add_features: bool = True):
        self.add_features = add_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not self.add_features:
            return X_df

        # Avoid zero division with safe epsilon
        eps = 1e-6
        X_df["rooms_per_household"] = X_df["total_rooms"] / (X_df["households"] + eps)
        X_df["bedrooms_per_room"] = X_df["total_bedrooms"] / (X_df["total_rooms"] + eps)
        X_df["population_per_household"] = X_df["population"] / (X_df["households"] + eps)
        return X_df


def build_preprocessor(include_feature_engineering: bool = False) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for preprocessing.
    - Missing values in numerical features are imputed using the median (calculated from train set only).
    - Numerical features are standardized (zero mean, unit variance).
    - Categorical feature 'ocean_proximity' is one-hot encoded with drop='first' to prevent dummy variable trap.
    """
    num_cols = list(NUMERICAL_FEATURES)
    if include_feature_engineering:
        num_cols = num_cols + list(ENGINEERED_FEATURES)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor
