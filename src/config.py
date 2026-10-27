"""
Configuration module for Property Price Prediction project.
Defines constants, paths, column lists, and model parameters.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data_file.xlsx"
DATA_INFO_PATH = PROJECT_ROOT / "Data Information.pdf"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_JSON_PATH = OUTPUTS_DIR / "metrics.json"
METRICS_CSV_PATH = OUTPUTS_DIR / "metrics_summary.csv"

# Target and Feature Columns
TARGET_COLUMN = "median_house_value"

NUMERICAL_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

CATEGORICAL_FEATURES = [
    "ocean_proximity",
]

ENGINEERED_FEATURES = [
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]

ALL_NUMERICAL_FEATURES_FE = NUMERICAL_FEATURES + ENGINEERED_FEATURES

# Predictor for Simple Linear Regression
SIMPLE_PREDICTOR = "median_income"

# Modeling Hyperparameters and Experiment Setup
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Known dataset thresholds
TARGET_CAP_VALUE = 500001.0
AGE_CAP_VALUE = 52.0
