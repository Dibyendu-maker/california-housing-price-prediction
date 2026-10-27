"""
Inference module for Property Price Prediction.
Loads saved trained pipeline and generates predictions for new property records.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR
from src.data import PropertyFeatureEngineer
from src.models import load_model_artifact

# Register PropertyFeatureEngineer into __main__ to guarantee backward-compatibility with notebook pickles
import __main__
__main__.PropertyFeatureEngineer = PropertyFeatureEngineer

DEFAULT_MODEL_PATH = MODELS_DIR / "final_model_pipeline.joblib"


def predict_property_prices(data: pd.DataFrame, model_path: Path = DEFAULT_MODEL_PATH) -> np.ndarray:
    """
    Generate median house value predictions for input DataFrame.
    Expects input dataframe to match schema:
    ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
     'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Train and save the model first.")

    pipeline = load_model_artifact(model_path)
    predictions = pipeline.predict(data)
    return predictions


def run_sample_inference():
    """Demonstrate inference on sample representative California districts."""
    sample_districts = pd.DataFrame([
        {
            "district_name": "San Francisco Bay Area Coastal District",
            "longitude": -122.25,
            "latitude": 37.85,
            "housing_median_age": 45,
            "total_rooms": 2500,
            "total_bedrooms": 450,
            "population": 1100,
            "households": 420,
            "median_income": 8.5,  # $85,000
            "ocean_proximity": "NEAR BAY",
        },
        {
            "district_name": "Inland Valley Suburban District",
            "longitude": -119.80,
            "latitude": 36.75,
            "housing_median_age": 20,
            "total_rooms": 3200,
            "total_bedrooms": 600,
            "population": 1800,
            "households": 580,
            "median_income": 3.2,  # $32,000
            "ocean_proximity": "INLAND",
        },
        {
            "district_name": "Southern California High-Density District (Missing Bedrooms)",
            "longitude": -118.30,
            "latitude": 34.05,
            "housing_median_age": 35,
            "total_rooms": 1800,
            "total_bedrooms": np.nan,  # tests automated median imputation!
            "population": 1500,
            "households": 490,
            "median_income": 4.1,  # $41,000
            "ocean_proximity": "<1H OCEAN",
        },
    ])

    print("=" * 75)
    print("DEMO: PREDICTING MEDIAN HOUSE VALUES FOR SAMPLE DISTRICTS")
    print("=" * 75)

    features = sample_districts.drop(columns=["district_name"])
    preds = predict_property_prices(features)

    for i, row in sample_districts.iterrows():
        print(f"\nDistrict {i+1}: {row['district_name']}")
        print(f"  Location: ({row['latitude']:.2f}° N, {row['longitude']:.2f}° W) | Proximity: {row['ocean_proximity']}")
        print(f"  Median Income: ${row['median_income']*10000:,.0f} | Median Age: {row['housing_median_age']} yrs")
        print(f"  Total Rooms: {row['total_rooms']:,} | Total Bedrooms: {row['total_bedrooms']} | Households: {row['households']:,}")
        print(f"  --> PREDICTED MEDIAN HOUSE VALUE: ${preds[i]:,.2f}")

    print("\nInference completed successfully.")


if __name__ == "__main__":
    run_sample_inference()
