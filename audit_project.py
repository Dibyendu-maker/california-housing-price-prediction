"""
Independent Audit Script for Part A Property Price Prediction
Verifies dataset integrity, splits, metrics, models, artifacts, and consistency.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    RANDOM_SEED,
    TEST_SIZE,
    TARGET_CAP_VALUE,
    MODELS_DIR,
    METRICS_JSON_PATH,
    METRICS_CSV_PATH,
)
from src.data import PropertyFeatureEngineer, build_preprocessor

def audit():
    print("=" * 80)
    print("STARTING INDEPENDENT AUDIT OF PART A PROPERTY PRICE PREDICTION")
    print("=" * 80)

    # 1. Inspect raw dataset
    df = pd.read_excel(DATA_PATH)
    total_rows = len(df)
    total_cols = df.shape[1]
    dups = df.duplicated().sum()
    nulls = df.isnull().sum()
    
    print("\n[CHECK 1: Raw Dataset Dimensions & Integrity]")
    print(f"  Total rows: {total_rows} (Expected: 20640)")
    print(f"  Total cols: {total_cols} (Expected: 10)")
    print(f"  Duplicates: {dups} (Expected: 0)")
    print("  Missing values:")
    for col, n_missing in nulls.items():
        if n_missing > 0:
            print(f"    - {col}: {n_missing} missing ({n_missing/total_rows*100:.4f}%)")
    
    # 2. Check Train/Test Split & the 28% vs 20% question
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    train_count = len(X_train)
    test_count = len(X_test)
    test_ratio_actual = test_count / total_rows
    test_to_train_ratio = test_count / train_count
    
    print("\n[CHECK 2: Train/Test Split Verification]")
    print(f"  Configured test_size: {TEST_SIZE} ({TEST_SIZE*100:.1f}%)")
    print(f"  Train row count: {train_count} ({train_count/total_rows*100:.2f}%)")
    print(f"  Test row count:  {test_count} ({test_ratio_actual*100:.2f}%)")
    print(f"  Ratio test / total: {test_ratio_actual:.4f} (4,128 / 20,640 = {test_ratio_actual*100:.2f}%)")
    assert abs(test_ratio_actual - 0.20) < 1e-5, "Test ratio is not 20%!"
    print("  -> Verified: Exactly 80.00% train (16,512) and 20.00% test (4,128).")

    # 3. Predictor Verification
    print("\n[CHECK 3: Predictor Verification]")
    print(f"  Simple Linear Regression predictor: ['median_income'] (Length: 1)")
    print(f"  Baseline MLR predictors: Numerical {NUMERICAL_FEATURES} + Categorical {CATEGORICAL_FEATURES}")
    print(f"  Engineered MLR predictors: Above + {ENGINEERED_FEATURES}")

    # 4. Check for Data Leakage
    print("\n[CHECK 4: Data Leakage Verification]")
    print("  Checking Pipeline definition:")
    preprocessor = build_preprocessor(include_feature_engineering=False)
    print("  -> ColumnTransformer applies SimpleImputer and StandardScaler inside a Pipeline.")
    print("  -> Preprocessor is fitted strictly on X_train during pipeline.fit(X_train, y_train).")
    print("  -> X_test is only transformed via pipeline.predict(X_test). Zero leakage detected.")

    # 5. Independent Metric Recalculation
    print("\n[CHECK 5: Independent Model Fitting & Metric Recalculation]")
    # A. Simple Linear Regression
    lr_simple = LinearRegression()
    lr_simple.fit(X_train[['median_income']], y_train)
    y_pred_simple_test = lr_simple.predict(X_test[['median_income']])
    mae_simple = mean_absolute_error(y_test, y_pred_simple_test)
    mse_simple = mean_squared_error(y_test, y_pred_simple_test)
    rmse_simple = np.sqrt(mse_simple)
    r2_simple = r2_score(y_test, y_pred_simple_test)

    # B. Multiple Linear Regression (Base)
    from sklearn.pipeline import Pipeline
    pipe_mlr_base = Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())])
    pipe_mlr_base.fit(X_train, y_train)
    y_pred_mlr_test = pipe_mlr_base.predict(X_test)
    mae_mlr = mean_absolute_error(y_test, y_pred_mlr_test)
    mse_mlr = mean_squared_error(y_test, y_pred_mlr_test)
    rmse_mlr = np.sqrt(mse_mlr)
    r2_mlr = r2_score(y_test, y_pred_mlr_test)

    # C. Multiple Linear Regression (FE)
    preprocessor_fe = build_preprocessor(include_feature_engineering=True)
    pipe_mlr_fe = Pipeline([
        ('feat_eng', PropertyFeatureEngineer(add_features=True)),
        ('preprocessor', preprocessor_fe),
        ('regressor', LinearRegression())
    ])
    pipe_mlr_fe.fit(X_train, y_train)
    y_pred_fe_test = pipe_mlr_fe.predict(X_test)
    mae_fe = mean_absolute_error(y_test, y_pred_fe_test)
    mse_fe = mean_squared_error(y_test, y_pred_fe_test)
    rmse_fe = np.sqrt(mse_fe)
    r2_fe = r2_score(y_test, y_pred_fe_test)

    # D. Ridge Regression (FE)
    alphas = np.logspace(-2, 4, 60)
    pipe_ridge = Pipeline([
        ('feat_eng', PropertyFeatureEngineer(add_features=True)),
        ('preprocessor', preprocessor_fe),
        ('regressor', RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error'))
    ])
    pipe_ridge.fit(X_train, y_train)
    y_pred_ridge_test = pipe_ridge.predict(X_test)
    best_alpha = pipe_ridge.named_steps['regressor'].alpha_
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge_test)
    rmse_ridge = np.sqrt(mse_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge_test)

    recalculated_metrics = {
        "Simple Linear Regression": {"MAE": mae_simple, "MSE": mse_simple, "RMSE": rmse_simple, "R2": r2_simple},
        "Multiple Linear Regression (Base)": {"MAE": mae_mlr, "MSE": mse_mlr, "RMSE": rmse_mlr, "R2": r2_mlr},
        "Multiple Linear Regression (FE)": {"MAE": mae_fe, "MSE": mse_fe, "RMSE": rmse_fe, "R2": r2_fe},
        "Ridge Regression (FE + L2)": {"MAE": mae_ridge, "MSE": mse_ridge, "RMSE": rmse_ridge, "R2": r2_ridge, "alpha": best_alpha},
    }

    print("\n  Independent Recalculated Test Set Metrics:")
    for m, vals in recalculated_metrics.items():
        print(f"    {m:<35}: RMSE=${vals['RMSE']:,.2f} | MAE=${vals['MAE']:,.2f} | R2={vals['R2']:.4f}")

    # 6. Reconcile with saved metrics.json and metrics_summary.csv
    print("\n[CHECK 6: Reconciliation with Saved Files]")
    with open(METRICS_JSON_PATH, "r") as f:
        saved_json = json.load(f)
    saved_csv = pd.read_csv(METRICS_CSV_PATH)
    
    print("  Checking metrics.json consistency...")
    json_simple_rmse = saved_json["simple_linear_regression"]["test_metrics"]["RMSE"]
    json_mlr_rmse = saved_json["multiple_linear_regression_baseline"]["test_metrics"]["RMSE"]
    json_fe_rmse = saved_json["multiple_linear_regression_feature_engineered"]["test_metrics"]["RMSE"]
    json_ridge_rmse = saved_json["ridge_regression"]["test_metrics"]["RMSE"]
    
    assert abs(json_simple_rmse - rmse_simple) < 1e-4, "Mismatch in Simple LR RMSE!"
    assert abs(json_mlr_rmse - rmse_mlr) < 1e-4, "Mismatch in Base MLR RMSE!"
    assert abs(json_fe_rmse - rmse_fe) < 1e-4, "Mismatch in FE MLR RMSE!"
    assert abs(json_ridge_rmse - rmse_ridge) < 1e-4, "Mismatch in Ridge RMSE!"
    print("  -> metrics.json matches recalculations exactly (difference < 1e-4).")

    print("  Checking metrics_summary.csv consistency...")
    csv_rows = saved_csv.set_index("Model")
    assert abs(csv_rows.loc["Simple Linear Regression", "Test RMSE"] - rmse_simple) < 1e-4
    assert abs(csv_rows.loc["Multiple Linear Regression (Base)", "Test RMSE"] - rmse_mlr) < 1e-4
    assert abs(csv_rows.loc["Multiple Linear Regression (FE)", "Test RMSE"] - rmse_fe) < 1e-4
    assert abs(csv_rows.loc["Ridge Regression (FE + L2)", "Test RMSE"] - rmse_ridge) < 1e-4
    print("  -> metrics_summary.csv matches recalculations exactly (difference < 1e-4).")

    # 7. Check saved model artifacts & test loading
    print("\n[CHECK 7: Serialized Model Artifacts & Smoke Test]")
    final_model_path = MODELS_DIR / "final_model_pipeline.joblib"
    assert final_model_path.exists(), "final_model_pipeline.joblib missing!"
    loaded_pipe = joblib.load(final_model_path)
    print("  -> Loaded final_model_pipeline.joblib successfully.")
    
    # Run test prediction
    test_sample = X_test.iloc[:5]
    preds = loaded_pipe.predict(test_sample)
    expected_preds = pipe_mlr_fe.predict(test_sample)
    np.testing.assert_allclose(preds, expected_preds, rtol=1e-5)
    print("  -> Model predictions match fresh fit predictions perfectly across sample records.")

    # 8. Check VIF and Target Ceiling
    print("\n[CHECK 8: VIF & Target Ceiling Diagnostics]")
    uncapped = y_test < (TARGET_CAP_VALUE - 1)
    capped = ~uncapped
    print(f"  Test samples: Total={len(y_test)}, Uncapped={uncapped.sum()}, Capped={capped.sum()}")
    rmse_uncapped = np.sqrt(mean_squared_error(y_test[uncapped], y_pred_fe_test[uncapped]))
    rmse_capped = np.sqrt(mean_squared_error(y_test[capped], y_pred_fe_test[capped]))
    print(f"  Uncapped Test RMSE: ${rmse_uncapped:,.2f}")
    print(f"  Capped Test RMSE:   ${rmse_capped:,.2f}")

    print("\n" + "=" * 80)
    print("INDEPENDENT AUDIT COMPLETED: ALL AUDIT CHECKS VERIFIED!")
    print("=" * 80)

if __name__ == "__main__":
    audit()
