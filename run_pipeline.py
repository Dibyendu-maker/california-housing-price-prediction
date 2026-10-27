"""
End-to-End Execution Pipeline for Property Price Prediction.
Runs EDA, Model Training, Cross-Validation, Evaluation, Diagnostics,
Saves Artifacts, Figures, and Metrics Summary.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    METRICS_JSON_PATH,
    METRICS_CSV_PATH,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    SIMPLE_PREDICTOR,
    RANDOM_SEED,
    TARGET_CAP_VALUE,
)
from src.data import (
    load_dataset,
    split_data,
    PropertyFeatureEngineer,
    build_preprocessor,
)
from src.eda import (
    compute_basic_info,
    compute_vif,
    plot_target_distribution,
    plot_numerical_distributions,
    plot_numerical_boxplots,
    plot_correlation_heatmap,
    plot_income_vs_price,
    plot_ocean_proximity_analysis,
    plot_california_geography,
    plot_vif_barchart,
)
from src.models import (
    train_simple_linear_regression,
    train_multiple_linear_regression,
    train_ridge_regression,
    save_model_artifact,
)
from src.evaluate import (
    calculate_metrics,
    evaluate_cross_validation,
    analyze_target_ceiling,
    plot_actual_vs_predicted,
    plot_residuals_vs_fitted,
    plot_residuals_distribution_qq,
    plot_model_comparison,
    plot_ceiling_effect,
    plot_mlr_coefficients_chart,
    export_metrics_summary,
)


def run_pipeline():
    print("=" * 80)
    print("STARTING PROPERTY PRICE PREDICTION WORKFLOW")
    print("=" * 80)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print(f"\n[1] Loading dataset from {DATA_PATH}...")
    df = load_dataset(DATA_PATH)
    print(f"    Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. EDA & Statistics
    print("\n[2] Computing EDA statistics & generating figures...")
    info = compute_basic_info(df)
    print(f"    Duplicates: {info['duplicates']}")
    print(f"    Missing values:\n{info['missing_summary']}")
    print(f"    Capped records (>= $500,000): {info['capped_count']} ({info['capped_pct']:.2f}%)")

    # Save EDA Plots
    print("    Saving EDA visualizations to outputs/figures/...")
    plot_target_distribution(df, FIGURES_DIR / "01_target_distribution.png")
    plot_numerical_distributions(df, FIGURES_DIR / "02_numerical_features_histograms.png")
    plot_numerical_boxplots(df, FIGURES_DIR / "03_numerical_features_boxplots.png")
    plot_correlation_heatmap(df, FIGURES_DIR / "04_correlation_heatmap.png")
    plot_income_vs_price(df, FIGURES_DIR / "05_median_income_vs_price_scatter.png")
    plot_ocean_proximity_analysis(df, FIGURES_DIR / "06_ocean_proximity_distribution_and_price.png")
    plot_california_geography(df, FIGURES_DIR / "07_california_housing_geography.png")

    # 3. Data Splitting (Strict 80/20 train/test split)
    print(f"\n[3] Splitting dataset (Train 80%, Test 20%, random_state={RANDOM_SEED})...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"    Training samples: {len(X_train)} ({len(X_train)/len(df):.1%})")
    print(f"    Testing samples:  {len(X_test)} ({len(X_test)/len(df):.1%})")

    # 4. Multicollinearity Analysis on Training Set
    print("\n[4] Computing Variance Inflation Factor (VIF) on training numerical features...")
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_num_scaled = pd.DataFrame(
        scaler.fit_transform(imputer.fit_transform(X_train[NUMERICAL_FEATURES])),
        columns=NUMERICAL_FEATURES,
    )
    vif_df = compute_vif(X_train_num_scaled)
    print("    VIF Summary on Training Data:")
    for _, row in vif_df.iterrows():
        print(f"      - {row['Feature']:<20}: {row['VIF']:.2f}")
    plot_vif_barchart(vif_df, FIGURES_DIR / "16_vif_comparison.png")

    # 5. Model 1: Simple Linear Regression
    print("\n[5] Fitting Model 1: Simple Linear Regression (Predictor: median_income)...")
    simple_model, simple_summary = train_simple_linear_regression(X_train, y_train)
    print(f"    Learned Formula: {simple_summary['formula']}")
    print(f"    Intercept (beta_0): ${simple_summary['intercept']:,.2f}")
    print(f"    Slope (beta_1):     ${simple_summary['coefficient']:,.2f}")

    y_train_pred_simple = simple_model.predict(X_train[[SIMPLE_PREDICTOR]])
    y_test_pred_simple = simple_model.predict(X_test[[SIMPLE_PREDICTOR]])
    metrics_simple_train = calculate_metrics(y_train, y_train_pred_simple)
    metrics_simple_test = calculate_metrics(y_test, y_test_pred_simple)

    # 6. Model 2: Multiple Linear Regression (Baseline)
    print("\n[6] Fitting Model 2: Multiple Linear Regression (Baseline with All Features)...")
    mlr_pipe = train_multiple_linear_regression(X_train, y_train, include_feature_engineering=False)
    y_train_pred_mlr = mlr_pipe.predict(X_train)
    y_test_pred_mlr = mlr_pipe.predict(X_test)
    metrics_mlr_train = calculate_metrics(y_train, y_train_pred_mlr)
    metrics_mlr_test = calculate_metrics(y_test, y_test_pred_mlr)

    # Extract feature names & coefficients for MLR
    cat_feature_names = list(mlr_pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names
    mlr_coefs = mlr_pipe.named_steps["regressor"].coef_
    plot_mlr_coefficients_chart(all_feature_names, mlr_coefs, FIGURES_DIR / "10_mlr_coefficients.png")

    # 7. Model 3: Multiple Linear Regression with Feature Engineering
    print("\n[7] Fitting Model 3: MLR with Engineered District Ratios...")
    mlr_fe_pipe = train_multiple_linear_regression(X_train, y_train, include_feature_engineering=True)
    y_train_pred_fe = mlr_fe_pipe.predict(X_train)
    y_test_pred_fe = mlr_fe_pipe.predict(X_test)
    metrics_fe_train = calculate_metrics(y_train, y_train_pred_fe)
    metrics_fe_test = calculate_metrics(y_test, y_test_pred_fe)

    # 8. Model 4: Ridge Regression (Regularized Linear Regression)
    print("\n[8] Fitting Model 4: Ridge Regression with 5-Fold Cross-Validated Alpha...")
    ridge_pipe, best_alpha = train_ridge_regression(X_train, y_train, include_feature_engineering=True)
    print(f"    Optimal L2 Regularization Alpha: {best_alpha:.4f}")
    y_train_pred_ridge = ridge_pipe.predict(X_train)
    y_test_pred_ridge = ridge_pipe.predict(X_test)
    metrics_ridge_train = calculate_metrics(y_train, y_train_pred_ridge)
    metrics_ridge_test = calculate_metrics(y_test, y_test_pred_ridge)

    # 9. Cross-Validation on Training Set
    print("\n[9] Running 5-Fold Cross-Validation on Training Data...")
    cv_simple = evaluate_cross_validation(simple_model, X_train[[SIMPLE_PREDICTOR]], y_train)
    cv_mlr = evaluate_cross_validation(mlr_pipe, X_train, y_train)
    cv_fe = evaluate_cross_validation(mlr_fe_pipe, X_train, y_train)
    cv_ridge = evaluate_cross_validation(ridge_pipe, X_train, y_train)

    print(f"    Simple Linear Reg CV RMSE: ${cv_simple['CV_RMSE_mean']:,.2f} +/- ${cv_simple['CV_RMSE_std']:,.2f}")
    print(f"    Baseline MLR CV RMSE:     ${cv_mlr['CV_RMSE_mean']:,.2f} +/- ${cv_mlr['CV_RMSE_std']:,.2f}")
    print(f"    MLR with FE CV RMSE:      ${cv_fe['CV_RMSE_mean']:,.2f} +/- ${cv_fe['CV_RMSE_std']:,.2f}")
    print(f"    Ridge with FE CV RMSE:    ${cv_ridge['CV_RMSE_mean']:,.2f} +/- ${cv_ridge['CV_RMSE_std']:,.2f}")

    # 10. Residual Diagnostics & Ceiling Effect Analysis
    print("\n[10] Analyzing Residual Diagnostics & Target Ceiling Effect...")
    residuals_simple_test = y_test - y_test_pred_simple
    residuals_mlr_test = y_test - y_test_pred_mlr

    ceiling_analysis_base = analyze_target_ceiling(y_test, y_test_pred_mlr)
    ceiling_analysis_fe = analyze_target_ceiling(y_test, y_test_pred_fe)
    print(f"    Baseline MLR Overall Test RMSE:  ${ceiling_analysis_base['overall_rmse']:,.2f}")
    print(f"      - Uncapped Districts RMSE:      ${ceiling_analysis_base['uncapped_rmse']:,.2f}")
    print(f"      - Capped Districts RMSE:        ${ceiling_analysis_base['capped_rmse']:,.2f}")
    print(f"    Champion MLR (FE) Overall RMSE:  ${ceiling_analysis_fe['overall_rmse']:,.2f}")
    print(f"      - Uncapped Districts RMSE:      ${ceiling_analysis_fe['uncapped_rmse']:,.2f}")
    print(f"      - Capped Districts RMSE:        ${ceiling_analysis_fe['capped_rmse']:,.2f}")

    # Evaluation figures
    plot_actual_vs_predicted(y_test, y_test_pred_simple, y_test_pred_mlr, FIGURES_DIR / "12_actual_vs_predicted.png")
    plot_residuals_vs_fitted(y_test_pred_simple, residuals_simple_test, y_test_pred_mlr, residuals_mlr_test, FIGURES_DIR / "13_residuals_vs_fitted.png")
    plot_residuals_distribution_qq(residuals_mlr_test, FIGURES_DIR / "14_residuals_distribution_qq.png")
    plot_ceiling_effect(y_test, y_test_pred_fe, FIGURES_DIR / "15_ceiling_effect_analysis.png")

    # 11. Model Comparison Table
    comparison_table = [
        {
            "Model": "Simple Linear Regression",
            "Predictors": "median_income (1 feature)",
            "Train RMSE": metrics_simple_train["RMSE"],
            "Test RMSE": metrics_simple_test["RMSE"],
            "Test MAE": metrics_simple_test["MAE"],
            "Test MSE": metrics_simple_test["MSE"],
            "Train R2": metrics_simple_train["R2"],
            "Test R2": metrics_simple_test["R2"],
            "CV RMSE Mean": cv_simple["CV_RMSE_mean"],
            "CV R2 Mean": cv_simple["CV_R2_mean"],
        },
        {
            "Model": "Multiple Linear Regression (Base)",
            "Predictors": "All 9 features (numerical + one-hot)",
            "Train RMSE": metrics_mlr_train["RMSE"],
            "Test RMSE": metrics_mlr_test["RMSE"],
            "Test MAE": metrics_mlr_test["MAE"],
            "Test MSE": metrics_mlr_test["MSE"],
            "Train R2": metrics_mlr_train["R2"],
            "Test R2": metrics_mlr_test["R2"],
            "CV RMSE Mean": cv_mlr["CV_RMSE_mean"],
            "CV R2 Mean": cv_mlr["CV_R2_mean"],
        },
        {
            "Model": "Multiple Linear Regression (FE)",
            "Predictors": "All features + 3 engineered ratios",
            "Train RMSE": metrics_fe_train["RMSE"],
            "Test RMSE": metrics_fe_test["RMSE"],
            "Test MAE": metrics_fe_test["MAE"],
            "Test MSE": metrics_fe_test["MSE"],
            "Train R2": metrics_fe_train["R2"],
            "Test R2": metrics_fe_test["R2"],
            "CV RMSE Mean": cv_fe["CV_RMSE_mean"],
            "CV R2 Mean": cv_fe["CV_R2_mean"],
        },
        {
            "Model": "Ridge Regression (FE + L2)",
            "Predictors": "All features + ratios + L2 penalty",
            "Train RMSE": metrics_ridge_train["RMSE"],
            "Test RMSE": metrics_ridge_test["RMSE"],
            "Test MAE": metrics_ridge_test["MAE"],
            "Test MSE": metrics_ridge_test["MSE"],
            "Train R2": metrics_ridge_train["R2"],
            "Test R2": metrics_ridge_test["R2"],
            "CV RMSE Mean": cv_ridge["CV_RMSE_mean"],
            "CV R2 Mean": cv_ridge["CV_R2_mean"],
        },
    ]

    comp_df = pd.DataFrame(comparison_table)
    print("\n--- MODEL PERFORMANCE COMPARISON TABLE ---")
    print(comp_df[["Model", "Test RMSE", "Test MAE", "Test R2", "CV RMSE Mean"]].to_string(index=False))

    plot_model_comparison(comp_df, FIGURES_DIR / "11_model_comparison_metrics.png")

    # 12. Save Models and Metrics
    print("\n[12] Saving serialized models and evaluation metrics...")
    save_model_artifact(mlr_fe_pipe, MODELS_DIR / "final_model_pipeline.joblib")
    save_model_artifact(simple_model, MODELS_DIR / "simple_linear_model.joblib")
    print(f"     Saved final pipeline to {MODELS_DIR / 'final_model_pipeline.joblib'}")

    # Compile comprehensive metrics dict
    full_metrics = {
        "dataset_summary": {
            "total_records": int(len(df)),
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "missing_total_bedrooms": int(info["missing_summary"].loc["total_bedrooms", "Missing_Count"]),
            "duplicates": int(info["duplicates"]),
            "target_capped_records": int(info["capped_count"]),
            "target_capped_percent": float(info["capped_pct"]),
        },
        "simple_linear_regression": {
            "summary": simple_summary,
            "train_metrics": metrics_simple_train,
            "test_metrics": metrics_simple_test,
            "cv_metrics": cv_simple,
        },
        "multiple_linear_regression_baseline": {
            "train_metrics": metrics_mlr_train,
            "test_metrics": metrics_mlr_test,
            "cv_metrics": cv_mlr,
        },
        "multiple_linear_regression_feature_engineered": {
            "train_metrics": metrics_fe_train,
            "test_metrics": metrics_fe_test,
            "cv_metrics": cv_fe,
        },
        "ridge_regression": {
            "best_alpha": best_alpha,
            "train_metrics": metrics_ridge_train,
            "test_metrics": metrics_ridge_test,
            "cv_metrics": cv_ridge,
        },
        "ceiling_effect_analysis": ceiling_analysis_fe,
        "ceiling_effect_analysis_baseline": ceiling_analysis_base,
        "ceiling_effect_analysis_feature_engineered": ceiling_analysis_fe,
        "vif_analysis": vif_df.to_dict(orient="records"),
        "comparison_table": comparison_table,
    }

    export_metrics_summary(full_metrics, METRICS_JSON_PATH, METRICS_CSV_PATH)
    print(f"     Saved metrics JSON to {METRICS_JSON_PATH}")
    print(f"     Saved metrics CSV  to {METRICS_CSV_PATH}")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
