"""
Evaluation and diagnostic routines for Property Price Prediction.
Computes MAE, MSE, RMSE, R2, K-Fold Cross-Validation, and generates diagnostic plots:
- Actual vs. Predicted plots
- Residuals vs. Fitted plots (Homoscedasticity)
- Residual Distribution and Normal Q-Q plots (Normality)
- Target Ceiling Impact Analysis
- Coefficient Interpretation plots
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from src.config import TARGET_CAP_VALUE


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calculate MAE, MSE, RMSE, and R2 score."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def evaluate_cross_validation(estimator, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> dict:
    """Perform K-Fold cross validation on training set for RMSE and R2."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    neg_mse_scores = cross_val_score(estimator, X_train, y_train, scoring="neg_mean_squared_error", cv=kf)
    rmse_scores = np.sqrt(-neg_mse_scores)
    r2_scores = cross_val_score(estimator, X_train, y_train, scoring="r2", cv=kf)

    return {
        "CV_RMSE_mean": float(np.mean(rmse_scores)),
        "CV_RMSE_std": float(np.std(rmse_scores)),
        "CV_R2_mean": float(np.mean(r2_scores)),
        "CV_R2_std": float(np.std(r2_scores)),
    }


def analyze_target_ceiling(y_test: pd.Series, y_pred: np.ndarray, cap_value: float = TARGET_CAP_VALUE) -> dict:
    """Examine model performance split by uncapped vs. capped target records."""
    uncapped_mask = y_test < (cap_value - 1)
    capped_mask = ~uncapped_mask

    overall_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    uncapped_rmse = float(np.sqrt(mean_squared_error(y_test[uncapped_mask], y_pred[uncapped_mask])))
    capped_rmse = float(np.sqrt(mean_squared_error(y_test[capped_mask], y_pred[capped_mask])))

    uncapped_mae = float(mean_absolute_error(y_test[uncapped_mask], y_pred[uncapped_mask]))
    capped_mae = float(mean_absolute_error(y_test[capped_mask], y_pred[capped_mask]))

    uncapped_r2 = float(r2_score(y_test[uncapped_mask], y_pred[uncapped_mask]))

    return {
        "total_test_samples": int(len(y_test)),
        "uncapped_samples": int(uncapped_mask.sum()),
        "capped_samples": int(capped_mask.sum()),
        "overall_rmse": overall_rmse,
        "uncapped_rmse": uncapped_rmse,
        "capped_rmse": capped_rmse,
        "uncapped_mae": uncapped_mae,
        "capped_mae": capped_mae,
        "uncapped_r2": uncapped_r2,
    }


def plot_actual_vs_predicted(
    y_test: pd.Series,
    y_pred_simple: np.ndarray,
    y_pred_mlr: np.ndarray,
    save_path: Path = None,
):
    """Plot Actual vs. Predicted for Simple and Multiple Linear Regression."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot 1: Simple Linear Regression
    ax1.scatter(y_test, y_pred_simple, alpha=0.3, color="#1f77b4", s=15, label="Test Observations")
    min_val = min(y_test.min(), y_pred_simple.min())
    max_val = max(y_test.max(), y_pred_simple.max())
    ax1.plot([0, 550000], [0, 550000], "r--", linewidth=2, label="Ideal Fit (y = x)")
    ax1.axvline(TARGET_CAP_VALUE, color="purple", linestyle=":", linewidth=1.5, label="Capped Ceiling")
    ax1.set_title("Simple Linear Regression\n(Predictor: median_income)", fontsize=12, weight="bold")
    ax1.set_xlabel("Actual Median House Value ($)")
    ax1.set_ylabel("Predicted Median House Value ($)")
    ax1.legend(loc="upper left", frameon=True)

    # Plot 2: Multiple Linear Regression
    ax2.scatter(y_test, y_pred_mlr, alpha=0.3, color="#2ca02c", s=15, label="Test Observations")
    ax2.plot([0, 550000], [0, 550000], "r--", linewidth=2, label="Ideal Fit (y = x)")
    ax2.axvline(TARGET_CAP_VALUE, color="purple", linestyle=":", linewidth=1.5, label="Capped Ceiling")
    ax2.set_title("Multiple Linear Regression\n(All Features)", fontsize=12, weight="bold")
    ax2.set_xlabel("Actual Median House Value ($)")
    ax2.legend(loc="upper left", frameon=True)

    plt.suptitle("Actual vs. Predicted Median House Value (Test Set)", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_residuals_vs_fitted(
    y_pred_simple: np.ndarray,
    residuals_simple: np.ndarray,
    y_pred_mlr: np.ndarray,
    residuals_mlr: np.ndarray,
    save_path: Path = None,
):
    """Plot Residuals vs. Fitted values to examine homoscedasticity and non-linearity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(y_pred_simple, residuals_simple, alpha=0.3, color="#1f77b4", s=15)
    ax1.axhline(0, color="red", linestyle="--", linewidth=2)
    ax1.set_title("Simple Linear Regression: Residuals vs. Fitted", fontsize=12, weight="bold")
    ax1.set_xlabel("Fitted Values ($)")
    ax1.set_ylabel("Residuals ($) [y - y_hat]")

    ax2.scatter(y_pred_mlr, residuals_mlr, alpha=0.3, color="#2ca02c", s=15)
    ax2.axhline(0, color="red", linestyle="--", linewidth=2)
    ax2.set_title("Multiple Linear Regression: Residuals vs. Fitted", fontsize=12, weight="bold")
    ax2.set_xlabel("Fitted Values ($)")
    ax2.set_ylabel("Residuals ($) [y - y_hat]")

    plt.suptitle("Residual Diagnostics: Homoscedasticity Check", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_residuals_distribution_qq(
    residuals_mlr: np.ndarray,
    save_path: Path = None,
):
    """Plot histogram + KDE of MLR residuals alongside Normal Q-Q plot."""
    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Histogram + KDE
    sns.histplot(residuals_mlr, kde=True, ax=ax_hist, color="#336699", bins=50)
    ax_hist.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Mean")
    ax_hist.set_title("Distribution of MLR Test Residuals", fontsize=12, weight="bold")
    ax_hist.set_xlabel("Residual ($ USD)")
    ax_hist.set_ylabel("Count")
    ax_hist.legend(frameon=True)

    # Q-Q Plot
    stats.probplot(residuals_mlr, dist="norm", plot=ax_qq)
    ax_qq.get_lines()[0].set_markerfacecolor("#336699")
    ax_qq.get_lines()[0].set_markersize(4.0)
    ax_qq.get_lines()[0].set_alpha(0.5)
    ax_qq.get_lines()[1].set_color("red")
    ax_qq.get_lines()[1].set_linewidth(2.0)
    ax_qq.set_title("Normal Q-Q Plot (MLR Test Residuals)", fontsize=12, weight="bold")

    plt.suptitle("Residual Diagnostics: Normality Analysis", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: Path = None):
    """Bar chart comparing test metrics across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Error Metrics (RMSE & MAE)
    err_df = comparison_df[["Model", "Test RMSE", "Test MAE"]].melt(
        id_vars="Model", var_name="Metric", value_name="Error ($ USD)"
    )
    sns.barplot(data=err_df, x="Model", y="Error ($ USD)", hue="Metric", ax=ax1, palette=["#d62728", "#1f77b4"])
    ax1.set_title("Comparison of Error Metrics (Lower is Better)", fontsize=12, weight="bold")
    ax1.tick_params(axis="x", rotation=15)

    for p in ax1.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax1.annotate(f"${height:,.0f}", (p.get_x() + p.get_width() / 2., height),
                         ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=8)

    # R2 Metric
    sns.barplot(data=comparison_df, x="Model", y="Test R2", ax=ax2, hue="Model", palette="Greens_r", legend=False)
    ax2.set_title("Comparison of Explained Variance (R2 - Higher is Better)", fontsize=12, weight="bold")
    ax2.set_ylim(0, 0.8)
    ax2.tick_params(axis="x", rotation=15)

    for p in ax2.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax2.annotate(f"{height:.4f}", (p.get_x() + p.get_width() / 2., height),
                         ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)

    plt.suptitle("Performance Comparison Across Evaluated Models (Test Set)", fontsize=14, weight="bold", y=1.03)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_ceiling_effect(y_test: pd.Series, y_pred_mlr: np.ndarray, save_path: Path = None):
    """Visualize target ceiling distortion and residuals for capped vs uncapped districts."""
    residuals = y_test - y_pred_mlr
    is_capped = y_test >= (TARGET_CAP_VALUE - 1)
    plot_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred_mlr,
        "Residual": residuals,
        "Absolute_Error": np.abs(residuals),
        "District_Group": np.where(is_capped, "Capped ($500,001) [n=184]", "Uncapped (<$500,000) [n=3,944]"),
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.boxplot(data=plot_df, x="District_Group", y="Absolute_Error", ax=ax1, hue="District_Group", palette=["#2ca02c", "#d62728"], legend=False)
    ax1.set_title("Absolute Error by Target Capping Status", fontsize=12, weight="bold")
    ax1.set_ylabel("Absolute Prediction Error ($ USD)")

    sns.scatterplot(
        data=plot_df,
        x="Actual",
        y="Predicted",
        hue="District_Group",
        palette=["#2ca02c", "#d62728"],
        alpha=0.45,
        s=20,
        ax=ax2,
    )
    ax2.plot([0, 550000], [0, 550000], "k--", linewidth=1.5, label="Ideal (y = x)")
    ax2.axvline(TARGET_CAP_VALUE, color="purple", linestyle=":", label="Cap Line")
    ax2.set_title("Actual vs. Predicted Highlighting Capped Ceiling", fontsize=12, weight="bold")
    ax2.set_xlabel("Actual Median House Value ($)")
    ax2.legend(loc="upper left", frameon=True)

    plt.suptitle("Target Ceiling Effect Analysis ($500,001 Cap)", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_mlr_coefficients_chart(feature_names: list, coefficients: np.ndarray, save_path: Path = None):
    """Bar plot of learned standardized MLR regression coefficients."""
    coef_series = pd.Series(coefficients, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in coef_series.values]

    ax.barh(coef_series.index, coef_series.values, color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.set_title("Standardized Multiple Linear Regression Coefficients", fontsize=13, weight="bold")
    ax.set_xlabel("Coefficient Magnitude ($ Impact per Standardized Unit)")

    for i, c in enumerate(coef_series.values):
        offset = 1500 if c >= 0 else -1500
        ha = "left" if c >= 0 else "right"
        ax.text(c + offset, i, f"${c:+,.0f}", va="center", ha=ha, fontsize=8.5, weight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def export_metrics_summary(metrics_dict: dict, json_path: Path, csv_path: Path):
    """Export calculated metrics to JSON and CSV formats."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # Convert comparison list to CSV
    if "comparison_table" in metrics_dict:
        df_comp = pd.DataFrame(metrics_dict["comparison_table"])
        df_comp.to_csv(csv_path, index=False)
