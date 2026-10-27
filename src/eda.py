"""
Exploratory Data Analysis (EDA) module for Property Price Prediction.
Generates comprehensive statistics and publication-quality visualizations.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import (
    FIGURES_DIR,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_CAP_VALUE,
)

# Set high-quality plotting aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def compute_basic_info(df: pd.DataFrame) -> dict:
    """Return dictionary of dataset shape, null counts, duplicate counts, and types."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    missing_summary = pd.DataFrame({
        "Missing_Count": null_counts,
        "Missing_Percent": null_pct,
        "Data_Type": df.dtypes.astype(str),
    })

    capped_count = (df[TARGET_COLUMN] >= TARGET_CAP_VALUE - 1).sum()
    capped_pct = (capped_count / len(df)) * 100

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicates": int(df.duplicated().sum()),
        "missing_summary": missing_summary,
        "capped_count": int(capped_count),
        "capped_pct": float(capped_pct),
        "descriptive_stats": df.describe().T,
        "skewness": df.select_dtypes(include=[np.number]).skew(),
    }


def compute_vif(df_num: pd.DataFrame) -> pd.DataFrame:
    """Calculate Variance Inflation Factor (VIF) for numerical features."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_num.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_num.values, i) for i in range(df_num.shape[1])
    ]
    return vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)


def plot_target_distribution(df: pd.DataFrame, save_path: Path = None):
    """Plot distribution of median_house_value with histogram, KDE, and boxplot."""
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [0.25, 0.75]}
    )

    # Boxplot
    sns.boxplot(x=df[TARGET_COLUMN], ax=ax_box, color="#4C72B0", fliersize=3)
    ax_box.set(xlabel="")
    ax_box.set_title("Distribution of Target Variable: Median House Value", pad=10, weight="bold")

    # Histogram & KDE
    sns.histplot(df[TARGET_COLUMN], kde=True, ax=ax_hist, color="#4C72B0", bins=50, stat="density")
    ax_hist.axvline(df[TARGET_COLUMN].median(), color="red", linestyle="--", linewidth=1.8, label=f"Median: ${df[TARGET_COLUMN].median():,.0f}")
    ax_hist.axvline(df[TARGET_COLUMN].mean(), color="green", linestyle="-.", linewidth=1.8, label=f"Mean: ${df[TARGET_COLUMN].mean():,.0f}")
    ax_hist.axvline(TARGET_CAP_VALUE, color="darkorange", linestyle=":", linewidth=2, label="Capped Ceiling ($500,001)")
    
    ax_hist.set_xlabel("Median House Value ($ USD)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_numerical_distributions(df: pd.DataFrame, save_path: Path = None):
    """Plot histograms and KDE curves for all 8 numerical features."""
    cols = NUMERICAL_FEATURES
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#336699", bins=35)
        axes[i].set_title(f"{col}\n(Skew: {df[col].skew():.2f})", fontsize=11, weight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

    plt.suptitle("Distributions and Skewness of Numerical Features", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_numerical_boxplots(df: pd.DataFrame, save_path: Path = None):
    """Plot standardized boxplots for numerical features to visualize outliers."""
    cols = NUMERICAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(y=df[col].dropna(), ax=axes[i], color="#55A868", fliersize=2)
        axes[i].set_title(col, fontsize=11, weight="bold")
        axes[i].set_ylabel("Value")

    plt.suptitle("Boxplots of Numerical Features (Outlier Inspection)", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path = None):
    """Generate annotated correlation heatmap for numerical variables and target."""
    num_cols = NUMERICAL_FEATURES + [TARGET_COLUMN]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Heatmap (Features and Median House Value)", fontsize=14, weight="bold", pad=15)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_income_vs_price(df: pd.DataFrame, save_path: Path = None):
    """Scatter plot with linear regression line between median_income and median_house_value."""
    fig, ax = plt.subplots(figsize=(9, 6))

    sample_df = df.sample(min(3000, len(df)), random_state=42)
    sns.regplot(
        data=sample_df,
        x="median_income",
        y="median_house_value",
        scatter_kws={"alpha": 0.35, "color": "#1f77b4", "s": 18},
        line_kws={"color": "#d62728", "linewidth": 2.5, "label": "OLS Regression Fit"},
        ax=ax,
    )
    ax.axhline(TARGET_CAP_VALUE, color="purple", linestyle="--", linewidth=1.5, label="Ceiling ($500,001)")

    ax.set_title("Relationship: Median Income vs. Median House Value", fontsize=13, weight="bold")
    ax.set_xlabel("Median Income (Tens of Thousands $ USD)")
    ax.set_ylabel("Median House Value ($ USD)")
    ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_ocean_proximity_analysis(df: pd.DataFrame, save_path: Path = None):
    """Analyze ocean_proximity: distribution and median house values across categories."""
    fig, (ax_count, ax_box) = plt.subplots(1, 2, figsize=(14, 5.5))

    order = df["ocean_proximity"].value_counts().index

    # Category Frequency
    sns.countplot(data=df, x="ocean_proximity", order=order, ax=ax_count, hue="ocean_proximity", palette="Blues_r", legend=False)
    ax_count.set_title("Frequency of Districts by Ocean Proximity", fontsize=12, weight="bold")
    ax_count.set_xlabel("Ocean Proximity")
    ax_count.set_ylabel("Number of Districts")
    for p in ax_count.patches:
        ax_count.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontsize=9)

    # Boxplot of Price by Category
    sns.boxplot(data=df, x="ocean_proximity", y="median_house_value", order=order, ax=ax_box, hue="ocean_proximity", palette="Blues_r", legend=False)
    ax_box.set_title("Median House Value by Ocean Proximity", fontsize=12, weight="bold")
    ax_box.set_xlabel("Ocean Proximity")
    ax_box.set_ylabel("Median House Value ($ USD)")

    plt.suptitle("Categorical Analysis: Ocean Proximity", fontsize=14, weight="bold", y=1.03)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_california_geography(df: pd.DataFrame, save_path: Path = None):
    """Geographic scatter plot of California housing prices and population."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df["median_house_value"],
        cmap="jet",
        alpha=0.45,
        s=df["population"] / 100,
        label="Population / 100",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Median House Value ($ USD)", fontsize=11)

    ax.set_title("California Housing Geography: Price & Population Distribution", fontsize=13, weight="bold")
    ax.set_xlabel("Longitude (Degrees)")
    ax.set_ylabel("Latitude (Degrees)")
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def plot_vif_barchart(vif_df: pd.DataFrame, save_path: Path = None):
    """Horizontal bar plot of Variance Inflation Factors (VIF) with multicollinearity thresholds."""
    fig, ax = plt.subplots(figsize=(9, 5))

    vif_sorted = vif_df.sort_values(by="VIF", ascending=True)
    colors = ["#d62728" if v > 10 else "#ff7f0e" if v > 5 else "#2ca02c" for v in vif_sorted["VIF"]]

    ax.barh(vif_sorted["Feature"], vif_sorted["VIF"], color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(5, color="orange", linestyle="--", linewidth=1.5, label="VIF = 5 (Moderate Multicollinearity)")
    ax.axvline(10, color="red", linestyle="--", linewidth=1.5, label="VIF = 10 (Severe Multicollinearity)")

    ax.set_title("Variance Inflation Factor (VIF) by Feature", fontsize=13, weight="bold")
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.legend(loc="lower right", frameon=True)

    for i, v in enumerate(vif_sorted["VIF"]):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9, weight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
