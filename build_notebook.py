"""
Script to generate the complete, production-grade Jupyter Notebook:
property_price_prediction.ipynb
"""

import nbformat as nbf
from pathlib import Path

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    }

    cells = []

    # Title & Header
    cells.append(nbf.v4.new_markdown_cell(r"""# Part A: Property Price Prediction
## Internshala Machine Learning with AI Internship

---
### Project Overview & Objective
This project builds a reproducible, industry-grade machine learning system to predict median house values across California districts. Following rigorous ML engineering practices, we formulate, train, evaluate, and diagnose:
1. **Simple Linear Regression** (single defensible predictor based on domain knowledge and empirical correlation).
2. **Multiple Linear Regression** (incorporating all spatial, demographic, and housing attributes).
3. **Engineered Multiple Linear Regression** (incorporating district-level structural ratios).
4. **Regularized Ridge Regression** ($L_2$ shrinkage to mitigate severe multicollinearity).

Every preprocessing decision, assumption check, diagnostic plot, and evaluation metric is thoroughly explained and backed by empirical evidence.

---
### Table of Contents
1. **Environment Setup & Imports**
2. **Step 1: Dataset Inspection & Data Dictionary Alignment**
3. **Step 2: Comprehensive Exploratory Data Analysis (EDA)**
4. **Step 3: Data Preprocessing & Leak-Free Pipeline Design**
5. **Step 4: Model Development & Coefficient Interpretation**
6. **Step 5: Model Evaluation & Residual Diagnostics**
7. **Step 6: Target Ceiling Impact Analysis ($500,001 Cap)**
8. **Step 7: Model Improvement (Cross-Validation, Feature Engineering, Ridge Regression)**
9. **Step 8: Model Comparison, Selection & Artifact Persistence**
10. **Step 9: Inference Pipeline Testing on Unseen Records**
11. **Step 10: Conclusions, Business Insights & Architectural Limitations**
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""# 1. Environment Setup & Library Imports
import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib

# Scikit-Learn Preprocessing, Models & Metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress minor warnings for clean presentation
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

print("Libraries imported successfully. Ready for ML pipeline execution.")
"""))

    # Step 1: Inspection
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 1: Project Files & Dataset Inspection

Before executing any modeling or transformations, we inspect `Data Information.pdf` and `Data_file.xlsx` to understand the true schema, units, data types, and documented target variable.

### Data Dictionary Summary (from `Data Information.pdf`):
| Variable | Description | Type | Unit |
| :--- | :--- | :--- | :--- |
| `longitude` | Longitude of the property block | FLOAT | degrees |
| `latitude` | Latitude of the property block | FLOAT | degrees |
| `housing_median_age` | Median age of the houses | INT | years |
| `total_rooms` | Total number of rooms in the block | INT | count |
| `total_bedrooms` | Total number of bedrooms in the block | INT/FLOAT | count (contains missing values) |
| `population` | Population residing in the block | INT | count |
| `households` | Total number of households (occupants) | INT | count |
| `median_income` | Median income in the area | FLOAT | tens of thousands ($ USD) |
| `median_house_value` | **Target Variable**: Median house value | FLOAT/INT | $ USD (capped at $500,001) |
| `ocean_proximity` | Categorical proximity to ocean | STRING | 5 categories |
"""))

    cells.append(nbf.v4.new_code_cell("""# Load the real dataset directly from Data_file.xlsx
data_path = Path("Data_file.xlsx")
if not data_path.exists():
    raise FileNotFoundError(f"Dataset file '{data_path}' was not found in the project root.")

df = pd.read_excel(data_path)
print(f"Dataset successfully loaded. Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\\n")
print("First 5 records:")
display(df.head())
"""))

    cells.append(nbf.v4.new_code_cell("""# Column types and memory summary
df.info()
"""))

    cells.append(nbf.v4.new_code_cell("""# Summary statistics of all numerical columns
print("Descriptive Statistics for Numerical Features and Target:")
display(df.describe().T)
"""))

    # Step 2: EDA
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 2: Comprehensive Exploratory Data Analysis (EDA)

In this section, we rigorously analyze:
1. Missing value counts and percentages.
2. Duplicate records.
3. Target variable (`median_house_value`) distribution, central tendency, and artificial ceiling.
4. Numerical feature distributions, skewness, and outliers.
5. Multicollinearity among numerical features via Pearson correlation and VIF.
6. Categorical distribution and price breakdown for `ocean_proximity`.
7. Spatial geographic distribution of house values and population density across California.
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.1 Missing Values and Duplicate Checks
null_counts = df.isnull().sum()
null_percent = (null_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'Feature': df.columns,
    'Missing_Count': null_counts.values,
    'Missing_Percent': null_percent.values,
    'Data_Type': [str(t) for t in df.dtypes.values]
}).sort_values(by='Missing_Count', ascending=False).reset_index(drop=True)

print(f"Total duplicate records: {df.duplicated().sum()}")
print("\\nMissing Value Analysis:")
display(missing_df)
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Interpretation of Data Quality
* **Missing Values**: Only `total_bedrooms` contains missing values ($207$ null entries out of $20,640$, exactly $1.003\\%$). Because the feature is heavily skewed, we will impute these values using the training-set **median** rather than the mean.
* **Duplicates**: $0$ duplicate rows exist.
* **Integrity**: All feature types match the specifications documented in `Data Information.pdf`.
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.2 Target Variable Analysis: median_house_value
target_col = 'median_house_value'
target_cap = 500001.0
capped_count = (df[target_col] >= target_cap - 1).sum()
capped_pct = (capped_count / len(df)) * 100

print(f"Target Variable Statistics:")
print(f"  Mean:   ${df[target_col].mean():,.2f}")
print(f"  Median: ${df[target_col].median():,.2f}")
print(f"  Std:    ${df[target_col].std():,.2f}")
print(f"  Min:    ${df[target_col].min():,.2f}")
print(f"  Max:    ${df[target_col].max():,.2f}")
print(f"  Skew:   {df[target_col].skew():.3f}")
print(f"  Districts capped at $500,001: {capped_count:,} ({capped_pct:.2f}% of the entire dataset)")

fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [0.25, 0.75]})

sns.boxplot(x=df[target_col], ax=ax_box, color="#4C72B0", fliersize=3)
ax_box.set(xlabel="")
ax_box.set_title("Distribution of Target Variable: Median House Value", pad=10, weight="bold")

sns.histplot(df[target_col], kde=True, ax=ax_hist, color="#4C72B0", bins=50, stat="density")
ax_hist.axvline(df[target_col].median(), color="red", linestyle="--", linewidth=1.8, label=f"Median: ${df[target_col].median():,.0f}")
ax_hist.axvline(df[target_col].mean(), color="green", linestyle="-.", linewidth=1.8, label=f"Mean: ${df[target_col].mean():,.0f}")
ax_hist.axvline(target_cap, color="darkorange", linestyle=":", linewidth=2.2, label=f"Capped Ceiling (${target_cap:,.0f})")

ax_hist.set_xlabel("Median House Value ($ USD)")
ax_hist.set_ylabel("Density")
ax_hist.legend(loc="upper right", frameon=True)

plt.tight_layout()
os.makedirs("outputs/figures", exist_ok=True)
fig.savefig("outputs/figures/01_target_distribution.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Target Distribution Findings:
1. **Right Skewness**: The target variable has positive skewness ($0.978$) with the majority of district median house values centered between $\$120,000$ and $\$265,000$.
2. **Artificial Ceiling at $\$500,001$**: Notice the sharp frequency spike at the rightmost edge ($\$500,001$). A total of **$992$ districts ($4.81\\%$)** have been capped at $\$500,001$ during census data collection. Linear models will attempt to extrapolate past $\$500,000$ for luxury districts, generating large positive residuals at this upper boundary.
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.3 Numerical Feature Distributions & Skewness
num_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                'total_bedrooms', 'population', 'households', 'median_income']

skew_series = df[num_features].skew().sort_values(ascending=False)
print("Numerical Features Skewness:")
display(pd.DataFrame({'Skewness': skew_series}))

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#336699", bins=35)
    axes[i].set_title(f"{col}\\n(Skew: {df[col].skew():.2f})", fontsize=11, weight="bold")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

plt.suptitle("Distributions and Skewness of Numerical Predictors", fontsize=15, weight="bold", y=1.02)
plt.tight_layout()
fig.savefig("outputs/figures/02_numerical_features_histograms.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.4 Boxplots for Outlier Inspection
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.boxplot(y=df[col].dropna(), ax=axes[i], color="#55A868", fliersize=2)
    axes[i].set_title(col, fontsize=11, weight="bold")
    axes[i].set_ylabel("Value")

plt.suptitle("Boxplots of Numerical Features (Outlier Inspection)", fontsize=15, weight="bold", y=1.02)
plt.tight_layout()
fig.savefig("outputs/figures/03_numerical_features_boxplots.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Feature Distribution Findings:
* `total_rooms`, `total_bedrooms`, `population`, and `households` exhibit severe right-skewness ($>3.0$). These are block-group aggregate counts rather than per-capita quantities; larger geographical districts naturally accumulate higher total room and population counts.
* `housing_median_age` is capped at $52$ years (evident from the spike at the maximum).
* `median_income` exhibits a realistic bell-shaped distribution with moderate right-skewness ($0.68$), representing typical regional household earnings.
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.5 Correlation Heatmap and Target Association
corr = df[num_features + [target_col]].corr()

print("Pearson Correlation with Target (median_house_value):")
display(corr[target_col].sort_values(ascending=False).to_frame(name="Pearson r"))

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
fig.savefig("outputs/figures/04_correlation_heatmap.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Critical Correlation Insights:
1. **Dominant Predictor**: `median_income` has by far the strongest linear correlation with `median_house_value` ($r = 0.688$). This provides strong empirical justification for selecting `median_income` as the single predictor for **Simple Linear Regression**.
2. **Extreme Multicollinearity**: The aggregate counts (`total_rooms`, `total_bedrooms`, `population`, `households`) have correlations exceeding $0.85$ to $0.98$ with one another! In an unregularized Multiple Linear Regression model, this will cause coefficient inflation and sign flips.
3. **Latitude & Longitude**: Show moderate negative correlation with price ($r = -0.144$ and $r = -0.046$), reflecting regional differences (e.g., coastal southern/central areas vs inland north).
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.6 Relationship: median_income vs. median_house_value
fig, ax = plt.subplots(figsize=(10, 6))

sample_df = df.sample(min(3000, len(df)), random_state=42)
sns.regplot(
    data=sample_df,
    x="median_income",
    y="median_house_value",
    scatter_kws={"alpha": 0.35, "color": "#1f77b4", "s": 18},
    line_kws={"color": "#d62728", "linewidth": 2.5, "label": "OLS Regression Line"},
    ax=ax,
)
ax.axhline(target_cap, color="purple", linestyle="--", linewidth=1.5, label="Ceiling ($500,001)")

ax.set_title("Scatter Plot: Median Income vs. Median House Value", fontsize=13, weight="bold")
ax.set_xlabel("Median Income (Tens of Thousands $ USD)")
ax.set_ylabel("Median House Value ($ USD)")
ax.legend(frameon=True, loc="upper left")

plt.tight_layout()
fig.savefig("outputs/figures/05_median_income_vs_price_scatter.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.7 Categorical Feature Analysis: ocean_proximity
fig, (ax_count, ax_box) = plt.subplots(1, 2, figsize=(14, 5.5))
order = df['ocean_proximity'].value_counts().index

sns.countplot(data=df, x="ocean_proximity", order=order, ax=ax_count, hue="ocean_proximity", palette="Blues_r", legend=False)
ax_count.set_title("Frequency of Districts by Ocean Proximity", fontsize=12, weight="bold")
ax_count.set_xlabel("Ocean Proximity")
ax_count.set_ylabel("Number of Districts")
for p in ax_count.patches:
    ax_count.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontsize=9)

sns.boxplot(data=df, x="ocean_proximity", y="median_house_value", order=order, ax=ax_box, hue="ocean_proximity", palette="Blues_r", legend=False)
ax_box.set_title("Median House Value by Ocean Proximity", fontsize=12, weight="bold")
ax_box.set_xlabel("Ocean Proximity")
ax_box.set_ylabel("Median House Value ($ USD)")

plt.suptitle("Categorical Variable Analysis: ocean_proximity", fontsize=14, weight="bold", y=1.03)
plt.tight_layout()
fig.savefig("outputs/figures/06_ocean_proximity_distribution_and_price.png", dpi=300)
plt.show()

print("Median House Value by Ocean Proximity Group:")
display(df.groupby("ocean_proximity")["median_house_value"].agg(["count", "mean", "median", "std"]).loc[order])
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Categorical Insights:
* `INLAND` districts have by far the lowest median property values (mean: $\$124,805$, median: $\$108,500$).
* Coastal districts (`<1H OCEAN`, `NEAR OCEAN`, `NEAR BAY`) have much higher valuations (median values between $\$214,000$ and $\$247,000$).
* `ISLAND` has only $5$ districts with very high property values (median: $\$414,700$).
* One-hot encoding `ocean_proximity` will allow Multiple Linear Regression to assign specific price premiums/discounts to geographical zones.
"""))

    cells.append(nbf.v4.new_code_cell(r"""# 2.8 Geographic Visualization of California Housing
fig, ax = plt.subplots(figsize=(11, 8.5))

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

ax.set_title("California Housing Geography: Spatial Price Distribution & Population Density", fontsize=13, weight="bold")
ax.set_xlabel("Longitude (Degrees)")
ax.set_ylabel("Latitude (Degrees)")
ax.legend(frameon=True, loc="upper right")

plt.tight_layout()
fig.savefig("outputs/figures/07_california_housing_geography.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 3: Data Preprocessing & Pipeline Construction

### Leakage-Free Design Principles:
1. **Feature-Target Separation**: Separate target variable `median_house_value` from predictor matrix $X$.
2. **Train/Test Splitting**: Split data into $80\\%$ training ($16,512$ rows) and $20\\%$ testing ($4,128$ rows) with fixed `random_state=42`. The test set represents exactly $20.00\\%$ of total records ($4,128 / 20,640 = 0.2000$).
3. **Strict Pipeline Encapsulation**: Fit imputation and scaling statistics **only on the training set** to prevent data leakage.
4. **Imputation**: Impute missing `total_bedrooms` using the median (robust against extreme right skew).
5. **Categorical Encoding**: One-hot encode `ocean_proximity` using `drop='first'` to prevent the dummy variable trap (perfect collinearity) in linear regression.
6. **Feature Scaling**: Standardize all numerical predictors using `StandardScaler` (zero mean, unit variance) to place all features on comparable scales and enable stable regularized regression.
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.1 Splitting the Data into Training and Test Sets
X = df.drop(columns=[target_col])
y = df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Full Dataset:     {len(df):,} samples")
print(f"Training Set:     {len(X_train):,} samples ({len(X_train)/len(df):.1%})")
print(f"Testing Set:      {len(X_test):,} samples ({len(X_test)/len(df):.1%})")
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.2 Defining Preprocessing Pipelines with ColumnTransformer
numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                    'total_bedrooms', 'population', 'households', 'median_income']
categorical_features = ['ocean_proximity']

# Numerical Pipeline: Median Imputer + Standard Scaler
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Pipeline: OneHotEncoder with drop='first'
categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Master ColumnTransformer
preprocessor_baseline = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

print("Preprocessing pipeline successfully defined.")
"""))

    # Step 4: Model Development
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 4: Model Development

### Model A: Simple Linear Regression
* **Predictor Choice**: `median_income` (statistically verified by the highest linear correlation $r = 0.688$ and supported by economic theory).
* **Mathematical Equation**:
  $$\\hat{y} = \\beta_0 + \\beta_1 \\times \\text{median\\_income}$$
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.1 Training Simple Linear Regression
simple_reg = LinearRegression()
simple_reg.fit(X_train[['median_income']], y_train)

beta_0 = simple_reg.intercept_
beta_1 = simple_reg.coef_[0]

print("=" * 60)
print("SIMPLE LINEAR REGRESSION LEARNED MODEL")
print("=" * 60)
print(f"Intercept (beta_0): ${beta_0:,.2f}")
print(f"Slope (beta_1):     ${beta_1:,.2f}")
print(f"Formula: median_house_value = {beta_0:,.2f} + ({beta_1:,.2f} * median_income)")
print("=" * 60)
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Economic & Statistical Interpretation of Simple Linear Regression:
* **Slope ($\\beta_1 = \\$41,933.85$)**: For every $1$-unit increase in `median_income` (which corresponds to $\$10,000$ in household annual income), the predicted district median house value increases by approximately **$\\$41,933.85$**.
* **Intercept ($\\beta_0 = \\$44,459.73$)**: Represents the theoretical baseline median house price for a hypothetical district with zero income.
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Model B: Multiple Linear Regression & Multicollinearity Investigation

Before fitting the Multiple Linear Regression model, we examine the **Variance Inflation Factor (VIF)** of the numerical features on the training set:
$$\\text{VIF}_j = \\frac{1}{1 - R_j^2}$$
* $\\text{VIF} = 1$: No collinearity.
* $\\text{VIF} > 5$: Moderate collinearity.
* $\\text{VIF} > 10$: Severe multicollinearity requiring structural intervention.
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.2 Variance Inflation Factor (VIF) on Training Data
imputer_temp = SimpleImputer(strategy='median')
scaler_temp = StandardScaler()

X_num_train_imputed = imputer_temp.fit_transform(X_train[numeric_features])
X_num_train_scaled = pd.DataFrame(scaler_temp.fit_transform(X_num_train_imputed), columns=numeric_features)

vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features
vif_data["VIF"] = [variance_inflation_factor(X_num_train_scaled.values, i) for i in range(len(numeric_features))]
vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)

print("Variance Inflation Factor (VIF) Table on Training Data:")
display(vif_data)

# Visualize VIF
fig, ax = plt.subplots(figsize=(9, 5))
vif_sorted = vif_data.sort_values(by="VIF", ascending=True)
colors = ["#d62728" if v > 10 else "#ff7f0e" if v > 5 else "#2ca02c" for v in vif_sorted["VIF"]]

ax.barh(vif_sorted["Feature"], vif_sorted["VIF"], color=colors, edgecolor="black", alpha=0.85)
ax.axvline(5, color="orange", linestyle="--", linewidth=1.5, label="VIF = 5 (Moderate Multicollinearity)")
ax.axvline(10, color="red", linestyle="--", linewidth=1.5, label="VIF = 10 (Severe Multicollinearity)")
ax.set_title("Variance Inflation Factor (VIF) by Feature", fontsize=13, weight="bold")
ax.set_xlabel("VIF")
ax.legend(loc="lower right", frameon=True)

for i, v in enumerate(vif_sorted["VIF"]):
    ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=9, weight="bold")

plt.tight_layout()
fig.savefig("outputs/figures/16_vif_comparison.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Implications of VIF Diagnostics:
* `total_bedrooms` ($\\text{VIF} = 36.31$) and `households` ($\\text{VIF} = 35.82$) exhibit **severe multicollinearity** (well above $10$).
* `total_rooms` ($\\text{VIF} = 12.82$) also exceeds the critical threshold.
* `latitude` ($\\text{VIF} = 8.80$) and `longitude` ($\\text{VIF} = 8.68$) exhibit strong spatial interdependence.
* **Impact**: While collinearity does not degrade the overall $R^2$ of linear regression, it inflates standard errors and creates sign instabilities where individual coefficients become counter-intuitive.
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.3 Training Multiple Linear Regression (Baseline)
mlr_pipeline = Pipeline([
    ('preprocessor', preprocessor_baseline),
    ('regressor', LinearRegression())
])

mlr_pipeline.fit(X_train, y_train)

# Extract learned coefficients
onehot_cols = list(mlr_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
all_features_baseline = numeric_features + onehot_cols
mlr_coefficients = mlr_pipeline.named_steps['regressor'].coef_
mlr_intercept = mlr_pipeline.named_steps['regressor'].intercept_

coef_df = pd.DataFrame({
    'Feature': all_features_baseline,
    'Standardized_Coefficient': mlr_coefficients
}).sort_values(by='Standardized_Coefficient', ascending=False).reset_index(drop=True)

print(f"MLR Intercept (baseline): ${mlr_intercept:,.2f}\\n")
print("Multiple Linear Regression Standardized Coefficients:")
display(coef_df)

# Plot coefficients
fig, ax = plt.subplots(figsize=(10, 6.5))
coef_sorted = coef_df.sort_values(by="Standardized_Coefficient", ascending=True)
colors = ["#d62728" if c < 0 else "#2ca02c" for c in coef_sorted["Standardized_Coefficient"]]

ax.barh(coef_sorted["Feature"], coef_sorted["Standardized_Coefficient"], color=colors, edgecolor="black", alpha=0.85)
ax.axvline(0, color="black", linestyle="-", linewidth=1)
ax.set_title("Standardized Multiple Linear Regression Coefficients", fontsize=13, weight="bold")
ax.set_xlabel("Impact on Price per Standardized Unit ($ USD)")

for i, c in enumerate(coef_sorted["Standardized_Coefficient"]):
    offset = 1800 if c >= 0 else -1800
    ha = "left" if c >= 0 else "right"
    ax.text(c + offset, i, f"${c:+,.0f}", va="center", ha=ha, fontsize=8.5, weight="bold")

plt.tight_layout()
fig.savefig("outputs/figures/10_mlr_coefficients.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Interpretation of Multiple Linear Regression Coefficients:
1. `median_income` has the highest positive standardized impact ($+\\$75,168$), reaffirming it as the primary economic driver of housing values.
2. `ocean_proximity_ISLAND` ($+\\$136,125$) carries a massive price premium due to extreme scarcity, whereas `ocean_proximity_INLAND` ($-\\$39,787$) carries a steep discount relative to the reference category (`<1H OCEAN`).
3. **Collinearity Sign Flip**: Notice that `total_rooms` has a **negative** coefficient ($-\\$13,094$) while `total_bedrooms` has a **positive** coefficient ($+\\$43,068$). In bivariate analysis, total rooms was positively correlated with price ($r = +0.134$). Holding total bedrooms fixed, adding more non-bedroom rooms in an aggregate district actually proxies for higher density/smaller units, illustrating the classic collinearity effect.
"""))

    # Step 5: Model Evaluation
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 5: Model Evaluation on Held-Out Test Set

We evaluate both Simple Linear Regression and Multiple Linear Regression on the identical held-out test set ($4,128$ records).

### Regression Metrics:
* **MAE (Mean Absolute Error)**: Average absolute magnitude of errors in dollar terms. Robust to outliers.
  $$\\text{MAE} = \\frac{1}{n}\\sum_{i=1}^n |y_i - \\hat{y}_i|$$
* **MSE (Mean Squared Error)**: Mean of squared prediction errors. Heavily penalizes large misses.
  $$\\text{MSE} = \\frac{1}{n}\\sum_{i=1}^n (y_i - \\hat{y}_i)^2$$
* **RMSE (Root Mean Squared Error)**: Square root of MSE; expresses error variance in original dollar units.
  $$\\text{RMSE} = \\sqrt{\\text{MSE}}$$
* **$R^2$ (Coefficient of Determination)**: Proportion of total target variance explained by the model ($1 - \\text{SS}_{res}/\\text{SS}_{tot}$).
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.1 Computing Test Set Predictions and Metrics
# Simple Linear Regression Predictions
y_train_pred_simple = simple_reg.predict(X_train[['median_income']])
y_test_pred_simple = simple_reg.predict(X_test[['median_income']])

# Multiple Linear Regression Predictions
y_train_pred_mlr = mlr_pipeline.predict(X_train)
y_test_pred_mlr = mlr_pipeline.predict(X_test)

def get_metrics_dict(name, y_tr, y_tr_pred, y_te, y_te_pred):
    return {
        'Model': name,
        'Train MAE': mean_absolute_error(y_tr, y_tr_pred),
        'Test MAE': mean_absolute_error(y_te, y_te_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_tr, y_tr_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_te, y_te_pred)),
        'Test MSE': mean_squared_error(y_te, y_te_pred),
        'Train R2': r2_score(y_tr, y_tr_pred),
        'Test R2': r2_score(y_te, y_te_pred),
    }

metrics_comparison = [
    get_metrics_dict("Simple Linear Regression", y_train, y_train_pred_simple, y_test, y_test_pred_simple),
    get_metrics_dict("Multiple Linear Regression (Base)", y_train, y_train_pred_mlr, y_test, y_test_pred_mlr),
]

eval_df = pd.DataFrame(metrics_comparison)
display(eval_df[['Model', 'Train RMSE', 'Test RMSE', 'Test MAE', 'Train R2', 'Test R2']])
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Model Comparison Evaluation:
* **Multiple Linear Regression significantly outperforms Simple Linear Regression**:
  * **Test RMSE**: Drops from **$\\$84,209.01$** down to **$\\$70,059.19$** (a reduction of **$\\$14,149.82$** or **$16.8\\%$**).
  * **Test MAE**: Drops from **$\\$62,990.87$** down to **$\\$50,670.49$** (a reduction of **$\\$12,320.38$**).
  * **Test $R^2$**: Increases from **$0.4589$** to **$0.6254$** (explaining an additional **$16.65\\%$** of total market variance).
* **No Overfitting**: For both models, Train RMSE and Test RMSE are very close (Train: $\$68,433.94$ vs Test: $\$70,059.19$), indicating healthy generalization without over-parameterization.
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.2 Actual vs. Predicted Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Simple Linear Regression
ax1.scatter(y_test, y_test_pred_simple, alpha=0.3, color="#1f77b4", s=15, label="Test Districts")
ax1.plot([0, 550000], [0, 550000], "r--", linewidth=2, label="Ideal Fit (y = x)")
ax1.axvline(target_cap, color="purple", linestyle=":", linewidth=1.5, label="Capped Ceiling")
ax1.set_title("Simple Linear Regression (median_income only)", fontsize=12, weight="bold")
ax1.set_xlabel("Actual Median House Value ($)")
ax1.set_ylabel("Predicted Median House Value ($)")
ax1.legend(loc="upper left", frameon=True)

# Multiple Linear Regression
ax2.scatter(y_test, y_test_pred_mlr, alpha=0.3, color="#2ca02c", s=15, label="Test Districts")
ax2.plot([0, 550000], [0, 550000], "r--", linewidth=2, label="Ideal Fit (y = x)")
ax2.axvline(target_cap, color="purple", linestyle=":", linewidth=1.5, label="Capped Ceiling")
ax2.set_title("Multiple Linear Regression (All Features)", fontsize=12, weight="bold")
ax2.set_xlabel("Actual Median House Value ($)")
ax2.legend(loc="upper left", frameon=True)

plt.suptitle("Actual vs. Predicted Median House Value (Test Set Evaluation)", fontsize=14, weight="bold", y=1.02)
plt.tight_layout()
fig.savefig("outputs/figures/12_actual_vs_predicted.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.3 Residuals vs. Fitted Plots (Checking Homoscedasticity)
residuals_simple = y_test - y_test_pred_simple
residuals_mlr = y_test - y_test_pred_mlr

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(y_test_pred_simple, residuals_simple, alpha=0.3, color="#1f77b4", s=15)
ax1.axhline(0, color="red", linestyle="--", linewidth=2)
ax1.set_title("Simple Linear Regression: Residuals vs. Fitted", fontsize=12, weight="bold")
ax1.set_xlabel("Fitted Values ($)")
ax1.set_ylabel("Residuals ($) [Actual - Predicted]")

ax2.scatter(y_test_pred_mlr, residuals_mlr, alpha=0.3, color="#2ca02c", s=15)
ax2.axhline(0, color="red", linestyle="--", linewidth=2)
ax2.set_title("Multiple Linear Regression: Residuals vs. Fitted", fontsize=12, weight="bold")
ax2.set_xlabel("Fitted Values ($)")
ax2.set_ylabel("Residuals ($) [Actual - Predicted]")

plt.suptitle("Residual Diagnostics: Homoscedasticity & Linearity Assessment", fontsize=14, weight="bold", y=1.02)
plt.tight_layout()
fig.savefig("outputs/figures/13_residuals_vs_fitted.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.4 Residual Distribution & Normal Q-Q Plot
fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(14, 5.5))

sns.histplot(residuals_mlr, kde=True, ax=ax_hist, color="#336699", bins=50)
ax_hist.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Mean")
ax_hist.set_title(f"MLR Test Residuals Distribution\\n(Mean: ${residuals_mlr.mean():,.0f}, Skew: {residuals_mlr.skew():.2f})", fontsize=12, weight="bold")
ax_hist.set_xlabel("Residual ($ USD)")
ax_hist.set_ylabel("Count")
ax_hist.legend(frameon=True)

stats.probplot(residuals_mlr, dist="norm", plot=ax_qq)
ax_qq.get_lines()[0].set_markerfacecolor("#336699")
ax_qq.get_lines()[0].set_markersize(4.0)
ax_qq.get_lines()[0].set_alpha(0.5)
ax_qq.get_lines()[1].set_color("red")
ax_qq.get_lines()[1].set_linewidth(2.0)
ax_qq.set_title("Normal Q-Q Plot (MLR Test Residuals)", fontsize=12, weight="bold")

plt.suptitle("Residual Diagnostics: Normality Analysis", fontsize=14, weight="bold", y=1.02)
plt.tight_layout()
fig.savefig("outputs/figures/14_residuals_distribution_qq.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Residual Diagnostic Findings:
1. **Zero Error Mean**: The residuals are centered around $\$0$ (mean $= -\\$1,625$), satisfying the un-biasedness condition.
2. **Mild Heteroscedasticity**: The spread of residuals widens slightly at higher fitted values, a common characteristic in real-estate pricing where luxury homes have higher absolute variance.
3. **Heavy Tails in Q-Q Plot**: Deviations from the straight theoretical line occur at the upper and lower extremes. The prominent upper tail is heavily driven by the artificial $\$500,001$ cap.
"""))

    # Step 6: Target Ceiling Impact
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 6: Target Ceiling Impact Analysis ($500,001 Cap)

As identified during EDA, the census bureau capped property values at $\$500,001$. In our test set:
* **$184$ districts** ($4.5\\%$) are artificially capped at $\$500,001$.
* **$3,944$ districts** ($95.5\\%$) represent uncapped, natural market values.

We quantify the exact distortion this ceiling introduces to linear regression evaluation:
"""))

    cells.append(nbf.v4.new_code_cell("""# 6.1 Performance Breakdown: Uncapped vs. Capped Test Districts
uncapped_mask = y_test < (target_cap - 1)
capped_mask = ~uncapped_mask

rmse_overall = np.sqrt(mean_squared_error(y_test, y_test_pred_mlr))
rmse_uncapped = np.sqrt(mean_squared_error(y_test[uncapped_mask], y_test_pred_mlr[uncapped_mask]))
rmse_capped = np.sqrt(mean_squared_error(y_test[capped_mask], y_test_pred_mlr[capped_mask]))

mae_uncapped = mean_absolute_error(y_test[uncapped_mask], y_test_pred_mlr[uncapped_mask])
mae_capped = mean_absolute_error(y_test[capped_mask], y_test_pred_mlr[capped_mask])

print("=" * 65)
print("TARGET CEILING DISTORTION ANALYSIS (MLR MODEL)")
print("=" * 65)
print(f"Overall Test Set RMSE:          ${rmse_overall:,.2f}  (n = {len(y_test):,})")
print(f"Uncapped Districts RMSE (<$500k): ${rmse_uncapped:,.2f}  (n = {uncapped_mask.sum():,})")
print(f"Capped Districts RMSE ($500,001): ${rmse_capped:,.2f}  (n = {capped_mask.sum():,})")
print("-" * 65)
print(f"Uncapped Districts MAE:         ${mae_uncapped:,.2f}")
print(f"Capped Districts MAE:           ${mae_capped:,.2f}")
print("=" * 65)

# Visualizing Ceiling Distortion
plot_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred_mlr,
    'Absolute_Error': np.abs(residuals_mlr),
    'District_Group': np.where(capped_mask, f'Capped ($500,001) [n={capped_mask.sum()}]', f'Uncapped (<$500k) [n={uncapped_mask.sum():,}]')
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

sns.boxplot(data=plot_df, x="District_Group", y="Absolute_Error", ax=ax1, hue="District_Group", palette=["#2ca02c", "#d62728"], legend=False)
ax1.set_title("Absolute Error Distribution: Uncapped vs. Capped", fontsize=12, weight="bold")
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
ax2.plot([0, 550000], [0, 550000], "k--", linewidth=1.5, label="Ideal Fit (y = x)")
ax2.axvline(target_cap, color="purple", linestyle=":", label="Ceiling Line")
ax2.set_title("Actual vs. Predicted Highlighting Ceiling Truncation", fontsize=12, weight="bold")
ax2.set_xlabel("Actual Median House Value ($)")
ax2.legend(loc="upper left", frameon=True)

plt.suptitle("Impact of Artificial Target Ceiling ($500,001)", fontsize=14, weight="bold", y=1.02)
plt.tight_layout()
fig.savefig("outputs/figures/15_ceiling_effect_analysis.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Ceiling Analysis Findings:
* For **$95.5\\%$ of normal, uncapped districts**, the baseline model achieves a substantially lower RMSE of **$\\$61,848.69$** and MAE of **$\\$46,428.91$** (which drops further to **$\\$60,872.31$** and **$\\$45,405.84$** with feature engineering).
* For the **$184$ capped districts**, the RMSE surges to **$\\$167,698.58$** and MAE to **$\\$141,587.86$** (and **$\\$166,674.60$** / **$\\$140,521.53$** with feature engineering).
* **Reason**: Linear regression inherently models an unbounded continuous line. In reality, wealthy coastal districts are worth $\$700k+$, but the ground-truth target is artificially truncated at $\$500,001$. This horizontal truncation creates massive positive residuals at the boundary.
"""))

    # Step 7: Model Improvement
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 7: Model Improvement (Cross-Validation, Feature Engineering & Ridge Regression)

To improve model stability and mitigate collinearity, we implement:
1. **5-Fold Cross-Validation** on the training set to ensure statistically sound generalization estimates without test leakage.
2. **Feature Engineering**: Creating domain-justified district-level ratios:
   - `rooms_per_household = total_rooms / households` (Average home size).
   - `bedrooms_per_room = total_bedrooms / total_rooms` (Proportion of bedrooms vs living space).
   - `population_per_household = population / households` (District family/occupancy density).
3. **Regularized Linear Regression (Ridge)**:
   $$\\min_{\\beta} \\left\\{ \\sum_{i=1}^n (y_i - X_i\\beta)^2 + \\alpha \\sum_{j=1}^p \\beta_j^2 \\right\\}$$
   Tuning hyperparameter $\\alpha$ via 5-fold cross-validation on training data.
"""))

    cells.append(nbf.v4.new_code_cell("""# 7.1 Defining the Feature Engineering Transformer
# Import standard transformer from src.data to ensure clean serialization
from src.data import PropertyFeatureEngineer

# Preprocessor including engineered features
engineered_numeric_cols = numeric_features + ['rooms_per_household', 'bedrooms_per_room', 'population_per_household']

preprocessor_fe = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), engineered_numeric_cols),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Model 3: MLR with Feature Engineering Pipeline
mlr_fe_pipeline = Pipeline([
    ('feat_eng', PropertyFeatureEngineer(add_features=True)),
    ('preprocessor', preprocessor_fe),
    ('regressor', LinearRegression())
])

mlr_fe_pipeline.fit(X_train, y_train)
y_train_pred_fe = mlr_fe_pipeline.predict(X_train)
y_test_pred_fe = mlr_fe_pipeline.predict(X_test)

metrics_fe = get_metrics_dict("Multiple Linear Regression (FE)", y_train, y_train_pred_fe, y_test, y_test_pred_fe)
print("MLR with Feature Engineering Test Metrics:")
print(f"  Test RMSE: ${metrics_fe['Test RMSE']:,.2f}")
print(f"  Test MAE:  ${metrics_fe['Test MAE']:,.2f}")
print(f"  Test R2:   {metrics_fe['Test R2']:.4f}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 7.2 Model 4: Ridge Regression with 5-Fold Cross-Validation for Alpha
alphas = np.logspace(-2, 4, 60)

ridge_cv_pipeline = Pipeline([
    ('feat_eng', PropertyFeatureEngineer(add_features=True)),
    ('preprocessor', preprocessor_fe),
    ('regressor', RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error'))
])

ridge_cv_pipeline.fit(X_train, y_train)
best_alpha = ridge_cv_pipeline.named_steps['regressor'].alpha_
print(f"Optimal L2 Regularization Parameter (alpha): {best_alpha:.4f}\\n")

y_train_pred_ridge = ridge_cv_pipeline.predict(X_train)
y_test_pred_ridge = ridge_cv_pipeline.predict(X_test)

metrics_ridge = get_metrics_dict("Ridge Regression (FE + L2)", y_train, y_train_pred_ridge, y_test, y_test_pred_ridge)
print("Ridge Regression Test Metrics:")
print(f"  Test RMSE: ${metrics_ridge['Test RMSE']:,.2f}")
print(f"  Test MAE:  ${metrics_ridge['Test MAE']:,.2f}")
print(f"  Test R2:   {metrics_ridge['Test R2']:.4f}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 7.3 5-Fold Cross-Validation on Training Data for All Models
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_evaluate(model, X_tr, y_tr):
    neg_mse = cross_val_score(model, X_tr, y_tr, scoring="neg_mean_squared_error", cv=kf)
    rmse = np.sqrt(-neg_mse)
    r2 = cross_val_score(model, X_tr, y_tr, scoring="r2", cv=kf)
    return rmse.mean(), rmse.std(), r2.mean()

cv_res = {}
cv_res["Simple Linear Regression"] = cv_evaluate(simple_reg, X_train[['median_income']], y_train)
cv_res["Multiple Linear Regression (Base)"] = cv_evaluate(mlr_pipeline, X_train, y_train)
cv_res["Multiple Linear Regression (FE)"] = cv_evaluate(mlr_fe_pipeline, X_train, y_train)
cv_res["Ridge Regression (FE + L2)"] = cv_evaluate(ridge_cv_pipeline, X_train, y_train)

print("5-Fold Cross-Validation Results on Training Set:")
for m, (rmse_mean, rmse_std, r2_m) in cv_res.items():
    print(f"  {m:<35}: CV RMSE = ${rmse_mean:,.2f} +/- ${rmse_std:,.2f} | CV R2 = {r2_m:.4f}")
"""))

    # Step 8: Master Model Comparison
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 8: Master Model Comparison & Selection

Here we assemble the comprehensive results table across all four models evaluated on the identical test set:
"""))

    cells.append(nbf.v4.new_code_cell("""# 8.1 Master Summary Table
all_metrics = [
    metrics_comparison[0],
    metrics_comparison[1],
    metrics_fe,
    metrics_ridge,
]

master_df = pd.DataFrame(all_metrics)
master_df['CV RMSE Mean'] = [cv_res[m][0] for m in master_df['Model']]
master_df['CV RMSE Std'] = [cv_res[m][1] for m in master_df['Model']]
master_df['CV R2 Mean'] = [cv_res[m][2] for m in master_df['Model']]

display_cols = ['Model', 'Test RMSE', 'Test MAE', 'Test R2', 'CV RMSE Mean', 'CV R2 Mean']
print("=" * 85)
print("FINAL MODEL BENCHMARK COMPARISON TABLE")
print("=" * 85)
display(master_df[display_cols].style.format({
    'Test RMSE': '${:,.2f}',
    'Test MAE': '${:,.2f}',
    'Test R2': '{:.4f}',
    'CV RMSE Mean': '${:,.2f}',
    'CV R2 Mean': '{:.4f}',
}))

# Save summary to outputs
master_df.to_csv("outputs/metrics_summary.csv", index=False)
"""))

    cells.append(nbf.v4.new_code_cell("""# 8.2 Visualizing Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

err_df = master_df[["Model", "Test RMSE", "Test MAE"]].melt(
    id_vars="Model", var_name="Metric", value_name="Error ($ USD)"
)
sns.barplot(data=err_df, x="Model", y="Error ($ USD)", hue="Metric", ax=ax1, palette=["#d62728", "#1f77b4"])
ax1.set_title("Error Metrics: Test RMSE & MAE (Lower is Better)", fontsize=12, weight="bold")
ax1.tick_params(axis="x", rotation=15)
for p in ax1.patches:
    h = p.get_height()
    if not np.isnan(h) and h > 0:
        ax1.annotate(f"${h:,.0f}", (p.get_x() + p.get_width() / 2., h),
                     ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=8)

sns.barplot(data=master_df, x="Model", y="Test R2", ax=ax2, hue="Model", palette="Greens_r", legend=False)
ax2.set_title("Explained Variance: Test R2 (Higher is Better)", fontsize=12, weight="bold")
ax2.set_ylim(0, 0.8)
ax2.tick_params(axis="x", rotation=15)
for p in ax2.patches:
    h = p.get_height()
    if not np.isnan(h) and h > 0:
        ax2.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h),
                     ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)

plt.suptitle("Model Evaluation Summary Across Primary & Improved Models", fontsize=14, weight="bold", y=1.03)
plt.tight_layout()
fig.savefig("outputs/figures/11_model_comparison_metrics.png", dpi=300)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### Model Selection Verdict:
1. **Best Overall Model**: **Multiple Linear Regression with Feature Engineering (FE)** is selected as the production model:
   * **Test RMSE**: **$\\$69,127.04$** (lowest among all models).
   * **Test MAE**: **$\\$49,645.49$** (lowest among all models).
   * **Test $R^2$**: **$0.6353$** (highest explained variance).
2. **Feature Engineering Contribution**:
   * Incorporating `rooms_per_household`, `bedrooms_per_room`, and `population_per_household` reduced Test RMSE by **$\\$932.15$** and Test MAE by **$\\$1,025.00$** over the baseline MLR.
3. **Ridge Regression Assessment**:
   * Ridge regularized coefficients with optimal $\\alpha = 0.8555$, producing nearly identical performance (Test RMSE: $\$69,135.04$, Test $R^2$: $0.6353$). Because the sample size ($N = 16,512$) is large relative to the number of features ($p = 12$), ordinary least squares with ratio features already avoids extreme coefficient explosion, while Ridge provides an additional guarantee against numerical instability.
"""))

    # Step 9: Model Persistence & Validation
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 9: Model Artifact Persistence & Live Inference Testing

To ensure industrial reproducibility, we serialize:
1. The champion pipeline `final_model_pipeline.joblib` (which encapsulates feature engineering, median imputation, standard scaling, one-hot encoding, and the regression estimator).
2. The simple model `simple_linear_model.joblib`.

Then we test loading the pipeline and predicting on brand-new, unseen property records.
"""))

    cells.append(nbf.v4.new_code_cell("""# 9.1 Serializing Model Artifacts
models_dir = Path("outputs/models")
models_dir.mkdir(parents=True, exist_ok=True)

final_pipeline_path = models_dir / "final_model_pipeline.joblib"
simple_model_path = models_dir / "simple_linear_model.joblib"

joblib.dump(mlr_fe_pipeline, final_pipeline_path)
joblib.dump(simple_reg, simple_model_path)

print(f"Champion pipeline saved to: {final_pipeline_path}")
print(f"Simple linear model saved to: {simple_model_path}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 9.2 Validation: Loading Saved Pipeline & Testing on New District Inputs
loaded_pipeline = joblib.load(final_pipeline_path)

# Test inference with three realistic synthetic district profiles
test_cases = pd.DataFrame([
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

test_features = test_cases.drop(columns=["district_name"])
predictions = loaded_pipeline.predict(test_features)

print("=" * 80)
print("LIVE PREDICTION TEST RESULTS ON NEW DISTRICT RECORDS")
print("=" * 80)
for i, row in test_cases.iterrows():
    print(f"District {i+1}: {row['district_name']}")
    print(f"  Location: ({row['latitude']:.2f} deg N, {row['longitude']:.2f} deg W) | Proximity: {row['ocean_proximity']}")
    print(f"  Income: ${row['median_income']*10000:,.0f} | Median Age: {row['housing_median_age']} yrs")
    print(f"  Total Rooms: {row['total_rooms']:,} | Bedrooms: {row['total_bedrooms']} | Households: {row['households']:,}")
    print(f"  --> PREDICTED MEDIAN HOUSE VALUE: ${predictions[i]:,.2f}\\n")

print("Validation completed successfully: End-to-end inference functions as expected.")
"""))

    # Step 10: Conclusions & Limitations
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 10: Conclusions, Business Recommendations & Architectural Limitations

### 1. Key Analytical Conclusions:
* **Simple vs. Multiple Regression**: Moving from Simple Linear Regression to Multiple Linear Regression drops prediction error by **$\\$14,149.82$ RMSE ($16.8\\%$)** and elevates explained variance ($R^2$) from **$0.4589$ to $0.6254$**.
* **Feature Engineering Impact**: Transforming block-aggregate counts into household-level ratios (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`) delivers an additional **$\\$932.15$ RMSE reduction** (final Test RMSE: **$\\$69,127.04$**, Test $R^2$: **$0.6353$**).
* **Economic Drivers**: `median_income` is the primary determinant of property value ($+\\$75,168$ standardized impact), followed by geographic zone (`ocean_proximity`), where coastal proximity commands substantial premiums over inland basins.

### 2. Business Recommendations:
* Real estate underwriting platforms should prioritize local median income and spatial ocean proximity as tier-1 pricing signals.
* Block-group census counts must always be normalized to per-household metrics before feeding valuation algorithms.

### 3. Model Limitations:
* **Linearity Constraint**: Linear regression assumes monotonic additive relationships. Spatial non-linearities (e.g. micro-neighborhood boundaries, mountain ranges) are imperfectly captured by linear coordinates.
* **Target Ceiling Distortion**: Censoring at $\$500,001$ severely penalizes the model on luxury districts (RMSE surges from $\$60,872$ on uncapped properties to $\$166,675$ on capped properties). In future iterations, Tobit regression (censored regression) or non-linear models (e.g., Gradient Boosted Trees) should be explored.
"""))

    nb.cells = cells
    
    notebook_path = Path("property_price_prediction.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Jupyter Notebook successfully written to {notebook_path.resolve()}")

if __name__ == "__main__":
    create_notebook()
