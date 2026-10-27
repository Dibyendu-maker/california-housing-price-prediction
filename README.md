# Part A: Property Price Prediction
## Internshala Machine Learning with AI Internship

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Overview & Objectives

This project builds an end-to-end, reproducible, production-grade Machine Learning solution to predict median property values across California districts. Developed for **Part A** of the **Internshala Machine Learning with AI** internship, the project evaluates:

1. **Simple Linear Regression**: Formulated using an empirical, domain-justified single predictor (`median_income`).
2. **Multiple Linear Regression (Baseline)**: Incorporating all geographical, structural, and demographic variables.
3. **Multiple Linear Regression (Feature Engineered)**: Enhanced with district-level structural ratios (`rooms_per_household`, `bedrooms_per_room`, and `population_per_household`).
4. **Regularized Ridge Regression ($L_2$ Regularization)**: Addressing severe multicollinearity identified via Variance Inflation Factor (VIF) analysis.

All steps strictly prevent data leakage by fitting transformers exclusively on training data, validate generalization using 5-fold cross-validation and a 20% held-out test set, and persist trained model artifacts for immediate inference.

---

## 2. Dataset Description & Schema

The dataset represents district-level census data from the 1990 U.S. Census for the State of California, provided in `Data_file.xlsx` and documented in `Data Information.pdf`.

* **Total Observations**: 20,640 districts
* **Target Variable**: `median_house_value` ($ USD)
* **Duplicate Records**: 0
* **Missing Values**: `total_bedrooms` contains 207 missing values (1.003%), imputed via median strategy on training data.
* **Target Censoring (Ceiling)**: 992 districts (4.81%) have their values capped at $500,001.

### Data Dictionary

| Variable Name | Description | Data Type | Physical Unit | Statistical Role |
| :--- | :--- | :--- | :--- | :--- |
| `longitude` | Longitudinal coordinate of district | FLOAT | degrees | Feature |
| `latitude` | Latitudinal coordinate of district | FLOAT | degrees | Feature |
| `housing_median_age` | Median age of housing units | INT | years (capped at 52) | Feature |
| `total_rooms` | Total room count across district | INT | count | Feature |
| `total_bedrooms` | Total bedroom count across district | FLOAT | count (207 nulls) | Feature |
| `population` | Resident population in district | INT | count | Feature |
| `households` | Occupied housing units/families | INT | count | Feature |
| `median_income` | Median household income | FLOAT | tens of thousands ($) | Primary Predictor |
| `ocean_proximity` | Categorical distance to ocean | STRING | 5 categories | Categorical Feature |
| `median_house_value` | **Median property value** | INT/FLOAT | $ USD (capped at $500,001) | **Target ($y$)** |

---

## 3. Directory Structure

```
Part A Property Price Prediction/
├── Data Information.pdf          # Official project specification & data dictionary
├── Data_file.xlsx                # Primary dataset (20,640 records)
├── property_price_prediction.ipynb # Fully executed Jupyter Notebook with all outputs & plots
├── requirements.txt              # Production dependency specifications
├── README.md                     # Project documentation & architectural report
├── run_pipeline.py               # Automated end-to-end training & evaluation pipeline
├── build_notebook.py             # Notebook generator script
├── src/                          # Modular Python source codebase
│   ├── __init__.py               # Package initializer
│   ├── config.py                 # Constants, paths, seed configurations
│   ├── data.py                   # Data ingestion, split, and pipeline builders
│   ├── eda.py                    # Exploratory statistical analysis & figure generators
│   ├── models.py                 # Estimator definitions & artifact persistence
│   ├── evaluate.py               # Metrics computation, diagnostics, & VIF analysis
│   └── predict.py                # Standalone inference & validation script
└── outputs/
    ├── figures/                  # 14 publication-quality visualizations (300 DPI)
    │   ├── 01_target_distribution.png
    │   ├── 02_numerical_features_histograms.png
    │   ├── 03_numerical_features_boxplots.png
    │   ├── 04_correlation_heatmap.png
    │   ├── 05_median_income_vs_price_scatter.png
    │   ├── 06_ocean_proximity_distribution_and_price.png
    │   ├── 07_california_housing_geography.png
    │   ├── 10_mlr_coefficients.png
    │   ├── 11_model_comparison_metrics.png
    │   ├── 12_actual_vs_predicted.png
    │   ├── 13_residuals_vs_fitted.png
    │   ├── 14_residuals_distribution_qq.png
    │   ├── 15_ceiling_effect_analysis.png
    │   └── 16_vif_comparison.png
    ├── metrics.json              # Full numerical metrics in JSON format
    ├── metrics_summary.csv       # Tabular benchmark comparison
    └── models/                   # Serialized joblib model artifacts
        ├── final_model_pipeline.joblib # Champion feature-engineered MLR pipeline
        └── simple_linear_model.joblib  # Simple Linear Regression model
```

---

## 4. Installation & Environment Setup

### Prerequisites
* Python 3.10, 3.11, 3.12, 3.13, or 3.14.
* Windows, macOS, or Linux.

### Setup Instructions

1. Clone or extract the project directory and open a terminal in the folder:
   ```bash
   cd "Part A Property Price Prediction"
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Execution Instructions

### A. Run Full End-to-End Pipeline
To re-run the entire data ingestion, EDA figure generation, model training, cross-validation, residual diagnostic plotting, and artifact export:
```bash
python run_pipeline.py
```

### B. Run Live Sample Inference
To test the saved production pipeline on new district records (including missing bedroom handling):
```bash
python src/predict.py
```

### C. Launch Interactive Jupyter Notebook
To inspect the fully executed notebook with inline plots and mathematical markdown:
```bash
python -m jupyter lab property_price_prediction.ipynb
# Or using classic notebook:
python -m jupyter notebook property_price_prediction.ipynb
```

---

## 6. Methodology & Key Findings

### Step 1: Exploratory Data Analysis (EDA)
* **Target Distribution**: Right-skewed ($0.978$) with an artificial ceiling at $\$500,001$ ($992$ districts, $4.81\%$).
* **Dominant Linear Predictor**: `median_income` exhibits the strongest linear correlation with `median_house_value` ($r = +0.688$), justifying its choice as the sole predictor for Simple Linear Regression.
* **Multicollinearity Diagnostic**: Variance Inflation Factor (VIF) calculations revealed extreme collinearity among district aggregates:
  * `total_bedrooms`: VIF = **$36.31$**
  * `households`: VIF = **$35.82$**
  * `total_rooms`: VIF = **$12.82$**
* **Geographical Influence**: Coastal districts (`<1H OCEAN`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`) command dramatic premiums over inland valley basins (`INLAND` median price is $\$108,500$ vs $\$214,000+$ for coastal).

### Step 2: Data Preprocessing & Leakage Prevention
* **Data Splitting**: Strict 80/20 train/test partition ($16,512$ train / $4,128$ test) with fixed random seed $42$. The held-out test set represents exactly $20.00\%$ of the total dataset ($4,128 / 20,640 = 0.2000$).
* **Imputation**: Missing values in `total_bedrooms` imputed via training-set median.
* **Encoding**: Categorical `ocean_proximity` encoded using `OneHotEncoder(drop='first')` to eliminate the dummy variable trap.
* **Scaling**: Standardized numerical features with `StandardScaler` to ensure coefficients reflect standardized effect sizes.
* **Pipeline Architecture**: All preprocessing encapsulated inside scikit-learn `Pipeline` and `ColumnTransformer` objects to eliminate data leakage.

### Step 3: Model Development & Mathematical Formulations

#### 1. Simple Linear Regression
$$\text{median\_house\_value} = 44,459.73 + 41,933.85 \times \text{median\_income}$$
* **Intercept ($\beta_0 = \$44,459.73$)**: Theoretical baseline value for zero income.
* **Slope ($\beta_1 = \$41,933.85$)**: For each $\$10,000$ increase in district median income, median house value increases by $\$41,933.85$.

#### 2. Multiple Linear Regression (Baseline)
Incorporates all 9 original predictors. Top standardized coefficients:
* `median_income`: $+\$75,168$
* `ocean_proximity_ISLAND`: $+\$136,125$
* `ocean_proximity_INLAND`: $-\$39,787$
* Collinearity artifact: `total_rooms` displays a negative coefficient ($-\$13,094$) while `total_bedrooms` displays $+\$43,068$.

#### 3. Multiple Linear Regression with Feature Engineering (Champion Model)
District count aggregates were normalized into ratio features:
* `rooms_per_household = total_rooms / households`
* `bedrooms_per_room = total_bedrooms / total_rooms`
* `population_per_household = population / households`

#### 4. Regularized Ridge Regression ($L_2$ Shrinkage)
Tuned using 5-fold cross-validation on training data ($\alpha = 0.8555$). Ridge penalizes inflated coefficients and provides robust stability against multicollinearity.

---

## 7. Model Evaluation & Benchmark Results

All models were evaluated on the **identical 20% held-out test set** ($4,128$ observations):

| Model Name | Predictors Used | Train RMSE ($) | Test RMSE ($) | Test MAE ($) | Test $R^2$ | 5-Fold CV RMSE ($) | 5-Fold CV $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple Linear Regression** | `median_income` (1 feature) | $83,614.87$ | $84,209.01$ | $62,990.87$ | $0.4589$ | $83,609.29 \pm 1,339$ | $0.4769$ |
| **Multiple Linear Regression (Base)** | All 9 original features | $68,433.94$ | $70,059.19$ | $50,670.49$ | $0.6254$ | $68,604.16 \pm 1,325$ | $0.6477$ |
| **Multiple Linear Regression (FE)** | All features + 3 ratio features | **$67,593.21$** | **$69,127.04$** | **$49,645.49$** | **$0.6353$** | **$67,858.78 \pm 1,312$** | **$0.6552$** |
| **Ridge Regression (FE + $L_2$)** | All features + ratios + $L_2$ ($\alpha=0.8555$) | $67,594.16$ | $69,135.04$ | $49,652.36$ | $0.6353$ | $67,884.26 \pm 1,336$ | $0.6550$ |

### Performance Insights:
1. **Simple vs. Multiple Regression**: Transitioning from Simple to Multiple Linear Regression reduced Test RMSE by **$\$14,149.82$ ($16.8\%$)** and improved $R^2$ from **$0.4589$ to $0.6254$**.
2. **Feature Engineering Impact**: Incorporating household-level ratios reduced Test RMSE by an additional **$\$932.15$** and Test MAE by **$\$1,025.00$**, yielding the top-performing model.
3. **No Overfitting**: Train RMSE ($\$67,593$) and Test RMSE ($\$69,127$) exhibit a negligible generalization gap ($2.2\%$), confirming solid model stability.

---

## 8. Diagnostic Findings & Target Ceiling Impact

### Residual Diagnostics
* **Mean Residual**: $-\$1,625$ (satisfies zero-error mean condition).
* **Homoscedasticity**: Residual variance increases slightly at higher price ranges (typical for high-value housing).
* **Normality**: Residuals follow a bell curve with moderate kurtosis and heavy tails at the boundary.

### Target Ceiling Distortion ($500,001 Cap)
Evaluating test performance separately across capped and uncapped districts demonstrates the significant impact of the census truncation:

* **Champion Model (Feature Engineered MLR)**:
  * **Uncapped Districts ($n = 3,944$, $95.5\%$ of test data)**:
    * **RMSE: $\$60,872.31$** | **MAE: $\$45,405.84$** | **$R^2: 0.6091$**
  * **Capped Districts ($n = 184$, $4.5\%$ of test data)**:
    * **RMSE: $\$166,674.60$** | **MAE: $\$140,521.53$**
* **Baseline Multiple Linear Regression**:
  * **Uncapped Districts ($n = 3,944$)**:
    * **RMSE: $\$61,848.69$** | **MAE: $\$46,428.91$**
  * **Capped Districts ($n = 184$)**:
    * **RMSE: $\$167,698.58$** | **MAE: $\$141,587.86$**

Linear regression inherently attempts to predict continuous market valuations above $\$500k$ for luxury coastal districts, which inflates the apparent error due to target truncation in the census survey.

---

## 9. Live Inference & Model Usage

To make predictions on new data using Python:

```python
import pandas as pd
import joblib

# Load production pipeline
pipeline = joblib.load("outputs/models/final_model_pipeline.joblib")

# New district observation
new_district = pd.DataFrame([{
    "longitude": -122.25,
    "latitude": 37.85,
    "housing_median_age": 45,
    "total_rooms": 2500,
    "total_bedrooms": 450,
    "population": 1100,
    "households": 420,
    "median_income": 8.5,  # $85,000
    "ocean_proximity": "NEAR BAY"
}])

# Predict
predicted_price = pipeline.predict(new_district)
print(f"Predicted Median House Value: ${predicted_price[0]:,.2f}")
# Output: Predicted Median House Value: $434,603.87
```

---

## 10. Limitations & Recommendations for Future Iterations

1. **Non-Linear Geographic Boundaries**: Linear regression models spatial coordinates as planar hyperplanes, failing to capture micro-neighborhood pricing boundaries or natural geographic barriers (e.g., mountains, coastlines).
2. **Censored Regression (Tobit Models)**: Because $4.8\%$ of properties are artificially capped at $\$500,001$, a Tobit model (censored regression) would better account for truncated target bounds.
3. **Advanced Tree-Based Ensembles**: Non-linear algorithms such as Random Forests or Gradient Boosted Trees (XGBoost/LightGBM) could capture complex feature interactions (e.g., location $\times$ income) that linear models inherently cannot represent.

---

## 11. Author & Internship Submission
* **Assignment**: Part A: Property Price Prediction
* **Internship**: Internshala Machine Learning with AI
* **Status**: Complete, Verified, & Reproducible
