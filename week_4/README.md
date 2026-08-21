# Week 4: Building a Proper ML Pipeline with Feature Engineering 🔧

## 📌 Overview

Professional ML code isn't a pile of notebook cells — it's a clean, reusable pipeline. This week replaces the manual preprocessing from Week 3 with a single `Pipeline` object built using `ColumnTransformer`, and adds two new engineered features to test whether they improve model performance.

**Notebook:** `01_ML_Pipeline_Feature_Engineering.ipynb`

**Dataset:** `telco_churn.csv` (same dataset as Week 3, scoped to the same 6 features for a valid manual-vs-pipeline comparison)

---

## 🔨 What Was Built

- A `ColumnTransformer` that applies `StandardScaler` to numerical columns (`Tenure in Months`, `Monthly Charge`, `Total Charges`) and a `SimpleImputer` + `OneHotEncoder` to categorical columns (`Contract`, `Payment Method`, `Internet Type`)
- Two full pipelines chaining preprocessing with a classifier: Logistic Regression and Decision Tree
- Two new engineered features:
  - `Charge_per_Tenure` = Monthly Charge / (Tenure + 1)
  - `Has_Internet` = binary flag for whether the customer has internet service
- Final best-performing pipeline saved with `joblib` and verified by reloading

---

## 📊 Results

### Pipeline vs. Week 3 Manual Approach

| Metric | Week 3 (Manual) | Week 4 (Pipeline) |
|---|---|---|
| Logistic Regression Accuracy | 80.3% | 80.13% |
| Decision Tree Accuracy | 74.0% | 74.24% |

Results closely match the manual approach, confirming the pipeline was built correctly.

### With Engineered Features

| Model | Without Engineered Features | With Engineered Features |
|---|---|---|
| Logistic Regression | 80.13% | 80.34% |
| Decision Tree | 74.24% | 74.66% |

Both models saw a small but consistent improvement after adding `Charge_per_Tenure` and `Has_Internet`.

---

## 🎯 Key Takeaways

- A `Pipeline` prevents data leakage and inconsistent preprocessing between train/test sets — everything happens in one fit/predict call.
- Missing values in `Internet Type` (customers without internet service) were handled inside the pipeline using `SimpleImputer`, rather than relying on implicit `pandas.get_dummies` behavior like in Week 3.
- Simple engineered features (ratios, binary flags) can provide small but real performance gains.
