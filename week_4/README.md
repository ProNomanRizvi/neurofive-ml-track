# Week 4: Building a Proper ML Pipeline with Feature Engineering

## Overview

Notebook cells that only work if you run them in the right order aren't real code. This week I rebuilt Week 3's manual preprocessing as a single `Pipeline` using `ColumnTransformer`, then added two engineered features to see if they'd actually move the needle.

**Notebook:** `01_ML_Pipeline_Feature_Engineering.ipynb`

**Dataset:** `telco_churn.csv` — same one from Week 3, same six features, so the pipeline numbers can be checked directly against the manual ones.

## What's in it

The `ColumnTransformer` scales the numerical columns (`Tenure in Months`, `Monthly Charge`, `Total Charges`) with `StandardScaler`, and runs the categorical columns (`Contract`, `Payment Method`, `Internet Type`) through a `SimpleImputer` and then `OneHotEncoder`. That feeds into two full pipelines — one ending in Logistic Regression, one in a Decision Tree.

I also added two features: `Charge_per_Tenure` (monthly charge divided by tenure+1) and `Has_Internet`, a simple binary flag. The best pipeline gets saved with `joblib` and reloaded to confirm it still predicts the same way.

## Results

First check: does the pipeline reproduce what I got manually in Week 3?

| Metric | Week 3 (manual) | Week 4 (pipeline) |
|---|---|---|
| Logistic Regression Accuracy | 80.3% | 80.13% |
| Decision Tree Accuracy | 74.0% | 74.24% |

Close enough that I trust the pipeline is doing what the manual code did.

Then with the engineered features added in:

| Model | Without new features | With new features |
|---|---|---|
| Logistic Regression | 80.13% | 80.34% |
| Decision Tree | 74.24% | 74.66% |

Small gains on both models, nothing dramatic, but consistent.

## What I took from this

Doing preprocessing inside a `Pipeline` means train and test data can't accidentally leak into each other — one `fit`, one `predict`, same steps every time. It also forced me to handle the missing values in `Internet Type` properly with `SimpleImputer` instead of letting `pandas.get_dummies` quietly paper over them like it did in Week 3. And the engineered features were a good reminder that you don't need anything complicated to get a small real improvement — a ratio and a flag were enough here.

---

# Week 4 (Task 2): Ensemble Learning — Random Forest vs XGBoost

## Overview

A single Logistic Regression or Decision Tree only gets you so far. This task adds two ensemble models — Random Forest and XGBoost — trained on the same churn data and same six features as Task 1, so all four models can be compared on equal footing.

**Notebook:** `02_Ensemble_RF_vs_XGBoost.ipynb`

**Dataset:** `telco_churn.csv`, same scope as Task 1.

## What's in it

Both `RandomForestClassifier` and `XGBClassifier` run through the same `ColumnTransformer` preprocessing from Task 1, so the only thing changing between models is the classifier itself. I plotted feature importances for both side by side, wrote up how the two methods actually differ in how they build their trees, and saved the better-performing one (XGBoost) with `joblib`, then reloaded it to confirm the predictions matched.

## Results

All four models together:

| Model | Accuracy | F1-score (Churn) |
|---|---|---|
| Logistic Regression | 80.13% | 0.64 |
| XGBoost | 78.71% | 0.58 |
| Random Forest | 77.08% | 0.53 |
| Decision Tree | 74.24% | 0.54 |

Both ensembles beat the single Decision Tree, which makes sense — combining trees, whether by bagging or boosting, cuts down the variance you get from one unpruned tree. XGBoost came out slightly ahead of Random Forest. Logistic Regression still won overall, though, which is a fair reminder that ensembles aren't automatically the better choice, especially with a small, mostly linear feature set like this one.

Feature importance told a different story than the accuracy numbers:

| Rank | Random Forest | XGBoost |
|---|---|---|
| 1 | Monthly Charge (0.28) | Contract_Month-to-Month (0.59) |
| 2 | Total Charges (0.28) | Internet Type_Fiber Optic (0.18) |
| 3 | Tenure in Months (0.18) | Internet Type_No Internet Service (0.04) |

Random Forest spreads its attention across the numerical columns fairly evenly. XGBoost puts almost 60% of its weight on one categorical split. That difference comes straight from how the two models are built — Random Forest trains a bunch of trees independently and averages them out, while XGBoost trains trees one after another, each one correcting whatever the last one got wrong, so it tends to latch onto whatever feature reduces error fastest early on.

## What I took from this

Ensembles reduced variance and beat the plain Decision Tree, but they didn't beat Logistic Regression here — the "better" model still depends on the dataset, not just on using a fancier method. The bigger thing I noticed was how differently Random Forest and XGBoost weight features even when trained on identical data, which says more about bagging vs. boosting than the accuracy numbers alone do. Saving and reloading the model with `joblib` was also a good check that the pipeline actually works outside the notebook, not just inside it.