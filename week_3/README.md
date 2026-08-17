# Week 3: Model Evaluation, Tuning & Business Application 🎯

## 📌 Overview
This week has two parts: first, going beyond raw accuracy to properly evaluate a classification model using precision, recall, and F1-score, and improving it through hyperparameter tuning. Second, applying these skills to a real business problem — customer churn prediction.

---

## 🔧 Part 1: Model Evaluation & Tuning
**Notebook:** `01_Model_Evaluation_Tuning.ipynb`

### Approach
* Revisited the Logistic Regression model from Week 2.
* Generated a full `classification_report` (precision, recall, F1-score) instead of relying on accuracy alone.
* Explained why accuracy can be misleading on the moderately imbalanced Titanic dataset (105 "Not Survived" vs. 74 "Survived" in the test set).
* Tuned 2 hyperparameters (`C` and `solver`) using `GridSearchCV` with 5-fold cross-validation.

### 📊 Results

| Metric | Original Model | Tuned Model |
|---|---|---|
| Accuracy | 82.1% | 80.4% |
| Precision (Survived) | 0.80 | 0.78 |
| Recall (Survived) | 0.76 | 0.73 |
| F1-score (Survived) | 0.78 | 0.76 |
| Best Parameters | Default | `C=1`, `solver='liblinear'` |

**Insight:** Hyperparameter tuning did **not** improve performance on this test set — a useful reminder that GridSearchCV optimizes for cross-validation performance, which doesn't always generalize to a better result on one specific split, especially on a small dataset like this.

---

## 📞 Part 2: Customer Churn Prediction
**Notebook:** `02_Customer_Churn_Prediction.ipynb`

### Approach
* Used the Telco Customer Churn dataset (7,043 customers, 50 columns).
* Selected 6 business-relevant features: `Tenure in Months`, `Monthly Charge`, `Contract`, `Payment Method`, `Internet Type`, `Total Charges`.
* Handled missing values in `Internet Type` (customers without internet service).
* Trained and compared two models: **Logistic Regression** and **Decision Tree**.
* Identified the top 3 churn-driving features using `.feature_importances_`.

### 📊 Results

| Metric | Logistic Regression | Decision Tree |
|---|---|---|
| Accuracy | 80.3% | 74.0% |
| Precision (Churn) | 0.66 | 0.54 |
| Recall (Churn) | 0.61 | 0.53 |
| F1-score (Churn) | 0.64 | 0.54 |

**Top 3 Churn Drivers:** Monthly Charge, Total Charges, Tenure in Months.

**Insight:** Logistic Regression outperformed the untuned Decision Tree across every metric. Customers who churn tend to be newer (avg. 18 months vs. 38 months for retained customers) and pay higher monthly bills (avg. $74 vs. $61) — actionable signals for a targeted retention strategy.