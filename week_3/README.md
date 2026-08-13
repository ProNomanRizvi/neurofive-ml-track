# Week 3: Model Evaluation & Tuning 🎯

## 📌 Overview
This week goes beyond raw accuracy to properly evaluate the Titanic classification model using precision, recall, and F1-score, then attempts to improve it through hyperparameter tuning.

## 🛠️ Approach
**Notebook:** `01_Model_Evaluation_Tuning.ipynb`

* Revisited the Logistic Regression model from Week 2.
* Generated a full `classification_report` (precision, recall, F1-score) instead of relying on accuracy alone.
* Explained why accuracy can be misleading on the moderately imbalanced Titanic dataset (105 "Not Survived" vs. 74 "Survived" in the test set).
* Tuned 2 hyperparameters (`C` and `solver`) using `GridSearchCV` with 5-fold cross-validation.

## 📊 Results

| Metric | Original Model | Tuned Model |
|---|---|---|
| Accuracy | 82.1% | 80.4% |
| Precision (Survived) | 0.80 | 0.78 |
| Recall (Survived) | 0.76 | 0.73 |
| F1-score (Survived) | 0.78 | 0.76 |
| Best Parameters | Default | `C=1`, `solver='liblinear'` |

**Insight:** Hyperparameter tuning did **not** improve performance on this test set — a useful reminder that GridSearchCV optimizes for cross-validation performance, which doesn't always generalize to a better result on one specific split, especially on a small dataset like this.