# Week 2: First Machine Learning Models (Classification & Regression) 🤖

## 📌 Overview
This week transitions from data exploration to predictive modeling. The goal is to solve two fundamental types of machine learning problems: **Classification** (predicting a category) and **Regression** (predicting a continuous number). 

---

## 🛠️ Part 1: Titanic Survival Prediction (Classification)
**Notebook:** `01_Titanic_Classification.ipynb`

### Approach & Preprocessing
* Loaded the cleaned dataset from Week 1.
* **Feature Engineering:** Encoded categorical variables (`Sex` and `Embarked`) using `pd.get_dummies(drop_first=True)` to convert them into machine-readable numerical formats.
* **Data Splitting:** Divided the dataset into training (80%) and testing (20%) sets using `train_test_split` to ensure the model is evaluated on unseen data.
* **Algorithm:** Trained a **Logistic Regression** model.

### 📊 Results & Evaluation
* **Final Accuracy Score:** **82.1%** 
* **Confusion Matrix Breakdown:**
* **True Negatives (91):** Correctly predicted passengers who did not survive.
* **True Positives (56):** Correctly predicted passengers who survived.
* **False Positives (14):** Model incorrectly predicted they survived.
* **False Negatives (18):** Model incorrectly predicted they did not survive.

**Insight:** The model achieved a solid 82.1% accuracy. It performs slightly better at identifying non-survivors, which is expected since the original dataset has a higher proportion of non-survivors (class imbalance). 

---

## 🏠 Part 2: House Price Prediction (Regression)
**Notebook:** `02_House_Price_Regression.ipynb`

### Approach & Preprocessing
* Used a Seattle-area house sales dataset (4,140 rows, 18 columns).
* **Feature Selection:** Chose 5 features believed to most affect price — `sqft_living`, `bedrooms`, `bathrooms`, `floors`, and `condition`.
* **Data Splitting:** Divided the dataset into training (80%) and testing (20%) sets using `train_test_split`.
* **Algorithm:** Trained a **Linear Regression** model.

### 📊 Results & Evaluation
* **RMSE:** ~267,796 (average prediction error in price units)
* **R² Score:** **0.32**
* A scatter plot of predicted vs. actual prices showed reasonable alignment at lower price points, with wider spread at higher price ranges.

**Insight:** The model explains about 32% of the variation in house prices using only structural features. The remaining variance is likely driven by factors not included in this model — most notably **location** (city/neighborhood), which can significantly affect price independent of a house's size or condition.