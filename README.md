# NeuroFive Machine Learning Internship Track 🚀

This repository contains my progress, assignments, and projects for the Machine Learning Fundamentals track at NeuroFive Solutions.

## 📂 Repository Structure

* **`data/`**: Contains raw and processed datasets (e.g., Titanic dataset, House Prices). *Note: Large datasets are tracked locally but should generally be added to `.gitignore` in production.*
* **`week_1/`**: Exploratory Data Analysis (EDA) and Data Cleaning. 
  * Handled missing values, detected outliers, and discovered key survival patterns using `matplotlib` and `seaborn`.
* **`week_2/`**: First Machine Learning Models.
  * Logistic Regression (Classification) for Titanic Survival Prediction — **82.1% accuracy**.
  * Linear Regression (Regression) for House Price Prediction — **R²: 0.32, RMSE: ~267,796**.
* **`week_3/`**: Model Evaluation & Hyperparameter Tuning.
  * Evaluated the Titanic model beyond accuracy using precision, recall, and F1-score.
  * Tuned hyperparameters (`C`, `solver`) via `GridSearchCV` — tuned accuracy: **80.4%** vs. original **82.1%**, showing tuning doesn't always improve results on a given split.

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

## ⚙️ How to run locally
To replicate this environment on your local machine, install the dependencies using the provided `requirements.txt` file:

`pip install -r requirements.txt`