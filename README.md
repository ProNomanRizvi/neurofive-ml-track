# NeuroFive Machine Learning Internship Track

This repository holds my work for the Machine Learning Fundamentals track at NeuroFive Solutions — assignments, notebooks, and the projects that came out of them.

## Repository Structure

**`data/`** — raw and processed datasets (Titanic, House Prices, Telco Churn). Large files are tracked locally for now; in a production setup these would go in `.gitignore` instead.

**`week_1/`** — exploratory data analysis and cleaning. Handled missing values, checked for outliers, and pulled out the early survival patterns using `matplotlib` and `seaborn`.

**`week_2/`** — first two models. Logistic Regression for Titanic survival prediction (82.1% accuracy), and Linear Regression for house price prediction (R² 0.32, RMSE ~267,796).

**`week_3/`** — model evaluation and a real business problem. Went past plain accuracy on the Titanic model using precision, recall, and F1-score, then tuned `C` and `solver` with `GridSearchCV` — tuned accuracy came out at 80.4% against the original 82.1%, a reminder that tuning doesn't always beat the baseline on a given split. Also built a churn prediction model on the Telco dataset, comparing Logistic Regression (80.3%) against a Decision Tree (74.0%), pulled out the top churn drivers, and wrote the findings up for a non-technical reader.

**`week_4/`** — two tasks.

Task 1 replaced the manual preprocessing from Week 3 with a single `Pipeline`: a `ColumnTransformer` running `StandardScaler` on the numerical columns and `SimpleImputer` + `OneHotEncoder` on the categorical ones, feeding into Logistic Regression and a Decision Tree. The pipeline's numbers matched the manual approach closely (80.13% vs. 80.3% for Logistic Regression), and adding two engineered features (`Charge_per_Tenure`, `Has_Internet`) pushed it up to 80.34%. Final pipeline saved with `joblib`.

Task 2 added Random Forest and XGBoost on the same feature set, so all four models line up against each other: Logistic Regression 80.13%, XGBoost 78.71%, Random Forest 77.08%, Decision Tree 74.24%. Both ensembles beat the single Decision Tree, and XGBoost edged out Random Forest, though Logistic Regression stayed on top overall. The two ensembles also disagreed sharply on feature importance — Random Forest split its attention across the numerical columns, while XGBoost put almost 60% of its weight on a single categorical feature (`Contract_Month-to-Month`), which comes down to bagging vs. boosting building trees in fundamentally different ways. Best model (XGBoost) saved with `joblib` and verified by reloading.

## Tech Stack & Tools

Python, with Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, and Seaborn, all in Jupyter Notebook.

## Running this locally

Install the dependencies from `requirements.txt`:

```
pip install -r requirements.txt
```