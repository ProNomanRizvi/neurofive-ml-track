# Neurofive ML Track — Week 1

## Task: Titanic Dataset EDA

First step in the ML learning track — setting up a Python data science environment 
and performing exploratory data analysis (EDA) on the Titanic dataset.

## What's Inside
- `notebook.ipynb` — Full EDA: dataset loading, missing values check, 
  column type breakdown, and key findings

## Key Finding
The dataset has 891 rows and 12 columns, with the Cabin column missing ~77% 
of its values — a strong candidate for feature engineering rather than direct use.

## Tools Used
Python, pandas, NumPy, Jupyter Notebook

## Week 2 — Classification Model

Built a Logistic Regression model to predict Titanic passenger survival.

**Approach:**
- Encoded categorical columns (`Sex`, `Embarked`) using `pd.get_dummies()`
- Split data into 80% training / 20% test sets
- Trained a Logistic Regression model
- Evaluated using accuracy score and a confusion matrix

**Result:**
- **Accuracy: 82.1%**
- The model correctly classified 147 out of 179 test passengers