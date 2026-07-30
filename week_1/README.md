# Week 1: Exploratory Data Analysis & Data Cleaning 🚢

## 📌 Overview
This module focuses on the foundational step of any machine learning pipeline: "listening to the data." Before building any predictive models, this week's tasks involved setting up the environment, exploring the **Titanic - Machine Learning from Disaster** dataset, handling messy data, and uncovering hidden patterns through visualization.

## 🧰 Tech Stack Used
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Matplotlib & Seaborn:** For statistical data visualization.

## 🧹 Data Cleaning Strategy
Real-world data is rarely perfect. The initial dataset had 891 rows and 12 columns, with significant missing values. Here is how they were handled and the rationale behind the decisions:

* **Age (19.9% missing):** Filled missing values using the `median`. **Reason:** The Age column contained high outliers (older passengers), so the median is more robust to skewness than the mean.
* **Embarked (0.2% missing):** Filled with the `mode` (most common value). **Reason:** Since only 2 rows were missing, dropping them or using complex imputation wasn't necessary.
* **Cabin (77.1% missing):** Created a new binary feature `Has_Cabin` (1 for Yes, 0 for No) and dropped the original column. **Reason:** With 77% missing data, filling it would introduce noise. However, the *recording* of a cabin itself might hold a signal (e.g., higher class passengers were more likely to have a cabin recorded).

## 📊 Key Insights & Data Story
Visualizing the data revealed crucial patterns that will inform the Week 2 Classification model:

1. **Outlier Detection (Fare):** A boxplot revealed massive outliers in the `Fare` column, with some passengers paying up to $512. This extreme variance indicates why median-based analysis is safer for this feature.
2. **The Strongest Predictors of Survival:**
   * **Sex:** A bar chart of Survival by Sex showed a stark contrast—females survived at a drastically higher rate than males, strictly aligning with the historical "women and children first" evacuation protocol.
   * **Pclass (Passenger Class):** The correlation heatmap showed a strong negative correlation (-0.34) between `Pclass` and `Survived`. First-class passengers had a noticeable survival advantage over 3rd-class passengers, likely due to cabin proximity to lifeboats.

**Conclusion:** `Sex` and `Pclass` are the most heavily weighted features determining passenger survival in this dataset.

---
*This folder contains two notebooks: `01_Titanic_EDA.ipynb` (Initial Inspection) and `02_Titanic_Cleaning_Vis.ipynb` (Cleaning, Imputation, and Visualization).*