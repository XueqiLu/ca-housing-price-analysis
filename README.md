# California Housing Price Analysis (Applied Data Science)

This repository contains an end-to-end data science project on predicting **median house value** using the 1990 California census housing dataset. The work was completed as an internship task and focuses on building a robust modeling pipeline, comparing multiple algorithms, and improving performance through feature engineering and optimization.

## What this project demonstrates

- Data cleaning and preprocessing for mixed feature types
- Exploratory data analysis (EDA) and visualization
- Outlier detection and removal using **Local Outlier Factor (LOF)**
- Correlation analysis and feature selection
- Dimensionality reduction using **PCA** (for comparison)
- Model comparison across:
  - Linear / Polynomial Regression
  - Decision Tree / Random Forest
  - Gradient Boosting (GBDT), XGBoost, LightGBM
  - SVR (tested and discussed)
- Evaluation using multiple metrics (R², MAE, MSE, RMSE, MAPE)
- Robustness testing via feature perturbation (sensitivity analysis)
- Hyperparameter optimization using **GridSearchCV**
- Basic interpretability via feature importance

## Dataset

- Source: 1990 California census housing data
- Size: ~20k rows, 10 columns (including target)
- Target: `median_house_value`
- Key features: location (longitude/latitude), housing age, rooms/bedrooms, population, households, median income, ocean proximity

The dataset file used in this repository is:
- `data/housing.csv`

## Approach summary

1. **Data preparation**
   - Handled missing values
   - Encoded categorical feature `ocean_proximity`
   - Checked distribution skewness and potential outliers

2. **Outlier handling**
   - Applied **Local Outlier Factor (LOF)** to remove anomalous records and tested the impact on downstream performance

3. **Modeling and evaluation**
   - Trained and evaluated multiple model families
   - Used a consistent train/test split and compared results across:
     - original data
     - LOF-filtered data
     - reduced/updated feature sets
     - PCA-transformed data

4. **Optimization**
   - Applied **GridSearchCV** to tune a Random Forest model
   - Identified model configurations that improve generalization performance

## Results (high-level)

Across the tested models, ensemble methods performed best. In particular:
- **XGBoost** achieved the strongest predictive performance among the tested models.
- Robustness testing highlighted that some models (e.g., XGBoost) can be sensitive to feature perturbations, motivating further refinement.

For full quantitative results, figures, and discussion, see the report PDF in `report/`.

