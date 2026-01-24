# Bike Sharing Demand Prediction with LightGBM

This project addresses the Bike Sharing Demand problem, aiming to predict the total number of bike rentals (`count`) using tree-based machine learning models, with a focus on **LightGBM**.

The dataset presents right-skewed target distributions, temporal dependencies, and missing values.

## 1. Problem Description

The goal is to predict bike rental demand based on the available features.

### Target variable
- `count` — total number of rentals  
  - `count = registered + casual`

Although `registered` and `casual` are also present in the dataset, the final prediction task focuses exclusively on `count`.

### Available Features
- `datetime` - hourly timestamp
- `season` - season category encoded as integers (1–4)
- `holiday` - binary holiday flag
- `workingday` - binary working day flag 
- `weather` - weather category encoded as integers (1–4)
- `temp` - temperature (°C)
- `atemp` - apparent temperature
- `humidity` - relative humidity
- `windspeed` - wind speed

## 2. Dataset Challenges

- **Temporal gaps**: The dataset spans two years with missing values for `count`, `registered`, and `casual`.
- **Right-skewed targets**: Median < mean for `casual`, `registered`, and `count`, indicating long right tails.
- **Temporal dependency**: Despite being modeled as a regression task, temporal ordering is preserved during the train–test split to ensure generalization.

## 3. Project Structure

1. Problem Description  
2. Imports  
3. Data Loading  
4. Data Overview  
5. Baseline Model  
6. Preprocessing & Feature Engineering  
   - 6.1 Data Validation and Consistency Checks  
   - 6.2 Temporal Structuring and Feature Engineering  
   - 6.3 Temporal Data Consistency and Coverage  
   - 6.4 Categorical Variable Quality and Handling  
   - 6.5 Forecast-Oriented Exploratory Data Analysis  
   - 6.6 Interaction Analysis and Feature Engineering  
7. Outlier Removal  
8. Model Development and Evaluation  
   - 8.1 Baseline Models (Random Forest, XGBoost, LightGBM)  
   - 8.2 Baseline Model Comparison  
   - 8.3 Feature Selection via Feature Importances  
   - 8.4 Evaluation After Feature Selection  
   - 8.5 Hyperparameter Tuning (Optuna)  
   - 8.6 Evaluation After Tuning  
9. Test Data Forecasting  

## 4. Modeling Strategy

### Key Constraints
- Tree-based models are preferred due to:
  - robustness to skewed distributions
  - ability to model non-linear interactions
- No feature engineering based on target-derived demand patterns. 

### Outlier Treatment
- Removal of extreme and clearly invalid observations.
- Application of **Isolation Forest** for multivariate outlier detection.

### Peak-based features  
- Although hourly demand peaks are well-known, features derived from the training target distribution were intentionally avoided to prevent data leakage.
- Instead, only external domain knowledge was considered:
    - Typical commuting hours on working days:
        - Morning: 07–09
        - Evening: 17–19

### Evaluation
- RMSLE

### Models Evaluated
- Random Forest (baseline)
- XGBoost
- LightGBM

### Model Selection
- XGBoost and LightGBM showed very similar performance
- **LightGBM** was selected

### Feature Selection & Hyperparameter Tuning
- Feature selection based on **feature importances**
- Hyperparameter tuning performed using **Optuna**

## 5. Model Performance 

Random Forest baseline performed well, confirming that tree-based models are suitable for this dataset. Outlier removal led to a improvement in RMSLE.

### Residual Analysis

- The initial baseline model exhibited a funnel-shaped residual pattern, characterized by overestimation at low target values and high variance at large target values
- After modeling, residuals became more uniform and symmetric, the distribution centered around zero reflecting improved predictive stability

## 6. Final Model & Forecasting

The final LightGBM model was trained on the full training dataset and applied to the test set to generate demand forecasts.