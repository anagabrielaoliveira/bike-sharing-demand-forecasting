import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Frameworks / Optimization
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb

import optuna

# Scikit-learn
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error


"""
class MachineLearningPipeline:
  def __init__(self, model, test_size=0.2, random_state=42):
    self.model = model
    self.test_size = test_size
    self.random_state = random_state

  def split(self, X, y):
    self.X_train, self.X_test,self.y_train, self.y_test = train_test_split(
        X, y, test_size=self.test_size, random_state=self.random_state
    )
    return self
  
  def fit(self):
    self.model.fit(self.X_train, self.y_train)
    return self

  def predict(self):
    return self.model.predict(self.X_test)
  
  def evaluate_rmsle(self):
     y_pred = self.predict()
     return mean_squared_error(self.y_test, y_pred, squared=False)


"""
def train_rf_model(X, y, test_size=0.2, random_state=42):
    """
    Split and train Random Forest baseline model
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    
    y_train = y_train.values.ravel()
    y_test  = y_test.values.ravel()

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred






def unique_values(columns, df):

  """
  Print unique values for each column in the list, binary and categorical

  Parameters:
  df : pandas.DataFrame
    The DataFrame to check unique values for
  columns : list of str
    The list of column names to check unique values for
  """

  for col in columns:
    unique_values = df[col].unique()
    print(f"{col}: {unique_values}\n")

def create_time_features(df, special_holidays=None):
    """
    Time features
    """
    df = df.copy()

    # Basic time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofmonth'] = df.index.day
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year

    # Binary feature indicating weekend
    df['weekend'] = (df['dayofweek'] >= 5).astype(int)

    # special holiday feature
    if special_holidays is not None:
        df['date'] = df.index.date
        df['special_holiday'] = df['date'].isin(special_holidays).astype(int)

    return df

def plot_daily_by_year(df, column):
    """
    Gera um gráfico de dispersão da coluna especificada agrupada por ano.
    """
    df = df.copy()
    df['year'] = df.index.year

    plt.figure(figsize=(12,5))
    for year in df['year'].unique():
        df_year = df[df['year'] == year]
        plt.scatter(df_year.index, df_year[column], s=10, label=str(year))

    plt.title(f"Daily mean of {column} by year")
    plt.ylabel(column)
    plt.xlabel("Date")
    plt.legend(title="Year")
    plt.show()

def create_peak_feature(df):
    peak_column = np.where(
        (df['workingday'] == 1) &
        (
            df['hour'].between(7, 9) |
            df['hour'].between(17, 19)
        ),
        1,
        0
    )

    return peak_column

def create_time_slots(df):
  return (df['hour'] // 3) + 1




# Defining RMSLE function
def rmsle(y_true, y_pred):
   y_true = np.maximum(0, y_true)
   y_pred = np.maximum(0, y_pred)
   return np.sqrt(mean_squared_log_error(y_true, y_pred))

def objective(trial, X_train, y_train):

  """
  X_train: X_train_reg, X_train_cas, X_train_cnt
  y_train: y_train_reg, y_train_cas, y_train_cnt

  """

  params = {
  'num_leaves': trial.suggest_int('num_leaves', 2, 1e3),
  'max_depth': trial.suggest_int('max_depth', 3, 15),
  'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
  'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
  'colsample_bytree': trial.suggest_float("colsample_bytree", 0.05, 1.0),
  'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
  'verbosity': -1
  }

  model = lgb.LGBMRegressor(**params)

  score = cross_val_score(
      model,
      X_train,
      y_train,
      cv=5,
      scoring='neg_mean_squared_error',
      n_jobs=-1
      )

  rmsle = np.sqrt(-score.mean())

  return rmsle

def rmsle_improvement_pct(rmsle_base, rmsle_final):
    return (rmsle_base - rmsle_final) / rmsle_base * 100

def plot_daily_train_test(df, column, mask_train, mask_test):
    """
    Plots a daily scatter plot using different colors for training data and predicted test data

    mask_train - mask for training data
    mask_test - mask for test data
    """

    plt.figure(figsize=(12,5))

    # Train
    plt.scatter(df.index[mask_train], df.loc[mask_train, column],
                s=10, color='blue', alpha=0.6, label='Train')

    # Test
    plt.scatter(df.index[mask_test], df.loc[mask_test, column],
                s=10, color='orange', alpha=0.6, label='Test')

    plt.title(f"Daily {column}: Train vs Test")
    plt.ylabel(column)
    plt.xlabel("Date")
    plt.legend()
    plt.show()



def rmsle_per_target(y_true, y_pred, targets):
    return {
        t: rmsle(y_true[t], y_pred[t])
        for t in targets
    }
