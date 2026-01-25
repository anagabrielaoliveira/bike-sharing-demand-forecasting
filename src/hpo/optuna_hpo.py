from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from config.config import RANDOM_STATE
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import json
import os
from datetime import datetime

def lgbm_objective(trial, X_train, y_train, x_test, y_test):
    """
    This function is used to optimize the model hyperparameters for LightGBM.

    ------------------------------------------------------------

    Args:
    trial : optuna.Trial
        The trial object
    X_train : pandas.DataFrame
        The training data
    y_train : pandas.DataFrame
        The training labels
    x_test : pandas.DataFrame
        The test data
    Returns:
    mape : float
        The mean absolute percentage error of the model
    """
    params = {
        'objective': 'regression',
        'num_leaves': trial.suggest_int('num_leaves', 2, 1e3),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.05, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'random_state': RANDOM_STATE
    }
    model = MultiOutputRegressor(LGBMRegressor(**params, verbosity=-1))
    model.fit(X_train, y_train)
    y_pred = model.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mape

def xgb_objective(trial, X_train, y_train, x_test, y_test):
    """
    This function is used to optimize the model hyperparameters for XGBoost.

    ------------------------------------------------------------

    Args:
    trial : optuna.Trial
        The trial object
    X_train : pandas.DataFrame
        The training data
    y_train : pandas.DataFrame
        The training labels
    x_test : pandas.DataFrame
        The test data
    Returns:
    mape : float
        The mean absolute percentage error of the model
    """
    params = {
        'objective': 'reg:squaredlogerror',
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.05, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.05, 1.0),
        'random_state': RANDOM_STATE
    }
    model = MultiOutputRegressor(XGBRegressor(**params, verbosity=0))
    model.fit(X_train, y_train)
    y_pred = model.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mape

def rf_objective(trial, X_train, y_train, x_test, y_test):
    """
    This function is used to optimize the model hyperparameters for Random Forest.

    ------------------------------------------------------------

    Args:
    trial : optuna.Trial
        The trial object
    X_train : pandas.DataFrame
        The training data
    y_train : pandas.DataFrame
        The training labels
    x_test : pandas.DataFrame
        The test data
    Returns:
    mape : float
        The mean absolute percentage error of the model
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'random_state': RANDOM_STATE
    }
    model = MultiOutputRegressor(RandomForestRegressor(**params))
    model.fit(X_train, y_train)
    y_pred = model.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mape

class OptunaHPO:
    def __init__(self, X_train, y_train, x_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def _create_study(self, model_name):
        """
        This function is used to create a study for the model.

        ------------------------------------------------------------

        Args:
        model_name : str
            The name of the model
        Returns:
        study : optuna.Study
            The study object
        """
        study = optuna.create_study(
            study_name=model_name,
            direction='minimize',
            sampler=TPESampler(seed=RANDOM_STATE),
            pruner=HyperbandPruner()
        )

        return study

    def _save_study_results(self, study, model_name):
        """
        This function is used to save the study results to a JSON file.

        ------------------------------------------------------------

        Args:
        study : optuna.Study
            The study object
        model_name : str
            The name of the model
        """
        os.makedirs('data/models/hpo', exist_ok=True)
        with open(f'data/models/hpo/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M")}_study.json', 'w') as f:
            json.dump({
                'model_name': model_name,
                'model_params': study.best_trial.params,
                'mape': study.best_value,
                'n_trials': len(study.trials)
            }, f, indent=4)


    def lgbm_optimize(self, n_trials: int):
        """
        This function is used to optimize the model hyperparameters for LightGBM.

        ------------------------------------------------------------

        Args:
        n_trials : int
            The number of trials to run
        """
        study = self._create_study(model_name='lgbm')
        study.optimize(
            lambda trial: lgbm_objective(
                trial,
                self.X_train,
                self.y_train,
                self.x_test,
                self.y_test
            ),
            n_trials=n_trials,
            show_progress_bar=True
        )

        self._save_study_results(study, model_name='lgbm')

    def xgb_optimize(self, n_trials: int):
        """
        This function is used to optimize the model hyperparameters for XGBoost.

        ------------------------------------------------------------

        Args:
        n_trials : int
            The number of trials to run
        """
        study = self._create_study(model_name='xgb')
        study.optimize(
            lambda trial: xgb_objective(
                trial,
                self.X_train,
                self.y_train,
                self.x_test,
                self.y_test
            ),
            n_trials=n_trials,
            show_progress_bar=True
        )

        self._save_study_results(study, model_name='xgb')

    def rf_optimize(self, n_trials: int):
        """
        This function is used to optimize the model hyperparameters for Random Forest.

        ------------------------------------------------------------

        Args:
        n_trials : int
            The number of trials to run
        """
        study = self._create_study(model_name='rf')
        study.optimize(
            lambda trial: rf_objective(
                trial,
                self.X_train,
                self.y_train,
                self.x_test,
                self.y_test
            ),
            n_trials=n_trials,
            show_progress_bar=True
        )

        self._save_study_results(study, model_name='rf')