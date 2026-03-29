from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
#from sklearn.metrics import mean_absolute_percentage_error
#from sklearn.multioutput import MultiOutputRegressor
from config.config import RANDOM_STATE
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import json
import os
from datetime import datetime
import joblib

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
    rmsle: 
        The root mean squared logarithmic error of the model
    """
    params = {
        #'objective': 'huber',
        'num_leaves': trial.suggest_int('num_leaves', 15, 100),
        #'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 120),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        #"alpha": trial.suggest_float("alpha", 0.7, 0.99),
        'random_state': RANDOM_STATE
    }
    model = LGBMRegressor(**params, verbosity=-1)
    model.fit(X_train, y_train)
    y_pred_log = model.predict(x_test)

    rmsle = np.sqrt(np.mean((y_test - y_pred_log) ** 2))

    trial.set_user_attr('model', model)
    
    return rmsle
    # mape = mean_absolute_percentage_error(y_test, y_pred)
    # return mape

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
    rmsle: 
        The root mean squared logarithmic error of the model
    """

    params = {
        #'objective': 'reg:pseudohubererror',
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        #"huber_slope": trial.suggest_float("huber_slope", 0.5, 2.0),
        'random_state': RANDOM_STATE
    }
    model = XGBRegressor(**params, verbosity=0)
    model.fit(X_train, y_train)
    y_pred_log = model.predict(x_test)

    rmsle = np.sqrt(np.mean((y_test - y_pred_log) ** 2))

    trial.set_user_attr('model', model)

    return rmsle
    # mape = mean_absolute_percentage_error(y_test, y_pred)
    # return mape

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
    rmsle: 
        The root mean squared logarithmic error of the model
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 30),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'random_state': RANDOM_STATE
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred_log = model.predict(x_test)

    rmsle = np.sqrt(np.mean((y_test - y_pred_log) ** 2))

    trial.set_user_attr('model', model)

    return rmsle
    # mape = mean_absolute_percentage_error(y_test, y_pred)
    # return mape

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
        os.makedirs(f'data/models/hpo', exist_ok=True)      
        joblib.dump({
            'model_name': model_name,
            'model_params':  study.best_trial.params,
            'rmsle': study.best_value,
            'n_trials': len(study.trials),
            'model': study.best_trial.user_attrs['model']
            }, f'data/models/hpo/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M")}.pkl')


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
