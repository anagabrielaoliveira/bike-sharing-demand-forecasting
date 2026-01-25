from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime
import json
import os

def train_rf_model(X, y):
    """
    This function is used to train a Random Forest baseline model.

    ------------------------------------------------------------

    Parameters:
    X : pandas.DataFrame
        The features to train the model on
    y : pandas.Series
        The target variable to train the model on
    Returns:
    model : sklearn.ensemble.RandomForestRegressor
        The trained model
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y_true):
    """
    This function is used to evaluate the model.

    ------------------------------------------------------------

    Parameters:
    model : sklearn.ensemble.RandomForestRegressor
        The model to evaluate
    X : pandas.DataFrame
    y_true : pandas.Series
        The target variable to evaluate the model on
    Returns:    
    mape : float
        The mean absolute percentage error of the model
    """
    y_pred = model.predict(X)
    return mean_absolute_percentage_error(y_true, y_pred)

def save_baseline_model_info(model_name, model, mape):
    """
    This function is used to save the baseline model information.

    ------------------------------------------------------------

    Parameters:
    model_name : str
    model : sklearn.ensemble.RandomForestRegressor
        The model to extract the information from
    rmsle : float
        The mean absolute percentage error of the model
    """
    os.makedirs('data/models', exist_ok=True)
    with open(f'data/models/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M")}.json', 'w') as f:
        json.dump({
            'model_name': model_name,
            'model_type': model.__class__.__name__,
            'model_params': model.get_params(),
            'mape': mape
        }, f)