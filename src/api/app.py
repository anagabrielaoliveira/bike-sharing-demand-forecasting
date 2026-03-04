from fastapi import FastAPI
import joblib

app = FastAPI()

@app.get("/health")
def healt_check():
    return {"status": "ok"}

@app.get("/model_info")
def model_info():
    """
    This function is used to get the model information.

    ------------------------------------------------------------

    Returns:
    dict
        A dictionary containing the model information
    """
    data = joblib.load('data/best_model.pkl')

    return {
        'model_name': data['model_name'],
        'model_params': data['model_params'],
        'rmsle': data['rmsle'],
        'n_trials': data['n_trials'],
        'model_type': type(data['model']).__name__
    }
