from pathlib import Path
import json
import joblib

class EvalResults:
    def __init__(self, hpo_results_folder, baseline_results_path):
        self.hpo_results_folder = hpo_results_folder
        self.baseline_results_path = baseline_results_path
        self.results = None
        self.best_model = None
    
    def _load_results(self):
        baseline_path = Path(self.baseline_results_path)
        baseline_files = list(baseline_path.glob('*.pkl'))

        baseline_data = []
        for file in baseline_files:
            with open(file, 'rb') as f:
                baseline_archive = joblib.load(f)
                
                baseline_data.append({
                    'model_name': baseline_archive['model_name'],
                    'model_params': baseline_archive['model_params'],
                    'rmsle': baseline_archive['rmsle'],
                    'model': baseline_archive['model']       
                })
        
        hpo_path = Path(self.hpo_results_folder)
        hpo_files = list(hpo_path.glob('*.pkl'))

        hpo_data = []
        for file in hpo_files:
            with open(file, 'rb') as f:
                data = joblib.load(f)
                hpo_data.append(data)
        
        self.results = [{
            'model_name': data['model_name'],
            'model_params': data['model_params'],
            'rmsle': data['rmsle'],
            'model': data['model']
            }
            for data in hpo_data
            ] + baseline_data   

    def _return_best_model(self):
        self.best_model = min(self.results, key=lambda x: x['rmsle'])
    
    def save_best_model(self):
        model_selected = self.best_model['model']

        model_path = Path("data/best_model.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {"model": model_selected},
            model_path
        )

    def evaluate_models(self):
        self._load_results()
        self._return_best_model()
        self.save_best_model()
        return self.best_model

# e = EvalResults()
# e._load_results()
# e._return_best_model()

# print(e.best_model)