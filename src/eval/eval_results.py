from pathlib import Path
import json

class EvalResults:
    def __init__(self, hpo_results_folder, baseline_results_path):
        self.hpo_results_folder = hpo_results_folder
        self.baseline_results_path = baseline_results_path
        self.results = None
        self.best_model = None
    
    def _load_results(self):
        baseline_path = Path(self.baseline_results_path)
        baseline_files = list(baseline_path.glob('*.json'))

        baseline_data = []
        for file in baseline_files:
            with open(file, 'r') as f:
                baseline_archive = json.load(f)
                baseline_data.append({
                    'model_name': baseline_archive['model_name'],
                    'rmsle': baseline_archive['rmsle']
                })

        hpo_path = Path(self.hpo_results_folder)
        hpo_results = list(hpo_path.glob('*.json'))

        hpo_data = []
        for archive in hpo_results:
            with open(archive, 'r') as f:
                data = json.load(f)
                hpo_data.append(data)
        
        self.results = [{'model_name': data['model_name'],
                         'rmsle': data['rmsle']} for data in hpo_data] + baseline_data

    def _return_best_model(self):
        self.best_model = min(self.results, key=lambda x: x['rmsle'])

    def evaluate_models(self):
        self._load_results()
        self._return_best_model()
        return self.best_model

# e = EvalResults()
# e._load_results()
# e._return_best_model()

# print(e.best_model)