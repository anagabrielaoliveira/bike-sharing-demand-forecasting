from metaflow import FlowSpec, IncludeFile, step, Parameter
from .preprocess import (
    OutlierRemoval,
    create_time_features,
    create_peak_feature,
    create_time_slots,
    create_interaction_features,
    update_workingday_and_holiday
)

from .train import (
    train_rf_model,
    evaluate_model,
    save_baseline_model_info
)

from .eval import EvalResults
from .hpo import OptunaHPO
from sklearn.model_selection import train_test_split
from config.config import QUANTITATIVE_COLS, PEAK_HOURS, SPECIAL_HOLIDAYS, Y_COLS, RANDOM_STATE
import pandas as pd

class BikeSharingDemandFlow(FlowSpec):

    n_trials = Parameter(
        name='n_trials',
        default=10
    )

    @step
    def start(self):
        self.train_data = pd.read_csv('data/train.csv')
        
        # Convert datetime column to datetime type and set as index (matching notebook)
        self.train_data['datetime'] = pd.to_datetime(self.train_data['datetime'])
        self.train_data.set_index('datetime', inplace=True)
        
        self.next(self.preprocess)

    @step
    def preprocess(self):
        train_data = create_time_features(self.train_data, special_holidays=SPECIAL_HOLIDAYS)
        train_data = update_workingday_and_holiday(train_data)
        train_data = create_interaction_features(train_data)
        train_data = create_peak_feature(train_data, peak_hours=PEAK_HOURS)
        train_data = create_time_slots(train_data)

        self.train_data = train_data.drop(columns=['date'])

        self.next(self.outlier_removal)

    @step
    def outlier_removal(self):

        ## train data outlier removal
        train_data = (
            OutlierRemoval(self.train_data)
            .temp_and_atemp_outlier_removal(
                min_temp_value=25,
                max_temp_value=37,
                min_atemp_value=10,
                max_atemp_value=15
            )
            .humidity_remove_zero_values()
            .temp_vs_windspeed_outlier_removal(value=1750)
            .train_if_model(columns=QUANTITATIVE_COLS)
            .if_removal_predict(columns=QUANTITATIVE_COLS)
            .get_dataframe()
        )

        self.train_data = train_data
        self.train_data.to_csv('data/processed_train_data.csv', index=False)

        self.next(self.train_baseline_model)

    @step
    def train_baseline_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.train_data.drop(columns=Y_COLS),
            self.train_data[Y_COLS],
            test_size=0.2,
            random_state=RANDOM_STATE
        )
        model = train_rf_model(X_train, y_train)
        mape = evaluate_model(model, X_test, y_test)
        save_baseline_model_info(model_name='baseline_model', model=model, mape=mape)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.next(self.hpo)

    @step
    def hpo(self):
        hpo = OptunaHPO(
            X_train=self.X_train,
            y_train=self.y_train,
            x_test=self.X_test,
            y_test=self.y_test
        )
        
        hpo.lgbm_optimize(n_trials=self.n_trials)
        hpo.xgb_optimize(n_trials=self.n_trials)
        hpo.rf_optimize(n_trials=self.n_trials)

        self.next(self.evaluate_models)

    @step
    def evaluate_models(self):
        eval_results = EvalResults(
            hpo_results_folder='data/models/hpo',
            baseline_results_path='data/models/baseline_model_20260125_1257.json'
        )
        best_model = eval_results.evaluate_models()
        print(pd.DataFrame(best_model))
        self.best_model = best_model
        self.next(self.end)

    @step
    def end(self):
        pass