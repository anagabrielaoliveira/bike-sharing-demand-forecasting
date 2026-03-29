import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import List


class OutlierRemoval:
    def __init__(self, data: pd.DataFrame):
        self.df = data.copy()

    def temp_and_atemp_outlier_removal(self, min_temp_value: float, max_temp_value: float, min_atemp_value: float, max_atemp_value: float):
        mask = (( 
            (self.df['temp'] > min_temp_value) & 
            (self.df['temp'] < max_temp_value)
            ) & (
            (self.df['atemp'] > min_atemp_value) & 
            (self.df['atemp'] < max_atemp_value)
            ))

        self.df = self.df[~mask]
        
        return self

    def humidity_remove_zero_values(self):
        self.df = self.df[self.df['humidity'] != 0]
        return self

    def temp_vs_windspeed_outlier_removal(self, value: float):
        self.df = self.df[self.df['temp_vs_windspeed'] < value]
        return self

    def train_if_model(self, columns: List[str]):
        self.model = IsolationForest(
            contamination=0.005,
            random_state=42,
            n_estimators=100,
        )
        self.model.fit(self.df[columns])
        return self
    
    def if_removal_predict(self, columns: List[str]):
        mask = self.model.predict(self.df[columns]) == 1
        self.df = self.df[mask]
        return self

    def get_dataframe(self):
        return self.df
