from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

train_df = pd.read_csv("train.csv")
census_df = pd.read_csv("census_starter.csv")


def month_number(date):
    year, month, _day = map(int, date.split("-"))
    start_year, start_month = 2019, 8
    return (year - start_year) * 12 + (month - start_month)


class TimeSeries:
    def __init__(self, n_timesteps, feature_names, target_name):
        self.n_timesteps = n_timesteps
        self.feature_names = feature_names
        self.target_name = target_name
        self.features = {
            name: np.full((self.n_timesteps,), np.nan) for name in self.feature_names
        }

    def build_dense(self):
        X = np.stack(
            [
                self.features[name]
                for name in self.feature_names
                if name != self.target_name
            ]
        )
        y = self.features[self.target_name]
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
        return X, y


series = defaultdict(
    lambda: TimeSeries(
        n_timesteps=39, feature_names=["active", "density"], target_name="density"
    )
)

for row in train_df.iterrows():
    row_id, cfips, county, state, date, density, active = row[1]
    t = month_number(date)
    series[cfips].features["active"][t] = active
    series[cfips].features["density"][t] = density

series = {name: value.build_dense() for name, value in series.items()}
"""
X: pd.DataFrame, (n_series, n_timesteps = 39, n_features)
y: pd.DataFrame, (n_series, n_timesteps = 39)
"""
X = np.stack([X for X, _y in series.values()])
y = np.stack([y for _X, y in series.values()])