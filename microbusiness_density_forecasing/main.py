from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

"""
TODO:
- Use revealed_test.csv for training too
- Try different data preprocessing
- Try predicting active instead of microbusiness_density
"""


class TimeSeriesBuilder:
    """
    A temporary object allowing to fill time series data iteratively.
    """

    def __init__(self, n_timesteps, feature_names, target_name):
        self.n_timesteps = n_timesteps
        self.feature_names = feature_names
        self.target_name = target_name
        self.features = {
            name: np.full((self.n_timesteps,), np.nan) for name in self.feature_names
        }

    def build(self) -> tuple[np.array, np.array]:
        """
        Puts all features into one matrix.

        Returns:
            X: np.array, float[n_series, n_timesteps = 39, n_features]
                Features. Not cleaned, imputed or normalized.
            y: np.array, float[n_series, n_timesteps = 39]
                Targets.
        """
        X = np.stack(
            [
                self.features[name]
                for name in self.feature_names
                if name != self.target_name
            ]
        )
        y = self.features[self.target_name]
        assert not np.isnan(y).any()
        return X, y


def month_number(date: str) -> int:
    """
    Converts a date into a number of month passed since 2019-08-01.

    Arguments:
        date: str
            Date in YYYY-MM-DD format.

    Returns:
        months: int
            Number of months since 2019-08-01.
            For "2019-09-01" it is 0.
    """
    year, month, _day = map(int, date.split("-"))
    start_year, start_month = 2019, 8
    months = (year - start_year) * 12 + (month - start_month)
    return months


def build_dataset() -> tuple[np.array, np.array]:
    """
    Returns:
        X: np.array, float[n_series, n_timesteps = 39, n_features]
            Features. Already sanitized, encoded properly into floats and normalized.
        y: np.array, float[n_series, n_timesteps = 39]
            Targets.
    """
    builder_by_cfips = defaultdict(
        lambda: TimeSeriesBuilder(
            n_timesteps=39,
            feature_names=[
                "active",
                "density",
                "pct_bb",
                "pct_college",
                "pct_foreign_born",
                "pct_it_workers",
                "median_hh_inc",
            ],
            target_name="density",
        )
    )

    train_df = pd.read_csv("train.csv")
    census_df = pd.read_csv("census_starter.csv").set_index("cfips")

    for train_row in tqdm(train_df.iterrows(), total=len(train_df)):
        _row_id, cfips, _county, _state, date, density, active = train_row[1]
        timestep, year = month_number(date), date[:4]
        feature_dict = builder_by_cfips[cfips].features
        feature_dict["active"][timestep] = active
        feature_dict["density"][timestep] = density
        for name, value in census_df.loc[cfips].to_dict().items():
            feature_name, feature_year = name[:-5], name[-4:]
            if feature_year == year:
                feature_dict[feature_name][timestep] = value

    series = {name: value.build() for name, value in builder_by_cfips.items()}

    X_train = np.stack([X for X, _y in series.values()])
    y_train = np.stack([y for _X, y in series.values()])

    """
    X_test should be in the same format as X_train,
    but starting from 1001_2022-11-01 (or 1001_2022-01-01) to 56045_2023-06-01.
    Putting together predictions on np.concatenate(X_train, X_test) will give values 
    to put into sample_submission table.
    """

    n_series, n_timesteps, n_features = X_train.shape
    X_train = X_train.reshape(n_series * n_timesteps, n_features)
    scaler = RobustScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = X_train.reshape(n_series, n_timesteps, n_features)

    assert np.isnan(X_train).sum() == 0

    return X_train, y_train


build_dataset()
