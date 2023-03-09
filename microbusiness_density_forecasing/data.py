from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

"""
TODO:
- Add "population" feature for test data.
- Try different data preprocessing
- Try predicting active instead of microbusiness_density
- More data from https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/372604
"""

T_AVAILABLE, T_PREDICT = 41, 6
LAG = 2
FEATURE_NAMES, TARGET_NAME = [
    # "active",
    "population",
    "density",
    "pct_bb",
    "pct_college",
    "pct_foreign_born",
    "pct_it_workers",
    "median_hh_inc",
], "density"


class TimeSeriesBuilder:
    """
    A temporary object allowing to fill time series data iteratively.
    """

    def __init__(self, n_timesteps):
        self.n_timesteps = n_timesteps
        self.features = {
            name: np.full((self.n_timesteps,), np.nan) for name in FEATURE_NAMES
        }

    def build_X(self) -> np.array:
        """
        Puts all features into one matrix.

        Returns:
            X: np.array, float[n_timesteps, n_features]
                Features. Not cleaned, imputed or normalized.

        """
        X = np.transpose(
            np.stack(
                [self.features[name] for name in FEATURE_NAMES if name != TARGET_NAME]
            )
        )
        assert X.ndim == 2
        # for name in FEATURE_NAMES:
        #     for step in range(len(self.features[name])):
        #         if np.isnan(self.features[name][step]):
        #             print(f"{name}_{step} is missing")
        return X

    def build_y(self) -> np.array:
        """
        Returns:
            y: np.array, float[n_timesteps = T_AVAILABLE]
                Targets.
        """
        y = self.features[TARGET_NAME]
        # assert not np.isnan(y).any()
        assert y.ndim == 1
        return y


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
    number = (year - start_year) * 12 + (month - start_month)
    return number


def month_date(number: int) -> str:
    """
    Inverse of month_number.
    """
    start_year, start_month = 2019, 8
    year, month, day = start_year + number // 12, start_month + number % 2, 1
    return f"{year}-{month}-{day}"


def merge_census(
    census_df: pd.DataFrame,
    cfips: int,
    feature_dict: dict[str, TimeSeriesBuilder],
    timestep: int,
    year: int,
):
    """
    Takes the census_df row for given cfips, iterates over features that match timestep by year,
    puts values of these features into feature_dict[...][timestep].
    """
    for name, value in census_df.loc[cfips].to_dict().items():
        feature_name, feature_year = name[:-5], int(name[-4:])
        if feature_year == year - LAG:
            feature_dict[feature_name][timestep] = value
    # for feature_name in feature_dict.keys():
    #     if np.isnan(feature_dict[feature_name][timestep]):
    #         print(f"Missing {feature_name} for cfips={cfips}, timestep={timestep}, year={year}")


def build_train(
    train_df: pd.DataFrame,
    census_df: pd.DataFrame,
    population_dict: dict[tuple[int, int], int],
) -> tuple[np.array, np.array]:
    train_builder = defaultdict(lambda: TimeSeriesBuilder(n_timesteps=T_AVAILABLE))

    for train_row in tqdm(train_df.iterrows(), total=len(train_df)):
        _row_id, cfips, _county, _state, date, density, active = train_row[1]
        timestep, year = month_number(date), int(date[:4])
        feature_dict = train_builder[cfips].features
        # feature_dict["active"][timestep] = active
        population = population_dict[(cfips, year - LAG)]
        feature_dict["population"][timestep] = population
        assert abs(density - active * 100 / population) < 1
        feature_dict["density"][timestep] = density
        merge_census(census_df, cfips, feature_dict, timestep, year)

    X_train = np.stack([series.build_X() for series in train_builder.values()])
    y_train = np.stack([series.build_y() for series in train_builder.values()])

    return X_train, y_train


class Reshaper:
    """
    Sklearn-style class to reshape data in a pipeline.
    """

    def __init__(self, shape: list[int]):
        self.shape = shape

    def fit(self, X, y=None):
        print("fit", X.shape, self.shape)
        return self

    def transform(self, X):
        print("transform", X.shape, self.shape)
        return X.reshape(*self.shape)


def clean_train(X: np.array, y: np.array) -> tuple[np.array, np.array, Pipeline]:
    """
    Cleans and normalizes the data, and saves the pipeline to do the same with test.

    Returns:
        X: torch.tensor, float32[n_series, n_timesteps = T_AVAILABLE, n_features]
            Features. Already sanitized, encoded properly into floats and normalized.
        y: torch.tensor, float32[n_series, n_timesteps = T_AVAILABLE]
            Targets.
        pipeline: Pipeline
            Pipeline that applies the same preprocessing as was done to X.
    """
    n_series, n_timesteps, n_features = X.shape
    pipeline = Pipeline(
        [
            ("A", Reshaper([-1, n_features])),
            ("B", RobustScaler()),
            ("C", SimpleImputer()),
            ("D", Reshaper([n_series, -1, n_features])),
        ]
    )
    X = pipeline.fit_transform(X)
    assert np.isnan(X).sum() == 0

    return X, y, pipeline


def build_test(
    sample_df: pd.DataFrame,
    census_df: pd.DataFrame,
    population_dict: dict[tuple[int, int], int],
) -> np.array:
    """
    X_test should be in the same format as X_train,
    but starting from 1001_2023-01-01 (or 1001_2022-01-01) to 56045_2023-06-01.
    Putting together predictions on np.concatenate(X_train, X_test) will give values
    to put into sample_submission table.
    """
    test_builder = defaultdict(lambda: TimeSeriesBuilder(n_timesteps=T_PREDICT))

    for train_row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        row_id, density = train_row[1]
        cfips, date = row_id.split("_")
        cfips = int(cfips)
        timestep, year = month_number(date) - T_AVAILABLE, int(date[:4])
        feature_dict = test_builder[cfips].features
        population = population_dict[(cfips, year - LAG)]
        feature_dict["population"][timestep] = population
        feature_dict["density"][timestep] = density
        merge_census(census_df, cfips, feature_dict, timestep, year)

    X_test = np.stack([series.build_X() for series in test_builder.values()])
    return X_test


def build_dataset() -> tuple[torch.tensor, torch.tensor]:
    """
    Returns:
        X: torch.tensor, float32[n_series, n_timesteps = T_AVAILABLE, n_features]
            Features. Already sanitized, encoded properly into floats and normalized.
        y: torch.tensor, float32[n_series, n_timesteps = T_AVAILABLE]
            Targets.
    """

    train_df = pd.concat([pd.read_csv("train.csv"), pd.read_csv("revealed_test.csv")])
    census_df = pd.read_csv("census_starter.csv").set_index("cfips")
    population_dict = torch.load("population.p")
    X_train, y_train = build_train(train_df, census_df, population_dict)
    X_train, y_train, pipeline = clean_train(X_train, y_train)
    sample_df = pd.read_csv("sample_submission.csv")
    X_test = build_test(sample_df, census_df, population_dict)
    X_test = pipeline.transform(X_test)

    f32 = lambda x: torch.tensor(x, dtype=torch.float32)
    return f32(X_train), f32(y_train), f32(X_test)


X_train, y_train, X_test = build_dataset()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

torch.save(X_train, "X_train.p")
torch.save(y_train, "y_train.p")
torch.save(X_test, "X_test.p")
os.system("rm data.zip && zip data.zip X_test.p X_train.p y_train.p")
