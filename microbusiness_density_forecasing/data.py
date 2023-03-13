import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

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
    # from census_starter.csv
    "population",
    "pct_bb",
    "pct_college",
    "pct_foreign_born",
    "pct_it_workers",
    "median_hh_inc",
    # from VF_indcom_counties_Q222.csv
    "avg_traffic",
    "gmv_rank",
    "merchants_rank",
    "orders_rank",
    # from big_data
    # "big_0",
    # "big_1",
    # "big_2",
    # "big_3",
    # "big_4",
    # "big_5",
    # "big_6",
    # "big_7",
    # "big_8",
    # "big_9",
    # "big_10",
    # "big_11",
    # "big_12",
    # "big_13",
    # "big_14",
    # "big_15",
    # "big_16",
    # "big_17",
    # "big_18",
    # "big_19",
    # "big_20",
    # "big_21",
    # "big_22",
    # "big_23",
    # "big_24",
    # "big_25",
    # "big_26",
    # "big_27",
    # "big_28",
    # "big_29",
    # "big_30",
    # "big_31",
    # "big_32",
    # "big_33",
    # "big_34",
    # target
    "density",
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
    month = start_month + number
    year, month, day = start_year + (month - 1) // 12, (month - 1) % 12 + 1, 1
    return f"{year}-{month}-{day}"


month_order = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]


def mmmyy_to_yyyymmdd(date):
    assert len(date) == 5
    month = month_order.index(date[:3]) + 1
    year = int(date[3:])
    return f"20{year}-{month:02}-01"


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


def build_VF_indcom(result: dict):
    """
    result: dict[(cfips, feature, timestep), float]
    """
    df = pd.read_csv("VF_indcom_counties_Q222.csv").set_index("cfips")
    for col in df.columns:
        parts = col.split("_")
        name = "_".join(parts[:-1])
        if name not in FEATURE_NAMES:
            continue
        t = month_number(mmmyy_to_yyyymmdd(parts[-1]))
        for cfips, value in df[col].items():
            if np.isnan(cfips) or np.isnan(value):
                continue
            if t < 12:
                result[(int(cfips), name, t)] = float(value)
            result[(int(cfips), name, t + 12)] = float(value)


def build_train(
    train_df: pd.DataFrame,
    census_df: pd.DataFrame,
    population_dict: dict[tuple[int, int], int],
    other_features: dict,
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

    for (cfips, name, t), value in other_features.items():
        if name not in FEATURE_NAMES:
            continue
        if t < T_AVAILABLE and cfips in train_builder.keys():
            train_builder[cfips].features[name][t] = value

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
    Cleans the data, and saves the pipeline to do the same with test.

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
            ("B", SimpleImputer(keep_empty_features=True)),
            ("C", Reshaper([n_series, -1, n_features])),
        ]
    )
    X = pipeline.fit_transform(X)
    assert np.isnan(X).sum() == 0

    return X, y, pipeline


def build_test(
    sample_df: pd.DataFrame,
    census_df: pd.DataFrame,
    population_dict: dict[tuple[int, int], int],
    other_features: dict,
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

    for (cfips, name, t), value in other_features.items():
        if name not in FEATURE_NAMES:
            continue
        if T_AVAILABLE <= t < T_AVAILABLE + T_PREDICT and cfips in test_builder.keys():
            test_builder[cfips].features[name][t - T_AVAILABLE] = value

    X_test = np.stack([series.build_X() for series in test_builder.values()])
    return X_test


def whiten(A, pipeline=None):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import Pipeline

    shape = A.shape
    A = A.reshape(-1, A.shape[-1])

    if not pipeline:
        pipeline = Pipeline(
            [
                ("scale", RobustScaler()),
                ("pca", PCA(n_components=min(A.shape[0], A.shape[1]))),
            ]
        ).fit(A)
    return torch.tensor(pipeline.transform(A).reshape(shape)), pipeline


def build_big(other_features: dict, cfips: list):
    arr = torch.load("big.p")
    print(len(cfips), "cfips")
    for c in tqdm(cfips):
        for i in range(arr.shape[1]):
            for t in range(arr.shape[0]):
                other_features[(c, f"big_{i}", t)] = arr[t, i]


def add_differences(X: np.array, feature_range: tuple[int, int], k: int) -> np.array:
    """
    Adds new features of form X[i, j, t] - X[i, j - k, t] for all features in feature_range.
    X: np.array, float32[n_series, n_timesteps, n_features]
        Array with features.
    feature_range: tuple[int, int]
        feature_range = (l, r) means features from X[:, :, l] to X[:, :, r - 1].
    k: int
        Lag with which to calculate differences.
    Returns
        X: np.array
    """
    n_series, n_timesteps, n_features = X.shape
    l, r = feature_range
    shifted_X = np.concatenate((np.zeros((n_series, k, n_features)), X[:, :-k]), axis=1)
    X = np.concatenate((X, (X - shifted_X)[:, :, l:r]), axis=2)
    return X


def build_dataset() -> tuple[torch.tensor, torch.tensor]:
    """
    Returns:
        X: torch.tensor, float32[n_series, n_timesteps = T_AVAILABLE, n_features]
            Features. Already sanitized and encoded properly into floats.
        y: torch.tensor, float32[n_series, n_timesteps = T_AVAILABLE]
            Targets.
    """

    train_df = pd.concat([pd.read_csv("train.csv"), pd.read_csv("revealed_test.csv")])
    census_df = pd.read_csv("census_starter.csv").set_index("cfips")
    population_dict = torch.load("population.p")
    other_features = dict()
    build_VF_indcom(other_features)
    # build_big(other_features, census_df.index)
    X_train, y_train = build_train(train_df, census_df, population_dict, other_features)
    print("train", X_train.shape, y_train.shape)
    X_train, y_train, pipeline = clean_train(X_train, y_train)
    print("clean train", X_train.shape, y_train.shape)
    sample_df = pd.read_csv("sample_submission.csv")
    X_test = build_test(sample_df, census_df, population_dict, other_features)
    print("test", X_test.shape)
    X_test = pipeline.transform(X_test)
    print("clean test", X_test.shape)

    X_train[:, :, 1:], pipeline = whiten(X_train[:, :, 1:])
    X_test[:, :, 1:], _ = whiten(X_test[:, :, 1:])

    X_merged = np.concatenate((X_train, X_test), axis=1)
    feature_range = (1, X_train.shape[2])
    X_merged = add_differences(X_merged, feature_range, 1)
    # X_merged = add_differences(X_merged, feature_range, 2)

    X_train, X_test = np.split(X_merged, [X_train.shape[1]], axis=1)

    f32 = lambda x: torch.tensor(x, dtype=torch.float32)
    return f32(X_train), f32(y_train), f32(X_test)


if __name__ == "__main__":
    X_train, y_train, X_test = build_dataset()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)

    torch.save(X_train, "X_train.p")
    torch.save(y_train, "y_train.p")
    torch.save(X_test, "X_test.p")
    os.system("rm data.zip && zip data.zip X_test.p X_train.p y_train.p")
