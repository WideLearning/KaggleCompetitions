import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm


def noise(
    X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor
) -> torch.Tensor:
    N_SERIES, N_FEATURES = X_train.size(0), X_train.size(2)
    T_AVAILABLE, T_PREDICT = X_train.size(1), X_test.size(1)
    assert X_train.shape == (N_SERIES, T_AVAILABLE, N_FEATURES)
    assert y_train.shape == (N_SERIES, T_AVAILABLE)
    assert X_test.shape == (N_SERIES, T_PREDICT, N_FEATURES)

    y_test = torch.randn((N_SERIES, T_PREDICT))
    return y_test


def last_density(
    X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor
) -> torch.Tensor:
    N_SERIES, N_FEATURES = X_train.size(0), X_train.size(2)
    T_AVAILABLE, T_PREDICT = X_train.size(1), X_test.size(1)
    assert X_train.shape == (N_SERIES, T_AVAILABLE, N_FEATURES)
    assert y_train.shape == (N_SERIES, T_AVAILABLE)
    assert X_test.shape == (N_SERIES, T_PREDICT, N_FEATURES)

    y_test = torch.empty((N_SERIES, T_PREDICT))
    for i in range(N_SERIES):
        for j in range(T_PREDICT):
            y_test[i, j] = y_train[i, -1]
    return y_test


def last_density(
    X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor
) -> torch.Tensor:
    N_SERIES, N_FEATURES = X_train.size(0), X_train.size(2)
    T_AVAILABLE, T_PREDICT = X_train.size(1), X_test.size(1)
    assert X_train.shape == (N_SERIES, T_AVAILABLE, N_FEATURES)
    assert y_train.shape == (N_SERIES, T_AVAILABLE)
    assert X_test.shape == (N_SERIES, T_PREDICT, N_FEATURES)

    y_test = torch.empty((N_SERIES, T_PREDICT))
    for i in range(N_SERIES):
        for j in range(T_PREDICT):
            y_test[i, j] = y_train[i, -1]
    return y_test


def last_density_corrected(
    X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor
) -> torch.Tensor:
    N_SERIES, N_FEATURES = X_train.size(0), X_train.size(2)
    T_AVAILABLE, T_PREDICT = X_train.size(1), X_test.size(1)
    assert X_train.shape == (N_SERIES, T_AVAILABLE, N_FEATURES)
    assert y_train.shape == (N_SERIES, T_AVAILABLE)
    assert X_test.shape == (N_SERIES, T_PREDICT, N_FEATURES)

    y_test = torch.empty((N_SERIES, T_PREDICT))
    for i in range(N_SERIES):
        for j in range(T_PREDICT):
            y_test[i, j] = y_train[i, -1] * (X_train[i, -1, 0] / X_test[i, j, 0])
    return y_test


class Reshaper:
    """
    Sklearn-style class to reshape data in a pipeline.
    """

    def __init__(self, shape: list[int]):
        self.shape = shape

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(*self.shape)


class LogFeatures:
    """
    Sklearn-style class to add logarithm of each feature as a new feature.
    """

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return torch.cat((X, X.log()), axis=-1)


def row_linear(
    X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor
) -> torch.Tensor:
    N_SERIES, N_FEATURES = X_train.size(0), X_train.size(2)
    T_AVAILABLE, T_PREDICT = X_train.size(1), X_test.size(1)
    assert X_train.shape == (N_SERIES, T_AVAILABLE, N_FEATURES)
    assert y_train.shape == (N_SERIES, T_AVAILABLE)
    assert X_test.shape == (N_SERIES, T_PREDICT, N_FEATURES)

    pipeline = Pipeline(
        [
            ("A", Reshaper([-1, N_FEATURES])),
            ("B", RobustScaler()),
        ]
    )

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    y_train = y_train.clamp_min(min=1).log().flatten()

    estimator = Lasso(alpha=1.0).fit(X_train, y_train.numpy())

    X_train = X_train.reshape(N_SERIES, T_AVAILABLE, N_FEATURES)
    X_test = X_test.reshape(N_SERIES, T_PREDICT, N_FEATURES)
    y_test = torch.empty((N_SERIES, T_PREDICT))
    for i in range(N_SERIES):
        for j in range(T_PREDICT):
            row = X_test[i, j].reshape(1, -1)
            p = estimator.predict(row).item()
            y_test[i, j] = p

    y_test = y_test.exp()
    return y_test


def row_linear_extra(
    X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor
) -> torch.Tensor:
    N_SERIES, N_FEATURES = X_train.size(0), X_train.size(2)
    T_AVAILABLE, T_PREDICT = X_train.size(1), X_test.size(1)
    assert X_train.shape == (N_SERIES, T_AVAILABLE, N_FEATURES)
    assert y_train.shape == (N_SERIES, T_AVAILABLE)
    assert X_test.shape == (N_SERIES, T_PREDICT, N_FEATURES)

    pipeline = Pipeline(
        [
            ("A", Reshaper([-1, N_FEATURES])),
            ("B", RobustScaler()),
            ("C", Reshaper([N_SERIES, -1, N_FEATURES])),
        ]
    )

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    y_train = y_train.clamp_min(min=1).log().numpy()
    y_shifted = np.roll(y_train, shift=1, axis=1)
    X_train = np.concatenate((X_train, y_shifted[:, :, np.newaxis]), axis=-1)
    estimator = Lasso(alpha=0.001).fit(
        X_train.reshape(-1, N_FEATURES + 1), y_train.flatten()
    )

    print(estimator.coef_)

    y_test = torch.empty((N_SERIES, T_PREDICT))
    for i in range(N_SERIES):
        for j in range(T_PREDICT):
            prev = y_train[i, -1]  # or make auto-regressive
            row = np.concatenate(
                (X_test[i, j].reshape(1, -1), prev.reshape(1, 1)), axis=-1
            )
            p = estimator.predict(row).item()
            y_test[i, j] = p

    y_test = y_test.exp()
    return y_test


def row_linear_log(
    X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor
) -> torch.Tensor:
    N_SERIES, N_FEATURES = X_train.size(0), X_train.size(2)
    T_AVAILABLE, T_PREDICT = X_train.size(1), X_test.size(1)
    assert X_train.shape == (N_SERIES, T_AVAILABLE, N_FEATURES)
    assert y_train.shape == (N_SERIES, T_AVAILABLE)
    assert X_test.shape == (N_SERIES, T_PREDICT, N_FEATURES)

    pipeline = Pipeline([("log", LogFeatures()), ("scale", RobustScaler())])

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    y_train = y_train.log()

    y_test = torch.empty((N_SERIES, T_PREDICT))
    for i in range(N_SERIES):
        for j in range(T_PREDICT):
            y_test[i, j] = y_train[i, -1] + (X_train[i, -1, 0] / X_test[i, j, 0]).log()

    y_test = y_test.exp()
    return y_test
