from collections import defaultdict
from itertools import islice
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
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
