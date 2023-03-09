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
from solve import noise, last_density

"""
TODO:
- set up local validation

- check https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/375802
"""

N_SERIES, N_FEATURES = 3135, 5
T_AVAILABLE, T_REVEALED, T_PREDICT = 41, 2, 6

X_train = torch.load("X_train.p")
assert X_train.shape == (N_SERIES, T_AVAILABLE, N_FEATURES)
y_train = torch.load("y_train.p")
assert y_train.shape == (N_SERIES, T_AVAILABLE)


def submission(data, name):
    assert data.shape == (N_SERIES, T_PREDICT)
    df = pd.read_csv("sample_submission.csv")
    revealed = y_train[:, -T_REVEALED:]
    total = torch.cat((revealed, data), dim=1).T
    assert total.shape == (2 + T_PREDICT, N_SERIES)
    df["microbusiness_density"] = total.ravel().numpy()
    df.to_csv(name, index=False)


X_test = torch.load("X_test.p")
assert X_test.shape == (N_SERIES, T_PREDICT, N_FEATURES)
y_test = last_density(X_train, y_train, X_test)

submission(y_test, "submissions/last_density.csv")
