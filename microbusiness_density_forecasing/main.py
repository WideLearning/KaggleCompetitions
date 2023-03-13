import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from subvis import visualize

from solve import (
    last_density,
    last_density_corrected,
    row_linear,
    row_linear_extra,
    row_linear_extra_log,
    row_linear_add,
    row_mlp_add,
    arima,
)

"""
TODO:
- set up local validation

- check https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/375802
"""

N_SERIES, N_FEATURES = 3135, 45
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


def smape(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1e-9
    return (200 * np.abs(a - b) / (np.abs(a) + np.abs(b) + eps)).mean()


X_test = torch.load("X_test.p")
assert X_test.shape == (N_SERIES, T_PREDICT, N_FEATURES)


def validate(
    solve, train_l: int, train_r: int, val_l: int, val_r: int, visualize: bool = False
):
    assert train_l < train_r <= val_l < val_r

    p = solve(
        X_train[:, train_l:train_r, :],
        y_train[:, train_l:train_r],
        X_train[:, val_l:val_r, :],
    )
    y = y_train[:, val_l:val_r]
    if visualize:
        eps = 1e-9
        delta = np.log(p + eps) - np.log(y + eps)
        print("100 log-MAE:", 100 * np.abs(delta).mean())
        samples = 256
        print(f"Sampling {samples} out of {delta.shape[0]} predictions")
        idx = np.random.choice(delta.shape[0], size=samples, replace=False)
        plt.subplot(1, 2, 1)
        for i in idx:
            plt.plot(delta[i], "-ob", ms=0.5, lw=0.5, alpha=0.2)
        plt.ylim((-0.2, 0.2))
        plt.grid("major")
        plt.subplot(1, 2, 2)
        plt.plot(100 * np.abs(delta).mean(axis=0), "-ob")
        plt.ylim((0, 10))
        plt.grid("major")
        plt.show()
    return smape(p, y)


val_len, skip, r = 3, 2, T_AVAILABLE
result = validate(row_linear_add, 0, r - val_len - skip, r - val_len, r, visualize=True)
print("SMAPE:", result)

# y_test = row_mlp_add(X_train, y_train, X_test)
# name = "row_mlp_add"
# submission(y_test, f"submissions/{name}.csv")
# visualize(name)
