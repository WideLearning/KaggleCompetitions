# pylint: disable=unused-import
import math

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch.nn.functional as F
import torch.optim as opt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

from tracker import FileTracker


def write_answer(answer, filename):
    test_df = pd.read_csv("test.csv")
    assert isinstance(answer, np.ndarray)
    assert answer.size == len(test_df)
    answer = answer.astype(int)
    answer_df = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": answer
    })
    answer_df.to_csv(filename, index=False)


def read_input():
    full_train = pd.read_csv("train.csv")
    X_train = full_train.drop(columns=["PassengerId", "Survived"])
    bad_columns = [c for c in X_train.columns if (
        X_train[c].dtype == "object" and X_train[c].unique().size > 5)]
    X_train = pd.get_dummies(X_train.drop(columns=bad_columns))
    y_train = full_train["Survived"].to_numpy()
    X_test = pd.read_csv("test.csv").drop(columns="PassengerId")
    X_test = pd.get_dummies(X_test.drop(columns=bad_columns))

    imp = IterativeImputer(max_iter=10, random_state=0).fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test


X_train, y_train, X_test = read_input()
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
    X_train, y_train, test_size=0.2)


def to_torch_tensor(arr):
    return torch.from_numpy(arr).to(torch.float32)


def to_torch_1d(arr):
    return torch.from_numpy(arr.reshape(arr.shape[0], 1)).to(torch.float32)


X_train = to_torch_tensor(X_train)
X_val = to_torch_tensor(X_val)
X_test = to_torch_tensor(X_test)

y_train = to_torch_1d(y_train)
y_val = to_torch_1d(y_val)


network = nn.Sequential(
    nn.Linear(X_train.shape[1], 200, bias=True),
    nn.Tanh(),
    nn.Linear(200, 200, bias=False),
    nn.Tanh(),
    nn.Linear(200, 200, bias=False),
    nn.Tanh(),
    nn.Linear(200, 200, bias=False),
    nn.Tanh(),
    nn.Linear(200, 200, bias=False),
    nn.Tanh(),
    nn.Linear(200, 200, bias=False),
    nn.Tanh(),
    nn.Linear(200, 200, bias=False),
    nn.Tanh(),
    nn.Linear(200, 1, bias=False),
    nn.Sigmoid()
)

tracker = FileTracker(k=5, filename="save.p")

optimizer = opt.AdamW(network.parameters(), lr=1e-3, weight_decay=100)

epochs_num = 100
logging_exp = 1.1
logging_points = set(int(logging_exp ** i)
                     for i in range(int(math.log(epochs_num, logging_exp))))

for i in tqdm(range(epochs_num)):
    with torch.no_grad():
        y_pred = network(X_val)
        val_loss = F.binary_cross_entropy(y_pred, y_val)

    y_pred = network(X_train)
    train_loss = F.binary_cross_entropy(y_pred, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if i in logging_points:
        print(f"{i}: {train_loss.item():.3f} {val_loss.item():.3f}")
        tracker.model(network)
        tracker.scalar("train/loss", train_loss.item())
        tracker.scalar("val/loss", val_loss.item())
tracker.dump()

# answer = network(X_test).detach().numpy().reshape(
#     X_test.shape[0]).round().astype(int)
# write_answer(answer, "submission.csv")
