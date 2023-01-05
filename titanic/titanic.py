import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import f1_score
import matplotlib as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


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


class Tracker:
    def __init__(self, project, api_token):
        import neptune.new as neptune
        self.run = neptune.init_run(project, api_token)

    def scalar(self, name, value):
        self.run[name].log(value)
    
    def model(self, model):
        for name, param in model.named_parameters():
            self.scalar(name, param.abs().log().mean())
            self.scalar(name + "_grad", param.grad.abs().log().mean())


tracker = Tracker(project="WideLearning/Titanic",
                  api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==")

network = nn.Sequential(
    nn.Linear(X_train.shape[1], 200),
    nn.PReLU(),
    nn.Linear(200, 1),
    nn.Sigmoid()
)

optimizer = torch.optim.Adam(network.parameters(), lr=2e-3)

epochs_nums = 500

for i in range(epochs_nums):
    y_pred = network(X_train)
    loss = F.binary_cross_entropy(y_pred, y_train)
    tracker.scalar("train/loss", loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    tracker.model(network)

    y_pred = network(X_val)
    loss = F.binary_cross_entropy(y_pred, y_val)
    tracker.scalar("val/loss", loss.item())

answer = network(X_test).detach().numpy().reshape(
    X_test.shape[0]).round().astype(int)
write_answer(answer, "answer.csv")
