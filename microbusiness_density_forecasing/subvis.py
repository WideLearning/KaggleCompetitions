from matplotlib import pyplot as plt
import pandas as pd
from sys import argv
from data import month_number, month_date
import torch


def visualize(filename):
    pop = torch.load("population.p")
    train = pd.read_csv("train.csv")
    sub = pd.read_csv(f"submissions/{filename}.csv")
    data = dict()

    for _idx, (name, value) in sub.iterrows():
        cfips, t = name.split("_")
        cfips = int(cfips)
        t = month_number(t)
        data[(cfips, t)] = value

    for _idx, (
        _row_id,
        cfips,
        _county,
        _state,
        t,
        value,
        _active,
    ) in train.iterrows():
        cfips = int(cfips)
        t = month_number(t)
        data[(cfips, t)] = value

    def get_year(t):
        date = month_date(t)
        return int(date[:4]) - 2

    k = list(data.keys())
    for (cfips, t) in k:
        data[(cfips, t)] *= pop[(cfips, get_year(t))] / 100

    T_AVAILABLE, T_REVEALED, T_PREDICT = 41, 2, 6

    def plot(cfips: int):
        x = list(range(T_AVAILABLE))
        y = [data[(cfips, t)] for t in x]
        plt.plot(x, y, "o-", color="blue", ms=3)
        x = list(range(T_AVAILABLE, T_AVAILABLE + T_PREDICT))
        y = [data[(cfips, t)] for t in x]
        plt.grid("major")
        plt.plot(x, y, "o-", color="green", ms=3)

    plt.subplot(2, 2, 1)
    plot(9001)
    plt.subplot(2, 2, 2)
    plot(6001)
    plt.subplot(2, 2, 3)
    plot(21087)
    plt.subplot(2, 2, 4)
    plot(48301)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize(argv[1])
