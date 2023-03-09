"""
Population data is taken from https://www.kaggle.com/datasets/cdeotte/census-data-for-godaddy?select=ACSST5Y2019.S0101-Data.csv
"""
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from io import StringIO


def read_data(year):
    filename = f"ACSST5Y{year}.S0101-Data.csv"
    with open(filename, encoding="utf-8") as file:
        buffer = file.read()
    _first, rest = buffer.split("\n", maxsplit=1)
    df = pd.read_csv(StringIO(rest), low_memory=False)
    # df = df.filter(regex="^Estimate!!Total!!.*", axis=1)
    df = df[
        [
            "Geography",
            "Estimate!!Total!!Total population!!SELECTED AGE CATEGORIES!!18 years and over",
        ]
    ]
    result = dict()
    for _index, (geo, pop) in df.iterrows():
        cfips = int(geo[-5:])
        over_18 = int(pop)
        assert 1001 <= cfips <= 72153 and 10 < over_18 < 10**7
        result[(cfips, year)] = over_18
    return result

population = dict()
for year in range(2017, 2021 + 1):
    population = population | read_data(year)

torch.save(population, "population.p")
