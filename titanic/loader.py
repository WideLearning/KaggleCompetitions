import re
from functools import lru_cache

import neptune.new as neptune
import numpy as np
import torch


class LoaderBase:
    def __init__(self):
        self.series = []

    @lru_cache(10**6)
    def get(self, name):
        return self._get(name)

    @lru_cache(10**6)
    def regex(self, regex):
        for name in self.series:
            if not re.fullmatch(regex, name):
                continue
            return self.get(name)

    @lru_cache(10**6)
    def distribution(self, regex, epoch):
        result = []
        for name in self.series:
            if not re.fullmatch(regex, name):
                continue
            value = self.get(name)
            if epoch not in range(len(value)):
                continue
            result.append(value[epoch])
        return np.array(result)

    @lru_cache(10**6)
    def layer_distributions(self, regex, epoch):
        result = []
        layer = 0
        while True:
            d = self.distribution(f"{layer}_.*:{regex}.*", epoch)
            if len(d):
                result.append(d)
                layer += 1
            else:
                return result

    def _get(self, _name):
        pass


class NeptuneLoader:
    def __init__(self, project, api_token, run):
        super().__init__()
        self.data = neptune.init_run(
            project,
            api_token,
            run, mode="read-only"
        ).get_structure()

        self.series = sorted([name for name, series in self.data.items()
                              if not isinstance(series, dict)])

    def _get(self, name):
        path = name.split("/")
        node = self.data
        for elem in path:
            node = node[elem]
        return node.fetch_values()["value"].to_numpy()


class FileLoader(LoaderBase):
    def __init__(self, filename):
        self.data = torch.load(filename)
        assert isinstance(self.data, dict)
        self.series = self.data.keys()
    
    def _get(self, name):
        return self.data.get(name, np.array([]))