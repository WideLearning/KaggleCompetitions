import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F


class QuantileDistribution:
    def __init__(self, points, k=100):
        assert points.ndim == 1
        assert points.size > 0 and k > 0
        points = points.astype(np.float32)
        self.k = k
        self.q = np.quantile(points, np.linspace(0, 1, k + 1), method="linear")

    def mean(self):
        return (self.q.sum() - (self.q[0] + self.q[-1]) / 2) / self.k

    def integrate(self, f):
        from scipy.integrate import quad

        def segment(l, r): return quad(f, l, r)[
            0] / (self.k * max(r - l, 1e-18))
        return sum(segment(l, r) for l, r in zip(self.q, self.q[1:]))

    def sample(self, size):
        segments = np.random.randint(0, self.k, size)
        uniforms = np.random.uniform(0, 1, size)
        return self.q[segments] + uniforms * (self.q[segments + 1] - self.q[segments])


class Tracker:
    def __init__(self, project, api_token):
        self.run = neptune.init_run(project, api_token)
        self.last = dict()
        self.outputs = dict()

    def _activation_hook(self, name):
        def hook(model, input, output):
            self.outputs[name] = output.detach()
        return hook

    def set_hooks(self, model):
        for name, module in model.named_modules():
            module.register_forward_hook(self._activation_hook(name))

    def scalar(self, name, value):
        self.run[name].log(value)

    def tensor(self, name, value, k=5):
        qd = QuantileDistribution(value.detach().cpu().numpy().ravel(), k)
        for i in range(k + 1):
            self.scalar(f"{name}%{i/k:.3f}", qd.q[i])

    def statistics(self, name, param, with_cosine=False):
        value = param.detach().clone()
        eps = 1e-8
        previous = self.last.get(name, value)
        self.tensor(f"{name}:log", (value.abs() + eps).log())
        if with_cosine:
            self.tensor(f"{name}:cosine_sim",
                        F.cosine_similarity(value, previous, dim=0))
        self.last[name] = value.detach()

    def model(self, model):
        for i, (name, param) in enumerate(model.named_parameters()):
            self.statistics(f"{i}_{name}:dweight", param -
                            self.last.get(f"{i}_{name}:weight", param))
            self.statistics(f"{i}_{name}:weight", param)
            self.statistics(f"{i}_{name}:gradient",
                            param.grad, with_cosine=True)

        for i, (name, module) in enumerate(model.named_modules()):
            self.tensor(f"{i}_{name}:output",
                        self.outputs.get(name, torch.zeros(1)))
