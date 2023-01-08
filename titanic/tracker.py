import neptune.new as neptune
import torch
import torch.nn.functional as F

from distributions import QuantileDistribution


def tensor_hash(tensor):
    vector = tensor.flatten()
    x = vector.sum()
    y = vector[0::2].sum() - vector[1::2].sum()
    return x, y


class Tracker:
    def __init__(self, project, api_token):
        self.run = neptune.init_run(project, api_token)
        self.last = {}
        self.outputs = {}

    def _activation_hook(self, name):
        def hook(_model, _input, output):
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

    def statistics(self, name, param, with_xy=False):
        value = param.detach().clone()
        eps = 1e-8
        previous = self.last.get(name, value)
        self.tensor(f"{name}:log", (value.abs() + eps).log())
        self.scalar(f"{name}:cosine_sim",
                    F.cosine_similarity(value.flatten(), previous.flatten(), dim=0).item())
        if with_xy:
            x, y = tensor_hash(value)
            self.scalar(f"{name}:x", x)
            self.scalar(f"{name}:y", y)
        self.last[name] = value.detach()

    def model(self, model):
        for i, (name, param) in enumerate(model.named_parameters()):
            self.statistics(f"{i}_{name}:dweight", param -
                            self.last.get(f"{i}_{name}:weight", param))
            self.statistics(f"{i}_{name}:weight", param, with_xy=True)
            self.statistics(f"{i}_{name}:gradient", param.grad)

        for i, (name, _module) in enumerate(model.named_modules()):
            self.tensor(f"{i}_{name}:output",
                        self.outputs.get(name, torch.zeros(1)))
