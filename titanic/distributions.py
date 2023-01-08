import numpy as np
from scipy.integrate import quad


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
        def segment(lef, rig):
            return quad(f, lef, rig)[0] / (self.k * max(rig - lef, 1e-18))
        return sum(segment(lef, rig) for lef, rig in zip(self.q, self.q[1:]))

    def sample(self, size):
        segments = np.random.randint(0, self.k, size)
        uniforms = np.random.uniform(0, 1, size)
        return self.q[segments] + uniforms * (self.q[segments + 1] - self.q[segments])
