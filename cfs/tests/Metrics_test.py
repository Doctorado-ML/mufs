import unittest
from sklearn.datasets import load_iris
from mdlp import MDLP
import numpy as np
from ..Selection import Metrics


class Metrics_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mdlp = MDLP(random_state=1)
        X, self.y = load_iris(return_X_y=True)
        self.X = mdlp.fit_transform(X, self.y).astype("int64")
        self.m, self.n = self.X.shape

    # @classmethod
    # def setup(cls):

    def test_entropy(self):
        metric = Metrics()
        datasets = [
            ([0, 0, 0, 0, 1, 1, 1, 1], 2, 1.0),
            ([0, 1, 0, 2, 1, 2], 3, 1.0),
            ([0, 0, 0, 0, 0, 0, 0, 2, 2, 2], 2, 0.8812908992306927),
            ([1, 1, 1, 5, 2, 2, 3, 3, 3], 4, 0.9455305560363263),
            ([1, 1, 1, 2, 2, 3, 3, 3, 5], 4, 0.9455305560363263),
            ([1, 1, 5], 2, 0.9182958340544896),
            (self.y, 3, 0.999999999),
        ]
        for dataset, base, entropy in datasets:
            computed = metric.entropy(dataset, base)
            self.assertAlmostEqual(entropy, computed)

    def test_conditional_entropy(self):
        metric = Metrics()
        results_expected = [
            0.490953458537736,
            0.7110077966379169,
            0.15663362014829718,
            0.13032469395094992,
        ]
        for expected, col in zip(results_expected, range(self.n)):
            computed = metric.conditional_entropy(self.X[:, col], self.y, 3)
            self.assertAlmostEqual(expected, computed)
        self.assertAlmostEqual(
            0.6309297535714573,
            metric.conditional_entropy(
                [1, 3, 2, 3, 2, 1], [1, 2, 0, 1, 1, 2], 3
            ),
        )
        # https://planetcalc.com/8414/?joint=0.4%200%0A0.2%200.4&showDetails=1
        self.assertAlmostEqual(
            0.5509775004326938,
            metric.conditional_entropy([1, 1, 2, 2, 2], [0, 0, 0, 2, 2], 2),
        )

    def test_information_gain(self):
        metric = Metrics()
        results_expected = [
            0.5090465414622638,
            0.28899220336208287,
            0.8433663798517026,
            0.8696753060490499,
        ]
        for expected, col in zip(results_expected, range(self.n)):
            computed = metric.information_gain(self.X[:, col], self.y, 3)
            self.assertAlmostEqual(expected, computed)
        # https://planetcalc.com/8419/
        # ?_d=FrDfFN2COAhqh9Pb5ycqy5CeKgIOxlfSjKgyyIR.Q5L0np-g-hw6yv8M1Q8_
        results_expected = [
            0.806819679,
            0.458041805,
            1.336704086,
            1.378402748,
        ]
        for expected, col in zip(results_expected, range(self.n)):
            computed = metric.information_gain(self.X[:, col], self.y, 2)
            self.assertAlmostEqual(expected, computed)

    def test_symmetrical_uncertainty(self):
        metric = Metrics()
        results_expected = [
            0.33296547388990266,
            0.19068147573570668,
            0.810724587460511,
            0.870521418179061,
        ]
        for expected, col in zip(results_expected, range(self.n)):
            computed = metric.symmetrical_uncertainty(self.X[:, col], self.y)
            self.assertAlmostEqual(expected, computed)
