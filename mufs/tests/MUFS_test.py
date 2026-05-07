import unittest
import os
import pandas as pd
import numpy as np
from fimdlp.mdlp import FImdlp
from sklearn.datasets import load_wine, load_iris
from ..Selection import MUFS
from .._version import __version__


class MUFSTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mdlp = FImdlp()
        self.X_wc, self.y_w = load_wine(return_X_y=True)
        self.X_w = mdlp.fit_transform(self.X_wc, self.y_w).astype("int64")
        self.X_ic, self.y_i = load_iris(return_X_y=True)
        mdlp = FImdlp()
        self.X_i = mdlp.fit_transform(self.X_ic, self.y_i).astype("int64")

    def test_version(self):
        """Check package version."""
        mufs = MUFS()
        self.assertEqual(__version__, mufs.version())

    def assertListAlmostEqual(self, list1, list2, tol=7):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, tol)

    def test_initialize(self):
        mufs = MUFS()
        mufs.fcbf(self.X_w, self.y_w, 0.05)
        mufs._initialize(self.X_w, self.y_w)
        self.assertIsNone(mufs.get_results())
        self.assertListEqual([], mufs.get_scores())
        self.assertDictEqual({}, mufs._su_features)
        self.assertIsNone(mufs._su_labels)

    def test_cfs_wine(self):
        mufs = MUFS()
        expected = [6, 9, 12, 11, 0, 10, 5, 4, 3, 1, 2, 8]
        self.assertListEqual(
            expected, mufs.cfs(self.X_w, self.y_w).get_results()
        )
        expected = [
            0.5917174272084998,
            0.6808496670794077,
            0.7388030054834412,
            0.7717412016033023,
            0.8038053387462911,
            0.8133745319939663,
            0.8107357936370617,
            0.8130002070466252,
            0.8119571211167845,
            0.809280339063435,
            0.8057178589188132,
            0.7997607122982983,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_cfs_wine_cont(self):
        mufs = MUFS(discrete=False)
        expected = [10, 6, 11, 0, 8, 9, 5, 2, 1, 12, 7, 4]
        self.assertListEqual(
            expected, mufs.cfs(self.X_wc, self.y_w).get_results()
        )
        expected = [
            0.735264150416997,
            0.8321684551546848,
            0.8679085907226346,
            0.8822305074419031,
            0.8924066667844589,
            0.9073194790747142,
            0.9206397519327469,
            0.9336434750486353,
            0.9308809576433106,
            0.914312752334637,
            0.8937370421625325,
            0.8632185502554365,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_cfs_max_features(self):
        mufs = MUFS(max_features=3)
        expected = [6, 9, 12]
        self.assertListAlmostEqual(
            expected, mufs.cfs(self.X_w, self.y_w).get_results()
        )
        expected = [
            0.5917174272084998,
            0.6808496670794077,
            0.7388030054834412,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_cfs_iris(self):
        mufs = MUFS()
        expected = [3, 2, 0, 1]
        computed = mufs.cfs(self.X_i, self.y_i).get_results()
        self.assertListEqual(expected, computed)
        expected = [
            0.870521418179061,
            0.8903750998814699,
            0.8410471990979477,
            0.799641283053033,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_fcbf_wine(self):
        mufs = MUFS()
        computed = mufs.fcbf(self.X_w, self.y_w, threshold=0.05).get_results()
        expected = [6, 11, 9, 12, 0, 10, 1, 3, 4, 2]
        self.assertListAlmostEqual(expected, computed)
        expected = [
            0.5917174272084998,
            0.5108760230166337,
            0.493818954279874,
            0.4705420577953077,
            0.4235377639459227,
            0.3691273879905201,
            0.2832112334579764,
            0.2267108713092456,
            0.2060537321202365,
            0.15899073131298294,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_fcbf_max_features(self):
        mufs = MUFS(max_features=3)
        computed = mufs.fcbf(self.X_w, self.y_w, threshold=0.05).get_results()
        expected = [6, 11, 9]
        self.assertListAlmostEqual(expected, computed)
        expected = [
            0.5917174272084998,
            0.5108760230166337,
            0.493818954279874,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_fcbf_iris(self):
        mufs = MUFS()
        computed = mufs.fcbf(self.X_i, self.y_i, threshold=0.05).get_results()
        expected = [3, 2]
        self.assertListAlmostEqual(expected, computed)
        expected = [0.870521418179061, 0.8164005165126134]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_compute_su_labels(self):
        mufs = MUFS()
        mufs.fcbf(self.X_i, self.y_i, threshold=0.05)
        expected = [0.0, 0.0, 0.8164005165126134, 0.870521418179061]
        self.assertListAlmostEqual(
            expected, mufs._compute_su_labels().tolist()
        )
        mufs._su_labels = [1, 2, 3, 4]
        self.assertListAlmostEqual([1, 2, 3, 4], mufs._compute_su_labels())

    def test_invalid_threshold(self):
        mufs = MUFS()
        with self.assertRaises(ValueError):
            mufs.fcbf(self.X_i, self.y_i, threshold=1e-15)

    def test_fcbf_exit_threshold(self):
        mufs = MUFS()
        computed = mufs.fcbf(self.X_w, self.y_w, threshold=0.4).get_results()
        expected = [6, 11, 9, 12, 0]
        self.assertListAlmostEqual(expected, computed)
        expected = [
            0.5917174272084998,
            0.5108760230166337,
            0.493818954279874,
            0.4705420577953077,
            0.4235377639459227,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_iwss_wine(self):
        mufs = MUFS()
        expected = [6, 11, 9, 12, 0, 5, 10, 1, 3, 8, 4, 7, 2]
        self.assertListEqual(
            expected, mufs.iwss(self.X_w, self.y_w, 0.2).get_results()
        )
        expected = [
            0.5917174272084998,
            0.63815174863856,
            0.7218306836003792,
            0.7717412016033023,
            0.8038053387462911,
            0.7995075934835203,
            0.8107357936370617,
            0.8058736346293363,
            0.8043602021283025,
            0.7938297203834654,
            0.8008460535746836,
            0.7904645789027258,
            0.7900022276486858,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_iwss_wine_max_features(self):
        mufs = MUFS(max_features=3)
        expected = [6, 11, 9]
        self.assertListEqual(
            expected, mufs.iwss(self.X_w, self.y_w, 0.4).get_results()
        )
        expected = [0.5917174272084998, 0.63815174863856, 0.7218306836003792]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_iwss_exception(self):
        mufs = MUFS()
        with self.assertRaises(ValueError):
            mufs.iwss(self.X_w, self.y_w, 0.51)
        with self.assertRaises(ValueError):
            mufs.iwss(self.X_w, self.y_w, -0.01)

    def test_iwss_better_merit_condition(self):
        folder = os.path.dirname(os.path.abspath(__file__))
        data = pd.read_csv(
            os.path.join(folder, "balloons_R.dat"),
            sep="\t",
            index_col=0,
        )
        X = data.drop("clase", axis=1).to_numpy()
        y = data["clase"].to_numpy()
        mufs = MUFS()
        expected = [0, 2, 3, 1]
        self.assertListEqual(expected, mufs.iwss(X, y, 0.3).get_results())

    def test_iwss_empty(self):
        mufs = MUFS()
        X = np.delete(self.X_i, [0, 1], 1)
        self.assertListEqual(mufs.iwss(X, self.y_i, 0.3).get_results(), [1, 0])
