import unittest
import os
import pandas as pd
import numpy as np
from mdlp import MDLP
from sklearn.datasets import load_wine, load_iris
from ..Selection import MUFS
from .._version import __version__


class MUFSTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mdlp = MDLP(random_state=1)
        self.X_wc, self.y_w = load_wine(return_X_y=True)
        self.X_w = mdlp.fit_transform(self.X_wc, self.y_w).astype("int64")
        self.X_ic, self.y_i = load_iris(return_X_y=True)
        mdlp = MDLP(random_state=1)
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

    def test_csf_wine(self):
        mufs = MUFS()
        expected = [6, 12, 9, 4, 10, 0]
        self.assertListEqual(
            expected, mufs.cfs(self.X_w, self.y_w).get_results()
        )
        expected = [
            0.5218299405215557,
            0.602513857132804,
            0.4877384978817362,
            0.3743688234383051,
            0.28795671854246285,
            0.2309165735173175,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_csf_wine_cont(self):
        mufs = MUFS(discrete=False)
        expected = [10, 6, 0, 2, 11, 9]
        self.assertListEqual(
            expected, mufs.cfs(self.X_wc, self.y_w).get_results()
        )
        expected = [
            0.735264150416997,
            0.8321684551546848,
            0.7439915858469107,
            0.6238883340158233,
            0.513637402071709,
            0.41596400981378984,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_csf_max_features(self):
        mufs = MUFS(max_features=3)
        expected = [6, 12, 9]
        self.assertListAlmostEqual(
            expected, mufs.cfs(self.X_w, self.y_w).get_results()
        )
        expected = [
            0.5218299405215557,
            0.602513857132804,
            0.4877384978817362,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_csf_iris(self):
        mufs = MUFS()
        expected = [3, 2, 0, 1]
        computed = mufs.cfs(self.X_i, self.y_i).get_results()
        self.assertListEqual(expected, computed)
        expected = [
            0.870521418179061,
            0.8968651482682227,
            0.5908278453318913,
            0.40371971570693366,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_fcbf_wine(self):
        mufs = MUFS()
        computed = mufs.fcbf(self.X_w, self.y_w, threshold=0.05).get_results()
        expected = [6, 9, 12, 0, 11, 4]
        self.assertListAlmostEqual(expected, computed)
        expected = [
            0.5218299405215557,
            0.46224298637417455,
            0.44518278979085646,
            0.38942355544213786,
            0.3790082191220976,
            0.24972405134844652,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_fcbf_max_features(self):
        mufs = MUFS(max_features=3)
        computed = mufs.fcbf(self.X_w, self.y_w, threshold=0.05).get_results()
        expected = [6, 9, 12]
        self.assertListAlmostEqual(expected, computed)
        expected = [
            0.5218299405215557,
            0.46224298637417455,
            0.44518278979085646,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_fcbf_iris(self):
        mufs = MUFS()
        computed = mufs.fcbf(self.X_i, self.y_i, threshold=0.05).get_results()
        expected = [3, 2]
        self.assertListAlmostEqual(expected, computed)
        expected = [0.870521418179061, 0.810724587460511]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_compute_su_labels(self):
        mufs = MUFS()
        mufs.fcbf(self.X_i, self.y_i, threshold=0.05)
        expected = [0.0, 0.0, 0.810724587460511, 0.870521418179061]
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
        expected = [6, 9, 12]
        self.assertListAlmostEqual(expected, computed)
        expected = [
            0.5218299405215557,
            0.46224298637417455,
            0.44518278979085646,
        ]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_iwss_wine(self):
        mufs = MUFS()
        expected = [6, 9, 12]
        self.assertListEqual(
            expected, mufs.iwss(self.X_w, self.y_w, 0.2).get_results()
        )
        expected = [0.5218299405215557, 0.5947822876110085, 0.4877384978817362]
        self.assertListAlmostEqual(expected, mufs.get_scores())

    def test_iwss_wine_max_features(self):
        mufs = MUFS(max_features=3)
        expected = [6, 9, 12]
        self.assertListEqual(
            expected, mufs.iwss(self.X_w, self.y_w, 0.4).get_results()
        )
        expected = [0.5218299405215557, 0.5947822876110085, 0.4877384978817362]
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
