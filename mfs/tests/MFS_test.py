import unittest
from mdlp import MDLP
from sklearn.datasets import load_wine

from ..Selection import MFS


class MFS_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mdlp = MDLP(random_state=1)
        X, self.y = load_wine(return_X_y=True)
        self.X = mdlp.fit_transform(X, self.y).astype("int64")
        self.m, self.n = self.X.shape

    # @classmethod
    # def setup(cls):
    #     pass

    def test_initialize(self):
        mfs = MFS()
        mfs.fcbs(self.X, self.y, 0.05)
        mfs._initialize()
        self.assertIsNone(mfs.get_results())
        self.assertListEqual([], mfs.get_scores())
        self.assertDictEqual({}, mfs._su_features)
        self.assertIsNone(mfs._su_labels)

    def test_csf(self):
        mfs = MFS()
        expected = [6, 4]
        self.assertListEqual(expected, mfs.cfs(self.X, self.y).get_results())
        expected = [0.5218299405215557, 2.4168234005280964]
        self.assertListEqual(expected, mfs.get_scores())

    def test_fcbs(self):
        mfs = MFS()
        computed = mfs.fcbs(self.X, self.y, threshold=0.05).get_results()
        expected = [6, 9, 12, 0, 11, 4]
        self.assertListEqual(expected, computed)
        expected = [
            0.5218299405215557,
            0.46224298637417455,
            0.44518278979085646,
            0.38942355544213786,
            0.3790082191220976,
            0.24972405134844652,
        ]
        self.assertListEqual(expected, mfs.get_scores())
