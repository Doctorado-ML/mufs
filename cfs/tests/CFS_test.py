import unittest

from ..Selection import CFS


class CFS_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # @classmethod
    # def setup(cls):
    #     pass

    def test_initial(self):
        cfs = CFS(a=1)
        self.assertEqual(cfs.a, 1)
