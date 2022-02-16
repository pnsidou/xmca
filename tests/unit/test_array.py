import unittest
import numpy as np

from xmca.array import MCA


class TestArray(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        self.A = np.random.rand(500, 20, 15)
        np.random.seed(8)
        self.B = np.random.rand(500, 15, 10)
        np.random.seed(9)
        self.C = np.random.rand(500, 10, 5)

        prodA = np.product(self.A.shape[1:])
        prodB = np.product(self.B.shape[1:])
        self.rank = np.min([prodA, prodB])

    def test_mca_input(self):
        MCA()
        MCA(self.A)
        MCA(self.A, self.B)
        with self.assertRaises(ValueError):
            MCA(self.A, self.B, self.A)

        with self.assertRaises(ValueError):
            MCA(self.A[:-1], self.B)

        with self.assertRaises(TypeError):
            MCA([1, 2, 3])

        A_with_nan = self.A.copy()
        A_with_nan[1, :] = np.nan
        with self.assertRaises(ValueError):
            MCA(A_with_nan, self.B)

    def test_pcs_shape(self):
        mca = MCA(self.A, self.B)
        mca.solve()
        pcs = mca.pcs()
        self.assertEqual((self.A.shape[0], self.rank), pcs['left'].shape)
        self.assertEqual((self.B.shape[0], self.rank), pcs['right'].shape)

    def test_eofs_shape(self):
        mca = MCA(self.A, self.B)
        mca.solve()
        eofs = mca.eofs()
        self.assertEqual(self.A.shape[1:] + (self.rank,), eofs['left'].shape)
        self.assertEqual(self.B.shape[1:] + (self.rank,), eofs['right'].shape)


    def tearDown(self):
        pass
