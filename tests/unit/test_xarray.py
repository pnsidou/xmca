import unittest

import numpy as np
import xarray as xr

from xmca.xarray import xMCA
from parameterized import parameterized

try:
    import dask.array
    from dask.distributed import Client
    Client()
    dask_support = True
except ImportError:
    dask_support = False


class TestXarray(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        np.random.seed(777)
        A = np.random.rand(500, 20, 15)
        np.random.seed(888)
        B = np.random.rand(500, 15, 10)
        np.random.seed(999)
        C = np.random.rand(500, 10, 5)
        self.A = xr.DataArray(A, dims=['time', 'lat', 'lon'])
        self.B = xr.DataArray(B, dims=['time', 'lat', 'lon'])
        self.C = xr.DataArray(C, dims=['time', 'lat', 'lon'])
        n_var_A = np.product(self.A.shape[1:])
        n_var_B = np.product(self.B.shape[1:])
        self.rank = np.min([n_var_A, n_var_B])


    def name_func(testcase_func, param_num, param):
        return "{:}_{:s}".format(
            testcase_func.__name__,
            parameterized.to_safe_name('_'.join([param.args[0]])),
        )

    def test_input(self):
        #xMCA()
        xMCA(self.A)
        xMCA(self.A, self.B)
        with self.assertRaises(ValueError):
            xMCA(self.A, self.B, self.A)

        with self.assertRaises(TypeError):
            xMCA(np.array([1, 2, 3]))

        if dask_support:

            temp = xr.tutorial.open_dataset(
                'air_temperature',
                chunks={'lat': 25, 'lon': 25, 'time': -1}
            )
            temp = temp.air
            xMCA(temp)
            xMCA(temp, temp)

    @classmethod
    def tearDownClass(self):
        pass
