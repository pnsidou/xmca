import pytest
import os
from pathlib import Path
import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

from xmca.xarray import xMCA

try:
    from dask.distributed import Client

    client = Client(scheduler_file=os.environ.get("DASK_SCHEDULER"))

    # relative tolerance = 0.1% for assert_allclose
    rtol = 0.0001

    # quantile -> 5% of outliers will be discarded for
    # purposes of testing with assert_allclose
    q = 0.05

    # extend period for complex solver
    period = 10

    # number of modes to compare
    modes = 200

    def remove_outliers_assert(numpy_da, dask_da, q=q, rtol=rtol):
        rel_error: xr.DataArray = (numpy_da - dask_da) / numpy_da
        cond = (rel_error >= rel_error.quantile(q / 2)) & (
            rel_error <= rel_error.quantile(1 - q / 2)
        )
        numpy_da, dask_da = numpy_da.where(cond), dask_da.where(cond)
        assert_allclose(numpy_da, dask_da, rtol=rtol)

    def correct_phase(numpy_da, dask_da):
        phase_diff = np.angle(numpy_da) - np.angle(dask_da)
        dask_da = dask_da * np.exp(phase_diff * 1j)
        return numpy_da, dask_da

    @pytest.fixture
    def path() -> Path:
        return Path(__file__).parent / "fixtures"

    @pytest.fixture
    def ds(path: Path) -> xr.Dataset:
        return xr.open_mfdataset(path.glob("*.nc")).load()

    @pytest.fixture
    def ds_dask(path: Path) -> xr.Dataset:
        chunks = {"lon": 5}
        return xr.open_mfdataset(path.glob("*.nc"), parallel=True, chunks=chunks)

    @pytest.fixture(params=["left", "right"])
    def side(request):
        return request.param

    @pytest.fixture()
    def mca(ds, ds_dask):
        mca = xMCA(ds.sst, ds.prcp)
        mca.solve()
        mca_dask = xMCA(ds_dask.sst, ds_dask.prcp)
        mca_dask.solve()
        return {"numpy": mca, "dask": mca_dask}

    @pytest.fixture(params=["exp", "theta", False])
    def mca_complex(request, ds, ds_dask):
        mca = xMCA(ds.sst, ds.prcp)
        mca.solve(complexify=True, extend=request.param, period=period)
        mca_dask = xMCA(ds_dask.sst, ds_dask.prcp)
        mca_dask.solve(complexify=True, extend=request.param, period=period)
        return {"numpy": mca, "dask": mca_dask}

    @pytest.mark.dask
    def test_sv_standard(mca: dict):
        """Test singular values"""
        sv = mca["numpy"].singular_values(modes)
        sv_dask = mca["dask"].singular_values(modes)
        assert_allclose(sv_dask, sv, rtol=rtol)

    @pytest.mark.dask
    def test_eof_standard(mca: dict, side: str):
        eof = mca["numpy"].eofs(modes)[side]
        eof_dask = mca["dask"].eofs(modes)[side]
        eof, eof_dask = np.abs(eof), np.abs(eof_dask)
        remove_outliers_assert(eof, eof_dask)

    @pytest.mark.dask
    def test_pcs_standard(mca: dict, side: str):
        pcs = mca["numpy"].pcs(modes)[side]
        pcs_dask = mca["dask"].pcs(modes)[side]
        pcs, pcs_dask = np.abs(pcs), np.abs(pcs_dask)
        remove_outliers_assert(pcs, pcs_dask)

    @pytest.mark.dask
    def test_sv_complex(mca_complex: dict):
        """Test singular values"""
        sv = mca_complex["numpy"].singular_values(modes)
        sv_dask = mca_complex["dask"].singular_values(modes)
        remove_outliers_assert(sv, sv_dask)

    @pytest.mark.dask
    def test_eof_complex(mca_complex: dict, side: str):
        eof = mca_complex["numpy"].eofs(modes)[side]
        eof_dask = mca_complex["dask"].eofs(modes)[side]
        eof, eof_dask = correct_phase(eof, eof_dask)
        remove_outliers_assert(eof, eof_dask)

    @pytest.mark.dask
    def test_pcs_complex(mca_complex: dict, side: str):
        pcs = mca_complex["numpy"].pcs(modes)[side]
        pcs_dask = mca_complex["dask"].pcs(modes)[side]
        pcs, pcs_dask = correct_phase(pcs, pcs_dask)
        remove_outliers_assert(pcs, pcs_dask)

except ImportError as e:
    print("Dask not supported")
