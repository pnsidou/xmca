import xarray as xr
import xmca.xarray as xmca

try:
    import dask.array
    from dask.distributed import LocalCluster, Client
    client = Client('127.0.0.1:41205')
except ImportError as e:
    print('dask is not supported')

prcp = xr.open_dataarray('prcp.nc')
sst = xr.open_dataarray('sst.nc')

model = xmca.xMCA(sst, prcp)
model.set_field_names('sst', 'prcp')
model.solve()
model.save_analysis('std')

model = xmca.xMCA(sst, prcp)
model.set_field_names('sst', 'prcp')
model.solve()
model.rotate(10, 1)
model.save_analysis('rot')

model = xmca.xMCA(sst, prcp)
model.set_field_names('sst', 'prcp')
model.solve(complexify=True)
model.save_analysis('cplx')
