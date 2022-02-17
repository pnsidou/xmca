#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Collection of tools for numpy.array modifications. '''

import warnings

import numpy as np
try:
    import dask.array as da
    dask_support = True
except ImportError:
    dask_support = False
import scipy.stats

# =============================================================================
# Tools
# =============================================================================

if dask_support:
    def dask_hilbert(x, N=None, axis=-1):
        x = da.asarray(x)
        if da.iscomplex(x).any():
            raise ValueError('x must be real.')

        if N is None:
            N = x.shape[axis]
        if N <= 0:
            raise ValueError("N must be positive.")

        Xf = da.fft.fft(x, N, axis=axis)
        h = np.zeros(N)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2

        if x.ndim > 1:
            ind = [np.newaxis] * x.ndim
            ind[axis] = slice(None)
            h = h[tuple(ind)]
        # maybe add dask support for h
        h = da.from_array(h)
        x = da.fft.ifft(Xf * h, axis=axis)
        return x


def hilbert(x, N=None, axis=-1):
    x = np.asarray(x)
    if np.iscomplex(x).any():
        raise ValueError('x must be real.')

    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = np.fft.fft(x, N, axis=axis)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = np.fft.ifft(Xf * h, axis=axis)
    return x


def arrs_are_equal(arr1, arr2):
    ''' True if arrays are the same. Also works for np.nan entries.'''
    if arr1.shape == arr2.shape:
        return ((np.isnan(arr1) & np.isnan(arr2)) | (arr1 == arr2)).all()
    else:
        return False


def is_arr(data):
    if (isinstance(data, np.ndarray)):
        return True
    else:
        raise TypeError('Data needs to be np.ndarray.')


def check_time_dims(arr1, arr2):
    if (arr1.shape[0] == arr2.shape[0]):
        pass
    else:
        raise ValueError('Both input fields need to have same time dimensions.')


def remove_mean(arr):
    '''Remove the mean of an array along the first dimension.

    If a variable (column) has at least 1 errorneous observation (row)
    the entire column will be set to NaN.

    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return arr - arr.mean(axis=0)


def get_nan_cols(arr: np.ndarray) -> np.ndarray:
    '''Get NaN columns from an array.

    Parameters
    ----------
    arr : ndarray
        Array to be scanned.

    Returns
    -------
    index : 1darray
        Index of columns with NaN entries from original data

    '''
    if isinstance(arr, np.ndarray):
        nan_index = np.isnan(arr).any(axis=0)
    elif isinstance(arr, da.Array):
        nan_index = da.isnan(arr).any(axis=0).compute()
    else:
        raise TypeError('Must be either `np.ndarray` or `dask.array.Array`')


    return nan_index


def remove_nan_cols(arr: np.ndarray) -> np.ndarray:
    '''Remove NaN columns in array.

    Parameters
    ----------
    arr : ndarray
        Array to be cleaned.

    Returns
    -------
    new_data : ndarray
        Array without NaN columns.

    '''

    idx = get_nan_cols(arr)
    new_arr  = arr[:, ~idx]
    return new_arr


def has_nan_time_steps(array):
    ''' Checks if an array has NaN time steps.

    Time is assumed to be on axis=0. The array is then reshaped to be 2D with
    time along axis 0 and variables along axis 1. A NaN time step is a row
    which contain NaN only.
    '''
    isnan = da.isnan if dask_support and isinstance(array, da.Array) else np.isnan
    arr = (isnan(array).all(axis=tuple(range(1, array.ndim))).any())
    return arr

def pearsonr(x, y):
    if x.shape[0] != y.shape[0]:
        raise ValueError('Time dimensions are different.')
    n = x.shape[0]

    corrcoef = da.corrcoef if dask_support and isinstance(x, da.Array) else np.corrcoef
    r = corrcoef(x, y, rowvar=False)
    r = r[:x.shape[1], x.shape[1]:]

    # get p-values
    dist = scipy.stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
    p = 2 * dist.cdf(-abs(r))

    return r, p


def block_bootstrap(
        arr: np.ndarray,
        axis : int = 0,
        block_size: int = 1,
        replace: bool = True) -> np.ndarray:
    '''Perform (moving-block) bootstrapping on a 2darray.

    Parameters
    ----------
    arr : np.ndarray
        Array to perform bootstrapping on. Must be 2d.
    axis : int
        Axis on which to bootstrap on. The default is 0.
    block_size : int
        Block size to keep intact when bootstrapping. By default 1.
    replace : bool
        Whether to resample with replacement (bootstrapping) or without
        (permutation). By default with replacement.

    Returns
    -------
    np.ndarray
        Resampled array.

    '''
    if axis == 0:
        pass
    elif axis == 1:
        arr = arr.T
    else:
        msg = '{:} not a valid axis. either 0 or 1.'.format(axis)
        raise ValueError(msg)

    n_obs, n_vars = arr.shape
    try:
        block_arr = arr.reshape(-1, block_size, arr.shape[1])
    except ValueError as err:
        msg = 'Length of data array ({:}) must be a multiple of block size {:}'
        msg = msg.format(n_obs, block_size)
        raise ValueError(msg) from err
    n_samples = block_arr.shape[0]
    idx_samples = np.random.choice(n_samples, size=n_samples, replace=replace)
    samples = block_arr[idx_samples]
    new_arr = samples.reshape(arr.shape)

    if axis == 1:
        new_arr = new_arr.T
    return new_arr
