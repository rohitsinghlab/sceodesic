import numpy as np 
import scipy.sparse

from functools import reduce


def split_computation_on_data_into_chunks(data, func, *args, nblock=30000, **kwargs):
    nobs = data.shape[0]
    lo = 0
    niter = (nobs // nblock) + (0 if nobs % nblock == 0 else 1)
    results = [None for _ in range(niter)]
    ct = 0
    while lo < nobs:
        hi = min(lo + nblock, nobs)
        print(f"computing results for observations {lo} through {hi-1}...", end=' ')
        results[ct] = func(data[lo:hi], *args, **kwargs)
        print("done!")
        lo = hi
        ct += 1
    return reduce(lambda x, y: np.vstack((x, y)), results)
