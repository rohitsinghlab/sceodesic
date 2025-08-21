import numpy as np 
import scipy.sparse


def split_computation_on_data_into_chunks(func):
    
    def inner(data, *args, block_size=10000, **kwargs):
        nobs = data.shape[0]
        lo = 0
        niter = (nobs // block_size) + (0 if nobs % block_size == 0 else 1)
        results = [None for _ in range(niter)]
        ct = 0
        while lo < nobs:
            hi = min(lo + block_size, nobs)
            print(f"computing results for observations {lo} through {hi-1}...", end=' ')
            results[ct] = func(data[lo:hi], *args, **kwargs)
            print("done!")
            lo = hi
            ct += 1
        return results

    return inner