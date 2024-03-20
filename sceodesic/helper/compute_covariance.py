
import numpy as np
from .condition_volume_normalization import _normalize_condition_volume

def compute_covariance_and_ncomps_pct_variance(data, max_condition_number, pvd_pct):
    """ Computes a symmetric positive definite sample covariance matrix.
    - `data` is a cell x gene 2D numpy array.
    """
    # Compute raw covariance.
    matrix = np.cov(data, rowvar=False)

    S,U = np.linalg.eigh(matrix)

    non_zero_count = np.sum(S != 0)
    var_explained_count = np.argmax(np.cumsum(S[::-1]) / np.sum(S) > pvd_pct) + 1
    ncomps_pct_variance = min(non_zero_count, var_explained_count)
    
    # normalize by condition-volume 
    matrix = _normalize_condition_volume(S, U, max_condition_number, log=False)
    
    return matrix, ncomps_pct_variance
