
import numpy as np


def _normalize_condition_volume(S, U, max_condition_number, log):
    """
    S, U is the eigendecomposition of the positive definite matrix which you want to normalize. 
    """
    # Floor on the minimum eigenvalue.
    floor_s = np.max(S)/max_condition_number
    
    # For the diagonal matrix with eigenvalues, replace eigenvalues below the floor
    S_floored = np.where(S < floor_s, floor_s, S)
    
    # Retrieve the smallest eigenvalue. Positivity condition redundant.
    min_ev = np.min(S_floored[S_floored > 0])
    
    # Scale all the eigenvalues so that the minimum eigenvalue is 1.
    S_scaled = S_floored / min_ev
    
    # Reconstruct the matrix, now SPD. Return log if desired.
    if log:
        SPD_matrix = U @ np.diag(np.log(S_scaled)) @ U.T
    else:
        SPD_matrix = U @ np.diag(S_scaled) @ U.T
    
    return SPD_matrix.real


def normalize_condition_volume(cov_matrix, max_condition_number, log=False):
    """
    Scales the covariance matrix so that its condition number (ratio of min to max eigenvalue) 
    is no greater than max_condition_number. Returns the log covariance matrix if desired. 
    """
    # Eigendecomposition of covariance matrix.
    S,U = np.linalg.eigh(cov_matrix)
    
    return _normalize_condition_volume(S, U, max_condition_number, log)