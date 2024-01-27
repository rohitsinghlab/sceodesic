
import numpy as np 


def determine_membership_matrix_threshold(A, quantiles, index):
    """
    Determines a "reasonable" threshold for the membership matrix. 
    
    We developed the following heuristic: for each cell, sort the 
    membership probabilities in ascending order, then look at the 
    -(index)th highest probability for each cell. The q-th quantile
    of this vector of probabilities is our threshold.
    """
    assert not index < 0 
    
    # too many damn zeroes!!! :(
    if index == 0: 
        return 0.0
    
    Asrt = np.sort(A, axis=1)
    thresholds = list(map(lambda q: np.quantile(Asrt[:, -index], q), quantiles))
    t = max(thresholds)
    
    if not t > 0: 
        return determine_membership_matrix_threshold(A, quantiles, index-1)
    
    return t
    
    
def threshold_membership_matrix(A, quantiles=[0.99, 0.98, 0.97, 0.96, 0.95], index=10):
    # ensure valid index number (not larger than number of cohorts)
    index = min(A.shape[1], index)
    t = determine_membership_matrix_threshold(A, quantiles, index)
    A[A < t] = 0
    return A / A.sum(axis=1)[:, np.newaxis]
