
import numpy as np 
from scipy import sparse 

import functools

# for chunking computations
from ..utils import split_computation_on_data_into_chunks


def compute_soft_embeddings(expr_data, num_hvg, 
                            cluster_info, kernel_func,
                            embeddings_dict, threshold=None, 
                            *args, **kwargs):
    
    # use default kernel function if not specified 
    if kernel_func is None: 
        kernel_func = rbf_kernel_func
    
    # numpy array for storing embeddings (not sparse) 
    results = np.zeros((expr_data.shape[0], num_hvg))
    
    # compute embeddings by stratification group
    for group_desc, group_indices in cluster_info['groups'].items():
        # parse cluster info - means and weights 
        means, cluster_offset = cluster_info['kmeans_centers'][group_desc]
        nclusters = means.shape[0]
        weights = np.zeros(nclusters)
        for i in range(nclusters):
            icluster = i + cluster_offset
            weights[i] = len(cluster_info['cell2cluster'][icluster])
        weights /= weights.sum()
        
        # prepare embeddings matrix 
        embeddings_matrix = np.zeros((nclusters, num_hvg))
        for i in range(nclusters):
            icluster = i + cluster_offset
            embeddings_matrix[i] = embeddings_dict[i + cluster_offset]
        
        # get pca info
        pca_means = cluster_info['cluster_pca_matrices'][group_desc]['means']
        pca_v = cluster_info['cluster_pca_matrices'][group_desc]['Vt']
        
        # compute a threshold 
        nobs = len(group_indices)
        sample_size = min(10000, nobs)
        group_sample = np.random.choice(group_indices, replace=False, size=sample_size)
        A = compute_resps(expr_data[group_sample], 
                          pca_means,
                          pca_v, 
                          means, 
                          weights, 
                          kernel_func, 
                          *args, **kwargs)
        threshold = determine_membership_matrix_threshold(A)
        
        # compute and assign embeddings
        results[group_indices] = functools.reduce(lambda x, y: np.vstack((x, y)),
                                                  _compute_soft_embeddings(expr_data, 
                                                                           pca_means, 
                                                                           pca_v, 
                                                                           means, 
                                                                           weights, 
                                                                           kernel_func, 
                                                                           embeddings_matrix, 
                                                                           threshold, 
                                                                           *args, 
                                                                           **kwargs))
        
    return results
        
    
@split_computation_on_data_into_chunks
def _compute_soft_embeddings(expr_data, pca_means, pca_v,
                             means, weights, kernel_func,
                             embeddings_matrix, 
                             threshold, 
                             *args, **kwargs): 
    resps = compute_resps(expr_data, pca_means, pca_v, 
                          means, weights, kernel_func, 
                          *args, **kwargs)
    resps[resps < threshold] = 0
    resps /= resps.sum(axis=1)[:, np.newaxis]
    return resps @ embeddings_matrix


def compute_resps(expr_data, pca_means, pca_v, means, 
                  weights, kernel_func, *args, **kwargs):
    pca_rep = np.array(expr_data - pca_means) @ pca_v.T
    resps = kernel_func(pca_rep, means, *args, **kwargs)
    resps *= weights
    return resps / resps.sum(axis=1)[:, np.newaxis]


def rbf_kernel_func(X, means, gamma=0.01, return_log=False):
    log_results = -gamma * np.power(X[:, np.newaxis, :] - means[np.newaxis, :], 2).sum(axis=2)
    return log_results if return_log else np.exp(log_results)
                            

def determine_membership_matrix_threshold(A, quantiles=[0.99, 0.98, 0.97, 0.96, 0.95], index=10):
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
    
    
def threshold_membership_matrix(A, quantiles=[0.99, 0.98, 0.97, 0.96, 0.95], index=10, t=None):
    if t is None:
        # ensure valid index number (not larger than number of cohorts)
        index = min(A.shape[1], index)
        t = determine_membership_matrix_threshold(A, quantiles, index)
    A[A < t] = 0
    sums = A.sum(axis=1)
    if len(sums.shape) == 1:
        sums = sums[:, np.newaxis]
    return A / sums