
import numpy as np 
from scipy import sparse 
from scipy.sparse import csr_array, coo_array

import functools

# for chunking computations
from ..utils import split_computation_on_data_into_chunks

N_NEIGHBORS = 5


def compute_soft_embeddings(num_hvg, cluster_info, kernel_func,
                            embeddings_dict, threshold=None, 
                            *args, **kwargs):
    pca_data = cluster_info['pca_results']
    
    # use default kernel function if not specified 
    if kernel_func is None: 
        kernel_func = rbf_kernel_func
    
    # output for storing embeddings 
    results = np.zeros((pca_data.shape[0], num_hvg))
    
    # embeddings matrix 
    embeddings_matrix = np.array([embeddings_dict[i] for i in range(len(embeddings_dict))])
    
    knn_graph = cluster_info['centroid_knn_graph']
    centers = cluster_info['kmeans_centers']
    for icluster, cluster_indices in cluster_info['cell2cluster'].items():
        ncluster = len(cluster_indices)
        xcluster = pca_data[cluster_indices]
        
        
        # get the soft embeddings for this cluster 
        soft_embeddings = _compute_soft_embeddings(xcluster, 
                                                   centers, 
                                                   knn_graph, 
                                                   embeddings_matrix, 
                                                   kernel_func, 
                                                   *args, 
                                                   **kwargs)
        
        results[cluster_indices] = functools.reduce(lambda x, y: np.vstack((x, y)), soft_embeddings)
        
    return results
        
    
@split_computation_on_data_into_chunks
def _compute_soft_embeddings(X, centers, knn_graph, 
                             embeddings_matrix, 
                             kernel_func,
                             *args, **kwargs): 
    N = len(X)
    nclusters = embeddings_matrix.shape[0]
    
    assert centers.shape == (nclusters, X.shape[1])
    
    # get nearest neighbors
    nn_matrix = np.array([knn_graph.get_nns_by_vector(x, N_NEIGHBORS) \
                          for x in X])
    
    assert nn_matrix.shape == (N, N_NEIGHBORS)
    assert np.all(np.isfinite(nn_matrix))
    
    resps = compute_resps(X, centers, nn_matrix, kernel_func)
    
    assert resps.shape == (N, N_NEIGHBORS)
    assert np.all(np.isfinite(nn_matrix))
    
    xidx = np.repeat(np.arange(N), N_NEIGHBORS)
    resps = coo_array((resps.flatten(), (xidx, nn_matrix.flatten())), 
                       shape=(N, nclusters)).tocsr()
    
    assert resps.shape == (N, nclusters)
    assert np.all(np.isfinite(nn_matrix))
    
    out = resps @ embeddings_matrix
    return out


def compute_resps(X, centers, nn_matrix, kernel_func, 
                  *args, **kwargs):
    means = np.array([centers[nn_matrix[i]] for i in range(len(X))])
    resps = rbf_kernel_func(X, means, *args, **kwargs)
    resps /= resps.sum(axis=1)[:, np.newaxis]
    return resps


def rbf_kernel_func(X, means, gamma=0.01, return_log=False):
    log_results = -gamma * np.power(X[:, np.newaxis, :] - means, 2).sum(axis=2)
    return log_results if return_log else np.exp(log_results)
                            
