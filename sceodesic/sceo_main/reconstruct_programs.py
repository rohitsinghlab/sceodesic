
import numpy as np
import scipy

import sys 
import pickle

from sklearn.decomposition import PCA
from sklearn.decomposition import MiniBatchSparsePCA

# package-specific modules 
from ..utils import fn_timer

from .default_keys import *

@fn_timer
def reconstruct_programs(adata, sparse_pca_lambda,
                         copy=False, return_results=False, 
                         results_coexp=None, 
                         uns_key=None):
    
    if uns_key is None:
        uns_key = UNS_KEY 
        
    # get results_coexp from adata.uns if not specified 
    if results_coexp is None:
        try:
            results_coexp = adata.uns[uns_key]
        except:
            message = ("Error: must either specify cluster-specific covariance matrices or "
                       "have run sceodesic.estimate_covariances beforehand.")
            print(message, file=sys.stderr)
            
            raise e
    
    return _reconstruct_programs(adata, sparse_pca_lambda,
                                 copy=copy, 
                                 return_results=return_results, 
                                 results_coexp=results_coexp, 
                                 uns_key=uns_key)
    
    
def _reconstruct_programs(adata, sparse_pca_lambda, embedding_filename=None, 
                          copy=False, return_results=False, results_coexp=None, 
                          uns_key=None, embeddings_dict_key=None, 
                          modules_key=None):
    
    if uns_key is None:
        uns_key = UNS_KEY

    if uns_key not in adata.uns:
        adata.uns[uns_key] = {}
    
    if embeddings_dict_key is None:
        embeddings_dict_key = EMBEDDINGS_DICT_KEY 
    
    if modules_key is None:
        modules_key = MODULES_KEY
    
    if copy:
        adata = adata.copy()
    
    if results_coexp is None:
        with open(kwargs.get('coexpression_filename'), 'rb') as f:
            results_coexp = pickle.load(f)
            
    covariance_matrices = results_coexp["cluster_covariances"]
    cluster_var_count = results_coexp["cluster_var_count"]

    cluster_eigendecomposition = {}
    for cluster_index in covariance_matrices:
        current_covariance = covariance_matrices[cluster_index]
        S,U = np.linalg.eigh(current_covariance)
        cluster_eigendecomposition[cluster_index] = [S,U]

    #### old code ####
    #### bug: taking least varied components, not most varied ####
    ### all_eigenvectors_horizontal = np.vstack([cluster_eigendecomposition[cluster_index][1][:, :cluster_var_count[cluster_index]].T for cluster_index in cluster_eigendecomposition])
    #### old code ####

    # new code: take most-varied components
    all_eigenvectors_horizontal = np.vstack([cluster_eigendecomposition[cluster_index][1][:, -cluster_var_count[cluster_index]:].T for cluster_index in cluster_eigendecomposition])
    #all_eigenvectors_horizontal = []
    #for cluster_index in cluster_eigendecomposition:
        #_, U = cluster_eigendecomposition[cluster_index]
        #ncomps = cluster_var_count[cluster_index]
        ## columns are eigenvectors, sorted in ascending order of eigenvalues
        #all_eigenvectors_horizontal.append(U[:, -ncomps:].T)
    ## transpose so that eigenvectors are horizontal
    #all_eigenvectors_horizontal = np.vstack(all_eigenvectors_horizontal)

    print("Concatenated eigenvectors matrix: ", all_eigenvectors_horizontal.shape)
    print(np.allclose(all_eigenvectors_horizontal, all_eigenvectors_horizontal.real))

    # set to regular pca if sparse pca is zero
    sparse_pca = None 
    if sparse_pca_lambda > 0.0: 
        sparse_pca = MiniBatchSparsePCA(alpha=sparse_pca_lambda)
    else:
        sparse_pca = PCA()

    # all imaginary parts are already zero per check (redundant)
    sparse_pca.fit(all_eigenvectors_horizontal.real) 
    sparse_pca_eigenvectors = sparse_pca.components_
    print("sparse_pca_eigenvectors dim: ", sparse_pca_eigenvectors.shape)
    print("Done training")
    embeddings = {}
    for cluster_index in cluster_eigendecomposition:
        sigma_i = covariance_matrices[cluster_index]
        M_star = sparse_pca_eigenvectors.T @ scipy.linalg.logm(sigma_i) @ sparse_pca_eigenvectors
        diagonal = np.diagonal(M_star)
        embedding = diagonal 
        embeddings[cluster_index] = embedding

    results_embedding = {"embedding_dictionary":embeddings, "cluster_svd": cluster_eigendecomposition, "modules": sparse_pca_eigenvectors}

    if embedding_filename:
        with open(embedding_filename, 'wb') as f:
            pickle.dump(results_embedding, f)
            
    # write to adata.uns
    adata.uns[uns_key][embeddings_dict_key] = results_embedding["embedding_dictionary"]
    adata.uns[uns_key][modules_key] = results_embedding["modules"]
        
    out = ()
    if copy:
        out += (adata,)
    if return_results:
        out += (results_embedding,)
        
    return out if len(out) > 1 else out[0] if len(out) == 1 else None
