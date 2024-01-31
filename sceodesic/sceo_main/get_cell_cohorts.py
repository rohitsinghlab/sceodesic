
import sys
import functools
import pickle

import numpy as np 
import fbpca 
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture 

# package-specific imports 
from ..utils import fn_timer
from .default_keys import *
from ..helper import threshold_membership_matrix 
from ..helper import compute_responsibilities

# hard-coded default parameters
VARIANCE_INFLATION_FACTOR = 1.0


@fn_timer
def get_cell_cohorts(adata, num_cohorts, stratify_cols='none', num_hvg=None, 
                     threshold_function=threshold_membership_matrix,
                     variance_inflation_factor=VARIANCE_INFLATION_FACTOR,
                     copy=False, return_results=False, n_init=1, 
                     uns_key=None):
    
    return _get_cell_cohorts(adata, num_cohorts, stratify_cols, num_hvg, 
                             threshold_function,
                             variance_inflation_factor,
                             copy, return_results, n_init, 
                             uns_key=uns_key)


def _get_cell_cohorts(adata, num_clusters, stratify_cols, num_hvg, 
                      threshold_function,
                      variance_inflation_factor,
                      copy, return_results, n_init, 
                      clustering_filename=None,
                      uns_key=None, cluster_key=None, stratify_key=None):
    
    if uns_key is None:
        uns_key = UNS_KEY
        
    if uns_key not in adata.uns:
        adata.uns[uns_key] = {}

    if cluster_key is None:
        cluster_key = CLUSTER_KEY
        
    if stratify_key is None:
        stratify_key = STRATIFY_KEY
        
    if copy:
        adata = adata.copy()

    # should we get rid of this (ROHIT)
    assert num_clusters == 'auto' or num_clusters > 10
    
    #if 'auto', we set num_clusters = ncells/nhvg
    if num_clusters == 'auto': 
        try: 
            num_clusters = _get_auto_num_clusters(adata, num_hvg=num_hvg)
        except Exception as e:
            print("Error: if num_clusters is 'auto', you must specify num_hvg.", file=sys.stderr)
            print(f"num_hvg is set to {num_hvg} and of type {type(num_hvg)}.", file=sys.stderr)
            raise e

    # cluster - either stratify or don't
    stratify_cols = [stratify_cols] if isinstance(stratify_cols, str) else stratify_cols 
    if len(stratify_cols) > 0 and stratify_cols[0].lower() != 'none':
        arrs = [adata.obs[c].ravel().astype(str) for c in stratify_cols]
        stratify_vec = functools.reduce(lambda x, y: np.char.add(np.char.add(x, '||'), y), arrs)
        unique_strats = np.unique(stratify_vec)
        
        print(f'{unique_strats.shape} groups to stratify clustering')
        
        groups = [(adata[stratify_vec == strat,:], np.where(stratify_vec==strat)[0], strat) \
                  for strat in unique_strats]
    else:
        groups = [(adata, np.arange(adata.shape[0]), 'all')]

    # storing PCA results as well
    pca_results = {}
    
    # storing kmeans objects as well
    kmeans_models = {}
    
    # cluster weights 
    cluster_weights = np.zeros((adata.shape[0], num_clusters))
    
    kmeans_cluster_dict = {}
    curr_cluster_count = 0
    for idx, (group_adata, orig_indices, strat_desc) in enumerate(groups):    
        # Cluster in PC space.    
        U, s, Vt = fbpca.pca(group_adata.X, k=100)         

        X_dimred = U[:,:100]* s[:100]
        print(f"PCA done for stratification group {idx+1} '{strat_desc}'")
        
        # save the pca results
        pca_means = np.array(group_adata.X.mean(axis=0)).squeeze()
        pca_results[strat_desc] = {'U': U, 's': s, 'Vt': Vt, 'means': pca_means}
        
        group_num_clusters = max(1, int(float(group_adata.shape[0])/adata.shape[0] * num_clusters))
        
        kmeans = GaussianMixture(n_components=group_num_clusters, n_init=n_init, covariance_type='diag', max_iter=100)
        kmeans.fit(X_dimred)
        print(f"Fitting done k means with {group_num_clusters} clusters for stratification group {idx+1} '{strat_desc}'")

        precisions = kmeans.precisions_ / variance_inflation_factor
        kmeans_cluster_assignments = compute_responsibilities(X_dimred, kmeans.weights_,
                                                              kmeans.means_,
                                                              precisions)

        print('max discrepancy:', np.abs(kmeans_cluster_assignments - kmeans.predict_proba(X_dimred)).max())
        
        # thresholding
        kmeans_cluster_assignments = threshold_function(kmeans_cluster_assignments)
        
        # save the kmeans model
        kmeans_models[strat_desc] = (kmeans, curr_cluster_count)

        # save the cluster assignments
        cluster_weights[orig_indices, curr_cluster_count:curr_cluster_count+group_num_clusters] = kmeans_cluster_assignments
        curr_cluster_count += group_num_clusters

    # cnt_sizeLT10 = len([v for v in kmeans_cluster_dict.values() if len(v) < 10])
    # cnt_sizeLT50 = len([v for v in kmeans_cluster_dict.values() if len(v) < 50])
    # print(f'Finished clustering with {len(kmeans_cluster_dict)} clusters (originally intended {num_clusters}). Size < 10: {cnt_sizeLT10}, Size < 50: {cnt_sizeLT50}')    
    clustering_results_dict = {"cell2cluster" : cluster_weights, 
                               "cluster_pca_matrices": pca_results,
                               "kmeans_models": kmeans_models,
                               "stratify_cols": stratify_cols}
    
    
    if clustering_filename:
        with open(clustering_filename, 'wb') as f:
            pickle.dump(clustering_results_dict, f)

    # write to adata
    adata.obsm[cluster_key] = cluster_weights
    adata.uns[uns_key]['obsm_cluster_assignment_key'] = 'cell2cluster'
    adata.uns[uns_key][stratify_key] = stratify_cols
    
    out = ()
    if copy:
        out += (adata,)
    if return_results:
        out += (clustering_results_dict,)
    
    return out if len(out) > 1 else out[0] if len(out) == 1 else None


def _get_auto_num_clusters(adata, *args, **kwargs):
    print("automatically determining number of cohorts")
    num_hvg = kwargs['num_hvg']
    num_cohorts = int(adata.X.shape[0]*1.0/num_hvg)
    print("number of cohorts:", num_cohorts)
    return num_cohorts
