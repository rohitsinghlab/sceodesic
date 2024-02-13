
import sys
import functools
import pickle

import numpy as np 
import fbpca 
from sklearn.cluster import KMeans

from scipy.sparse import csr_matrix
import scipy.sparse

# package-specific imports 
from ..utils import fn_timer
from .default_keys import *

@fn_timer
def get_cell_cohorts(adata, num_cohorts, stratify_cols='none', num_hvg=None, 
                     soft=False, soft_kernel_func=None,
                     copy=False, return_results=False, n_init=1, 
                     uns_key=None):

    if soft and soft_kernel_func is None:
        gamma = 0.25
        soft_kernel_func = lambda x: np.exp(-gamma * np.power(x, 2).sum(axis=1))
    
    return _get_cell_cohorts(adata, num_cohorts, stratify_cols, num_hvg, 
                             soft, soft_kernel_func,
                             copy, return_results, n_init, 
                             uns_key=uns_key)


def _get_cell_cohorts(adata, num_clusters, stratify_cols, num_hvg, 
                      soft, soft_kernel_func,
                      copy, return_results, n_init, 
                      clustering_filename=None,
                      uns_key=None, cluster_key=None, stratify_key=None):
    
    if uns_key is None:
        uns_key = UNS_KEY
        
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
    
    # store responsibilities if soft
    if soft:
        resps = []
        temp = []

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
        
        kmeans = KMeans(n_clusters=group_num_clusters, n_init=n_init)
        kmeans.fit(X_dimred)
        print(f"Fitting done k means with {group_num_clusters} clusters for stratification group {idx+1} '{strat_desc}'")
        kmeans_cluster_assignments = kmeans.labels_

        # compute cluster responsibilities
        if soft:
            # compute cluster probabilities
            cluster_probs = [(kmeans_cluster_assignments == i).sum() for i in range(group_num_clusters)]
            cluster_probs = np.array(cluster_probs, dtype='float64')
            cluster_probs /= cluster_probs.sum()

            # compute conditional probs (as per kernel)
            centers = kmeans.cluster_centers_
            group_resps = np.apply_along_axis(lambda x: soft_kernel_func(centers-x), 1, X_dimred)

            # multiply by cluster weights 
            group_resps *= cluster_probs

            group_resps /= group_resps.sum(axis=1)[:, np.newaxis]

            temp.append(group_resps)

            # get threshold and zero out small probabilities 
            t = np.median(np.sort(group_resps, axis=1)[-min(10, group_resps.shape[1])])
            group_resps = np.where(group_resps > t, group_resps, 0)
            group_resps /= group_resps.sum(axis=1)[:, np.newaxis]



            resps.append(csr_matrix(group_resps))
        
        # save the kmeans model
        kmeans_models[strat_desc] = (kmeans, curr_cluster_count)

        for i in range(group_num_clusters):
            # save keys as strings so we can save to .h5ad
            kmeans_cluster_dict[str(curr_cluster_count)] = orig_indices[np.where(kmeans_cluster_assignments == i)[0]].tolist()
            curr_cluster_count += 1

    cnt_sizeLT10 = len([v for v in kmeans_cluster_dict.values() if len(v) < 10])
    cnt_sizeLT50 = len([v for v in kmeans_cluster_dict.values() if len(v) < 50])
    print(f'Finished clustering with {len(kmeans_cluster_dict)} clusters (originally intended {num_clusters}). Size < 10: {cnt_sizeLT10}, Size < 50: {cnt_sizeLT50}')    
    clustering_results_dict = {"cell2cluster" : kmeans_cluster_dict, 
                               "cluster_pca_matrices": pca_results,
                               "kmeans_models": kmeans_models,
                               "kmeans_centers": {k: (v[0].cluster_centers_, v[1]) \
                                                  for k, v in kmeans_models.items()},
                               "stratify_cols": stratify_cols}
    
    
    if clustering_filename:
        with open(clustering_filename, 'wb') as f:
            pickle.dump(clustering_results_dict, f)

    # save pca to obsm if soft - will need later 
    if soft:
        resps = functools.reduce(lambda x, y: scipy.sparse.hstack((x, y)), resps)
        adata.uns[uns_key]['sceo_resps'] = resps
        temp = functools.reduce(lambda x, y: scipy.sparse.hstack((x, y)), temp)
        adata.uns[uns_key]['temp'] = temp

    # write to adata.uns 
    adata.uns[uns_key][cluster_key] = kmeans_cluster_dict
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
