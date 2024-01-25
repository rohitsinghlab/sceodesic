
import sys
import pickle

import scanpy as sc 
import numpy as np 

# package-specific modules
from ..utils import fn_timer
from .default_keys import *


@fn_timer
def get_locally_variable_genes(adata, num_hvg, num_hvg_per_cluster=100, global_hvg=False,
                               copy=False, return_results=False, cohort_weights=None, uns_key=None):
    
    if uns_key is None:
        uns_key = UNS_KEY

    if uns_key not in adata.uns:
        adata.uns[uns_key] = {}
    
    # removing ability to specify key
    cluster_key = CLUSTER_KEY
    
    # can either pass in soft cell cohort 'assignment' (array cohort_weights with cell[i] having cluster weight cohort_weights[i])
    # or the cluster_key 
    clustering_results = None
    if cohort_weights is None:
        try:
            clustering_results = adata.obsm[adata.uns[uns_key]['obsm_cluster_assignment_key']]
        except Exception as e:
            message = ("Error: must either specify a cell cohort assignment or "
                       "have run sceodesic.get_cell_cohorts beforehand.")
            print(message, file=sys.stderr)
            
            raise e
    else:
        clustering_metadata = {'obsm_cluster_assignment_key': 'cell2cluster', 'stratify_cols': '***NOT SPECIFIED***'}
        adata.uns[uns_key].update(clustering_metadata)
        adata.obsm['cell2cluster'] = cohort_weights
        clustering_results = cohort_weights
        
    return _get_locally_variable_genes(adata, num_hvg, num_hvg_per_cluster, global_hvg, 
                                       copy=copy, return_results=return_results, 
                                       clustering_results=clustering_results, 
                                       uns_key=uns_key)
        
    
def _get_locally_variable_genes(adata, num_hvg, num_hvg_per_cluster, global_hvg, hvg_filename=None, 
                copy=False, return_results=False, clustering_results=None, 
                uns_key=None, hvg_key=None):
    
    if uns_key is None:
        uns_key = UNS_KEY
    
    if hvg_key is None: 
        hvg_key = HVG_KEY
        
    if copy:
        adata = adata.copy()
        
    # change later 
    results_clustering = clustering_results
    
    # Store the cluster data matrices.
    c2c = results_clustering
    c2c = np.argmax(c2c, axis=1).tolist()
    cell2cluster = {}
    for i, c in enumerate(c2c):
        cell2cluster[c] = cell2cluster.get(c, []) + [i]

    full_data_clusters = []
    for key in cell2cluster.keys():
        cluster_indices = cell2cluster[key]
        ## full_data_clusters.append(adata[cluster_indices,:])
        full_data_clusters.append(cluster_indices)

    if global_hvg:
        sc.pp.highly_variable_genes(adata, layer=None, n_top_genes=num_hvg)
        top_gene_idxs = np.where(adata.var['highly_variable'])[0]
        top_gene_names = adata.var_names[top_gene_idxs]
    else:
        # Now compute bottoms up hvgs.
        hvg_count_vec = np.zeros(adata.shape[1])
        for i, clusterids in enumerate(full_data_clusters):
            cluster = adata[clusterids,:].copy()
            sc.pp.highly_variable_genes(cluster, layer=None, n_top_genes=num_hvg_per_cluster)
            hvg_count_vec += np.where(cluster.var['highly_variable'], 1, 0)

        # select the indices of the first num_hvg highest values in hvg_count_vec
        top_gene_idxs =  np.argsort(hvg_count_vec)[-num_hvg:]
        top_gene_names = adata.var_names[top_gene_idxs]
        
    if hvg_filename:
        with open(hvg_filename, 'wb') as f:
            pickle.dump((top_gene_idxs, top_gene_names), f)
            
    adata.uns[uns_key][hvg_key] = top_gene_names.tolist()

    out = ()
    if copy:
        out += (adata,)
    if return_results:
        out += ((top_gene_idxs, top_gene_names.tolist()),)
    
    return out if len(out) > 1 else out[0] if len(out) == 1 else None
