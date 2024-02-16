
import sys
import pickle

import scanpy as sc 
import numpy as np 

# package-specific modules
from ..utils import fn_timer
from .default_keys import *


@fn_timer
def get_locally_variable_genes(adata, num_hvg, num_hvg_per_cluster=100, global_hvg=False,
                               copy=False, return_results=False, cohort_assn=None, uns_key=None):
    
    if uns_key is None:
        uns_key = UNS_KEY

    # removing ability to specify key
    cluster_key = CLUSTER_KEY
    
    # can either pass in a cell cohort assignment (array cohort_assn with cell[i] having cluster assn cohort_assn[i])
    # or the cluster_key 
    clustering_results = None
    if cohort_assn is None:
        try:
            clustering_results = adata.uns[uns_key]
        except:
            message = ("Error: must either specify a cell cohort assignment or "
                       "have run sceodesic.get_cell_cohorts beforehand.")
            print(message, file=sys.stderr)
            
            raise e
    else:
        c2c = {}
        for i, c in enumerate(cohort_assn):
            c2c[c] = c2c.get(c, []) + [i]
        clustering_results = {'cell2cluster': c2c, 'stratify_cols': '***NOT SPECIFIED***'}
        adata.uns[uns_key].update(clustering_results)
        
    return _get_locally_variable_genes(adata, num_hvg, num_hvg_per_cluster, global_hvg, 
                                       copy=copy, return_results=return_results, 
                                       clustering_results=clustering_results, 
                                       uns_key=uns_key)
        
    
def _get_locally_variable_genes(adata, num_hvg, num_hvg_per_cluster, global_hvg, hvg_filename=None, 
                copy=False, return_results=False, clustering_results=None, 
                uns_key=None, hvg_key=None):
    
    if uns_key is None:
        uns_key = UNS_KEY
    
    if uns_key not in adata.uns:
        adata.uns[uns_key] = {}
    
    if hvg_key is None: 
        hvg_key = HVG_KEY
        
    if copy:
        adata = adata.copy()
        
    # Store the cluster data matrices.
    cell2cluster = clustering_results["cell2cluster"]
    
    if global_hvg:
        sc.pp.highly_variable_genes(adata, layer=None, n_top_genes=num_hvg)
        top_gene_idxs = np.where(adata.var['highly_variable'])[0]
        top_gene_names = adata.var_names[top_gene_idxs]
    else:
        # Now compute bottoms up hvgs.
        hvg_count_vec = np.zeros(adata.shape[1])
        for clusterids in cell2cluster.values():
            hvgs = sc.pp.highly_variable_genes(adata[clusterids], 
                                               layer=None, 
                                               n_top_genes=num_hvg_per_cluster,
                                               inplace=False)['highly_variable']
            hvg_count_vec += np.where(hvgs, 1, 0)

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
