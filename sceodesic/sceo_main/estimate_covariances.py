
import scipy

import pickle 
import sys 

# package-specific modules 
from ..utils import fn_timer
from ..helper import compute_covariance_and_ncomps_pct_variance

from .default_keys import *

@fn_timer
def estimate_covariances(adata, max_condition_number, pvd_pct=0.9, 
                         copy=False, return_results=False,
                         top_genes=None, cohort_assn=None,
                         uns_key=None):
    
    if uns_key is None:
        uns_key = UNS_KEY
    
    # not able to be passed in
    hvg_key = HVG_KEY
    
    # top_genes can either be passed in anew or be precomputed using get_locally_variable_genes
    if top_genes is None: 
        try:
            top_genes = adata.uns[uns_key][hvg_key]
        except Exception as e:
            message = ("Error: must either specify a set of genes to consider or "
                       "have run sceodesic.get_locally_variable_genes beforehand.")
            print(message, file=sys.stderr)
            
            raise e
    else:
        adata.uns[uns_key][hvg_key] = top_genes
            
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
    
    return _estimate_covariances(adata, max_condition_number, pvd_pct,
                                 copy, return_results, 
                                 top_genes=top_genes,
                                 results_clustering=clustering_results,
                                 uns_key=uns_key)
    
    


def _estimate_covariances(adata, max_condition_number, pvd_pct=0.9, 
                   copy=False, return_results=False, coexpression_filename=None,
                   top_genes=None, results_clustering=None, 
                   uns_key=None, cluster_covar_key=None,
                   cluster_var_ct_key=None):
    
    if uns_key is None:
        uns_key = UNS_KEY
        
    if cluster_covar_key is None:
        cluster_covar_key = CLUSTER_COVAR_KEY
        
    if cluster_var_ct_key is None:
        cluster_var_ct_key = CLUSTER_VAR_CT_KEY
    
    if copy:
        adata = adata.copy()
        
    # change later 
    top_genes = top_genes
    results_clustering = results_clustering
    
    cell2cluster = results_clustering["cell2cluster"]
    
    filtered_data = adata[:,top_genes]
    
    # Get the clusters from the reduced data.
    clusters = {}

    processed_data = None
    if scipy.sparse.issparse(filtered_data.X):
        processed_data = filtered_data.X.A
    else:
        processed_data = filtered_data.X

    for key in cell2cluster.keys():
        cluster_indices = cell2cluster[key]
        clusters[key] = processed_data[cluster_indices,:]
    
    cluster_covariances = {}
    cluster_var_count = {}  
    for i,cluster in clusters.items():
        cluster_covar, var_count = compute_covariance_and_ncomps_pct_variance(cluster, max_condition_number, pvd_pct)
        cluster_covariances[i] = cluster_covar # Ensures a PSD matrix.
        cluster_var_count[i] = var_count

    ### invariant based programming: put in asserts on what you expect the shape to be

    results_coexp = {"cluster_covariances": cluster_covariances, "cluster_var_count": cluster_var_count}

    if coexpression_filename:
        with open(coexpression_filename, 'wb') as f:
            pickle.dump(results_coexp, f)
            
    # write to adata.uns
    adata.uns[uns_key][cluster_covar_key] = results_coexp['cluster_covariances']
    adata.uns[uns_key][cluster_var_ct_key] = results_coexp['cluster_var_count']
    
    out = ()
    if copy:
        out += (adata,)
    if return_results:
        out += (results_coexp,)
    
    return out if len(out) > 1 else out[0] if len(out) == 1 else None 
