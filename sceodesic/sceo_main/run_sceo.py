
import sys 

import scanpy as sc 
import numpy as np
import pandas as pd

from .get_cell_cohorts import get_cell_cohorts
from .get_locally_variable_genes import get_locally_variable_genes
from .estimate_covariances import estimate_covariances 
from .reconstruct_programs import reconstruct_programs

from .default_keys import UNS_KEY
from .default_keys import SCEO_CONFIG_KEY
from .default_keys import *

from ..utils import fn_timer
from ..utils import order_by_second_moment


@fn_timer
def run_sceo(adata, num_hvg=-1, num_cohorts='auto', sparse_pca_lambda=0.03, 
             max_condition_number=50, stratify_cols='none', 
             num_hvg_per_cohort=100, pvd_pct=0.9, do_global_hvg=False, 
             cohort_weights=None, top_genes=None,
             copy=False, n_init=1, key_added=None, uns_key=None):
    """
    Computes sceodesic embeddings and saves them in adata.obsm[key_added], 
    sceodesic programs are stored in adata.obsm[key_added],
    and sceodesic programs and metadata stored in adata.uns[uns_key]. 
    
    Note that programs are also stored in adata.uns[uns_key]['sceo_programs'].
    
    :param adata: 
        An adata file with log-normalized gene expression data in adata.X.
    
    :type adata: anndata.AnnData
    
    :param num_hvg:
        The final number of locally variable genes to select out of the union of 
        locally variable genes across all cohorts. Must be specified unless top_genes 
        is specified.
    
    :type num_hvg: int > 0
    
    :param num_cohorts:
        The number of cell cohorts to create. Must be at least 10. 
    
    :type num_cohorts: int > 10
        
    :param sparse_pca_lambda:
        Sparsity parameter for sparse pca during module reconstruction. 
    
    :type sparse_pca_lambda: float >= 0
        
    :param max_condition_number: 
        The maximum condition number of each estimated cohort-specific covariance matrix.
    
    :type max_condition_number: float > 0
    
    :param stratify_cols: 
        Columns of adata.obs by which to stratify observations when constructing cell cohorts.
        If none, no stratification is performed. 
    
    :type stratify_cols: string or list of strings 
    
    :param num_hvg_per_cohort: 
        Number of locally variable genes to estimate per cohort
    
    :type num_hvg_per_cohort: int > 0
    
    :param pvd_pct: 
        Take eigenvectors 1,...,k, where k is the minimum integer for which 
        sum(lambda_1,...,lambda_k) >= pvd_pct
    
    :type pvd_pct: float in (0, 1)
    
    :param do_global_hvg: 
        If True, do a global hvg estimation from the entire gene expression matrix 
        rather than getting locally variable genes per cohort. 
    
    :type do_global_hvg: boolean 
    
    :param cohort_weights: 
        Optionally, one can specify the cohort assignment rather than have them computed. 
        If not specified, the cohort assignments will be computed according to the parameters 
        num_cohorts, stratify_cols, num_hvg. 
        
    :type cohort_weights: np.array[str] of length n_cells
    
    :param top_genes:
        If specified, use these genes as the gene set for estimating covariances and 
        computing sceodesic programs. If not specified, will be computed according to 
        the parameters num_hvg, num_hvg_per_cohort, and do_global_hvg. 
        
    :type top_genes: list[str]
    
    :param copy: 
        Return a copy of anndata if True 
    
    :type copy: boolean 
    
    :param n_init:
        Number of initializations for k-means clustering
    
    :type n_init: int > 0
    
    :param key_added: 
        If specified, sceodesic embeddings stored in adata.obsm[key_added], 
        and sceodesic programs are stored in adata.varm[key_added]. 
        
        Otherwise, (by default) sceodesic embeddings are stored in adata.obsm['sceo_embeddings']
        and sceodesic programs are stored in adata.varm['sceo_programs'].
        
    :type key_added: str
    
    :param uns_key:
        Where to store the sceodesic gene programs, parameter information, and metadata. 
        
    :type uns_key: str
    
    :returns: a copy of adata if copy=True, else modifies adata in place and returns None. 
    """
    
    if key_added is None:
        obsm_key_added = SCEO_EMBEDDINGS_KEY
        varm_key_added = MODULES_KEY
    else:
        obsm_key_added = key_added
        varm_key_added = key_added
        
    if uns_key is None:
        uns_key = UNS_KEY
    
    if copy:
        adata = adata.copy()
        
    adata.uns[uns_key] = {}
     
    # these are hard-coded for now (fix later)
    cluster_key = CLUSTER_KEY
    embeddings_dict_key = EMBEDDINGS_DICT_KEY
    modules_key = MODULES_KEY
    hvg_key = HVG_KEY
    
    # can give cohort_weights or top_genes optionally
    if cohort_weights is not None and top_genes is None:
        num_cohorts = cohort_weights.shape[1]
        stratify_cols = '***NOT SPECIFIED***'
        cluster_wts = cohort_weights
        get_locally_variable_genes(adata, num_hvg, num_hvg_per_cohort, do_global_hvg,
                                   cohort_weights=cohort_weights, uns_key=uns_key) 
        estimate_covariances(adata, max_condition_number, pvd_pct, uns_key=uns_key)
    elif cohort_weights is None and top_genes is not None:
        num_hvg = len(top_genes)
        adata.uns[uns_key][hvg_key] = top_genes
        cluster_wts = get_cell_cohorts(adata, num_cohorts, stratify_cols, num_hvg, n_init=n_init, uns_key=uns_key, return_results=True)['inflated_cluster_resps']
        estimate_covariances(adata, max_condition_number, pvd_pct, top_genes=top_genes, uns_key=uns_key)
    elif cohort_weights is not None and top_genes is not None:
        num_cohorts = cohort_weights.shape[1]
        stratify_cols = '***NOT SPECIFIED***'
        num_hvg = len(top_genes)
        adata.uns[uns_key][hvg_key] = top_genes
        cluster_wts = cohort_weights
        estimate_covariances(adata, max_condition_number, pvd_pct, top_genes=top_genes, 
                             cohort_weights=cohort_weights, uns_key=uns_key)
    else:
        try:
            assert num_hvg > 0
        except Exception as e:
            message = ("Error: you must either specify the number of variable genes to select"
                       " or input your own list of genes of interest.")
            print(message, file=sys.stderr)
            raise e
        cluster_wts = get_cell_cohorts(adata, num_cohorts, stratify_cols, num_hvg, n_init=n_init, uns_key=uns_key, return_results=True)['inflated_cluster_resps']
        get_locally_variable_genes(adata, num_hvg, num_hvg_per_cohort, do_global_hvg, uns_key=uns_key)
        estimate_covariances(adata, max_condition_number, pvd_pct, uns_key=uns_key)
    
    # will do the same thing here no matter which of cohort_weights/top_genes is pre-specified
    reconstruct_programs(adata, sparse_pca_lambda, uns_key=uns_key)
        
    embeddings = adata.uns[uns_key][embeddings_dict_key]
    modules = adata.uns[uns_key][modules_key]
    top_genes = adata.uns[uns_key][hvg_key]

    embeddings_matrix = np.zeros((num_hvg, cluster_wts.shape[1])).T
    for i in embeddings:
        embeddings_matrix[i, :] = embeddings[i]

    # making the .obsm object 
    observation_count = adata.n_obs    
    data_embedding = cluster_wts @ embeddings_matrix
    #data_embedding = np.zeros((observation_count, num_hvg))
    #for i, embed in embeddings.items():
        #cluster_indices = cell2cluster[i]
        #for cell in cluster_indices:
            #data_embedding[cell, :] = embed
            
    data_embedding, modules = order_by_second_moment(data_embedding, modules)
    
    # make the .varm data matrix 
    not_top_genes = adata.var_names[~np.isin(adata.var_names, top_genes)]
    tdf = pd.DataFrame(modules, index=top_genes)
    ntdf = pd.DataFrame(np.zeros((adata.shape[1]-len(top_genes), num_hvg)), index=not_top_genes)
    edf = pd.DataFrame(index=adata.var_names)
    varm = edf.join(pd.concat([tdf, ntdf]))

    adata.varm[varm_key_added] = varm.values
    adata.obsm[obsm_key_added] = data_embedding
    adata.uns[uns_key][modules_key] = modules
    
    # save config settings to anndata object 
    config = {
        'num_cohorts': num_cohorts,
        'num_hvg': num_hvg,
        'sparse_pca_lambda': sparse_pca_lambda, 
        'max_condition_number': max_condition_number, 
        'stratify_cols': stratify_cols, 
        'num_hvg_per_cohort': num_hvg_per_cohort, 
        'pvd_pct': pvd_pct, 
        'do_global_hvg': do_global_hvg,
        'n_init': n_init
    }
    config_key = SCEO_CONFIG_KEY
    adata.uns[uns_key][config_key] = config

    if copy:
        return adata
