
import scanpy as sc 
import pandas as pd 
import numpy as np 

import sys 
import pickle


# package-specific modules 
from ..utils import fn_timer
from ..utils import order_by_second_moment

from .default_keys import *


@fn_timer
def write_embedding(adata, num_hvg, config=None, 
                    results_clustering=None,
                    results_hvg=None, 
                    results_embedding=None,
                    sceodesic_adata_filename=None,
                    key_added=None,
                    config_key=SCEO_CONFIG_KEY,
                    modules_key=MODULES_KEY,
                    uns_key=UNS_KEY,
                    copy=False):
    
    if key_added is None:
        obsm_key_added = SCEO_EMBEDDINGS_KEY
        varm_key_added = MODULES_KEY
    else:
        obsm_key_added = key_added
        varm_key_added = key_added
        
    if copy:
        adata = adata.copy() 
    
    num_hvg = num_hvg
    
    # only difference between this and _write_embeddings
    top_genes = results_hvg
        
    cell2cluster = results_clustering["cell2cluster"]
        
    embeddings = results_embedding["embedding_dictionary"]
    modules = results_embedding['modules']
    
    # making the .obsm object 
    observation_count = adata.n_obs    
    data_embedding = np.zeros((observation_count, num_hvg))
    for i, embed in embeddings.items():
        cluster_indices = cell2cluster[i]
        for cell in cluster_indices:
            data_embedding[cell, :] = embed
    
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
    results_embedding["modules"] = modules
    
    # save config settings to anndata object 
    if config:
        adata.uns[config_key] = config

    if sceodesic_adata_filename:
        adata.write_h5ad(sceodesic_adata_filename)
    
    if copy:
        return adata


def _write_embedding(adata, num_hvg, config=None, 
                     results_clustering=None,
                     results_hvg=None, 
                     results_embedding=None,
                     sceodesic_adata_filename=None,
                     key_added=None,
                     config_key=SCEO_CONFIG_KEY,
                     modules_key=MODULES_KEY,
                     uns_key=UNS_KEY,
                     copy=False):
    
    if key_added is None:
        obsm_key_added = SCEO_EMBEDDINGS_KEY
        varm_key_added = MODULES_KEY
    else:
        obsm_key_added = key_added
        varm_key_added = key_added
        
    if copy:
        adata = adata.copy() 
    
    num_hvg = num_hvg
    
    top_gene_idxs, top_gene_names = results_hvg
        
    cell2cluster = results_clustering["cell2cluster"]
        
    embeddings = results_embedding["embedding_dictionary"]
    modules = results_embedding["modules"]
    _, top_genes = results_hvg
    
    # making the .obsm object 
    observation_count = adata.n_obs    
    data_embedding = np.zeros((observation_count, num_hvg))
    for i, embed in embeddings.items():
        cluster_indices = cell2cluster[i]
        for cell in cluster_indices:
            data_embedding[cell, :] = embed
            
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
    results_embedding["modules"] = modules
    
    # save config settings to anndata object 
    if config:
        adata.uns[config_key] = config

    if sceodesic_adata_filename:
        adata.write_h5ad(sceodesic_adata_filename)
        
    return adata
    
#     filtered_data = adata[:,top_gene_idxs].copy()
#     filtered_data.var_names = [f'P{i}' for i in range(1,num_hvg+1)]
    
#     df = pd.DataFrame(modules, columns = filtered_data.var_names)
#     df.index = top_gene_names
#     filtered_data.var = df.transpose()
    
#     observation_count = filtered_data.n_obs
        
#     data_embedding = np.zeros((observation_count, num_hvg))
        
#     for i, embed in embeddings.items():
#         cluster_indices = cell2cluster[i]
#         for cell in cluster_indices:
#             data_embedding[cell, :] = embed
        
#     filtered_data.X = data_embedding

#     # save config settings to anndata object 
#     if config:
#         filtered_data.uns['riem_config'] = config

#     if sceodesic_adata_filename:
#         filtered_data.write_h5ad(sceodesic_adata_filename)
        
#     return filtered_data