#!/usr/bin/env python

import numpy as np
from numpy.linalg import eig, eigh
import pandas as pd

import scipy
from scipy.linalg import logm, svd, expm
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, kendalltau, rankdata 

import scanpy as sc
import anndata

import fbpca

import sklearn
from sklearn.decomposition import SparsePCA, MiniBatchSparsePCA, PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

import os, sys, yaml, pickle
import functools, random
import time

## package-specific modules
from .utils import fn_timer

from .sceo_io.sceo_command_line_parser import parse_sceo_command_line_args
from .sceo_io.sceo_load_input import load_input

from .helper import compute_covariance_and_ncomps_pct_variance 

from .sceo_main.get_cell_cohorts import _get_cell_cohorts
from .sceo_main.get_locally_variable_genes import _get_locally_variable_genes
from .sceo_main.estimate_covariances import _estimate_covariances
from .sceo_main.reconstruct_programs import _reconstruct_programs
from .sceo_main.write_embedding import _write_embedding


# Default configuration
DEFAULT_CONFIG = {
    'num_clusters': 500,
    'num_hvg': 300,
    'max_condition_number': 50,
    'sparse_pca_lambda': 0.03,

    'stratify_clustering_by_columns': 'none',
    'filepath': '',

    'num_hvg_per_cluster': 100,
    'pvd_pct': 0.90,
    'do_global_hvg': False,
    
    # for very advanced users 
    'n_init': 1  # K-Means
}


def main():
    args = parse_sceo_command_line_args(DEFAULT_CONFIG)
    
    ### TESTING ###
    print("config:", args.config)
    ### TESTING ###

    output_identifier = "%s_%d_hvgs_%s_clusters_%g_sparsity" % (args.output_prefix, args.num_hvg,
                                                                str(args.config.get('num_clusters')), args.config.get('sparse_pca_lambda'))
    args.config['output_identifier'] = output_identifier
    
    filepath = args.config.get('filepath', DEFAULT_CONFIG['filepath']) + '/' #if the backslash is extra, it won't hurt
    
    args.config['clustering_filename'] = f"{filepath}clustering_results_{output_identifier}.pkl"
    args.config['hvg_filename'] = f"{filepath}hvg_results_{output_identifier}.pkl"
    args.config['coexpression_filename'] = f"{filepath}coexpression_results_{output_identifier}.pkl"
    args.config['embedding_filename'] = f"{filepath}embedding_results_{output_identifier}.pkl"
    
    # in case we want a custom output name for our output file
    if args.adata_output_name:
        args.config['sceodesic_adata_filename'] = args.adata_output_name
    else:
        args.config['sceodesic_adata_filename'] = f"{filepath}sceodesic_adata_results_{output_identifier}.h5ad"
    
    # run info file output 
    run_info_file_fname = f"{filepath}run_info_{output_identifier}.yaml"

    results_coexp = None
    results_embedding = None
    
    # Data preprocessing.
    adata = load_input(args.inp_data)
        
    # Flag 1: Clustering
    if args.action <= 1:
        print("At FLAG 1: clustering")
        num_clusters = args.config['num_clusters']
        stratify_cols = args.config['stratify_clustering_by_columns']
        num_hvg = args.config['num_hvg']
        n_init = args.config['n_init']
        clustering_filename = args.config['clustering_filename']
        
        clustering_results_dict = _get_cell_cohorts(
            adata, num_clusters, 
            stratify_cols=stratify_cols, 
            num_hvg=num_hvg,
            n_init=n_init,
            clustering_filename=clustering_filename, 
            copy=False, return_results=True
        )

    # Flag 2: Compute Covariances
    if args.action <= 2:
        print("At FLAG 2: compute covariances")
        # compute hvg 
        num_hvg = args.config['num_hvg']
        do_global_hvg = args.config['do_global_hvg'], 
        num_hvg_per_cluster = args.config['num_hvg_per_cluster']
        hvg_filename = args.config['hvg_filename']
        
        top_genes, top_gene_names = _get_locally_variable_genes(
            adata, num_hvg, 
            num_hvg_per_cluster=num_hvg_per_cluster,
            global_hvg=do_global_hvg,
            hvg_filename=hvg_filename,
            copy=False,
            return_results=True,
            clustering_results=clustering_results_dict
        )
        
        # compute coexpression results
        max_condition_number = args.config['max_condition_number']
        pvd_pct = args.config['pvd_pct']
        coexpression_filename = args.config['coexpression_filename']
        
        results_coexp = _estimate_covariances(
            adata, max_condition_number, 
            pvd_pct=pvd_pct,
            coexpression_filename=coexpression_filename,
            copy=False, 
            return_results=True, 
            top_genes=top_gene_names,
            results_clustering=clustering_results_dict
        )

    # Flag 3: Embeddings/Modules
    if args.action <= 3:
        print("At FLAG 3: common PCA")
        sparse_pca_lambda = args.config['sparse_pca_lambda']
        embedding_filename = args.config['embedding_filename']
        results_embedding = _reconstruct_programs(
            adata, sparse_pca_lambda, 
            embedding_filename=embedding_filename, 
            copy=False, 
            return_results=True,
            results_coexp=results_coexp
        )
        
    # Flag 4: Final Embeddings
    if args.action <= 4:
        print("At FLAG 4: writting embeddings")
        num_hvg = args.config['num_hvg']
        results_clustering = clustering_results_dict
        results_hvg = (top_genes, top_gene_names)
        results_embedding = results_embedding
        sceodesic_adata_filename = args.config['sceodesic_adata_filename']
        filtered_data = _write_embedding(
            adata, num_hvg, 
            config=args.config,
            results_clustering=results_clustering,
            results_hvg=results_hvg, 
            results_embedding=results_embedding,
            sceodesic_adata_filename=sceodesic_adata_filename,
            copy=False
        )

    print("Successfully ran sceodesic. Writing run information!")
    with open(run_info_file_fname, 'wt') as f:
        yaml.dump(args, f, default_flow_style=False)

    
    return { 'clustering_results': clustering_results_dict,
             'top_genes': [top_genes, top_gene_names],
             'results_coexp': results_coexp,
             'results_embedding': results_embedding,
             'filtered_data': filtered_data }


def run_main():
    start = time.time()
    main()
    end = time.time()
    print(f'Total time taken: {end-start:0.3f} seconds.')
    
    
if __name__ == '__main__':
    run_main()
