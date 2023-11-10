
from ..utils import fn_timer 

import anndata as ad
import scanpy as sc

@fn_timer
def load_input(inp_data):
    adata = ad.read_h5ad(inp_data)
    print("Anndata read in")
    
    # should we change this?  (ROHIT)
    if adata.X.max() > 20:
        sc.pp.normalize_total(adata, target_sum=1e4) #normalize counts
        sc.pp.log1p(adata) # log 1+
        print("log-cp10k data computed")
    else:
        print('data was already log-normalized')
        
    return adata