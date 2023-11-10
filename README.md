
# Sceodesic

Sceodesic is a Python package that implements the gene program discovery algorithm described by Sinan Ozbay, Aditya Parekh, and Rohit Singh in “Navigating the manifold of single-cell gene coexpression to discover interpretable gene programs.” Given a single-cell gene expression dataset, Seodesic allows the user to:
1. Discover meaningful gene programs from any single-cell gene expression data.
2. Re-interpet gene expression data as levels of expression of the aforementioned gene programs. 

This repository contains an implementation of the algorithm.  Some examples of use can be found below in "Example Usage".

## Installation

You can install Sceodesic by running the following in your command line:

\```
git clone https://github.com/rohitsinghlab/sceodesic.git
cd ./sceodesic  # to directory containing pyproject.toml file 
pip install .
\```

## API Example Usage

Below is example usage of sceodesic in Python. First, make sure your single-cell gene expression data is in an anndata object, as below.

\```python
import anndata

adata = [single cell gene expression anndata object]
\```

Now, you are ready to discover some programs!

\```python
from sceodesic import run_sceo

run_sceo(adata, num_hvg=300)

embeddings = adata.obsm['sceo_embeddings']

# programs (loadings) stored in .varm, as a numpy array
programs = adata.varm['sceo_programs']
\```

`embeddings` is a 2D numpy array where rows represent cells and columsn represent levels of gene program expression. `programs` is a 2D numpy array where columns represent individual gene programs, with each entry of that column giving the weight of each individual gene\ in that program. After you have generated programs and the embeddings, it becomes possible to do a wide variety of downstream single cell analysis using these embeddings.  For example, if our data is labelled by cell type with an `obs` column called "cell_type" , we can compute differential gene program expression across cell types as follows:

\```python

# Create new anndata object with embeddings
adata_sceo = anndata.AnnData(embeddings, obs=adata.obs)

sc.get.rank_genes_groups_df(adata = adata_sceo, group = “cell_type”)
\```
