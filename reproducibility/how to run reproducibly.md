1. set environment variables (copy and paste from `environment_variables.sh`)
2. install singhlab package from `https://github.com/rohitsinghlab/singhlab-main`
3. use python code as follows:

```python
from singhlab.utils.decorators import run_with_seed
from sceodesic import run_sceo

seeds = [123, 444, 33, 1234, 99, 14, 37]
results = {}

adata = sc.read_h5ad('adata.h5ad')

for seed in seeds:
    run_with_seed(seed)(run_sceo)(adata, num_hvgs=300)
    sceo_embeddings = adata.obsm['sceo_embeddings']
    sceo_programs = adata.varm['sceo_programs]
    results[seed] = (sceo_embeddings, sceo_programs)
```
