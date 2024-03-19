import numpy as np

def order_by_second_moment(data_embedding, modules):
    # reorder modules and embeddings by second moment
    # compute -E(X^2) because argsort sorts in increasing order
    metric_vals = np.apply_along_axis(lambda x: -np.sum(np.power(x, 2)), 0, data_embedding)
    index_ordered = np.argsort(metric_vals)
    data_embedding = data_embedding[:, index_ordered]
    modules = modules[index_ordered, :]
    
    return data_embedding, modules
