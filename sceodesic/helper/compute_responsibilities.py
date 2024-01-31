import numpy as np 
from scipy.special import logsumexp


def compute_responsibilities(X, weights, means, precisions):
    # precisions must be of dimension (ncluster, nfeatures)
    # calculating numertors
    XX = X[:, np.newaxis, :]
    XX_scaled = (XX - means[np.newaxis, :]) * np.sqrt(precisions)
    log_wts = np.apply_along_axis(lambda x: np.sum(np.power(x, 2)), 
                                  2, XX_scaled)
    log_wts *= -0.5
    log_wts -= 0.5 * len(weights) * np.log(2 * np.pi)
    log_wts += np.log(weights)
    log_wts += 0.5 * np.sum(np.log(precisions), axis=1)
    
    # calculating denominators
    sums = logsumexp(log_wts, axis=1, keepdims=True)
    return np.exp(log_wts - sums)
