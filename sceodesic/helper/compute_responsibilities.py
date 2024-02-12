import numpy as np
from scipy.special import logsumexp

from functools import reduce

# scipy
from scipy.sparse import csr_matrix
import scipy.sparse
from scipy.special import logsumexp


def compute_responsibilities(X, weights, means, precisions, t=None):

    if t is None:
        t = 0

    return _compute_responsibilities(X, weights, means, precisions, t)

def _compute_responsibilities(X, weights, means, precisions, threshold,
                              start0=None, end0=None, start1=None,
                              end1=None, chunk_size_0=10000,
                              chunk_size_1=300):
    if start0 is None:
        start0 = 0
    if end0 is None:
        end0 = X.shape[0]

    if start1 is None:
        start1 = 0
    if end1 is None:
        end1 = len(weights)

    matrices_list = list()

    irow = start0
    icol = start1

    ct = 1

    while irow < end0:
        # get end index row pointer
        iirow = irow + chunk_size_0
        if iirow > end0:
            iirow = end0

        # compute responsibilities for all cells in current block
        cell_matrices_list = []

        # the portion of the observations for which
        # we are calculating responsibilities
        x = X[irow:iirow]

        while icol < end1:

            # get end index column pointer
            iicol = icol + chunk_size_1
            if iicol > end1:
                iicol = end1

            # compute responsibilities for these cells for the chosen clusters
            a = weights[icol:iicol]
            b = means[icol:iicol]
            c = precisions[icol:iicol]
            cell_resps = _compute_responsibilities_calc(x, a, b, c, len(weights))
            cell_matrices_list.append(cell_resps)

            # update column pointer
            icol = iicol

            ct += 1

        # normalize
        cell_matrix = reduce(lambda x, y: np.hstack((x, y)),
                             cell_matrices_list)

        sums = logsumexp(cell_matrix, axis=1, keepdims=True)

        # exponentiate
        cell_matrix = np.exp(cell_matrix - sums)

        # apply thresholding
        cell_matrix = np.where(cell_matrix > threshold, cell_matrix, 0)
        cell_matrix /= cell_matrix.sum(axis=1)[:, np.newaxis]

        # convert to sparse matrix and append to matrix list
        matrices_list.append(csr_matrix(cell_matrix))

        # update row pointer
        irow = iirow

        print('computed responsibilities for', irow, 'cells out of', end0)

        # reset column pointers
        icol = 0

    # vstack
    matrix = reduce(lambda x, y: scipy.sparse.vstack((x, y)),
                    matrices_list)

    return matrix


def _compute_responsibilities_calc(X, weights, means, precisions, nclusters):
    # precisions must be of dimension (ncluster, nfeatures)
    # calculating numertors
    XX = X[:, np.newaxis, :]
    XX_scaled = (XX - means[np.newaxis, :]) * np.sqrt(precisions)
    log_wts = np.apply_along_axis(lambda x: np.sum(np.power(x, 2)),
                                  2, XX_scaled)
    log_wts *= -0.5
    log_wts -= 0.5 * nclusters * np.log(2 * np.pi)
    log_wts += np.log(weights)
    log_wts += 0.5 * np.sum(np.log(precisions), axis=1)

    # return unnormalized log responsibilities
    return log_wts


