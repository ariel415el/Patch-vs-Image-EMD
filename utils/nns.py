import numpy as np
import torch


def efficient_compute_distances(X, Y):
    """
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e (X[:, None] - Y[None, :])**2
    Get the nearest neighbor index from Y for each X
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
    d = X.shape[1]
    dist /= d # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
    return dist


def compute_distances_batch(X, Y, b):
    """
    Computes distance matrix in batches of rows to reduce memory consumption from (n1 * n2 * d) to (d * n2 * d)
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    :param b: rows batch size
    Returns a (n2, n1) matrix of L2 distances
    """
    """"""
    b = min(b, len(X))
    dist_mat = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float32)
    n_batches = len(X) // b
    for i in range(n_batches):
        dist_mat[i * b:(i + 1) * b] = efficient_compute_distances(X[i * b:(i + 1) * b], Y)
    if len(X) % b != 0:
        dist_mat[n_batches * b:] = efficient_compute_distances(X[n_batches * b:], Y)

    return dist_mat

def get_NN_indices_low_memory(X, Y, b):
    """
    Get the nearest neighbor index from Y for each X.
    Avoids holding a (n1 * n2) amtrix in order to reducing memory footprint to (b * max(n1,n2)).
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    NNs = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    n_batches = len(X) // b
    for i in range(n_batches):
        dists = efficient_compute_distances(X[i * b:(i + 1) * b], Y)
        NNs[i * b:(i + 1) * b] = dists.min(1)[1]
    if len(X) % b != 0:
        dists = efficient_compute_distances(X[n_batches * b:], Y)
        NNs[n_batches * b:] = dists.min(1)[1]
    return NNs


