import os

import numpy as np
from torchvision.utils import save_image
import torch
from torch.nn import functional as F

def get_nns(queries, refs):
    """
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e (X[:, None] - Y[None, :])**2
    Get the nearest neighbor index from Y for each X
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices

    """
    X = refs
    Y = queries
    dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
    d = X.shape[1]
    dist /= d # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)

    NNs = np.argmin(dist, axis=0)  # find_NNs
    return X[NNs]


def dump_images(imgs, b, d, c, fname):
    save_image(torch.from_numpy(imgs).reshape(b, d, d, c).permute(0,3,1,2), fname, normalize=True, nrow=int(np.sqrt(b)))


def get_centroids(data, n_centroids, fname):
    if os.path.exists(fname):
        centroids = np.load(fname)
    else:
        import faiss
        kmeans = faiss.Kmeans(data.shape[1], n_centroids, niter=100, verbose=False, gpu=True)
        kmeans.train(data)
        # kmeans = KMeans(n_clusters=n_centroids, random_state=0, verbose=0).fit(data)
        centroids = kmeans.centroids
        np.save(fname, centroids)
    return centroids


def compute_n_patches_in_image(d, c, p, s):
    dummy_img = torch.zeros((1, c, d, d))
    dummy_patches = F.unfold(dummy_img, kernel_size=p, stride=s)
    patches_per_image = dummy_patches.shape[-1]  # shape (1, c*p*p, N_patches)
    return patches_per_image


def to_patches(x, d, c, p=8, s=4):
    xp = x.reshape(-1, d, d, c)
    xp = torch.from_numpy(xp).permute(0, 3, 1, 2)
    patches = F.unfold(xp, kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    patches = patches.permute(0, 2, 1)
    patches = patches.reshape(-1, patches.shape[-1])
    return patches.numpy()


def patches_to_image(patches, d, c, p=8, s=4):
    patches_per_image = compute_n_patches_in_image(d, c, p, s)

    patches = torch.from_numpy(patches)
    patches = patches.reshape(-1, patches_per_image, c * p ** 2)
    patches = patches.permute(0, 2, 1)
    img = F.fold(patches, (d, d), kernel_size=p, stride=s)
    img = img.permute(0, 2, 3, 1).numpy()
    return img


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


def get_patch_centroids(train_data, batch_size, d, c, p, s, out_dir):
    # Train patch K-means
    patches_per_batch = compute_n_patches_in_image(d, c, p=p, s=s) * batch_size
    patch_centroids_path = f"{out_dir}/patch_centroids-{patches_per_batch}.npy"
    patches = to_patches(train_data, d, c, p=p, s=s)
    patch_centroids = get_centroids(patches, n_centroids=patches_per_batch, fname=patch_centroids_path)
    return patch_centroids


def reconstructed_with_patch_centroids(batch1, patch_centroids, d, c, p, s):
    reconstructions = []
    for i in range(len(batch1)):
        img = batch1[i]
        img_patches = to_patches(img[None,], d, c, p, s)
        new_patches = get_nns(queries=img_patches, refs=patch_centroids)

        new_patches = new_patches - new_patches.mean(1)[:, None] + img_patches.mean(1)[:, None]
        new_img = patches_to_image(new_patches, d, c, p, s)
        reconstructions.append(new_img)

    reconstructions = np.stack(reconstructions)
    return reconstructions


def plot(res_dict, fname, std=True):
    from matplotlib import pyplot as plt
    from matplotlib.pyplot import cm

    batch_sizes = list(res_dict.keys())
    metrics = list(res_dict[batch_sizes[0]])
    plt.figure(figsize=(10, 8), dpi=80)
    color = cm.nipy_spectral(np.linspace(0, 1, len(metrics)))
    for metric, c in zip(metrics, color):
        vals = []
        stds = []
        mins = []
        maxs = []
        for b in batch_sizes:
            vals.append(np.mean(res_dict[b][metric]))
            stds.append(np.std(res_dict[b][metric]))
            mins.append(np.min(res_dict[b][metric]))
            maxs.append(np.max(res_dict[b][metric]))
        vals = np.array(vals)
        stds = np.array(stds)
        type = '--' if 'patch' in metric else '-'
        plt.plot(batch_sizes, vals, type, alpha=1, color=c, label=metric)
        if std:
            plt.fill_between(batch_sizes, vals-stds/2, vals+stds/2, alpha=0.15, color=c)
        else:
            plt.fill_between(batch_sizes, mins, maxs, alpha=0.15, color=c)

    plt.xlabel("Batch-size")
    plt.ylabel("Distance")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)
    plt.tight_layout()
    plt.savefig(fname)
    plt.clf()
