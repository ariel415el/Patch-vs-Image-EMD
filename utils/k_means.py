import os

import numpy as np

from utils.image import compute_n_patches_in_image, to_patches


def get_centroids(data, n_centroids, fname, use_faiss=True):
    if os.path.exists(fname):
        centroids = np.load(fname)
    else:
        if use_faiss:
            import faiss
            kmeans = faiss.Kmeans(data.shape[1], n_centroids, niter=100, verbose=False, gpu=True)
            kmeans.train(data)
            centroids = kmeans.centroids
        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_centroids, random_state=0, verbose=0).fit(data)
            centroids = kmeans.cluster_centers_

        np.save(fname, centroids)
    return centroids


def get_patch_centroids(train_data, batch_size, d, c, p, s, out_dir):
    # Train patch K-means
    patches_per_batch = compute_n_patches_in_image(d, c, p=p, s=s) * batch_size
    patch_centroids_path = f"{out_dir}/patch_centroids-{patches_per_batch}.npy"
    patches = to_patches(train_data, d, c, p=p, s=s)
    patch_centroids = get_centroids(patches, n_centroids=patches_per_batch, fname=patch_centroids_path)
    return patch_centroids


