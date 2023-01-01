import os

import numpy as np

from utils.image import compute_n_patches_in_image, to_patches


def get_centroids(data, n_centroids, fname, use_faiss=True):
    if fname and os.path.exists(fname):
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

        if fname is not None:
            np.save(fname, centroids)
    return centroids


def get_patch_centroids(train_data, batch_size, d, c, p, s, out_dir):
    # Train patch K-means
    patches_per_batch = compute_n_patches_in_image(d, c, p=p, s=s) * batch_size
    patch_centroids_path = f"{out_dir}/patch_centroids-{patches_per_batch}.npy"
    patches = to_patches(train_data, d, c, p=p, s=s)
    patch_centroids = get_centroids(patches, n_centroids=patches_per_batch, fname=patch_centroids_path)
    return patch_centroids


if __name__ == '__main__':
    from Gradient_Kmeans import load_dataset
    import torch
    from torchvision.utils import save_image

    im_size = 64
    b = 64
    for b  in [1, 2, 4, 16, 64, 128]:
        data_path = '/home/ariel/university/repos/DataEfficientGANs/square_data/all'
        data = load_dataset(data_path, im_size=im_size, limit_data=None, gray=False, normalize_data=False).cuda()
        kmeans = get_centroids(data.cpu().numpy(), n_centroids=b, fname=None, use_faiss=False)
        kmeans = torch.from_numpy(kmeans).reshape(b, 3,im_size,im_size)
        save_image(kmeans, f"kmeans-{b}.png", nrow=int(np.sqrt(b)))

