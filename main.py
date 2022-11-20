import os
from collections import defaultdict

import numpy as np
import pandas as pd

from data import get_data
from utils import to_patches, get_patch_centroids, \
    reconstructed_with_patch_centroids, plot
from metrics import emd
from utils import dump_images, get_centroids


def batch_wassestein(train_data, train_data2, test_data, metric, batch_size, d, c, p, s, n_exp, out_dir, sample_patches=True):
    print("Batch-wassestein")
    mean_train_image_batch = np.mean(train_data, axis=0, keepdims=True).repeat(batch_size, 0)

    # Train image K-means
    fname = f"{out_dir}/centroids-{batch_size}.npy"
    print("\t- fitting images...", end='')
    centroids = get_centroids(train_data, n_centroids=batch_size, fname=fname)
    print("done")

    # Train patch K-means
    assert s == p, "Otherwise Grid artefacts in reconstructions have to be dealt with"
    print("\t- fitting patches...", end='')
    patch_centroids = get_patch_centroids(train_data, 256, d, c, p, s, out_dir)
    print("done")

    loss_dict = defaultdict(list)

    for i in range(n_exp):
        print(f"\t- rep: {i}:", end='')
        test_batch = test_data[np.random.choice(len(test_data), size=batch_size, replace=False)]
        real_batch = train_data2[np.random.choice(len(train_data2), size=batch_size, replace=False)]
        reconstructed_batch = reconstructed_with_patch_centroids(real_batch, patch_centroids, d, c, p, s).reshape(batch_size, -1)
        batches = {f"real": real_batch, f"mean": mean_train_image_batch, "centroids": centroids, 'reconstructed': reconstructed_batch}
        for name, batch in batches.items():
            print(f"{name}, ", end='')
            loss_dict[f'{metric.name}-{name}'].append(metric(batch, test_batch))

            print(f"{name}-patches, ", end='')
            x = to_patches(batch, d, c, p, s)
            y = to_patches(test_batch, d, c, p, s)
            if sample_patches:
                assert(len(x) == len(y))
                patch_indices = np.random.choice(len(x), size=batch_size, replace=False)
                x = x[patch_indices]
                y = y[patch_indices]
            loss_dict[f'patch-{metric.name}-{p}-{s}-{name}'].append(metric(x, y))

            dump_images(batch, batch_size, d, c, f"{out_dir}/{name}-{batch_size}.png")
        print("done")
    # loss_dict = {k: np.mean(v) for k,v in loss_dict.items()}

    return loss_dict

def main():
    data_path = '/mnt/storage_ssd/datasets/FFHQ_128'
    d = 64
    p, s = 8, 8
    gray = True
    normalize_data = False
    c = 1 if gray else 3
    metric = emd()

    out_dir = f"batch_wassestein_{d}x{d}{'_Gray' if gray else ''}{'_Normalized' if normalize_data else ''}_M-{metric.name}"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...", end='')
    data = get_data(data_path, resize=d, limit_data=20*1024, gray=gray, normalize_data=normalize_data)
    data = data[np.random.permutation(len(data))]
    print("done")
    train_data = data[4096:]
    train_data2 = data[2048:4096]
    test_batch = data[:2048]

    # patch_prior(train_data, test_data, d, c, tag)

    res_dict_full = dict()
    res_dict = dict()
    for batch_size in [2**i for i in range(4, 11)]:
        res = batch_wassestein(train_data, train_data2, test_batch, metric, batch_size, d, c, p, s, n_exp=3, out_dir=out_dir)
        res_dict_full[batch_size] = res

        plot(res_dict_full, f"{out_dir}/plot-std.png")
        plot(res_dict_full, f"{out_dir}/plot-minmax.png", std=False)

        res_dict[batch_size] = {k: np.mean(v) for k,v in res.items()}
        df = pd.DataFrame(res_dict)
        df.to_csv(f"{out_dir}/final.csv")


if __name__ == '__main__':
    main()