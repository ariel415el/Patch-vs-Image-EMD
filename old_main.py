import os
from collections import defaultdict

import numpy as np
import pandas as pd

from data import get_data, to_patches, patches_to_image
from metrics import emd
from utils import get_nns, dump_images, get_centroids


def batch_wassestein(train_data, test_data, batch_size, d, c, n_exp, tag):
    mean_train_image_batch = np.mean(train_data, axis=0, keepdims=True).repeat(batch_size, 0)

    os.makedirs("batch_wassestein", exist_ok=True)
    centroids_path = f"batch_wassestein/centroids-{tag}-{batch_size}.npy"
    centroids = get_centroids(train_data, batch_size, centroids_path)

    loss_dict = defaultdict(list)

    for i in range(n_exp):
        print("Rep:", i)
        test_indices = np.random.choice(np.arange(len(test_data)), size=batch_size, replace=False)
        test_batch = test_data[test_indices]
        train_indices = np.random.choice(np.arange(len(train_data)), size=batch_size, replace=False)

        batches = {f"train_batch": train_data[train_indices], f"mean": mean_train_image_batch, "centroids": centroids}

        for name, batch in batches.items():
            loss_dict[f'emd-{name}'].append(emd(batch, test_batch))
            # loss_dict[f'swd-{name}'].append(swd(batch, test_batch))
            # loss_dict[f'mmd-rbf-{name}'].append(mmd_rbf(batch, test_batch))
            # loss_dict[f'sinkhorn-10-{name}'].append(sinkhorn(batch, test_batch, reg=10))
            # loss_dict[name]['mmd-rbf-1'].append(mmd_rbf(batch, test_batch))

            x = to_patches(batch, d, c, p=16, s=16)
            y = to_patches(test_batch, d, c, p=16, s=16)
            loss_dict[f'patch_emd-16-16-{name}'].append(emd(x, y))
            # loss_dict[f'patch_swd-5-5-{name}'].append(swd(x, y))
            # loss_dict[f'patch_mmd-5-5-{name}'].append(mmd_linear(x, y))
            # loss_dict[f'mmd-rbf-5-5-{name}'].append(mmd_rbf(x, y))

            # x = to_patches(batch, s, 8, 8)
            # y = to_patches(test_batch, s, 8, 8)
            # loss_dict[f'patch_emd-8-8-{name}'].append(emd(x, y))
            # loss_dict[f'patch_swd-8-8-{name}'].append(swd(x, y))
            # loss_dict[f'patch_mmd-8-8-{name}8'].append(mmd_linear(x, y))

            dump_images(batch, batch_size, d, c, f"batch_wassestein/{name}-{batch_size}.png")
        # dump_images(test_batch, b, s, f"test_batch-{b}.png")

    loss_dict = {k: np.mean(v) for k,v in loss_dict.items()}

    return loss_dict


def patch_prior(train_data, test_data, d, c, tag, p=16, s=16):
    os.makedirs("patch_prior", exist_ok=True)
    pb = 1024
    patch_centroids_path = f"patch_prior/patch_centroids-{tag}-{pb}.npy"
    patches = to_patches(train_data, d, c, p=p, s=s)
    patch_centroids = get_centroids(patches, pb, patch_centroids_path)

    for i in range(5):
        img = test_data[i]
        img_patches = to_patches(img[None,], d, c, p, s)
        new_patches = get_nns(queries=img_patches, refs=patch_centroids)

        new_patches = new_patches - new_patches.mean(1)[:, None] + img_patches.mean(1)[:, None]
        new_img = patches_to_image(new_patches, d, c, p, s)

        dump_images(img, 1, d, c, f"patch_prior/img-{i}.png")
        dump_images(new_img, 1, d, c, f"patch_prior/rec-{i}.png")

def main():
    data_path = '/mnt/storage_ssd/datasets/FFHQ_128'
    d = 128
    gray = True
    normalize_data = False
    c = 1 if gray else 3
    tag = f"{d}x{d}{'_Gray' if gray else ''}{'_Normalized' if normalize_data else ''}"

    print("Loading data...", end='')
    data = get_data(data_path, resize=d, limit_data=3*1024, gray=gray, normalize_data=normalize_data)
    print("done")

    test_data = data[:1025]
    train_data = data[1025:]

    patch_prior(train_data, test_data, d, c, tag)

    # res_dict = dict()
    # for batch_size in [2**i for i in range(3, 10)]:
    #     res_dict[batch_size] = batch_wassestein(train_data, test_data, batch_size, d, c, n_exp=1, tag=tag)
    #     df = pd.DataFrame(res_dict)
    #     df.to_csv(f"results-{tag}.csv")





if __name__ == '__main__':
    main()