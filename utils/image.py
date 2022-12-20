import os

import numpy as np
from torchvision.utils import save_image
import torch
from torch.nn import functional as F


def dump_images(imgs, b, d, c, fname):
    if type(imgs) == np.ndarray:
        imgs = torch.from_numpy(imgs)
    save_image(imgs.reshape(b, c, d, d), fname, normalize=True, nrow=int(np.sqrt(b)))


def compute_n_patches_in_image(d, c, p, s):
    dummy_img = torch.zeros((1, c, d, d))
    dummy_patches = F.unfold(dummy_img, kernel_size=p, stride=s)
    patches_per_image = dummy_patches.shape[-1]  # shape (1, c*p*p, N_patches)
    return patches_per_image


def to_patches(x, d, c, p=8, s=4):
    xp = x.reshape(-1, c, d, d)  # shape  (b,c,d,d)
    is_np = type(x) == np.ndarray
    if is_np:
        xp = torch.from_numpy(xp)
    patches = F.unfold(xp, kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    patches = patches.permute(0, 2, 1)               # shape (b, N_patches, c*p*p)
    patches = patches.reshape(-1, patches.shape[-1]) # shape (b, N_patches * c*p*p)

    if is_np:
        patches = patches.numpy()
    return patches


def patches_to_image(patches, d, c, p=8, s=4):
    patches_per_image = compute_n_patches_in_image(d, c, p, s)

    is_ndarray = type(patches) == np.ndarray

    if is_ndarray:
        patches = torch.from_numpy(patches)
    patches = patches.reshape(-1, patches_per_image, c * p ** 2)
    patches = patches.permute(0, 2, 1)
    img = F.fold(patches, (d, d), kernel_size=p, stride=s)

    # normal fold matrix
    input_ones = torch.ones((1, c, d, d), dtype=patches.dtype, device=patches.device)
    divisor = F.unfold(input_ones, kernel_size=p, dilation=(1, 1), stride=s, padding=(0, 0))
    divisor = F.fold(divisor, output_size=(d, d), kernel_size=p, stride=s)

    divisor[divisor == 0] = 1.0
    return (img / divisor).squeeze(dim=0).unsqueeze(0)


    return img

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
