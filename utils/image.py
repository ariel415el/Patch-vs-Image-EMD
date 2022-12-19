import os

import numpy as np
from torchvision.utils import save_image
import torch
from torch.nn import functional as F

def dump_images(imgs, b, d, c, fname):
    save_image(torch.from_numpy(imgs).reshape(b, d, d, c).permute(0,3,1,2), fname, normalize=True, nrow=int(np.sqrt(b)))

def compute_n_patches_in_image(d, c, p, s):
    dummy_img = torch.zeros((1, c, d, d))
    dummy_patches = F.unfold(dummy_img, kernel_size=p, stride=s)
    patches_per_image = dummy_patches.shape[-1]  # shape (1, c*p*p, N_patches)
    return patches_per_image


def to_patches(x, d, c, p=8, s=4):
    xp = x.reshape(-1, d, d, c).permute(0, 3, 1, 2)  # shape  (b,c,d,d)
    if type(x) == np.ndarray:
        xp = torch.from_numpy(xp)
    patches = F.unfold(xp, kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    patches = patches.permute(0, 2, 1)               # shape (b, N_patches, c*p*p)
    patches = patches.reshape(-1, patches.shape[-1]) # shape (b, N_patches * c*p*p)

    if type(x) == np.ndarray:
        patches = patches.numpy()
    return patches


def patches_to_image(patches, d, c, p=8, s=4):
    patches_per_image = compute_n_patches_in_image(d, c, p, s)

    was_ndarray = type(patches) == np.ndarray

    if was_ndarray:
        patches = torch.from_numpy(patches)
    patches = patches.reshape(-1, patches_per_image, c * p ** 2)
    patches = patches.permute(0, 2, 1)
    img = F.fold(patches, (d, d), kernel_size=p, stride=s)
    if was_ndarray:
        img = img.permute(0, 2, 3, 1).numpy()
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
