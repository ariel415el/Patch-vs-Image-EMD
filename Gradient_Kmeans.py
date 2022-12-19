import os
from math import sqrt
import numpy as np
import torch
from PIL import Image

import wandb
from torchvision.utils import save_image, make_grid

from data import get_data
from metrics import emd
import torch.nn.functional as F

from utils.image import compute_n_patches_in_image, patches_to_image, to_patches
from utils.nns import get_NN_indices_low_memory


def get_log_image(tensor):
    grid = make_grid(tensor.reshape(-1, d, d, c).permute(0, 3, 1, 2), normalize=True)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(ndarr)

def init_patch_kmeans(data, data_patches, n_images, init_mode):
    if init_mode == "data":
        clusters_images = data[torch.randperm(len(data))[:n_images]]
    elif init_mode == "mean":
        clusters_images = torch.mean(data, dim=0, keepdims=True)
        clusters_images = clusters_images.repeat(n_images, 1, 1, 1)
    elif init_mode == "rand":
        clusters_images = torch.randn((n_images, *data.shape[1:]))
    elif init_mode == "rand_assignment":
        n = compute_n_patches_in_image(d, c, p, s) * n_images
        rand_assignmets = torch.randint(n, size=(len(data_patches),))
        patches = torch.stack([data_patches[rand_assignmets == j].mean(0) for j in torch.unique(rand_assignmets)])
        clusters_images = patches_to_image(patches, d, c, p, s).permute(0,2,3,1)
    return clusters_images


def train_gradient_kmeans(data, n_images):
    data_patches = to_patches(data, d, c, p, s)
    clusters_images = init_patch_kmeans(data, data_patches, n_images, init_mode).to(device)
    x = clusters_images.requires_grad_()
    opt = torch.optim.Adam([x], lr=0.001)
    for i in range(n_steps):
        save_image(x.reshape(-1, d, d, c).permute(0, 3, 1, 2), f"{out_dir}/clusters-{i}.png", normalize=True, nrow=int(sqrt(n_images)))

        x_patches = to_patches(x, d, c, p, s)
        with torch.no_grad():
            assignments = get_NN_indices_low_memory(data_patches, x_patches, b=nn_batch_size)
            means = x_patches.clone()
            for j in range(len(x_patches)):
                if torch.any(assignments == j):
                    means[j] = data_patches[assignments == j].mean(0)
        # loss = torch.mean(torch.abs(x_patches - means))
        loss = torch.mean((x_patches - means)**2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # wandb.log({"examples": wandb.Image(get_log_image(x), caption="Top: Output, Bottom: Input")})
        wandb.log({"Loss": loss.item(),
                   "EMD-256-patches":emd(n_patches=256)(x_patches.detach().cpu().numpy(), data_patches.detach().cpu().numpy())})

        print(f"iter-{i}: Loss {loss.item()}")


if __name__ == '__main__':

    device = torch.device("cpu")
    # device = torch.device("cpu")
    data_path = '/mnt/storage_ssd/datasets/FFHQ_128/FFHQ_128'
    d = 64
    p, s = 8, 4
    init_mode = "mean"
    gray = False
    normalize_data = False
    c = 1 if gray else 3
    limit_data = 1024
    n_images = 16
    nn_batch_size = 128
    n_steps = 1000

    out_dir = f"Kmeans_I-{init_mode}_D-{d}_P-{p}_S-{s}"
    os.makedirs(out_dir, exist_ok=True)

    wandb.init(project="Patch-EMD-experiments", name=out_dir)

    print("Loading data...", end='')
    data = get_data(data_path, resize=d, limit_data=limit_data, gray=gray, normalize_data=normalize_data)
    data = torch.from_numpy(data).to(device)

    train_gradient_kmeans(data, n_images)
