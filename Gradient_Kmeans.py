import os
from math import sqrt
import numpy as np
import torch
from torchvision.utils import save_image

from data import get_data
from metrics import emd
import torch.nn.functional as F

def efficient_compute_distances(X, Y):
    """
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e (X[:, None] - Y[None, :])**2
    Get the nearest neighbor index from Y for each X
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    d = X.shape[1]
    dist /= d # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
    return dist


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


def extract_patches(images, patch_size, stride):
    """
    Splits the image to overlapping patches and returns a pytorch tensor of size (N_patches, 3*patch_size**2)
    """
    channels = images.shape[1]
    patches = F.unfold(images, kernel_size=patch_size, stride=stride) # shape (b, 3*p*p, N_patches)
    patches = patches.permute(0, 2, 1).reshape(-1, channels * patch_size**2)
    return patches


def compute_n_patches_in_image(c, d, p, s):
    dummy_img = torch.zeros((1, c, d, d))
    dummy_patches = F.unfold(dummy_img, kernel_size=p, stride=s)
    patches_per_image = dummy_patches.shape[-1]  # shape (1, c*p*p, N_patches)
    return patches_per_image


def patches_to_image(patches, d, c, p, s):
    patches_per_image = compute_n_patches_in_image(c, d, p, s)

    patches = patches.reshape(-1, patches_per_image, c * p ** 2)
    patches = patches.permute(0, 2, 1)
    imgs = F.fold(patches, (d, d), kernel_size=p, stride=s)

    if s < p:
        # normal fold matrix
        input_ones = torch.ones_like(imgs)
        divisor = F.unfold(input_ones, kernel_size=p, stride=s)
        divisor = F.fold(divisor, output_size=imgs.shape[-2:], kernel_size=p, stride=s)

        divisor[divisor == 0] = 1.0
        return imgs / divisor

    return imgs


def init_patch_kmeans(data, data_patches, n_images,  init_mode):
    if init_mode == "data":
        clusters_images = data[torch.randperm(len(data))[:n_images]]
    elif init_mode == "mean":
        clusters_images = torch.mean(data, dim=0, keepdims=True).repeat(n_images, 1,1,1)
    elif init_mode == "rand":
        clusters_images = torch.randn((n_images, *data.shape[1:]))
    elif init_mode == "rand_assignment":
        n = compute_n_patches_in_image(c, d, p, s) * n_images
        rand_assignmets = torch.randint(n, size=(len(data_patches),))
        patches = torch.stack([data_patches[rand_assignmets == j].mean(0) for j in torch.unique(rand_assignmets)])
        clusters_images = patches_to_image(patches, d, c, p, s)
    return clusters_images


def train_gradient_kmeans(data, n_images):

    data_patches = extract_patches(data, p, s)
    clusters_images = init_patch_kmeans(data, data_patches, n_images, init_mode).to(device)
    losses = []
    emds = []
    x = clusters_images.requires_grad_()
    opt = torch.optim.Adam([x], lr=0.1)
    for i in range(n_steps):
        save_image(x, f"{out_dir}/clusters-{i}.png", normalize=True, nrow=int(sqrt(n_images)))
        x_patches = extract_patches(x, p, s)
        with torch.no_grad():
            assignments = get_NN_indices_low_memory(data_patches, x_patches, b=nn_batch_size)
            means = x_patches.clone()
            for j in range(len(x_patches)):
                if torch.any(assignments == j):
                    means[j] = data_patches[assignments == j].mean(0)
        loss = torch.norm(x_patches - means)
        opt.zero_grad()
        loss.backward()
        opt.step()


        losses.append(loss.item())
        print(f"iter-{i}: Loss {loss.item()}")



if __name__ == '__main__':
    device = torch.device("cuda:1")
    # device = torch.device("cpu")
    data_path = '/cs/labs/yweiss/ariel1/data/FFHQ_128'
    d = 64 
    p, s = 16, 8
    init_mode = "rand_assignment"
    gray = False
    normalize_data = True
    c = 1 if gray else 3
    limit_data = 10240
    n_images = 16
    nn_batch_size = 128
    n_steps = 1000

    out_dir = f"Kmeans_I-{init_mode}_D-{d}_P-{p}_S-{s}"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...", end='')
    data = get_data(data_path, resize=d, limit_data=limit_data, gray=gray, normalize_data=normalize_data)
    data = torch.from_numpy(data.reshape(len(data), d, d, c)).permute(0, 3, 1, 2).to(device)

    train_gradient_kmeans(data, n_images)
