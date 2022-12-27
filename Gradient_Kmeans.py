import os
from math import sqrt
import torch
from PIL import Image
from tqdm import tqdm

import wandb
from torchvision.utils import save_image, make_grid

from utils.data import get_data
from metrics import emd

from utils.image import compute_n_patches_in_image, patches_to_image, to_patches
from utils.k_means import get_patch_centroids
from utils.nns import get_NN_indices_low_memory


def get_log_image(tensor):
    grid = make_grid(tensor.reshape(-1, c, d, d), normalize=True)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(ndarr)


def init_patch_kmeans(data, data_patches, n_images, init_mode, p, s):
    if init_mode == "data":
        clusters_images = data[torch.randperm(len(data))[:n_images]]
    elif init_mode == "mean":
        clusters_images = torch.mean(data, dim=0, keepdims=True)
        clusters_images = clusters_images.repeat(n_images, 1, 1, 1)
    elif init_mode == "rand":
        clusters_images = torch.randn((n_images, *data.shape[1:]))
    elif init_mode == "rand_assignment":
        n = compute_n_patches_in_image(d, c, p, s) * n_images
        rand_assignmets = torch.cat([torch.randint(n, size=(len(data_patches) - n,)), torch.arange(n)]) # Ensure each patch is initiated
        rand_assignmets = rand_assignmets[torch.randperm(len(rand_assignmets))]
        patches = torch.stack([data_patches[rand_assignmets == j].mean(0) for j in range(n)])
        clusters_images = patches_to_image(patches, d, c, p, s)
    return clusters_images


def get_loss(data_patches, patch_centroids):
    assignments = get_NN_indices_low_memory(data_patches, patch_centroids.detach(), b=128)
    loss = 0
    for j in range(len(patch_centroids)):
        if torch.any(assignments == j):
            new_centroid = data_patches[assignments == j].mean(0)
            loss += ((patch_centroids[j] - new_centroid) ** 2).mean()
    return loss


def train_gradient_kmeans(data, n_images, init_mode, p, s, lr=0.01, n_steps=300):
    data_patches = to_patches(data, d, c, p, s)
    clusters_images = init_patch_kmeans(data, data_patches, n_images, init_mode, p, s).to(device)
    x = clusters_images.requires_grad_()
    opt = torch.optim.Adam([x], lr=lr)
    pbar = tqdm(range(n_steps))
    for i in pbar:
        save_image(x.reshape(-1, c, d, d), f"{out_dir}/clusters-{i}.png", normalize=True, nrow=int(sqrt(n_images)))
        x_patches = to_patches(x, d, c, p, s)

        loss = get_loss(data_patches, x_patches)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if log:
            # wandb.log({"examples": wandb.Image(get_log_image(x), caption="Top: Output, Bottom: Input")})
            wandb.log({"Loss": loss.item(),
                       "EMD-256-patches": emd(n_samples=256)(x_patches.detach().cpu().numpy(),
                                                             data_patches.detach().cpu().numpy())})

        pbar.set_description(f"iter-{i}: Loss {loss.item()}")

        # if i % 50:
        #     for g in opt.param_groups:
        #         g['lr'] *= 0.9

    return to_patches(x.detach(), d, c, p, s)


def load_dataset(data_path, limit_data, gray, normalize_data):
    print("Loading data...", end='')
    data = get_data(data_path, im_size=d, limit_data=limit_data, gray=gray, normalize_data=normalize_data).to(device)
    print(f"{len(data)} samples loaded", )
    return data


def create_dataset_from_image(data_path, limit_data, gray, normalize_data):
    data = get_data(data_path, im_size=None, limit_data=limit_data, gray=gray, normalize_data=normalize_data, flatten=False).to(device)
    data = torch.nn.functional.unfold(data, kernel_size=d, stride=d//2)  # shape (b, c*p*p, N_patches)
    data = data.permute(0, 2, 1)  # shape (b, N_patches, c*p*p)
    data = data.reshape(-1, data.shape[-1])  # shape (b * N_patches, c*p*p)
    data = data[torch.randperm(len(data))[:limit_data]]

    return data


if __name__ == '__main__':
    device = torch.device("cuda:0")
    # data_path = '/cs/labs/yweiss/ariel1/data/FFHQ_128'
    d = 64
    log = False
    gray = False
    normalize_data = False
    c = 1 if gray else 3
    limit_data = 128
    p = 7
    s = 1
    n_images = 8
    lr = 0.02
    n_steps = 300

    for data_path in [
                      'image_sample/12.jpg'
                      ]:
        data = create_dataset_from_image(data_path, limit_data, gray, normalize_data)
        for init_mode in [
                        # "data",
                        "rand_assignment"
            ]:
            out_dir = f"run_I-{init_mode}_D-{d}_P-{p}_S-{s}_{os.path.basename(data_path)}"
            os.makedirs(out_dir, exist_ok=True)
            print(out_dir)
            if log:
                wandb.init(project="Patch-EMD-experiments", name=out_dir, reinit=True)

            GKmeans_centroids = train_gradient_kmeans(data, n_images, init_mode, p, s, lr, n_steps)

            Kmeans_centroids = torch.from_numpy(get_patch_centroids(data.cpu().numpy(), n_images, d, c, p, s, out_dir)).to(device)

            # Evaluate
            data_patches = to_patches(data, d, c, p, s=p)
            GKmeans_centroids_loss = get_loss(data_patches, GKmeans_centroids).item()
            Kmeans_centroids_loss = get_loss(data_patches, Kmeans_centroids).item()
            print('GK-means final loss: ', GKmeans_centroids_loss)
            print('K-means final loss: ', Kmeans_centroids_loss)
            if log:
                wandb.log({'GKmeans_centroids_loss': GKmeans_centroids_loss, "Kmeans_centroids_loss": Kmeans_centroids_loss})

            GKmeans_centroids = patches_to_image(GKmeans_centroids, d, c, p, s).reshape(-1, c, d, d)
            kmeans_centroids = patches_to_image(Kmeans_centroids, d, c, p, s).reshape(-1, c, d, d)
            save_image(GKmeans_centroids, f"{out_dir}/GKmeans_centroids.png", normalize=True,
                       nrow=int(sqrt(len(GKmeans_centroids))))
            save_image(kmeans_centroids, f"{out_dir}/kmeans_centroids.png", normalize=True,
                       nrow=int(sqrt(len(kmeans_centroids))))
            save_image(data.reshape(-1, c, d, d), f"{out_dir}/data.png", normalize=True, nrow=int(sqrt(len(data))))
