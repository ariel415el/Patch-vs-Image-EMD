import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T


def get_data(data_path, im_size=None, limit_data=10000, gray=False, normalize_data=True, flatten=True):
    print("Loading data...", end='')

    """Read 'n_images' random images"""
    image_paths = sorted([os.path.join(data_path, x) for x in os.listdir(data_path)])[:limit_data]

    transforms = [T.ToTensor(), T.Normalize((0.5,), (0.5,))]
    if im_size is not None:
        transforms = [T.Resize(im_size, antialias=True)] + transforms

    transforms = T.Compose(transforms)

    data = []
    for i, path in enumerate(tqdm(image_paths)):
        im = Image.open(path)
        im = transforms(im)
        data.append(im)

    data = torch.stack(data, dim=0)

    if gray:
        data = torch.mean(data, dim=1, keepdim=True)

    if flatten:
        data = data.reshape(len(data), -1)

        if normalize_data:
            data = data - torch.mean(data, dim=1, keepdim=True)
            data /= torch.std(data, dim=1, keepdim=True)

    assert not ((not flatten) and normalize_data), "Not implemented"

    print("done")

    return data

