import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image(im_path, resize=None, gray=False):
    im = Image.open(im_path).resize((resize, resize))
    im = np.array(im).astype(np.float32) / 255.
    if gray:
        im = np.mean(im, axis=-1)
    return im


def get_data(data_path, resize=None, limit_data=10000, gray=False, normalize_data=True):
    """Read 'n_images' random images"""
    image_paths = sorted([os.path.join(data_path, x) for x in os.listdir(data_path)])[:limit_data]
    data = []
    for i, path in enumerate(tqdm(image_paths)):
        im = load_image(path, resize=resize, gray=gray)
        data.append(im)
    data = np.stack(data, axis=0)

    data = data.reshape(len(data), -1)
    if normalize_data:
        # data = (data - data.mean(0)) / data.std(0)
        data = data - np.mean(data, axis=1, keepdims=True)
        data /= np.std(data, axis=1, keepdims=True)

    return data

