import numpy as np
import ot


class swd:
    def __init__(self, num_proj=1024):
        self.name = f'swd-{num_proj}'
        self.num_proj = num_proj

    def __call__(self, x, y):
        b, c = x.shape[0], np.prod(x.shape[1:])

        # Sample random normalized projections
        rand = np.random.randn(self.num_proj, c)
        rand = rand / np.linalg.norm(rand, axis=1, keepdims=True)  # noramlize to unit directions

        # Sort and compute L1 loss
        projx = x @ rand.T
        projy = y @ rand.T

        projx = np.sort(projx, axis=0)
        projy = np.sort(projy, axis=0)

        loss = np.abs(projx - projy).mean()

        return loss


class emd:
    def __init__(self, n_samples=None):
        self.n_samples = n_samples
        self.name = f'emd_n-{n_samples}'

    def emd(self, x, y):
        uniform_x = np.ones(len(x)) / len(x)
        uniform_y = np.ones(len(y)) / len(y)
        M = ot.dist(x, y) / x.shape[1]
        # from utils import compute_distances_batch
        # M = compute_distances_batch(x, y, b=1024)
        return ot.emd2(uniform_x, uniform_y, M)

    def __call__(self, x, y):
        if self.n_samples is not None:
            samples_x = np.random.choice(len(x), size=min(len(x), self.n_samples), replace=False)
            samples_y = np.random.choice(len(y), size=min(len(y), self.n_samples), replace=False)
            return self.emd(x[samples_x], y[samples_y])
        else:
            return self.emd(x, y)





