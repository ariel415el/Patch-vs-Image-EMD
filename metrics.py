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
    name = 'emd'

    def __call__(self, x, y):
        uniform_x = np.ones(len(x)) / len(x)
        uniform_y = np.ones(len(y)) / len(y)
        M = ot.dist(x, y) / x.shape[1]
        # from utils import compute_distances_batch
        # M = compute_distances_batch(x, y, b=1024)
        return ot.emd2(uniform_x, uniform_y, M)


class sinkhorn:
    def __init__(self, reg=100):
        self.name = f'sinkhorn-{reg}'
        self.reg = reg

    def __call__(self, x, y):
        uniform_x = np.ones(len(x)) / len(x)
        uniform_y = np.ones(len(y)) / len(y)
        M = ot.dist(x, y) / x.shape[1]
        return ot.sinkhorn2(uniform_x, uniform_y, M, reg=self.reg)

class mmd_linear:
    name = 'mmd_linear'

    def __call__(self, x, y):
        delta = x.mean(0) - y.mean(0)
        return delta.dot(delta.T)

class mmd_rbf:
    def __init__(self, gamma=1.0):
        self.name = f'mmd_rbf-{gamma}'
        self.gamma = gamma

    def __call__(self, X, Y):
        from sklearn import metrics
        XX = metrics.pairwise.rbf_kernel(X, X, self.gamma)
        YY = metrics.pairwise.rbf_kernel(Y, Y, self.gamma)
        XY = metrics.pairwise.rbf_kernel(X, Y, self.gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()
