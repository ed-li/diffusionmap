import numpy as np
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import euclidean
from covertree import CoverTree


def gaussian_kernel(x, y, **kernel_params):
    eps = kernel_params.get('eps', 1.0)
    distance = kernel_params.get('distance', euclidean)
    return np.exp(-((distance(x, y))**2)/eps)


class DiffusionMap:
    def __init__(self, data, kernel=gaussian_kernel, kernel_params={}, neighbors=10, eps=1e-6):
        if isinstance(data, str):
            data = np.load(data)

        self.data = data
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.neighbors = neighbors
        self.eps = eps
        self.P = None

    def _compute_matrix(self):
        if self.P is not None:
            return

        data = self.data
        N = len(data)
        P = np.zeros((N, N), float)
        index = range(N)

        tree = CoverTree(data, self.kernel_params.get('distance', euclidean))
        near_points = tree.query(data, self.neighbors, self.eps)

        for i in index:
            x = data[i]
            for j in near_points[1][i]:
                P[i, j] = self.kernel(x, data[j], **self.kernel_params)

        for i in index:
            for j in range(i+1, N):
                P[i, j] = P[j, i]

        self.P = (P.T / P.sum(axis=1)).T.copy()

    def map(self, dimensions=2, time=1):
        self._compute_matrix()

        values, vectors = eigs(self.P, k=dimensions+1)

        values = values[1:dimensions+1].real
        vectors = (values.real**time)*np.array(vectors[:, 1:dimensions+1].real.astype(float))

        return values, vectors
