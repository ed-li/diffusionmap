import numpy as np
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import mahalanobis
from scipy.spatial import KDTree
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt


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

        tree = KDTree(data)
        near_points = tree.query(data, self.neighbors, self.eps)

        for i in index:
            x = data[i]
            for j in near_points[1][i]:
                P[i, j] = self.kernel(x, data[j], **self.kernel_params)

        for i in index:
            for j in range(i+1, N):
                P[i, j] = P[j, i]

        self.P = (P.T / P.sum(axis=1)).T.copy()
        print(self.P)
        plt.imshow(self.P)
        plt.show()
        print(self.P.shape)
        print("completed diffusion")

    # def _compute_matrix_local_mahalanobis_old(self):
    #     if self.P is not None:
    #         return
    #
    #     data = self.data
    #     N = len(data)
    #     P = np.zeros((N, N), float)
    #     index = range(N)
    #
    #     tree = KDTree(data)
    #     near_points = tree.query(data, self.neighbors, self.eps)
    #     nearer_points = tree.query(data, 100, self.eps)
    #     inv_cov = [np.linalg.inv(np.cov(data[nearer_points[1][i]], rowvar=False)) for i in index]
    #
    #     for i in index:
    #         x = data[i]
    #         x_inv_cov = inv_cov[i]
    #         for j in near_points[1][i]:
    #             P[i, j] = np.exp(-((mahalanobis(x, data[j], x_inv_cov + inv_cov[j])) ** 2) / self.kernel_params.get('eps', 1.0))
    #
    #     for i in index:
    #         for j in range(i + 1, N):
    #             P[i, j] = P[j, i]
    #
    #     self.P = (P.T / P.sum(axis=1)).T.copy()

    def _compute_matrix_local_mahalanobis(self, clusters):
        if self.P is not None:
            return

        data = self.data
        N = len(data)
        P = np.zeros((N, N), float)
        index = range(N)

        tree = KDTree(data)
        near_points = tree.query(data, self.neighbors, self.eps)

        gmm = GaussianMixture(n_components=clusters)
        gmm.fit(data)
        print("fitted GMM")
        labels = gmm.predict(data)
        print("predicted labels")
        print(gmm.covariances_)
        print(np.shape(gmm.covariances_))

        # inv_cov = [np.linalg.inv(cov) for cov in gmm.covariances_]

        inv_cov = [None] * clusters
        for i in range(len(gmm.covariances_)):
            inv_cov[i] = np.linalg.inv(gmm.covariances_[i])
            print(i)

        for i in index:
            x = data[i]
            x_inv_cov = inv_cov[labels[i]]
            for j in near_points[1][i]:
                P[i, j] = np.exp(-((mahalanobis(x, data[j], x_inv_cov + inv_cov[labels[j]])) ** 2) / self.kernel_params.get('eps', 1.0))

        for i in index:
            for j in range(i + 1, N):
                P[i, j] = P[j, i]

        self.P = (P.T / P.sum(axis=1)).T.copy()
        print(self.P)
        plt.imshow(self.P)
        plt.show()
        print(self.P.shape)
        print("completed diffusion")

    def map(self, dimensions=2, time=1, local_mahalanobis=False, clusters=10):
        if local_mahalanobis:
            self._compute_matrix_local_mahalanobis(clusters)
        else:
            self._compute_matrix()

        print("passed through diffusion")
        values, vectors = eigs(self.P, k=dimensions+1)
        print("calculated eigs")

        values = values[1:dimensions+1].real
        print("assigned values")
        vectors = (values.real**time)*np.array(vectors[:, 1:dimensions+1].real.astype(float))
        print("assigned vectors")

        return values, vectors
