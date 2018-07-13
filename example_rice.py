import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import mahalanobis
from diffusionmap import DiffusionMap

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.cm as cm


if __name__ == '__main__':
    # Load data
    data = np.loadtxt('rice.txt')

    # Plot luminance
    plt.imshow(data.mean(axis=1).reshape((93,56)), cmap='gray', origin='lower')
    plt.show()

    # Diffusion map clustering based on Euclidean distances
    e_dm = DiffusionMap(data, kernel_params={'eps': 1}, neighbors=100)
    e_w, e_v = e_dm.map(3)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(e_v)
    e_y = kmeans.predict(e_v)

    plt.imshow(e_y.reshape((93,56)), origin='lower')
    plt.show()

    # Diffusion map clustering based on Mahalanobis distances with overall covariances
    inv_cov = np.linalg.inv(np.cov(data, rowvar=False))
    def mdistance(x, y):
        return mahalanobis(x, y, inv_cov)
    m_dm = DiffusionMap(data, kernel_params={'eps': 1e9, 'distance': mdistance}, neighbors=100)
    m_w, m_v = m_dm.map(3)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(m_v)
    m_y = kmeans.predict(m_v)

    plt.imshow(m_y.reshape((93,56)), origin='lower')
    plt.show()