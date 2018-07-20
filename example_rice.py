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
    plt.imshow(data.mean(axis=1).reshape((93, 56)), cmap='gray', origin='lower')
    plt.savefig('rice_luminance.png')
    plt.show()

    # # Diffusion map clustering based on Euclidean distances
    # e_dm = DiffusionMap(data, kernel_params={'eps': 1}, neighbors=500)
    # e_w, e_v = e_dm.map(3)
    #
    # kmeans = KMeans(n_clusters=3, n_init=100)
    # kmeans.fit(e_v)
    # e_y = kmeans.predict(e_v)
    #
    # plt.imshow(e_y.reshape((93, 56)), origin='lower')
    # plt.savefig('rice_euclidean.png')
    # plt.show()
    #
    # # Diffusion map clustering based on Mahalanobis distances with overall covariances
    # inv_cov = np.linalg.inv(np.cov(data, rowvar=False))
    # def mdistance(x, y):
    #     return mahalanobis(x, y, VI=inv_cov)
    # m_dm = DiffusionMap(data, kernel_params={'eps': 1, 'distance': mdistance}, neighbors=500)
    # m_w, m_v = m_dm.map(3)
    #
    # kmeans = KMeans(n_clusters=3, n_init=100)
    # kmeans.fit(m_v)
    # m_y = kmeans.predict(m_v)
    #
    # plt.imshow(m_y.reshape((93, 56)), origin='lower')
    # plt.savefig('rice_mahalanobis.png')
    # plt.show()

    # Diffusion map clustering based on Mahalanobis distances with local covariances
    lm_dm = DiffusionMap(data, kernel_params={'eps': 1}, neighbors=500)
    lm_w, lm_v = lm_dm.map(3, local_mahalanobis=True, clusters=25)

    kmeans = KMeans(n_clusters=3, n_init=100)
    kmeans.fit(lm_v)
    lm_y = kmeans.predict(lm_v)

    plt.imshow(lm_y.reshape((93, 56)), origin='lower')
    plt.savefig('rice_local_gmm_mahalanobis.png')
    plt.show()
