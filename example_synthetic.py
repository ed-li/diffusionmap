import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import mahalanobis
from diffusionmap import DiffusionMap

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.cm as cm


if __name__ == '__main__':
    # Load data
    covariance = [[100, 0.5, 0.5],
                  [0.5, 50, 0.5],
                  [0.5, 0.5, 10]]
    data = np.concatenate([np.random.multivariate_normal([0,0,0], covariance, 1000),
                           np.random.multivariate_normal([100,0,0], covariance, 1000),
                           np.random.multivariate_normal([0,50,0], covariance, 1000)])

    # Plot luminance
    plt.imshow(data.reshape((60, 50, 3)), origin='lower')
    plt.savefig('synthetic.png')
    plt.show()

    # # Diffusion map clustering based on Euclidean distances
    # e_dm = DiffusionMap(data, kernel_params={'eps': 1e3}, neighbors=500)
    # e_w, e_v = e_dm.map(3)
    #
    # kmeans = KMeans(n_clusters=3, n_init=100)
    # kmeans.fit(e_v)
    # e_y = kmeans.predict(e_v)
    #
    # plt.imshow(e_y.reshape((60, 50)), origin='lower')
    # plt.savefig('synthetic_euclidean.png')
    # plt.show()
    #
    # # Diffusion map clustering based on Mahalanobis distances with overall covariances
    # inv_cov = np.linalg.inv(np.cov(data, rowvar=False))
    # def mdistance(x, y):
    #     return mahalanobis(x, y, VI=inv_cov)
    # m_dm = DiffusionMap(data, kernel_params={'eps': 1e3, 'distance': mdistance}, neighbors=500)
    # m_w, m_v = m_dm.map(3)
    #
    # kmeans = KMeans(n_clusters=3, n_init=100)
    # kmeans.fit(m_v)
    # m_y = kmeans.predict(m_v)
    #
    # plt.imshow(m_y.reshape((60, 50)), origin='lower')
    # plt.savefig('synthetic_mahalanobis.png')
    # plt.show()

    # Diffusion map clustering based on Mahalanobis distances with local covariances
    lm_dm = DiffusionMap(data, kernel_params={'eps': 1e3}, neighbors=500)
    lm_w, lm_v = lm_dm.map(3, local_mahalanobis=True, clusters=25)

    kmeans = KMeans(n_clusters=3, n_init=100)
    kmeans.fit(lm_v)
    lm_y = kmeans.predict(lm_v)

    plt.imshow(lm_y.reshape((60, 50)), origin='lower')
    plt.savefig('synthetic_local_gmm_mahalanobis.png')
    plt.show()

