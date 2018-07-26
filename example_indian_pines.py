import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import mahalanobis
from diffusionmap import DiffusionMap

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.cm as cm


if __name__ == '__main__':
    # Load data
    data = np.loadtxt('indian_pines_corrected.txt', delimiter=',')[13775:]

    # Plot luminance
    plt.imshow(data.mean(axis=1).reshape((50, 145)), cmap='gray', origin='lower')
    plt.axis('off')
    plt.savefig('indian_pines_luminance.png', bbox_inches='tight')
    plt.show()

    # Diffusion map clustering based on Euclidean distances
    e_dm = DiffusionMap(data, kernel_params={'eps': 1e7}, neighbors=250)
    e_w, e_v = e_dm.map(10, 30)

    kmeans = KMeans(n_clusters=8)
    kmeans.fit(e_v)
    e_y = kmeans.predict(e_v)

    plt.imshow(e_y.reshape((50, 145)), origin='lower')
    plt.axis('off')
    plt.savefig('indian_pines_euclidean.png', bbox_inches='tight')
    plt.show()

    # Diffusion map clustering based on Mahalanobis distances with overall covariances
    inv_cov = np.linalg.inv(np.cov(data, rowvar=False))
    def mdistance(x, y):
        return mahalanobis(x, y, VI=inv_cov)
    m_dm = DiffusionMap(data, kernel_params={'eps': 1e7, 'distance': mdistance}, neighbors=250)
    m_w, m_v = m_dm.map(10, 30)

    kmeans = KMeans(n_clusters=8)
    kmeans.fit(m_v)
    m_y = kmeans.predict(m_v)

    plt.imshow(m_y.reshape((50, 145)), origin='lower')
    plt.axis('off')
    plt.savefig('indian_pines_mahalanobis.png', bbox_inches='tight')
    plt.show()

    # Diffusion map clustering based on Mahalanobis distances with local covariances
    lm_dm = DiffusionMap(data, kernel_params={'eps': 1e7}, neighbors=250)
    lm_w, lm_v = lm_dm.map(10, 30, local_mahalanobis=True, clusters=10)

    kmeans = KMeans(n_clusters=8)
    kmeans.fit(lm_v)
    lm_y = kmeans.predict(lm_v)

    plt.imshow(lm_y.reshape((50, 145)), origin='lower')
    plt.axis('off')
    plt.savefig('indian_pines_local_mahalanobis.png', bbox_inches='tight')
    plt.show()
