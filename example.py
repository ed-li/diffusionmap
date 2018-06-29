import numpy as np
from scipy.spatial.distance import mahalanobis
from diffusionmap import DiffusionMap


#def distance(x,y):
#    inverse_covariance_matrix = np.linalg.inv(np.cov(data))


if __name__ == '__main__':
    data = np.load('rice.txt')
    inverse_covariance_matrix = np.linalg.inv(np.cov(data))
    dm = DiffusionMap(data, kernel_params={'distance':mahalanobis(VI=inverse_covariance_matrix)})
    dm.map()