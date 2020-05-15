import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis

class Dataset(object):
    def __init__(self, sample_size, x_dim, y_dim, x_mean, noise_level):
        self.sample_size = sample_size
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_mean = x_mean
        self.noise_level = noise_level
        self.X = None
        self._generate()
    
    def _generate(self):
        A = np.random.rand(self.x_dim, self.y_dim)
        X = np.zeros((self.sample_size, self.x_dim))
        for t in range(self.sample_size):
            y_t = np.random.randn(self.y_dim)
            e_t = np.random.normal(0, np.sqrt(self.noise_level), self.x_dim)
            X[t] = np.dot(A, y_t) + self.x_mean + e_t
        self.X = X

def runFA(dataset, k):
    transformer = FactorAnalysis(n_components=k)
    X_transformed = transformer.fit_transform(dataset.X)
    score = transformer.score(dataset.X)
    aic = score * dataset.sample_size - k
    bic = score * dataset.sample_size - np.log(dataset.sample_size) * k * 0.5
    return aic, bic

def modelSelection(dataset, k_range):
    aic_scores = []
    bic_scores = []
    for k in k_range:
        aic, bic = runFA(dataset, k)
        aic_scores.append(aic)
        bic_scores.append(bic)
    bestK_aic = np.argmax(aic_scores) + 1
    bestK_bic = np.argmax(bic_scores) + 1
    return bestK_aic, bestK_bic, np.max(aic), np.max(bic)

def main():
    N_range = [50, 100, 200, 500, 1000, 2000, 5000]
    n_range = [2, 3, 5, 8, 10, 15, 20, 50, 100]
    m_range = [1, 2, 3, 5, 8, 10, 15, 20, 50]
    mu_range = [-2, -1, -0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8, 1, 2]
    square_sigma_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    k_range = [i for i in range(1, 100)]

    # N = ?, n = 10, m = 3, mu = 0, sigma^2 = 0.1
    for N in N_range:
        data = Dataset(sample_size=N, x_dim=10, y_dim=3, x_mean=0, noise_level=0.1)
        bestK_aic, bestK_bic, best_aic, best_bic = modelSelection(data, k_range)
        with open('res/N.txt', 'a') as f:
            f.write('N=%d, n=10, m=3, mu=0, sigma^2=0.1, best_k_aic=%d, best_k_bic=%d, best_aic=%f, best_bic=%f\n' % (N, bestK_aic, bestK_bic, best_aic, best_bic))
    
    # N = 500, n = ?, m = 3, mu = 0, sigma^2 = 0.1
    for n in n_range:
        data = Dataset(sample_size=500, x_dim=n, y_dim=3, x_mean=0, noise_level=0.1)
        bestK_aic, bestK_bic, best_aic, best_bic = modelSelection(data, k_range)
        with open('res/n.txt', 'a') as f:
            f.write('N=500, n=%d, m=3, mu=0, sigma^2=0.1, best_k_aic=%d, best_k_bic=%d, best_aic=%f, best_bic=%f\n' % (n, bestK_aic, bestK_bic, best_aic, best_bic))

    # N = 500, n = 10, m = ?, mu = 0, sigma^2 = 0.1
    for m in m_range:
        data = Dataset(sample_size=500, x_dim=10, y_dim=m, x_mean=0, noise_level=0.1)
        bestK_aic, bestK_bic, best_aic, best_bic = modelSelection(data, k_range)
        with open('res/m.txt', 'a') as f:
            f.write('N=500, n=10, m=%d, mu=0, sigma^2=0.1, best_k_aic=%d, best_k_bic=%d, best_aic=%f, best_bic=%f\n' % (m, bestK_aic, bestK_bic, best_aic, best_bic))

    # N = 500, n = 10, m = 3, mu = ?, sigma^2 = 0.1
    for mu in mu_range:
        data = Dataset(sample_size=500, x_dim=10, y_dim=3, x_mean=mu, noise_level=0.1)
        bestK_aic, bestK_bic, best_aic, best_bic = modelSelection(data, k_range)
        with open('res/mu.txt', 'a') as f:
            f.write('N=500, n=10, m=3, mu=%f, sigma^2=0.1, best_k_aic=%d, best_k_bic=%d, best_aic=%f, best_bic=%f\n' % (mu, bestK_aic, bestK_bic, best_aic, best_bic))

    # N = 500, n = 10, m = 3, mu = 0, sigma^2 = ?
    for square_sigma in square_sigma_range:
        data = Dataset(sample_size=500, x_dim=10, y_dim=3, x_mean=0, noise_level=square_sigma)
        bestK_aic, bestK_bic, best_aic, best_bic = modelSelection(data, k_range)
        with open('res/square_sigma.txt', 'a') as f:
            f.write('N=500, n=10, m=3, mu=0, sigma^2=%f, best_k_aic=%d, best_k_bic=%d, best_aic=%f, best_bic=%f\n' % (square_sigma, bestK_aic, bestK_bic, best_aic, best_bic))

if __name__ == '__main__':
    main()