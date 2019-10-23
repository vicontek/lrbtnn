import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def find_max_mode(means, cov, sample_size=1000, quantile=0.1, verbose=False):
    if verbose:
        print('Sampling...')
    sample = np.vstack([multivariate_normal.rvs(means[i], cov, size=sample_size) for i in range(means.shape[0])])
    if verbose:
        print('Estimating bandwith...')
    bandwidth = estimate_bandwidth(sample, quantile=quantile, n_samples=1000, n_jobs=-1)
    if verbose:
        print('Fitting MeanShift...')
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    ms.fit(sample)
    cluster_centers = ms.cluster_centers_
    if verbose:
        plt.scatter(sample[:, 0], sample[:, 1], s=3)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1])
    return max(
        cluster_centers,
        key=lambda center: sum(multivariate_normal.pdf(center, mean=means[i], cov=cov) for i in range(means.shape[0]))
    )


def find_max_mode2(means, cov, sample_size=1000, verbose=False):
    # Dumb way, works bad for high dimensional data
    if verbose:
        print('Sampling...')
    sample = np.vstack([multivariate_normal.rvs(means[i], cov, size=sample_size) for i in range(means.shape[0])])
    if verbose:
        plt.scatter(sample[:, 0], sample[:, 1], s=3)
    if verbose:
        print('Searching...')
    return max(
        sample,
        key=lambda center: sum(multivariate_normal.pdf(center, mean=means[i], cov=cov) for i in range(means.shape[0]))
    )
