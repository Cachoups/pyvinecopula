import numpy as np
import pandas as pd
import scipy
from scipy.stats import ks_2samp
from scipy.spatial import distance
from scipy.spatial.distance import cdist
# Functions for model evaluation

def empcdf_tail(data):
            
    # Compute cdf for alpha in 0 0.1
    alpha = np.linspace(0, 0.1, 1000)
    
    upper = []
    lower = []
    
    for a in alpha:
        lower.append(np.mean((data <= a)))
        upper.append(np.mean((data <= 1-a)))
    return lower, upper



# compute empirical cdf for 2 variables on tail upper and lower
def qqplot(data):
    n,d = data.shape
            
    # Compute cdf observed and simulated data
    w = np.zeros(n)
    
    for i in range(n):
        # d = 2
        w[i] = np.mean(np.stack([data[:,0] <= data[i,0],(data[:,1] <= data[i,1])]).min(0))
    return w


def count_similar_variables(data1, data2, threshold = 0.05):
    count = 0

    for column in data1.columns:
        var1 = data1[column]
        var2 = data2[column]
        _, p_value = ks_2samp(var1, var2)
        if p_value <= threshold:
            count += 1

    return count


def mahalanobis_distance(data1, data2):
    # Calculate the covariance matrix for each dataset
    cov1 = np.cov(data1, rowvar=False)
    cov2 = np.cov(data2, rowvar=False)

    # Calculate the inverse of the covariance matrix for data2
    inv_cov2 = np.linalg.inv(cov2)

    # Calculate the mean difference vector
    mean_diff = np.mean(data1, axis=0) - np.mean(data2, axis=0)

    # Calculate the Mahalanobis distance
    mahalanobis_dist = distance.mahalanobis(mean_diff, np.zeros_like(mean_diff), inv_cov2)

    return mahalanobis_dist

def mahalanobis_distance1(dataset1, dataset2):
    mean_dataset1 = np.mean(dataset1.values, axis=0)
    cov_dataset1 = np.cov(dataset1.values.T)
    inv_cov_dataset1 = np.linalg.inv(cov_dataset1)

    distances = cdist(dataset2.values, mean_dataset1.reshape(1, -1), metric='mahalanobis', VI=inv_cov_dataset1)

    return distances.flatten()