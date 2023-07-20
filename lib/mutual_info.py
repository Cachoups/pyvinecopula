import numpy as np
from math import log
from sklearn.neighbors import NearestNeighbors
import sklearn
import pyvinecopulib as py

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = sklearn.metrics.mutual_info_score(None, None, contingency=c_xy)
    return mi

# data : the data matrix
# bons : the number of bins
def mutual_info_pairs(data, bins):
    n = data.shape[1]
    matMI = np.zeros((n, n))

    for ix in np.arange(n):
        for jx in np.arange(ix+1,n):
            matMI[ix,jx] = calc_MI(data[:,ix], data[:,jx], bins)
    return matMI

def mutual_info_level_(vinecopsearch):
    try : 
        dim = vinecopsearch.d
    except :
        dim=2
    def mc_integrate(func, a, b,level, n = 1000):
        # Monte Carlo integration of given function over domain from a to b (for each parameter)
        # dim: dimensions of function
        c_t = np.ones(n)
        if level == "0":
            tree_indep_list = vinecopsearch.tree_indep
            for i in range(0,len(tree_indep_list)):
                d = len(tree_indep_list[i])
                x_list = np.random.uniform(a, b, (n, d))
                c =  func(vinecopsearch.pair_copula[i], vinecopsearch.structure[i], x_list)
                c_t *=  c
                idx = np.where((c_t==0))
                res = c_t*np.log(c_t)
                res[idx] = 0 # handling log(0) value
                res[np.argwhere(np.isnan(res))] = 0 #handling nan value
        
        domain = np.power(b-a, dim)
        integ = domain * res.sum()/dim
            
        return integ
    def func1(pair_copula, structure, x) :
        sub_model = pv.Vinecop(structure, pair_copula)
        c = sub_model.pdf(x)
        return c
    
    level = vinecopsearch.level
        
    return mc_integrate(func1, 0, 1, level, 100000)

def mutual_info(vinecop):
    def mc_integrate(func, a, b, dim, n = 1000):
        # Monte Carlo integration of given function over domain from a to b (for each parameter)
        # dim: dimensions of function
            
        x_list = np.random.uniform(a, b, (n, dim))
        y = func(x_list)
        y_mean =  y.sum()/len(y)
        domain = np.power(b-a, dim)
            
        integ = domain * y_mean
            
        return integ
    try : 
        d = vinecop.dim
    except :
        d=2
    def func1(x) :
        c = vinecop.pdf(x)
        idx = np.where(c==0)
        res = c*np.log(c)
        res[idx] = 0
        return res
    return mc_integrate(func1, 0, 1, d ,100000)


def py_fast_digamma(x):
    "Faster digamma function assumes x > 0."
    r = 0
    while (x<=5):
        r -= 1/x
        x += 1
    f = 1/(x*x)
    t = f*(-1/12.0 + f*(1/120.0 + f*(-1/252.0 + f*(1/240.0 + f*(-1/132.0
        + f*(691/32760.0 + f*(-1/12.0 + f*3617/8160.0)))))))
    return r + np.log(x) - 0.5/x + t

def unsupervised_knn(data, n):
    N,d = data.shape
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree', p = float("inf")).fit(data)
    distances, indices = nbrs.kneighbors(data)
    return distances, indices

# KSG estimator
def mutual_info_ksg(data, k):
    n,d = data.shape
    digamma_n = py_fast_digamma(n)
    digamma_k = py_fast_digamma(k)
    distances, indices = unsupervised_knn(data, k )
    digamma = np.zeros((n,d))
    for i in range(n):
        for j in range(d):
            x_j = data[i,j]
            epsilon = np.max(np.abs(data[indices[i],j] - x_j))
            count = len(np.where(np.abs(data[:,j] - x_j) <= epsilon)[0])
            digamma[i] = py_fast_digamma(count)
    return (d-1)*digamma_n + digamma_k - (d-1)/k - digamma.sum(1).mean()