import numpy as np
from scipy import stats
from time import time
from tensorly.decomposition import parafac

#%% GENERAL FUNCTIONS

def fixed_point(f, x0, delta=1e-6, time_lim=5):
    ''' Fixed point algorithm '''
    xp, xm = f(x0), x0
    t0 = time()
    while np.max(np.abs(xp-xm)) > delta:
        print(xp)
        xp, xm = f(xp), xp
        if time()-t0 > time_lim:
            xp = np.nan
            break
    return xp

#%% TENSOR OPERATIONS

def outer_vec(vec_list):
    ''' Outer product of d vectors '''
    l = []
    for i, v in enumerate(vec_list):
        l.append(v)
        l.append([i])
    return np.einsum(*l)

def tensor_contraction(X, a_list, axes):
    ''' Contraction of a tensor X against vectors a on given axes. '''
    l = [X, np.arange(len(X.shape))]
    for a, axis in zip(a_list, axes):
        l.append(a)
        l.append([axis])
    return np.einsum(*l)

#%% TENSOR CONSTRUCTION

def make_T(n, x, beta):
    ''' Construction of a tensor T = rank-1 signal + noise '''
    P = beta*outer_vec(x)
    W = stats.norm.rvs(size=n)/np.sqrt(np.sum(n))
    return P+W

def make_B(n, eps):
    ''' Construction of a Bernoulli mask '''
    return stats.bernoulli.rvs(eps, size=n)

#%% TENSOR TRANSFORMATIONS

def Phi(X, a_list):
    ''' Phi mapping '''
    n = X.shape
    d, N = len(n), np.sum(n)
    PhiM = np.zeros((N, N))
    d_range = np.arange(d)
    idx = np.append(0, np.cumsum(n))
    for i in range(d):
        for j in range(i+1, d):
            a_list_ij = a_list[:i]+a_list[i+1:j]+a_list[j+1:]
            idx_arr_ij = d_range[(d_range != i) & (d_range != j)]
            Xij = tensor_contraction(X, a_list_ij, idx_arr_ij)
            PhiM[idx[i]:idx[i+1], idx[j]:idx[j+1]] = Xij
            PhiM[idx[j]:idx[j+1], idx[i]:idx[i+1]] = Xij.T
    return PhiM

def CPD1(X):
    ''' Best rank-1 approximation of X '''
    sigma, svecs = parafac(X, rank=1, normalize_factors=True)
    return sigma[0], [v[:, 0] for v in svecs]

#%% STIELTJES TRANSFORM

def stieltjes(z, c, delta=1e-6):
    gg = 1j*np.ones_like(c)
    gp, gm = np.sum(gg), 1j
    while np.abs(gp-gm) > delta:
        print(gp)
        gg = (gp+z-np.sqrt(4*c+(gp+z)**2))/2
        gp, gm = np.sum(gg), gp
    return gp, gg
