import numpy as np
from scipy import stats
from itertools import permutations
from tensorly.decomposition import parafac

#%% GENERIC FUNCTIONS

def plot_interv(xmin, xmax, alpha=.05):
    ''' Extend a given interval by a factor alpha, for plotting purposes '''
    delta = alpha*(xmax-xmin)/2
    return xmin-delta, xmax+delta

def get_classif_error(k, partition, true_partition):
    ''' Compute classification error '''
    permut = np.array(list(permutations(range(k))))
    c_err_list = [np.mean(pp[partition] != true_partition) for pp in permut]
    per = permut[np.argmin(c_err_list)]
    per_inv = np.argsort(per)
    c_err = np.min(c_err_list)
    return c_err, per, per_inv

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
    Z = stats.norm.rvs(size=n)/np.sqrt(np.sum(n))
    return P+Z

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

#%% LSD AND ALIGNMENTS

def stieltjes_scalar(z, c, eps, delta=1e-6):
    ''' Stieltjes transform of the LSD '''
    gi = 1j*np.ones(c.size)
    gp, gm = np.sum(gi), 1j
    while np.abs(gp-gm) > delta:
        gi = -c/(eps*(gp-gi)+z)
        gp, gm = np.sum(gi), gp
    return gp, gi

def stieltjes(zz, c, eps, delta=1e-6, maxiter=1000):
    ''' Stieltjes transform of the LSD '''
    gi = 1j*np.ones((c.size, zz.size))
    gp, gm = np.sum(gi, axis=0), 1j*np.ones_like(zz)
    n_iter = 0
    while np.max(np.abs(gp-gm)) > delta and n_iter < maxiter:
        gi = -c[:, None]/(eps*(gp-gi)+zz)
        gp, gm = np.sum(gi, axis=0), gp
        n_iter += 1
    return {'g': gp, 'gi': gi, 'delta': np.max(np.abs(gp-gm)), 'n_iter': n_iter}

def alignments(sigma, c, eps, tol=1e-5):
    ''' Asymptotic singular value and alignments '''
    g, gi = stieltjes_scalar(sigma, c, eps)
    if np.abs(g.imag) > tol:
        return np.nan, np.zeros_like(c)
    q = np.sqrt(1-eps*(gi.real**2)/c)
    beta = (sigma/eps+g.real)/np.prod(q)
    return beta, q

def phase_transition(c, eps, eta=1e-5, delta=1e-5, tol0=1e-3):
    ''' Finds the right edge of the bulk to compute beta and the alignments at phase transition '''
    # Initialisation
    x0 = np.sum(np.sqrt(eps*c))
    z = x0+1j*eta
    g, gi = stieltjes_scalar(z, c, eps, delta)
    while g.imag > delta: # while z is not on the right side of the bulk
        z += x0 # push z to the right
        g, gi = stieltjes_scalar(z, c, eps, delta)
    step = -z.real/10 # initialise step towards left
    # Compute position of the right edge
    while np.abs(step) > delta:
        z += step # make a step
        (g, gi), g_old = stieltjes_scalar(z, c, eps, delta), g
        if (g.imag > tol0 and g_old.imag < tol0) or (g.imag < tol0 and g_old.imag > tol0): # if the edge is crossed
            step = -step/2 # change direction and reduce step size
    # Return the corresponding beta and alignments
    align = np.sqrt(1-eps*(gi.real**2)/c)
    beta = (z.real/eps+g.real)/np.prod(align)
    return beta, align
