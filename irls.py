import numpy as np
from scipy.linalg import lstsq
import uncurl

from link_functions import *
from prob_families import *


def irls(A, b, error=Poisson(), link=IDLink(), maxiters=100, tol=1e-8, restrict=True, init=None, eps=1e-10):
    """
    implementation of the iteratively reweighted least squares algorithm to
    solve generalized linear models.
    """
    x = init
    if init is None:
        x = np.random.random(A.shape[1])
    #x = np.zeros(A.shape[1])
    linkfn = link.inv()
    dlink = link.inv_deriv()
    for i in range(maxiters):
        eta = A.dot(x)
        #print(eta)
        g = linkfn(eta)
        gprime = dlink(eta)
        z = eta + (b - g)/(gprime+eps)
        W = gprime**2/(error.var(g)+eps)
        W_ = np.sqrt(W)
        x_old = x
        # TODO: lstsq is unbearably inefficient
        # scipy.sparse.linalg.lsqr might be helpful?
        x_new = lstsq(A*W_[:,np.newaxis], W_*z)[0]
        # truncate to zero (nonnegativity constraint)
        if restrict:
            x_new[x_new<0] = 0
        #print(x_new)
        if np.sqrt(np.sum((x_new - x_old)**2)) < tol:
            return x_new
        x = x_new
    return x

def irls_uncurl(data, k, init_m, init_w, max_iters=10, inner_max_iters=25, tol=1e-8):
    # 1. initialization
    genes, cells = data.shape
    # update w
    w_new = init_w.copy()
    m_new = init_m.copy()
    for i in range(max_iters):
        w_old = w_new.copy()
        m_old = m_new.copy()
        print('iter: ' + str(i))
        for c in range(cells):
            #print(c)
            w_new[:,c] = irls(m_new, data[:,c].toarray().flatten(), init=w_new[:,c], maxiters=inner_max_iters)
            #print(w_new[:,c])
        for g in range(genes):
            #print(g)
            m_new[g,:] = irls(w_new.T, data[g,:].toarray().flatten(), init=m_new[g,:], maxiters=inner_max_iters)
        if tol > 0:
            if np.sqrt(np.sum((w_new - w_old)**2)) < tol and np.sqrt(np.sum((m_new-m_old)**2)) < tol:
                break
        print(uncurl.state_estimation._call_sparse_obj(data, m_new, w_new))
    return m_new, w_new



if __name__ == '__main__':
    from scipy import sparse
    from scipy.io import loadmat
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
    # TODO: load dataset
    dat = loadmat('data/10x_pooled_400.mat')
    data = sparse.csc_matrix(dat['data'])
    labs = dat['labels'].flatten()
    genes = uncurl.max_variance_genes(data) 
    data_subset = data[genes,:] 
    data_subset = uncurl.preprocessing.cell_normalize(data_subset)
    m, w, ll = uncurl.run_state_estimation(data_subset, 8, max_iters=0, inner_max_iters=1)
    m1, w1, ll1 = uncurl.run_state_estimation(data_subset, 8, max_iters=10, inner_max_iters=25)
    b = data_subset[:,0].toarray()
    results = irls(m, b.flatten(), Normal(), IDLink(), restrict=False)
    print(results)
    print(lstsq(m, b.flatten())[0])
    print('poisson error, id link')
    results = irls(m, b.flatten(), tol=1e-8)
    print(results)
    m_new, w_new = irls_uncurl(data_subset, 8, m, w)
    #print('poisson error, log link')
    #results = irls(m, b.flatten(), link=LogLink())
    #print(results)
    print(nmi(w.argmax(0), labs))
    print(nmi(w1.argmax(0), labs))
    print(nmi(w_new.argmax(0), labs))
