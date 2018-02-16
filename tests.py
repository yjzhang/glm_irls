# test examples, datasets, for irls

import numpy as np
from scipy import sparse
from scipy.io import loadmat
import uncurl

from irls import irls

if __name__ == '__main__':
    # TODO: load dataset
    dat = loadmat('data/10x_pooled_400.mat')
    data = sparse.csc_matrix(dat['data'])
    labs = dat['labels'].flatten()
    genes = uncurl.max_variance_genes(data)
    data_subset = data[genes,:]
    m, w, ll = uncurl.run_state_estimation(data_subset, 8, max_iters=20, inner_max_iters=50)
    b = data_subset[:,0].toarray()
    results = irls(m, b.flatten())
