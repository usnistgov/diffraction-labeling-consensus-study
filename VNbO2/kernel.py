""" mlmr.kernel: kernel methods helper functions """
import os
import tempfile
import subprocess
import numpy as np
from sklearn import neighbors
from sklearn import decomposition
from sklearn.metrics import pairwise
from scipy import signal

dtw_bin = os.path.expanduser('~/Software/trillion/src/dtw')

def locally_scaled_rbf(D, nk=7, average_scale=False):
    """ compute locally-scaled RBF kernel given distance matrix D
    D: (n,n) distance matrix
    nk: int use the nk'th nearest neighbor distance to set the scale
    average_scale: use the average nk'th nearest neighbor distance instead local scaling

    compute nearest neighbor distances via brute force
    TODO: switch to ball tree to scale up...
    Zelnik-Manor and Perona: http://www.vision.caltech.edu/lihi/Demos/SelfTuningClustering.html
    """
    N, _ = D.shape

    if nk >= N:
        nk = N-1

    # locally scale by distance to nk'th nearest neighbor
    scale = np.sort(D, axis=1)[:,nk]
    scale = scale[:,None]

    if average_scale:
        scale = np.mean(scale, axis=0)
        print(scale)

    # form the affinity matrix A
    return np.exp(- np.square(D) / (scale @ scale.T))

def locally_scaled_cosine_rbf(X, nk=7, average_scale=False):
    """ compute locally-scaled RBF kernel with cosine distance replacing squared euclidean distance
    D: (n,n) distance matrix
    nk: int use the nk'th nearest neighbor distance to set the scale
    average_scale: use the average nk'th nearest neighbor distance instead local scaling

    compute nearest neighbor distances via brute force
    TODO: switch to ball tree to scale up...
    Zelnik-Manor and Perona: http://www.vision.caltech.edu/lihi/Demos/SelfTuningClustering.html
    """
    D = pairwise.cosine_distances(X)

    N, _ = D.shape

    if nk >= N:
        nk = N-1

    # locally scale by distance to nk'th nearest neighbor
    scale = np.sort(np.sqrt(D), axis=1)[:,nk]
    scale = scale[:,None]

    if average_scale:
        scale = np.mean(scale, axis=0)
        print(scale)

    # form the affinity matrix A
    return np.exp(- D / (scale @ scale.T))

def run_dtw(cwd, datafile, queryfile, m, R):
    """ Run UCR dynamic time warping in a subprocess

    Thanawin Rakthanmanon, Bilson Campana, Abdullah Mueen, Gustavo Batista,
    Brandon Westover, Qiang Zhu, Jesin Zakaria, Eamonn Keogh (2012).
    Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping SIGKDD 2012.
    """
    output = subprocess.check_output([dtw_bin, datafile, queryfile, str(m), str(R)], cwd=cwd)
    outdata = {}
    for line in output.decode().split('\n'):
        if len(line):
            key, val = line.split(' : ')
            if key == 'Distance':
                return float(val)

def dtw_kernel(X, query_length=None, warping_window=0.05, verbose=False, print_freq=100):
    """ run UCR dynamic time warping on data and querydata
    write arrays to files in temporary directory, then parse the output

    X: (N, d) design matrix containing diffraction data or similar
    query_length: DTW query size -- default to matching the full spectrum
    warping_window: float [0.0,1.0], fractional DTW warping window size

    Thanawin Rakthanmanon, Bilson Campana, Abdullah Mueen, Gustavo Batista,
    Brandon Westover, Qiang Zhu, Jesin Zakaria, Eamonn Keogh (2012).
    Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping SIGKDD 2012.
    """
    N, d = X.shape

    if query_length is None:
        query_length = d

    with tempfile.TemporaryDirectory() as tmpdir:

        if verbose:
            print('write all data files to tmpdir...')

        for idx in range(N):
            with open(os.path.join(tmpdir, 'data{}.txt'.format(idx)), 'w') as f:
                print(' ', end='', file=f)
                for x in X[idx]:
                    print(x, end=' ', file=f)

        # run DTW to build up the lower triangular portion of the distance matrix
        # TODO: parallelize this...

        ii, jj = np.tril_indices(N)
        D_dtw = np.zeros((N,N))

        if verbose:
            print('running DTW for {} distance matrix entries'.format(ii.size))

        for count, (i, j) in enumerate(zip(ii, jj)):

            if verbose and count % print_freq == 0:
                print('{} / {} entries computed'.format(count, ii.size))

            D_dtw[i,j] = run_dtw(tmpdir, 'data{}.txt'.format(i), 'data{}.txt'.format(j), query_length, warping_window)

        return D_dtw + D_dtw.T

def wavelet_transform_distance(X):
    N, _ = X.shape

    widths = np.arange(5, 50)

    wt = [
        signal.cwt(x, signal.ricker, widths) for x in X
    ]

    w = np.array(wt)
    Y = w.reshape((N,-1))
    return pairwise.cosine_distances(Y)

def chi2_distance(X):
    """ chi2 distance D = 1/2 sum( (x - y)^2 / (x + y) )

    scikit-learn uses a strange sign convention for chi2 kernel...
    """
    return -0.5 * pairwise.additive_chi2_kernel(X)

def hellinger_kernel(X, Y=None):
    """ compute pairwise hellinger kernel """

    _hellinger = lambda x, y: np.sum(np.sqrt(x*y))

    if Y is None:
        K = pairwise.pairwise_distances(X, X, _hellinger)
    else:
        K = pairwise.pairwise_distances(X, Y, _hellinger)

    return K

def hellinger_distance(X, Y=None):
    """ pairwise hellinger distances """
    _hellinger_dist = lambda x, y: np.linalg.norm(np.sqrt(x) - np.sqrt(y))

    if Y is None:
        D = pairwise.pairwise_distances(X, X, _hellinger_dist)
    else:
        D = pairwise.pairwise_distances(X, Y, _hellinger_dist)

    return D

def kl(P, Q):
    """ Kullback-Leibler divergence (ignores entries with zero support) """
    return np.sum(np.where((P != 0) & (Q != 0), P * np.log(P/Q), 0))

def _js_div(P, Q):
    """ jensen-shannon divergence """
    M = 0.5 * (P + Q)
    return 0.5 * kl(P, M) + kl(Q, M)

def js_divergence(X, Y=None):
    """ construct the pairwise jensen-shannon divergence matrix """
    D = pairwise.pairwise_distances(X, X, _js_div)
    return D
